import warnings
warnings.filterwarnings("ignore", message="`torch.nn.utils.weight_norm` is deprecated")

import argparse
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import resource

import torch
import torchaudio
# DDP imports removed
from torch.utils.data import DataLoader
# autocast removed - handled by Accelerate
# clip_grad_norm_ removed - handled by Accelerate
from transformers import get_scheduler
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import parametrize
# import bitsandbytes as bnb # Removed for TPU compatibility
from tqdm import tqdm
from huggingface_hub import hf_hub_download
import gc
import wandb
import time
import re
import glob

from accelerate import Accelerator
from accelerate.utils import set_seed

# TPU-specific imports
try:
    import torch_xla.core.xla_model as xm
    HAS_XLA = True
except ImportError:
    HAS_XLA = False

import dac
from dia.config import DiaConfig
from dia.layers import DiaModel
from dia.model import Dia
from dia.audio import build_delay_indices, apply_audio_delay
from dia.dataset import MusicDataset, PreEncodedDACDataset


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


CODEBOOK_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

NONFINITE_HIT_LIMIT = 3
_nonfinite_hits = 0


TAG_SHUFFLE = True
TAG_DROPOUT = 0.0
TAG_LIMIT = None  # type: int | None


def _augment_tags(text: str) -> str:
    try:
        tags = [t.strip() for t in text.split(',') if t.strip()]
        if not tags:
            return text
        if TAG_DROPOUT and TAG_DROPOUT > 0.0:
            kept = [t for t in tags if random.random() > TAG_DROPOUT]
            if kept:
                tags = kept
        if TAG_SHUFFLE and len(tags) > 1:
            random.shuffle(tags)
        if TAG_LIMIT is not None and TAG_LIMIT > 0:
            tags = tags[:TAG_LIMIT]
        return ', '.join(tags)
    except Exception:
        return text


def seed_everything(seed: int):
    """Set seeds for reproducible training."""
    set_seed(seed)
    # Additional manual seeding if needed, but set_seed covers most
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_vocabulary_coverage(dataset, max_valid_token=1023):
    """
    Compute vocabulary coverage statistics for a dataset.
    
    Returns dict with:
        - unique_tokens: set of unique token IDs found
        - coverage_pct: percentage of vocabulary covered (out of 1024 DAC codes)
        - token_counts: Counter of token frequencies
    """
    from collections import Counter
    
    all_tokens = set()
    token_counts = Counter()
    
    logger.info(f"Computing vocabulary coverage for {len(dataset)} samples...")
    
    for i in range(len(dataset)):
        try:
            _, encoded, _ = dataset[i]
            # encoded is (T, C) tensor of audio codes
            tokens = encoded.flatten().tolist()
            # Filter to valid audio tokens (0-1023), exclude special tokens
            valid_tokens = [t for t in tokens if 0 <= t <= max_valid_token]
            all_tokens.update(valid_tokens)
            token_counts.update(valid_tokens)
        except Exception as e:
            logger.warning(f"Error processing sample {i} for vocab coverage: {e}")
            continue
    
    coverage_pct = len(all_tokens) / (max_valid_token + 1) * 100
    
    return {
        'unique_tokens': all_tokens,
        'num_unique': len(all_tokens),
        'coverage_pct': coverage_pct,
        'token_counts': token_counts,
        'total_tokens': sum(token_counts.values()),
    }


def strip_weight_norms(module: torch.nn.Module) -> int:
    """
    Remove weight_norm parametrizations so XLA doesn't fall back to CPU for
    aten::_weight_norm_interface (unsupported op on TPU).
    """
    removed = 0
    for m in module.modules():
        if parametrize.is_parametrized(m) and "weight" in getattr(m, "parametrizations", {}):
            try:
                parametrize.remove_parametrizations(m, "weight", leave_parametrized=False)
                removed += 1
                continue
            except Exception:
                pass
        if hasattr(m, "weight_g") or hasattr(m, "weight_orig"):
            try:
                torch.nn.utils.remove_weight_norm(m)
                removed += 1
            except ValueError:
                pass
    return removed


def verify_dac_encoding(audio_path: Path, dac_model: dac.DAC, seconds: float = 1.0, target_sr: int = 44100):
    """
    Run a quick DAC encode on a short snippet to ensure the pipeline works.
    Handles mono or stereo by encoding each channel separately (DAC expects 1ch).
    Raises on failure; logs the produced code shape on success.
    """
    device = next(dac_model.parameters()).device
    try:
        waveform, sr = torchaudio.load(audio_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load audio for DAC verification: {audio_path}") from e
    
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    
    # Use at most the requested seconds to keep verification fast
    max_samples = int(seconds * target_sr)
    if waveform.shape[1] > max_samples:
        waveform = waveform[:, :max_samples]
    
    # Encode each channel separately; DAC conv expects a single channel input
    num_ch = waveform.shape[0]
    if num_ch > 2:
        raise RuntimeError(f"DAC verification only supports mono/stereo, got {num_ch} channels in {audio_path}")
    
    channel_codes = []
    with torch.no_grad():
        for ch in range(min(num_ch, 2)):
            mono = waveform[ch:ch + 1, :]
            audio_tensor = dac_model.preprocess(mono.unsqueeze(0), target_sr).to(device)
            _, enc, *_ = dac_model.encode(audio_tensor, n_quantizers=None)
            if enc is None or enc.numel() == 0:
                raise RuntimeError("DAC verification produced empty codes.")
            channel_codes.append(enc.squeeze(0).transpose(0, 1))  # (T, 9)

    # If mono, duplicate to both sides so shape matches stereo training path
    if len(channel_codes) == 1:
        enc_stereo = torch.cat([channel_codes[0], channel_codes[0]], dim=1)
    else:
        enc_stereo = torch.cat(channel_codes, dim=1)
    
    logger.info("DAC verification succeeded on %s -> codes shape %s", audio_path, tuple(enc_stereo.shape))
    return enc_stereo


def cleanup_old_checkpoints(output_dir: Path, keep_last_n: int):
    """Keep only the last N checkpoints, delete older ones to save space."""
    if keep_last_n is None:
        return
    
    # Find all checkpoint files (both step and epoch checkpoints)
    step_checkpoints = sorted(glob.glob(str(output_dir / "ckpt_step*.pth")), 
                            key=lambda x: int(re.search(r'ckpt_step(\d+).pth', x).group(1)) if re.search(r'ckpt_step(\d+).pth', x) else 0)
    epoch_checkpoints = sorted(glob.glob(str(output_dir / "ckpt_epoch*.pth")),
                             key=lambda x: int(re.search(r'ckpt_epoch(\d+).pth', x).group(1)) if re.search(r'ckpt_epoch(\d+).pth', x) else 0)
    
    # Keep latest N of each type
    if len(step_checkpoints) > keep_last_n:
        for old_ckpt in step_checkpoints[:-keep_last_n]:
            try:
                os.remove(old_ckpt)
                logger.info(f"Removed old checkpoint: {old_ckpt}")
            except Exception as e:
                logger.warning(f"Failed to remove {old_ckpt}: {e}")
    
    if len(epoch_checkpoints) > keep_last_n:
        for old_ckpt in epoch_checkpoints[:-keep_last_n]:
            try:
                os.remove(old_ckpt)
                logger.info(f"Removed old checkpoint: {old_ckpt}")
            except Exception as e:
                logger.warning(f"Failed to remove {old_ckpt}: {e}")


# Removed DDP setup functions as they are replaced by Accelerator


@dataclass
class TrainConfig:
    epochs: int = 500
    batch_size: int = 4
    grad_accum_steps: int = 1
    learning_rate: float = 1e-5
    weight_decay: float = 0.1
    warmup_steps: int = 100
    unconditional_frac: float = 0.15
    save_step: int = 2000
    seed: int = 786
    runs_dir: Path = Path("runs")
    run_name: str = "dia_finetune_cv"
    output_dir: Path = None
    no_decay_embed: bool = False


def load_train_config(config_path: Path) -> dict:
    import json
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        cfg = json.load(f)
    flat = {}
    flat['experiment_id'] = cfg.get('experiment_id')
    if 'data' in cfg:
        flat['preencoded_dir'] = cfg['data'].get('preencoded_dir')
        flat['audio_folder'] = cfg['data'].get('audio_folder')
        flat['config'] = cfg['data'].get('config')
    if 'training' in cfg:
        flat['batch_size'] = cfg['training'].get('batch_size')
        flat['grad_accum_steps'] = cfg['training'].get('grad_accum_steps')
        flat['epochs'] = cfg['training'].get('epochs')
        flat['learning_rate'] = cfg['training'].get('learning_rate')
        flat['warmup_steps'] = cfg['training'].get('warmup_steps')
        flat['unconditional_frac'] = cfg['training'].get('unconditional_frac')
        flat['weight_decay'] = cfg['training'].get('weight_decay')
    if 'output' in cfg:
        flat['output_dir'] = cfg['output'].get('output_dir')
        flat['run_name'] = cfg['output'].get('run_name')
        flat['save_every'] = cfg['output'].get('save_every')
        flat['save_after_epoch'] = cfg['output'].get('save_after_epoch')
        flat['save_last'] = cfg['output'].get('save_last')
    if 'flags' in cfg:
        flat['scratch'] = cfg['flags'].get('scratch')
        flat['tag_no_shuffle'] = cfg['flags'].get('tag_no_shuffle')
        flat['force_single_gpu'] = cfg['flags'].get('force_single_gpu')
        flat['use_sliding_window'] = cfg['flags'].get('use_sliding_window')
        flat['require_prompts'] = cfg['flags'].get('require_prompts')
        flat['no_decay_embed'] = cfg['flags'].get('no_decay_embed')
    
    # Root level or training level seed
    if 'seed' in cfg:
        flat['seed'] = cfg['seed']
    elif 'training' in cfg and 'seed' in cfg['training']:
        flat['seed'] = cfg['training']['seed']
        
    return {k: v for k, v in flat.items() if v is not None}


def get_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--train_config", type=Path, default=Path("configs/train_config.json"),
                            help="Path to training config JSON (defaults loaded from here, CLI overrides)")
    pre_args, _ = pre_parser.parse_known_args()
    
    cfg_defaults = load_train_config(pre_args.train_config)
    
    parser = argparse.ArgumentParser(description="Train the Dia audio model")
    parser.add_argument("--train_config", type=Path, default=Path("configs/train_config.json"),
                        help="Path to training config JSON (defaults loaded from here, CLI overrides)")
    parser.add_argument("--config",    type=Path, default=Path(cfg_defaults.get('config', 'configs/architecture/model.json')))
    parser.add_argument("--hub_model", type=str,  default="nari-labs/Dia-1.6B")
    parser.add_argument("--local_ckpt", type=str,  default=None)
    parser.add_argument("--audio_folder", type=Path, default=cfg_defaults.get('audio_folder'),
                        help="Path to audio folder (expects audio_prompts folder at same level).")
    parser.add_argument("--preencoded_dir", type=Path, default=cfg_defaults.get('preencoded_dir'),
                        help="Directory with pre-encoded DAC codes (encoded_audio/*.pt) and optional metadata.json.")
    parser.add_argument("--run_name",  type=str,  default=cfg_defaults.get('run_name'))
    parser.add_argument("--output_dir",type=Path, default=cfg_defaults.get('output_dir'),
                        help="Output directory for checkpoints.")
    parser.add_argument("--seed", type=int, default=cfg_defaults.get('seed', 42),
                        help="Random seed for reproducibility.")
    parser.add_argument("--half", action="store_true", help="enable bf16 mixed precision (TPU-safe)")
    parser.add_argument("--compile", action="store_true", help="torch compile model")
    parser.add_argument("--wandb_project", type=str, default="dia-music-finetuning",
                        help="Weights & Biases project name.")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="Weights & Biases entity/team name.")
    parser.add_argument("--save_every", type=int, default=cfg_defaults.get('save_every'),
                        help="Save checkpoint every N steps (overrides TrainConfig.save_step).")
    parser.add_argument("--save_last", type=int, default=cfg_defaults.get('save_last'),
                        help="Keep only the last N checkpoints (e.g., --save_last 4). Saves disk space.")
    parser.add_argument("--force_single_gpu", action="store_true", default=cfg_defaults.get('force_single_gpu', False),
                        help="Force single GPU training even with multiple GPUs available")
    parser.add_argument("--tag_shuffle", action="store_true", default=True,
                        help="Shuffle comma-separated tags in prompts (default: on)")
    parser.add_argument("--tag_no_shuffle", action="store_true", default=cfg_defaults.get('tag_no_shuffle', False),
                        help="Disable tag shuffling (overrides --tag_shuffle)")
    parser.add_argument("--tag_dropout", type=float, default=0.0,
                        help="Per-tag dropout probability in [0,1] (default: 0.0)")
    parser.add_argument("--tag_limit", type=int, default=None,
                        help="Keep at most this many tags after shuffle/dropout (default: unlimited)")
    
    parser.add_argument("--epochs", type=int, default=cfg_defaults.get('epochs', 500),
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=cfg_defaults.get('batch_size', 4),
                        help="Batch size per GPU.")
    parser.add_argument("--grad_accum_steps", type=int, default=cfg_defaults.get('grad_accum_steps', 1),
                        help="Gradient accumulation steps.")
    parser.add_argument("--learning_rate", type=float, default=cfg_defaults.get('learning_rate', 1e-5),
                        help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=cfg_defaults.get('weight_decay', 0.1),
                        help="AdamW weight decay coefficient.")
    parser.add_argument("--warmup_steps", type=int, default=cfg_defaults.get('warmup_steps', 500),
                        help="Number of warmup steps.")
    parser.add_argument("--unconditional_frac", type=float, default=cfg_defaults.get('unconditional_frac'),
                        help="Fraction of unconditional training steps.")
    parser.add_argument("--save_every_epochs", type=int, default=None,
                        help="Save checkpoint at the end of every N epochs (overrides step-based save).")
    parser.add_argument("--save_after_epoch", type=int, default=cfg_defaults.get('save_after_epoch', 0),
                        help="Only start saving checkpoints after this epoch (default: 0, save from start).")
    parser.add_argument("--early_stop_loss", type=float, default=None,
                        help="Early stop when training loss <= this value; saves final checkpoint then exits.")
    parser.add_argument("--skip_dac_verify", action="store_true", default=False,
                        help="Skip one-off DAC encoding verification when using --audio_folder.")
    
    parser.add_argument("--use_sliding_window", action="store_true", default=cfg_defaults.get('use_sliding_window', False),
                        help="Enable sliding window: random cropping for data augmentation (default: off for deterministic training)")
    parser.add_argument("--no_decay_embed", action="store_true", default=cfg_defaults.get('no_decay_embed', False),
                        help="Exclude nn.Embedding parameters from weight decay")
    
    parser.add_argument("--require_prompts", action="store_true", default=cfg_defaults.get('require_prompts', False),
                        help="Fail if audio file is missing a corresponding prompt file (default: skip missing)")
    parser.add_argument("--skip_tags", type=str, default=None,
                        help="Comma-separated list of tags to skip (e.g. 'vocals,speech')")
    parser.add_argument("--scratch", action="store_true", default=cfg_defaults.get('scratch', False),
                        help="Train from scratch (random initialization) instead of loading a checkpoint.")
    parser.add_argument("--no_data_sharding", action="store_true", default=False,
                        help="Disable distributed sampler and replicate data on every device (useful for tiny/overfit datasets).")

    args = parser.parse_args()
    
    if args.output_dir is None:
        parser.error("--output_dir is required (set in train_config.json or via CLI)")
    if args.unconditional_frac is None:
        parser.error("--unconditional_frac is required (set in train_config.json or via CLI)")
    
    return args



def collate_fn(batch, config: DiaConfig, device: torch.device, use_sliding_window: bool = True):
    texts, encodings, waveforms = zip(*batch)

    window_size = config.data.audio_length
    cropped_encodings = []
    for e in encodings:
        if e.size(0) > window_size:
            if use_sliding_window:
                start = random.randint(0, e.size(0) - window_size)
            else:
                start = 0
            cropped_encodings.append(e[start : start + window_size])
        else:
            cropped_encodings.append(e)
    encodings = cropped_encodings

    max_text = config.data.text_length
    pad_tok = config.data.text_pad_value
    text_ids = []
    for txt in texts:
        txt_aug = _augment_tags(txt)
        b_full = txt_aug.encode('utf-8')
        bts = b_full[:max_text]
        arr = list(bts) + [pad_tok] * (max_text - len(bts))
        text_ids.append(torch.tensor(arr, dtype=torch.long))
    src = torch.stack(text_ids).to(device)
    src_pos = torch.arange(max_text, device=device).unsqueeze(0).expand(src.size(0), -1)
    src_pad = src.ne(pad_tok)
    enc_self_attn_mask = (src_pad.unsqueeze(2) & src_pad.unsqueeze(1)).unsqueeze(1)

    # For TPU efficiency, we must have fixed tensor shapes to prevent XLA recompilation.
    # We force the batch length to always match the window_size (config.data.audio_length).
    batch_max = window_size
    
    padded_encodings = []
    for e in encodings:
        if e.size(0) < batch_max:
            pad_length = batch_max - e.size(0)
            pad_value = config.data.audio_pad_value
            padding = torch.full((pad_length, e.size(1)), pad_value, dtype=e.dtype, device=e.device)
            padded_e = torch.cat([e, padding], dim=0)
        else:
            padded_e = e
        padded_encodings.append(padded_e)
    
    seq_lens = [e.size(0) for e in encodings]
    codes = torch.stack(padded_encodings).to(device)

    B, T, C = codes.shape
    t_idx, idxs = build_delay_indices(B, T, C, config.data.delay_pattern)
    delayed = apply_audio_delay(
        codes,
        config.data.audio_pad_value,
        config.data.audio_bos_value,
        (t_idx, idxs)
    )

    max_tgt_len = batch_max + 2
    pad_val = config.data.audio_pad_value
    bos_val = config.data.audio_bos_value
    eos_val = config.data.audio_eos_value

    tgt = torch.full((B, max_tgt_len, C), pad_val, dtype=torch.long, device=device)
    tgt[:, 0, :] = bos_val
    tgt_lens = []
    for i, L in enumerate(seq_lens):
        tgt[i, 1:1 + L, :] = delayed[i, :L, :]
        tgt[i, 1 + L, :] = eos_val
        tgt_lens.append(1 + L + 1)

    tgt_pos = torch.arange(max_tgt_len, device=device).unsqueeze(0).expand(B, -1)
    tgt_pad = tgt.ne(pad_val).any(-1)

    causal = torch.tril(torch.ones((max_tgt_len, max_tgt_len),
                                    dtype=torch.bool,
                                    device=device))
    dec_self_attn_mask = (tgt_pad.unsqueeze(2) & tgt_pad.unsqueeze(1) & causal).unsqueeze(1)
    dec_cross_attn_mask = (tgt_pad.unsqueeze(2) & src_pad.unsqueeze(1)).unsqueeze(1)

    return {
        'src_tokens': src,
        'src_positions': src_pos,
        'enc_self_attn_mask': enc_self_attn_mask,
        'tgt_tokens': tgt,
        'tgt_positions': tgt_pos,
        'dec_self_attn_mask': dec_self_attn_mask,
        'dec_cross_attn_mask': dec_cross_attn_mask,
        'waveforms': waveforms,
        'raw_text': texts[0],
        'tgt_lens': torch.tensor(tgt_lens, dtype=torch.long, device=device),
    }

from functools import partial

def setup_loaders(dataset, dia_cfg: DiaConfig, train_cfg: TrainConfig, use_sliding_window=True):
    collate = partial(collate_fn, config=dia_cfg, device=torch.device("cpu"), use_sliding_window=use_sliding_window)
    
    ds_len = len(dataset)
    if ds_len == 0:
        raise ValueError("Dataset is empty. Check your --audio_folder/--preencoded_dir paths and filtering settings.")

    train_ds = dataset

    # Accelerator handles distribution, we just provide standard loaders
    # If the dataset is smaller than the requested batch size, drop_last=True would yield 0 steps.
    drop_last = True
    if len(train_ds) < train_cfg.batch_size:
        drop_last = False
        logger.warning(
            "Dataset size (%d) is smaller than batch_size (%d); disabling drop_last so training still runs.",
            len(train_ds),
            train_cfg.batch_size,
        )
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=train_cfg.batch_size, 
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
        pin_memory=False,
        drop_last=drop_last,
        persistent_workers=False,
    )
    
    steps_per_epoch = len(train_loader)
    if steps_per_epoch == 0:
        raise ValueError(
            f"DataLoader produced zero steps (len(train_ds)={len(train_ds)}, "
            f"batch_size={train_cfg.batch_size}, drop_last={drop_last}). "
            "Reduce batch_size or disable drop_last."
        )
    train_loader.steps_per_epoch = steps_per_epoch
    
    return train_loader



def setup_optimizer(model, train_cfg):
    norm_types = [
        torch.nn.LayerNorm,
        torch.nn.GroupNorm,
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
    ]
    rmsnorm_cls = getattr(torch.nn.modules.normalization, "RMSNorm", None)
    if rmsnorm_cls is not None:
        norm_types.append(rmsnorm_cls)

    no_decay_params, decay_params = [], []
    seen = set()
    for module in model.modules():
        for name, p in module.named_parameters(recurse=False):
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            is_bias = name.endswith("bias")
            is_norm = any(isinstance(module, nt) for nt in norm_types)
            is_embed = isinstance(module, torch.nn.Embedding)
            lname = name.lower()
            is_lora_alpha = ("lora" in lname and "alpha" in lname) or lname.endswith("lora_alpha")
            if is_bias or is_norm or is_lora_alpha or (train_cfg.no_decay_embed and is_embed):
                no_decay_params.append(p)
            else:
                decay_params.append(p)
    for _, p in model.named_parameters():
        if p.requires_grad and id(p) not in seen:
            decay_params.append(p)
            seen.add(id(p))

    param_groups = [
        {"params": decay_params, "weight_decay": train_cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    # Use standard AdamW for TPU compatibility instead of bnb 8-bit
    opt = optim.AdamW(
        param_groups,
        lr=train_cfg.learning_rate,
        weight_decay=0.0, # Weight decay handled in param_groups
        betas=(0.9, 0.999)
    )
    return opt


def setup_scheduler(opt, train_cfg, steps_per_epoch: int):
    total_training_steps = steps_per_epoch * train_cfg.epochs
    return get_scheduler(
        'cosine', opt,
        num_warmup_steps=train_cfg.warmup_steps // train_cfg.grad_accum_steps,
        num_training_steps=total_training_steps // train_cfg.grad_accum_steps
    )




def train(model, dia_cfg: DiaConfig, dac_model: dac.DAC, dataset, train_cfg: TrainConfig, args, accelerator: Accelerator):
    
    if accelerator.is_main_process:
        train_cfg.output_dir.mkdir(parents=True, exist_ok=True)
        (train_cfg.runs_dir / train_cfg.run_name).mkdir(parents=True, exist_ok=True)
        
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=train_cfg.run_name,
            config={
                "model": "Dia-1.6B",
                "dataset_size": len(dataset) if hasattr(dataset, '__len__') else "streaming",
                "epochs": train_cfg.epochs,
                "batch_size": train_cfg.batch_size,
                "grad_accum_steps": train_cfg.grad_accum_steps,
                "learning_rate": train_cfg.learning_rate,
                "warmup_steps": train_cfg.warmup_steps,
                "unconditional_frac": train_cfg.unconditional_frac,
                "seed": train_cfg.seed,
                "num_processes": accelerator.num_processes,
                "mixed_precision": accelerator.mixed_precision,
            }
        )
    
    # No need for barrier or manual DDP wrap or manual device move
    
    use_sliding_window = args.use_sliding_window
    # Note: setup_loaders now returns a standard DataLoader; Accelerator wraps it later
    train_loader = setup_loaders(dataset, dia_cfg, train_cfg, use_sliding_window)
    opt = setup_optimizer(model, train_cfg)

    # Cache lengths before Accelerator potentially shards loaders
    pre_steps_per_epoch = getattr(train_loader, 'steps_per_epoch', None)
    if pre_steps_per_epoch is None:
        try:
            pre_steps_per_epoch = len(train_loader)
        except Exception:
            pre_steps_per_epoch = None
    pre_dataset_len = None
    try:
        pre_dataset_len = len(train_loader.dataset)
    except Exception:
        pass

    # Decide whether to shard the dataloaders; for tiny datasets (e.g., 1 sample) we disable sharding
    shard_dataloaders = not args.no_data_sharding
    if pre_dataset_len == 1:
        shard_dataloaders = False
        if accelerator.is_main_process:
            logger.warning("Dataset has exactly 1 sample; disabling sharding so every process reads it.")
    if shard_dataloaders and pre_dataset_len is not None and pre_dataset_len < accelerator.num_processes:
        shard_dataloaders = False
        if accelerator.is_main_process:
            logger.warning(
                "Dataset has %d samples but %d processes; disabling data sharding so each device sees all data.",
                pre_dataset_len,
                accelerator.num_processes,
            )

    # PREPARE EVERYTHING WITH ACCELERATOR
    if shard_dataloaders:
        model, opt, train_loader = accelerator.prepare(
            model, opt, train_loader
        )
        use_xla_native = False
    else:
        # No sharding: keep original loaders; run native XLA stepping
        model = model.to(accelerator.device)
        use_xla_native = True
        if pre_steps_per_epoch is not None:
            train_loader.steps_per_epoch = pre_steps_per_epoch

    # Note: Don't wrap with MpDeviceLoader when using Accelerate - they conflict
    # Accelerate should handle TPU data loading internally

    model.train()
    
    # CRITICAL FOR TPU: Initialize optimizer state before training to avoid
    # graph recompilation on step 2. AdamW lazily creates momentum/variance
    # tensors on first opt.step(), which changes the graph structure.
    if HAS_XLA:
        for param_group in opt.param_groups:
            for p in param_group['params']:
                if p.requires_grad:
                    state = opt.state[p]
                    if len(state) == 0:
                        device = p.device
                        state['step'] = torch.zeros((), device=device)
                        state['exp_avg'] = torch.zeros_like(p, device=device)
                        state['exp_avg_sq'] = torch.zeros_like(p, device=device)

    if shard_dataloaders:
        steps_per_epoch = getattr(train_loader, 'steps_per_epoch', None)
        if steps_per_epoch is None:
            try:
                steps_per_epoch = len(train_loader)
            except Exception:
                steps_per_epoch = None
        dataset_len = None
        try:
            dataset_len = len(train_loader.dataset)
        except Exception:
            dataset_len = pre_dataset_len
    else:
        steps_per_epoch = pre_steps_per_epoch
        dataset_len = pre_dataset_len

    if steps_per_epoch is None or steps_per_epoch == 0:
        raise RuntimeError(
            f"Training dataloader is empty (steps_per_epoch={steps_per_epoch}, dataset_len={dataset_len}). "
            "Check input paths/filters; TPU runs cannot proceed without batches."
        )
    # Keep an explicit attribute for downstream logging regardless of sharding mode
    try:
        train_loader.steps_per_epoch = steps_per_epoch
    except Exception:
        pass

    sched = setup_scheduler(opt, train_cfg, steps_per_epoch)

    stop_training = False
    for epoch in range(train_cfg.epochs):
        # No manual sampler set_epoch needed; Accelerator handles it if it wrapped the loader
        
        loss = 0.0 # Initialize loss to avoid UnboundLocalError if loop is empty
        
        if accelerator.is_main_process:
            loader_iter = tqdm(
                train_loader,
                desc=f"E{epoch+1}",
                total=steps_per_epoch,
                disable=not accelerator.is_local_main_process
            )
        else:
            loader_iter = train_loader
            
        for step, batch in enumerate(loader_iter):
            if not shard_dataloaders:
                batch = {k: (v.to(accelerator.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            global_step = epoch * (steps_per_epoch or 0) + step
            
            batch_start = time.time()
            
            # Updated train step signature
            loss = train_step(model, batch, dia_cfg, train_cfg, opt, sched, step, global_step, accelerator, use_xla_native)
            
            total_step_time = time.time() - batch_start

            if accelerator.is_main_process:
                # VRAM stats might be GPU specific, but let's keep basic logging
                # For TPU we might not get torch.cuda stats
                if torch.cuda.is_available():
                    cur_alloc = torch.cuda.memory_allocated()
                    peak_alloc = torch.cuda.max_memory_allocated()
                    cur_gb  = cur_alloc  / 1024**3
                    peak_gb = peak_alloc / 1024**3
                    vram_str = f"{cur_gb:.2f}/{peak_gb:.2f}"
                    torch.cuda.reset_peak_memory_stats()
                else:
                    vram_str = "N/A"
                
                current_lr = sched.get_last_lr()[0]
                
                if isinstance(loader_iter, tqdm):
                    loader_iter.set_postfix({
                        'loss': f"{loss:.4f}",
                        'VRAM (GB)': vram_str,
                        'step_time': f"{total_step_time:.1f}s"
                    })
                
                # Log training metrics
                wandb.log({
                    'train_loss': loss,
                    'learning_rate': current_lr,
                }, step=global_step)

            should_save = False
            past_save_threshold = (epoch + 1) > args.save_after_epoch
            if args.save_every_epochs is None and past_save_threshold:
                if args.save_every is not None:
                    should_save = global_step > 0 and global_step % args.save_every == 0
                else:
                    should_save = global_step > 0 and global_step % train_cfg.save_step == 0
            
            if should_save:
                # Wait for everyone before saving
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    ckpt = train_cfg.output_dir / f"ckpt_step{global_step}.pth"
                    unwrapped_model = accelerator.unwrap_model(model)
                    torch.save(unwrapped_model.state_dict(), ckpt)
                    logger.info(f"Saved checkpoint: {ckpt}")
                    
                    # Cleanup old checkpoints
                    cleanup_old_checkpoints(train_cfg.output_dir, args.save_last)

            if args.early_stop_loss is not None:
                # Check local loss condition
                local_stop = 1.0 if loss <= args.early_stop_loss else 0.0
                flag = torch.tensor(local_stop, device=accelerator.device)
                # Reduce: if any process wants to stop, all stop (max > 0)
                flag = accelerator.reduce(flag, reduction="max")
                
                if flag.item() > 0.5:
                    if accelerator.is_main_process:
                        # Save final checkpoints
                        final_ckpt = train_cfg.output_dir / f"ckpt_final_step{global_step}.pth"
                        unwrapped_model = accelerator.unwrap_model(model)
                        torch.save(unwrapped_model.state_dict(), final_ckpt)
                        latest_ckpt = train_cfg.output_dir / "latest.pth"
                        torch.save(unwrapped_model.state_dict(), latest_ckpt)
                        logger.info(f"Early stop triggered (loss <= {args.early_stop_loss}). Saved final checkpoints.")
                        cleanup_old_checkpoints(train_cfg.output_dir, args.save_last)
                    
                    stop_training = True

        if stop_training:
            break

        if accelerator.is_main_process:
            is_last_epoch = (epoch + 1) == train_cfg.epochs
            past_save_threshold = (epoch + 1) > args.save_after_epoch
            
            should_save_epoch = False
            if args.save_every_epochs is not None:
                should_save_epoch = ((epoch + 1) % args.save_every_epochs == 0) and past_save_threshold
            else:
                should_save_epoch = (args.save_every is None and past_save_threshold) or is_last_epoch
            
            if should_save_epoch:
                ckpt_e = train_cfg.output_dir / f"ckpt_epoch{epoch+1}.pth"
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                torch.save(unwrapped_model.state_dict(), ckpt_e)
                logger.info(f"Saved end-of-epoch checkpoint: {ckpt_e}")
                if is_last_epoch:
                    latest_ckpt = train_cfg.output_dir / "latest.pth"
                    torch.save(unwrapped_model.state_dict(), latest_ckpt)
                    logger.info(f"Saved latest checkpoint: {latest_ckpt}")
                
                # Cleanup old checkpoints
                cleanup_old_checkpoints(train_cfg.output_dir, args.save_last)
    
    accelerator.wait_for_everyone()


def train_step(model, batch, dia_cfg, train_cfg, opt, sched, step, global_step, accelerator: Accelerator, use_xla_native: bool):
    global _nonfinite_hits

    gen_val = ((global_step * 997 + train_cfg.seed) % 10000) / 10000.0
    if gen_val < train_cfg.unconditional_frac:
        pad_tok = dia_cfg.data.text_pad_value
        batch['src_tokens'].fill_(pad_tok)
        batch['enc_self_attn_mask'] = torch.zeros_like(batch['enc_self_attn_mask'])
        batch['dec_cross_attn_mask'] = torch.zeros_like(batch['dec_cross_attn_mask'])

    # Manual gradient accumulation (like original Nari Labs script)
    # Don't use accelerator.accumulate() - it causes graph fragmentation on TPU
    with accelerator.autocast():
        logits = model(
            src_BxS=batch['src_tokens'],
            tgt_BxTxC=batch['tgt_tokens'],
            src_positions=batch['src_positions'],
            tgt_positions=batch['tgt_positions'],
            enc_self_attn_mask=batch['enc_self_attn_mask'],
            dec_self_attn_mask=batch['dec_self_attn_mask'],
            dec_cross_attn_mask=batch['dec_cross_attn_mask'],
            enable_dropout=False,
        )
        # Note: tgt_lens not used for TPU - we rely on audio_token_mask and ignore_index
        
        logits = logits[:, :-1]
        target = batch['tgt_tokens'][:, 1:, :]
        
        B, Tm1, C = target.shape
        pad_val = dia_cfg.data.audio_pad_value
        
        # For TPU: Use fixed-shape operations only.
        # Don't use tgt_lens for masking - let ignore_index handle padding.
        # The audio_token_mask will exclude special tokens (BOS, EOS, PAD).
        
        channel_weights = []
        num_groups = C // 9
        if num_groups > 0:
            for _ in range(num_groups):
                channel_weights.extend(CODEBOOK_WEIGHTS)
        else:
            channel_weights = [1.0] * C

        loss_c = 0.0
        _, _, _, V = logits.size()
        
        # Only compute loss on valid audio tokens (0-1023), exclude special tokens
        audio_token_mask = (target >= 0) & (target <= 1023)
        
        # For TPU: Set non-audio tokens to pad_val so cross_entropy ignores them
        target_masked = torch.where(audio_token_mask, target, torch.full_like(target, pad_val))
        
        for c, w in enumerate(channel_weights):
            lc = logits[:, :, c, :].reshape(-1, V)
            tc = target_masked[:, :, c].reshape(-1)
            loss_c += w * F.cross_entropy(
                lc, tc,
                ignore_index=pad_val
            )
        loss = loss_c / sum(channel_weights)

    # Scale loss for gradient accumulation
    loss = loss / train_cfg.grad_accum_steps

    loss_detached = loss.detach()
    if not torch.isfinite(loss_detached):
        _nonfinite_hits += 1
        logger.warning(
            f"Non-finite loss at step {global_step} (hit {_nonfinite_hits}/{NONFINITE_HIT_LIMIT}); skipping backward"
        )
        opt.zero_grad()
        if HAS_XLA:
            xm.mark_step()
        if _nonfinite_hits >= NONFINITE_HIT_LIMIT:
            raise RuntimeError(f"Aborting: non-finite loss encountered {NONFINITE_HIT_LIMIT} times.")
        return float('nan')

    # Backward pass
    if use_xla_native and HAS_XLA:
        loss.backward()
    else:
        accelerator.backward(loss)
    
    # Manual gradient accumulation: only step optimizer every grad_accum_steps
    if (step + 1) % train_cfg.grad_accum_steps == 0:
        if use_xla_native and HAS_XLA:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            xm.optimizer_step(opt, barrier=False)
        else:
            accelerator.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
        sched.step()
        opt.zero_grad()
    
    # CRITICAL for TPU: mark_step tells XLA to execute the accumulated graph
    if HAS_XLA:
        xm.mark_step()

    # Return loss value
    return float(loss_detached) * train_cfg.grad_accum_steps


def run_training(args):
    # Initialize Accelerator
    # For TPU: Don't pass gradient_accumulation_steps - we handle it manually
    # to avoid graph fragmentation from accelerator.accumulate()
    accelerator = Accelerator(
        mixed_precision="bf16" if args.half else "no"
    )
    
    # Set seed
    seed_everything(args.seed)
    
    # Optional: improve bf16 throughput on Ampere+ GPUs
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    
    if accelerator.is_main_process:
        logger.info(f"Accelerator initialized. Device: {accelerator.device}, Num processes: {accelerator.num_processes}")

    dia_cfg = DiaConfig.load(args.config)
    
    global TAG_SHUFFLE, TAG_DROPOUT, TAG_LIMIT
    TAG_SHUFFLE = False if getattr(args, 'tag_no_shuffle', False) else bool(getattr(args, 'tag_shuffle', True))
    TAG_DROPOUT = float(getattr(args, 'tag_dropout', 0.0))
    TAG_LIMIT = getattr(args, 'tag_limit', None)

    if accelerator.is_main_process:
        print(f"Loaded config from: {args.config}")
        print("Loading DAC model...")
        
    # Load DAC model
    # We need it on the correct device
    with accelerator.main_process_first():
        dac_ckpt = dac.utils.download()
    dac_model = dac.DAC.load(dac_ckpt).eval()
    removed_dac_wn = strip_weight_norms(dac_model)
    dac_model = dac_model.to(accelerator.device)
    if accelerator.is_main_process and removed_dac_wn:
        logger.info("Removed %d weight_norm wrappers from DAC model for XLA compatibility", removed_dac_wn)

    use_sliding_window = args.use_sliding_window
    allow_empty_prompts = args.unconditional_frac >= 1.0 and not args.require_prompts
    if allow_empty_prompts and accelerator.is_main_process:
        logger.warning("unconditional_frac >= 1.0; allowing missing prompts by using empty strings.")
    if args.preencoded_dir:
        dataset = PreEncodedDACDataset(args.preencoded_dir, dia_cfg, use_sliding_window)
    elif args.audio_folder:
        skip_tags_list = [t.strip() for t in args.skip_tags.split(',')] if args.skip_tags else None
        dataset = MusicDataset(args.audio_folder, dia_cfg, dac_model, use_sliding_window,
                             ignore_missing_prompts=not args.require_prompts,
                             skip_tags=skip_tags_list,
                             allow_empty_prompts=allow_empty_prompts)
        # Quick one-off verification that DAC encoding works on the current environment
        if accelerator.is_main_process and not args.skip_dac_verify:
            try:
                sample_audio = dataset.valid_files[0]
                verify_dac_encoding(sample_audio, dac_model)
            except Exception as e:
                logger.error("DAC encoding verification failed: %s", e)
                raise
    else:
        raise ValueError("Must specify either --audio_folder or --preencoded_dir")

    train_cfg = TrainConfig(
        epochs = args.epochs,
        batch_size = args.batch_size,
        grad_accum_steps = args.grad_accum_steps,
        learning_rate = args.learning_rate,
        weight_decay = args.weight_decay,
        warmup_steps = args.warmup_steps,
        unconditional_frac = args.unconditional_frac,
        run_name = args.run_name or TrainConfig.run_name,
        output_dir = args.output_dir,
        seed = args.seed,
        no_decay_embed = args.no_decay_embed,
    )

    model = DiaModel(dia_cfg)

    if args.scratch:
        if accelerator.is_main_process:
            print("Initializing model from scratch (random weights)...")
        if hasattr(model, '_init_weights'):
            model._init_weights()
        expanded_stereo = False
    else:
        if args.local_ckpt:
            ckpt_file = args.local_ckpt
        else:
            ckpt_file = hf_hub_download(args.hub_model, filename="dia-v0_1.pth")
        
        state = torch.load(ckpt_file, map_location="cpu")
        expanded_stereo = False

        try:
            key = "decoder.logits_dense.weight"
            if key in state:
                expected_W = model.decoder.logits_dense.weight
                W_ckpt = state[key]
                if (
                    W_ckpt is not None
                    and expected_W is not None
                    and W_ckpt.dim() == 3
                    and expected_W.dim() == 3
                    and W_ckpt.shape[0] == expected_W.shape[0]
                    and W_ckpt.shape[2] == expected_W.shape[2]
                    and W_ckpt.shape[1] != expected_W.shape[1]
                ):
                    if W_ckpt.shape[1] * 2 == expected_W.shape[1]:
                        state[key] = torch.cat([W_ckpt, W_ckpt], dim=1)
                        expanded_stereo = True
                        if accelerator.is_main_process:
                            logger.info(
                                f"Expanded {key} from {tuple(W_ckpt.shape)} to {tuple(state[key].shape)} by duplication"
                            )
                    else:
                        del state[key]
                        if accelerator.is_main_process:
                            logger.warning(
                                f"Removed {key} from checkpoint due to incompatible shape {tuple(W_ckpt.shape)} -> expected {tuple(expected_W.shape)}"
                            )
        except Exception as e:
            if accelerator.is_main_process:
                logger.warning(f"While adapting checkpoint weights: {e}")
        
        missing, unexpected = model.load_state_dict(state, strict=False)
        if accelerator.is_main_process:
            logger.info(f"Loaded checkpoint with strict=False; missing={len(missing)}, unexpected={len(unexpected)}")
        
        if expanded_stereo:
            try:
                if dia_cfg.data.channels == 18:
                    if accelerator.is_main_process:
                        logger.info("Warm-starting stereo channels (copying Left -> Right)...")
                    if hasattr(model.decoder, "embeddings") and len(model.decoder.embeddings) >= 18:
                        for i in range(9, 18):
                            model.decoder.embeddings[i].weight.data.copy_(model.decoder.embeddings[i - 9].weight.data)
                    W = model.decoder.logits_dense.weight
                    if W.dim() == 3 and W.shape[1] >= 18:
                        W.data[:, 9:18, :].copy_(W.data[:, 0:9, :])
            except Exception as e:
                if accelerator.is_main_process:
                    logger.warning(f"Stereo warm-start duplication skipped: {e}")
        else:
            if dia_cfg.data.channels == 18 and accelerator.is_main_process:
                logger.info("Skipping stereo warm-start duplication (loaded checkpoint appears to be stereo already)")
        del state
        gc.collect()

        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        if accelerator.is_main_process:
            logger.info(f"Rank {accelerator.process_index} after checkpoint load RSS ~{rss_mb:.1f} MB (PID {os.getpid()})")

    removed_model_wn = strip_weight_norms(model)
    if accelerator.is_main_process and removed_model_wn:
        logger.info("Removed %d weight_norm wrappers from Dia model for XLA compatibility", removed_model_wn)

    # Optionally freeze encoder
    try:
        if train_cfg.unconditional_frac >= 0.9:
            for p in model.encoder.parameters():
                p.requires_grad = False
            if accelerator.is_main_process:
                print("Frozen encoder parameters due to high unconditional_frac")
    except Exception:
        pass

    if args.compile:
        if torch.cuda.is_available() and not HAS_XLA:
            model = torch.compile(model, backend="inductor")
        else:
            if accelerator.is_main_process:
                logger.warning("Skipping --compile: torch.compile(inductor) is only supported on CUDA (not TPU/XLA).")
    
    accelerator.wait_for_everyone()
    
    # Launch training
    train(model, dia_cfg, dac_model, dataset, train_cfg, args, accelerator)


def main():
    args = get_args()
    run_training(args)


if __name__ == "__main__":
    main()
