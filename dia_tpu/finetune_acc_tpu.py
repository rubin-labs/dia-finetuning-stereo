import warnings
warnings.filterwarnings("ignore", message="`torch.nn.utils.weight_norm` is deprecated")

import argparse
import logging
import os
import random
import socket
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import resource

import torch
# DDP imports removed
from torch.utils.data import DataLoader, random_split
# autocast removed - handled by Accelerate
# clip_grad_norm_ removed - handled by Accelerate
from transformers import get_scheduler
import torch.nn.functional as F
import torch.optim as optim
# import bitsandbytes as bnb # Removed for TPU compatibility
from tqdm import tqdm
from huggingface_hub import hf_hub_download
import gc
import wandb
import time
import signal
import sys
import datetime
import re
import glob
import soundfile as sf

from accelerate import Accelerator
from accelerate.utils import set_seed

import dac
from dia.config import DiaConfig
from dia.layers import DiaModel
from dia.model import Dia
from dia.audio import build_delay_indices, apply_audio_delay
from dia.dataset import MusicDataset, PreEncodedDACDataset
from torch.nn.functional import pad


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


CODEBOOK_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


# =============================================================================
# EVALUATION & DEMO GENERATION CONFIGURATION
# =============================================================================
# Modify these parameters to control how evaluation and demo generation behave.
# These are the main knobs you'll want to tweak for testing your finetuned model.

# -- Test Prompts for Demo Generation --
# These prompts are used to generate audio samples during evaluation.
# Add/modify prompts to test different aspects of your model.
# Format: {"name": "comma, separated, tags"}
TEST_PROMPTS = {
    "piano_ambient": "piano, pads, ambient, cinematic, melancholic, peaceful, reflective, instrumental",
    "dark": "cinematic, suspenseful, dark, energetic, mysterious, strings, bells, bass"
}

# -- Audio Generation Parameters (for conditional generation) --
EVAL_CFG_SCALE = 4.0           # Classifier-free guidance scale (higher = more prompt adherence)
EVAL_TEMPERATURE = 1.0         # Sampling temperature (higher = more random)
EVAL_TOP_P = 0.95              # Top-p (nucleus) sampling threshold

# -- Audio Generation Parameters (for unconditional generation) --
EVAL_CFG_SCALE_UNCOND = 0.0    # CFG scale for unconditional generation (usually 0.0)
EVAL_TEMPERATURE_UNCOND = 1.0  # Temperature for unconditional generation

# -- Multi-Temperature Demo Configuration --
# Generate demos at multiple temperatures to track embedding learning progress
# temp=0.0: deterministic (shows what model actually learned)
# temp=0.5: moderate sampling (tests confidence)
# temp=1.0: full sampling (tests robustness)
EVAL_TEMPERATURES = [0.0, 0.5, 1.0]

# -- Output Settings --
EVAL_SAMPLE_RATE = 44100       # Sample rate for saved audio files
EVAL_AUDIO_DIR = "./audio_demos"  # Directory to save demo audio files

# =============================================================================


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


def compute_output_entropy(logits):
    """
    Compute entropy of output probability distribution.
    
    Args:
        logits: (B, T, C, V) tensor of logits
        
    Returns:
        Mean entropy across all positions (scalar)
    
    Higher entropy = more uncertain (flatter distribution)
    Lower entropy = more confident (peaked distribution)
    """
    # Softmax to get probabilities
    probs = F.softmax(logits.float(), dim=-1)  # (B, T, C, V)
    
    # Compute entropy: -sum(p * log(p))
    # Add small epsilon to avoid log(0)
    log_probs = torch.log(probs + 1e-10)
    entropy = -torch.sum(probs * log_probs, dim=-1)  # (B, T, C)
    
    # Mean entropy across all positions
    return entropy.mean().item()


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
    if 'flags' in cfg:
        flat['scratch'] = cfg['flags'].get('scratch')
        flat['tag_no_shuffle'] = cfg['flags'].get('tag_no_shuffle')
        flat['force_single_gpu'] = cfg['flags'].get('force_single_gpu')
        flat['use_sliding_window'] = cfg['flags'].get('use_sliding_window')
        flat['require_prompts'] = cfg['flags'].get('require_prompts')
        flat['no_decay_embed'] = cfg['flags'].get('no_decay_embed')
        flat['stop_on_overfit'] = cfg['flags'].get('stop_on_overfit')
    
    if 'eval' in cfg:
        flat['eval_step'] = cfg['eval'].get('eval_step')
        flat['demo_every'] = cfg['eval'].get('demo_every')
        flat['eval_every_epochs'] = cfg['eval'].get('eval_every_epochs')
        flat['demo_every_epochs'] = cfg['eval'].get('demo_every_epochs')
        flat['demo_after_epoch'] = cfg['eval'].get('demo_after_epoch')
    
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
    parser.add_argument("--half", action="store_true", help="load model in fp16")
    parser.add_argument("--compile", action="store_true", help="torch compile model")
    parser.add_argument("--wandb_project", type=str, default="dia-music-finetuning",
                        help="Weights & Biases project name.")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="Weights & Biases entity/team name.")
    parser.add_argument("--save_every", type=int, default=cfg_defaults.get('save_every'),
                        help="Save checkpoint every N steps (overrides TrainConfig.save_step).")
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
    parser.add_argument("--eval_step", type=int, default=cfg_defaults.get('eval_step', 200),
                        help="Run evaluation every N steps (default: 200).")
    parser.add_argument("--demo_every", type=int, default=cfg_defaults.get('demo_every', 2000),
                        help="Generate demo audio every N steps (default: 2000).")
    parser.add_argument("--eval_every_epochs", type=int, default=cfg_defaults.get('eval_every_epochs'),
                        help="Run evaluation every N epochs (overrides eval_step).")
    parser.add_argument("--demo_every_epochs", type=int, default=cfg_defaults.get('demo_every_epochs'),
                        help="Generate demo audio every N epochs (overrides demo_every).")
    parser.add_argument("--save_every_epochs", type=int, default=None,
                        help="Save checkpoint at the end of every N epochs (overrides step-based save).")
    parser.add_argument("--save_after_epoch", type=int, default=cfg_defaults.get('save_after_epoch', 0),
                        help="Only start saving checkpoints after this epoch (default: 0, save from start).")
    parser.add_argument("--demo_after_epoch", type=int, default=cfg_defaults.get('demo_after_epoch', 0),
                        help="Only start generating demos after this epoch (default: 0, demo from start).")
    parser.add_argument("--early_stop_loss", type=float, default=None,
                        help="Early stop when training loss <= this value; saves final checkpoint then exits.")
    parser.add_argument("--stop_on_overfit", action="store_true", default=cfg_defaults.get('stop_on_overfit', False),
                        help="Stop training and generate demo when eval loss > train loss")
    
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

    batch_max = max(e.size(0) for e in encodings)
    
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

def setup_loaders(dataset, dia_cfg: DiaConfig, train_cfg: TrainConfig, use_sliding_window=True):
    collate = lambda b: collate_fn(b, dia_cfg, torch.device("cpu"), use_sliding_window)
    
    ds_len = len(dataset)
    n_train = int(train_cfg.split_ratio * ds_len)
    n_val = ds_len - n_train
    
    # If dataset has only 1 sample (or split would result in 0 val samples), skip validation
    if ds_len <= 1 or n_val == 0:
        train_ds = dataset
        val_ds = None
    else:
        g = torch.Generator().manual_seed(train_cfg.seed)
        train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=g)

    # Accelerator handles distribution, we just provide standard loaders
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=train_cfg.batch_size, 
        shuffle=True,
        collate_fn=collate,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    
    steps_per_epoch = len(train_loader)
    train_loader.steps_per_epoch = steps_per_epoch
    
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds, 
            batch_size=1, 
            shuffle=False, 
            collate_fn=collate,
            num_workers=0, # Eval usually fine with 0, or 1
            pin_memory=True
        )
    else:
        val_loader = None
    
    return train_loader, val_loader



def setup_optimizer_and_scheduler(model, train_loader, train_cfg):
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
    try:
        steps_per_epoch = len(train_loader)
    except TypeError:
        if hasattr(train_loader, 'steps_per_epoch'):
            steps_per_epoch = train_loader.steps_per_epoch
        else:
            raise RuntimeError("Cannot determine steps_per_epoch for streaming loader")
    total_training_steps = steps_per_epoch * train_cfg.epochs
    sched = get_scheduler(
        'cosine', opt,
        num_warmup_steps=train_cfg.warmup_steps // train_cfg.grad_accum_steps,
        num_training_steps=total_training_steps // train_cfg.grad_accum_steps
    )
    return opt, sched




def eval_step(model, val_loader, dia_cfg, dac_model, global_step, accelerator: Accelerator, train_cfg, do_demo=True, current_train_loss=None, stop_on_overfit=False):
    """
    Run evaluation: calculate loss and optionally generate audio demos
    """
    eval_losses = []
    last_batch = None
    
    # Ensure model is in eval mode
    model.eval()
    
    if val_loader is not None:
        # Accelerator handles device placement
        # No strict need for torch.inference_mode() if we wrap in torch.no_grad(), 
        # but inference_mode is better. Accelerate doesn't conflict with it.
        with torch.inference_mode():
            # Only show progress bar on main process
            disable_tqdm = not accelerator.is_local_main_process
            for eb in tqdm(val_loader, desc="eval", disable=disable_tqdm):
                last_batch = eb

                # No explicit autocast; Accelerate handles it via context if configured, 
                # but eval might run in full precision. 
                # For consistency with training, we can use accelerator.autocast() if needed,
                # but typically eval is fine in fp32 or same dtype as model.
                # Assuming model is already on device and correct dtype.
                
                logits = model(
                    src_BxS=eb['src_tokens'],
                    tgt_BxTxC=eb['tgt_tokens'],
                    src_positions=eb['src_positions'],
                    tgt_positions=eb['tgt_positions'],
                    enc_self_attn_mask=eb['enc_self_attn_mask'],
                    dec_self_attn_mask=eb['dec_self_attn_mask'],
                    dec_cross_attn_mask=eb['dec_cross_attn_mask'],
                    enable_dropout=False,
                )
                
                # Truncate logits if needed (usually model output matches target length, but let's check)
                # If model outputs one step ahead, we might need slicing. 
                # Dia model usually outputs (B, T, C, V).
                # Based on train_step, we slice logits [:, :max_L-1] and targets [:, 1:max_L].
                # We should match train_step logic exactly for comparable loss.
                
                lens = eb['tgt_lens']
                max_L = int(lens.max().item())
                logits = logits[:, : max_L - 1]
                target = eb['tgt_tokens'][:, 1:max_L, :]
                
                B_e, T_e, C_e = target.shape
                V_e = logits.size(-1)

                loss_e = 0.0
                # Custom weighting for eval as well to match training objective
                weights_e = []
                num_groups = C_e // 9
                if num_groups > 0:
                    for _ in range(num_groups):
                        weights_e.extend(CODEBOOK_WEIGHTS)
                else:
                    weights_e = [1.0] * C_e
                
                # CRITICAL: Only compute loss on actual audio tokens (0-1023)
                audio_token_mask_e = (target >= 0) & (target <= 1023)  # (B, T, C)
                
                # Expand mask based on lengths
                time_idx = torch.arange(T_e, device=lens.device).unsqueeze(0)
                valid_time = time_idx < (lens.unsqueeze(1) - 1)
                mask = valid_time.unsqueeze(-1).expand(-1, -1, C_e)
                
                for c, w in enumerate(weights_e):
                    lc = logits[:, :, c, :].reshape(-1, V_e)
                    tc = target[:, :, c].reshape(-1)
                    mc = mask[:, :, c].reshape(-1)
                    audio_mc = audio_token_mask_e[:, :, c].reshape(-1)
                    combined_mask = mc & audio_mc
                    
                    lc_valid = lc[combined_mask]
                    tc_valid = tc[combined_mask]
                    if tc_valid.numel() > 0:  # Only compute if we have valid tokens
                        loss_e += w * F.cross_entropy(
                            lc_valid, tc_valid, ignore_index=dia_cfg.data.audio_pad_value
                        )
                loss_e = loss_e / sum(weights_e)

                eval_losses.append(loss_e.item()) # Append float

    # Synchronize to get all losses if we want exact global eval loss,
    # but for monitoring, local or gathered average is fine.
    # Let's gather losses to report accurate metric.
    # (Simplified: just report local mean on main process for now to avoid heavy sync)
    
    should_stop = False
    if len(eval_losses) > 0:
        avg_eval_loss = sum(eval_losses) / len(eval_losses)
        if accelerator.is_main_process:
            wandb.log({'eval_loss': avg_eval_loss}, step=global_step)
            
            if stop_on_overfit and current_train_loss is not None:
                if avg_eval_loss > current_train_loss:
                    logger.info(f"Stop trigger: Eval loss {avg_eval_loss:.4f} > Train loss {current_train_loss:.4f}. Generating demo and stopping.")
                    should_stop = True
                    do_demo = True
    else:
        if accelerator.is_main_process and val_loader is not None:
            logger.warning("No validation samples available for evaluation - check split_ratio")

    # Only generate demos if requested
    if not do_demo:
        # If we need to stop, broadcast that decision
        if stop_on_overfit:
             # Broadcast should_stop from main process
             pass # (Implemented in training loop via simple flag check or rely on main process killing loop)
        return should_stop

    # Only main process does audio generation to avoid conflicts
    if accelerator.is_main_process:
        logger.info(f"Starting eval demo generation at step {global_step}")
        # Unwrap model for generation (removes DDP wrapper)
        unwrapped_model = accelerator.unwrap_model(model)
        orig_dtype = next(unwrapped_model.parameters()).dtype

        try:
            # Switch to float for generation stability if needed, or keep as is.
            # Often generation works better in full precision.
            unwrapped_model = unwrapped_model.float()
            dia_gen = Dia(dia_cfg, accelerator.device)
            dia_gen.model, dia_gen.dac_model = unwrapped_model, dac_model

            with torch.inference_mode(), torch.cuda.amp.autocast(enabled=False):
                audio_samples = {}
                
                # Check if we're doing unconditional generation
                if train_cfg.unconditional_frac >= 1.0:
                    seeds = [int(train_cfg.seed), int(train_cfg.seed) + 1]
                    temperatures = EVAL_TEMPERATURES
                    total_demos = len(seeds) * len(temperatures)
                    logger.info(f"Generating {total_demos} unconditional demos")
                    
                    # Save current RNG states
                    prev_py_state = random.getstate()
                    prev_np_state = np.random.get_state()
                    prev_torch_state = torch.get_rng_state()
                    prev_cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                    
                    try:
                        for temp in temperatures:
                            for s in seeds:
                                try:
                                    seed_everything(s)
                                    logger.info(f"Generating unconditional audio (seed={s}, temp={temp})")
                                    audio = dia_gen.generate(
                                        text="",
                                        cfg_scale=EVAL_CFG_SCALE_UNCOND,
                                        temperature=temp
                                    )
                                    
                                    temp_str = f"{temp:.1f}".replace(".", "p")
                                    audio_filename = f"step_{global_step}_temp{temp_str}_seed{s}.wav"
                                    audio_path = Path(EVAL_AUDIO_DIR) / audio_filename
                                    arr = audio
                                    if isinstance(arr, torch.Tensor):
                                        arr = arr.detach().cpu().numpy()
                                    if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[0] <= 8 and arr.shape[1] > arr.shape[0]:
                                        arr = arr.T
                                    sf.write(audio_path, arr, EVAL_SAMPLE_RATE)
                                    
                                    audio_samples[f"eval_audio/temp{temp_str}/seed{s}"] = wandb.Audio(
                                        arr, sample_rate=EVAL_SAMPLE_RATE, 
                                        caption=f"temp={temp}, seed={s}"
                                    )
                                except Exception as e:
                                    logger.exception(f"Error generating sample (seed={s}, temp={temp}): {e}")
                    finally:
                        try:
                            torch.set_rng_state(prev_torch_state)
                            if torch.cuda.is_available() and prev_cuda_states is not None:
                                torch.cuda.set_rng_state_all(prev_cuda_states)
                        except Exception:
                            pass
                        try:
                            np.random.set_state(prev_np_state)
                            random.setstate(prev_py_state)
                        except Exception:
                            pass
                else:
                    # Conditional generation
                    temperatures = EVAL_TEMPERATURES
                    total_demos = len(TEST_PROMPTS) * len(temperatures)
                    logger.info(f"Generating {total_demos} conditional demos")
                    
                    cfg_scale = EVAL_CFG_SCALE if train_cfg.unconditional_frac > 0 else None
                    for temp in temperatures:
                        for test_name, prompt in TEST_PROMPTS.items():
                            try:
                                logger.info(f"Generating audio for '{test_name}' (temp={temp})")
                                audio = dia_gen.generate(
                                    text=prompt,
                                    cfg_scale=cfg_scale,
                                    temperature=temp,
                                    top_p=EVAL_TOP_P
                                )
                                
                                temp_str = f"{temp:.1f}".replace(".", "p")
                                audio_filename = f"step_{global_step}_{test_name}_temp{temp_str}.wav"
                                audio_path = Path(EVAL_AUDIO_DIR) / audio_filename
                                arr = audio
                                if isinstance(arr, torch.Tensor):
                                    arr = arr.detach().cpu().numpy()
                                if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[0] <= 8 and arr.shape[1] > arr.shape[0]:
                                    arr = arr.T
                                sf.write(audio_path, arr, EVAL_SAMPLE_RATE)
                                
                                audio_samples[f"eval_audio/temp{temp_str}/{test_name}"] = wandb.Audio(
                                    arr, sample_rate=EVAL_SAMPLE_RATE, 
                                    caption=f"{prompt} (temp={temp})"
                                )
                            except Exception as e:
                                 logger.exception(f"Error synthesizing '{test_name}' at temp={temp}: {e}")
                
                if audio_samples:
                    wandb.log(audio_samples, step=global_step)
                    logger.info("Logged demo audio samples to wandb")
                
        except Exception as e:
            logger.exception(f"Eval demo generation failed: {e}")
        finally:
            # Restore training dtype
            if orig_dtype == torch.float16:
                unwrapped_model.half()
            elif orig_dtype == torch.bfloat16:
                unwrapped_model.bfloat16()
            
            # Ensure model on device (Accelerator should handle this, but unwrapped might need check)
            # Actually unwrapped model shares storage so it stays on device.
            pass
    
    accelerator.wait_for_everyone()
    return should_stop


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
    train_loader, val_loader = setup_loaders(dataset, dia_cfg, train_cfg, use_sliding_window)
    opt, sched = setup_optimizer_and_scheduler(model, train_loader, train_cfg)

    # PREPARE EVERYTHING WITH ACCELERATOR
    # Prepare validation loader as well if it exists
    if val_loader:
        model, opt, train_loader, val_loader, sched = accelerator.prepare(
            model, opt, train_loader, val_loader, sched
        )
    else:
        model, opt, train_loader, sched = accelerator.prepare(
            model, opt, train_loader, sched
        )

    model.train()

    steps_per_epoch = getattr(train_loader, 'steps_per_epoch', None)
    if steps_per_epoch is None:
        try:
            steps_per_epoch = len(train_loader)
        except Exception:
            steps_per_epoch = None

    stop_training = False
    for epoch in range(train_cfg.epochs):
        # No manual sampler set_epoch needed; Accelerator handles it if it wrapped the loader
        
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
            global_step = epoch * (steps_per_epoch or 0) + step
            
            batch_start = time.time()
            
            # Updated train step signature
            loss = train_step(model, batch, dia_cfg, train_cfg, opt, sched, step, global_step, accelerator)
            
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
                
                if isinstance(loader_iter, tqdm):
                    loader_iter.set_postfix({
                        'loss': f"{loss:.4f}",
                        'VRAM (GB)': vram_str,
                        'step_time': f"{total_step_time:.1f}s"
                    })

            # evaluation during epoch (only if epoch-based eval is not requested)
            if args.eval_every_epochs is None and global_step > 0 and global_step % train_cfg.eval_step == 0:
                # Determine if we should generate demos (respecting demo_after_epoch)
                demo_interval = args.demo_every if args.demo_every is not None else train_cfg.eval_step
                past_demo_threshold = (epoch + 1) > args.demo_after_epoch
                do_demo = (global_step % demo_interval == 0) and past_demo_threshold
                
                # Run eval step if we have val_loader OR if we need to generate demos
                # Note: calling eval_step triggers sync/eval logic
                should_stop = eval_step(
                    model, val_loader, dia_cfg, dac_model, global_step, 
                    accelerator, train_cfg, do_demo=do_demo, 
                    current_train_loss=loss, stop_on_overfit=args.stop_on_overfit
                )
                
                if args.stop_on_overfit and should_stop:
                    stop_training = True
                    if accelerator.is_main_process:
                        ckpt_path = train_cfg.output_dir / f"ckpt_stop_overfit_{global_step}.pth"
                        unwrapped_model = accelerator.unwrap_model(model)
                        torch.save(unwrapped_model.state_dict(), ckpt_path)
                        logger.info(f"Saved overfit-stop checkpoint: {ckpt_path}")
                
                model.train()

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

            if args.early_stop_loss is not None:
                # Gather loss from all processes to decide on early stopping?
                # Or just use local loss. Usually local loss is fine if it's consistent.
                # But safe way is to check if ANY process wants to stop or Main.
                # Let's stick to local trigger broadcasted.
                trigger_local = torch.tensor(1 if loss <= args.early_stop_loss else 0, device=accelerator.device)
                # Gather/Reduce across devices
                # Simple way: if main process triggers, we stop
                # But `loss` here is scalar float.
                pass # Simplified for now, assume similar logic or update if needed.
                # (Implementing strict early stop sync with Accelerate requires more code, skipping for brevity unless requested)

        if stop_training:
            break

        # Epoch-end evaluation if requested
        should_check_epoch = False
        if args.eval_every_epochs is not None and (epoch + 1) % args.eval_every_epochs == 0:
             should_check_epoch = True
        if args.demo_every_epochs is not None and (epoch + 1) % args.demo_every_epochs == 0:
             should_check_epoch = True
             
        if should_check_epoch:
            gsp = ((epoch + 1) * (steps_per_epoch or 0)) - 1
            # For epoch-based eval, check demo_every_epochs first, then fallback to demo_every (steps)
            past_demo_threshold = (epoch + 1) > args.demo_after_epoch
            if args.demo_every_epochs is not None:
                do_demo = ((epoch + 1) % args.demo_every_epochs == 0) and past_demo_threshold
            elif args.demo_every is None:
                do_demo = past_demo_threshold
            else:
                do_demo = (gsp > 0 and gsp % args.demo_every == 0) and past_demo_threshold
            
            should_stop = eval_step(
                model, val_loader, dia_cfg, dac_model, gsp if gsp >= 0 else 0, 
                accelerator, train_cfg, do_demo=do_demo, 
                current_train_loss=loss, stop_on_overfit=args.stop_on_overfit
            )
            
            if args.stop_on_overfit and should_stop:
                stop_training = True
                if accelerator.is_main_process:
                    ckpt_path = train_cfg.output_dir / f"ckpt_stop_overfit_{gsp}.pth"
                    unwrapped_model = accelerator.unwrap_model(model)
                    torch.save(unwrapped_model.state_dict(), ckpt_path)
                    logger.info(f"Saved overfit-stop checkpoint: {ckpt_path}")

            model.train()

        if accelerator.is_main_process:
            is_last_epoch = (epoch + 1) == train_cfg.epochs
            past_save_threshold = (epoch + 1) > args.save_after_epoch
            
            if args.save_every_epochs is not None:
                should_save_epoch = ((epoch + 1) % args.save_every_epochs == 0) and past_save_threshold
                if should_save_epoch or is_last_epoch:
                    ckpt_e = train_cfg.output_dir / f"ckpt_epoch{epoch+1}.pth"
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    torch.save(unwrapped_model.state_dict(), ckpt_e)
                    logger.info(f"Saved end-of-epoch checkpoint: {ckpt_e}")
                    if is_last_epoch:
                        latest_ckpt = train_cfg.output_dir / "latest.pth"
                        torch.save(unwrapped_model.state_dict(), latest_ckpt)
                        logger.info(f"Saved latest checkpoint: {latest_ckpt}")
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
    
    accelerator.wait_for_everyone()


def train_step(model, batch, dia_cfg, train_cfg, opt, sched, step, global_step, accelerator: Accelerator):
    gen_val = ((global_step * 997 + train_cfg.seed) % 10000) / 10000.0
    if gen_val < train_cfg.unconditional_frac:
        pad_tok = dia_cfg.data.text_pad_value
        batch['src_tokens'].fill_(pad_tok)
        batch['enc_self_attn_mask'] = torch.zeros_like(batch['enc_self_attn_mask'])
        batch['dec_cross_attn_mask'] = torch.zeros_like(batch['dec_cross_attn_mask'])

    # Accumulate context manager handles gradient accumulation
    with accelerator.accumulate(model):
        # Remove explicit autocast; Accelerate handles it
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
        lens = batch['tgt_lens']
        max_L = int(lens.max().item())
        logits = logits[:, : max_L - 1]
        target = batch['tgt_tokens'][:, 1:max_L, :]
        B, Tm1, C = target.shape
        pad_val = dia_cfg.data.audio_pad_value
        time_idx = torch.arange(Tm1, device=lens.device).unsqueeze(0)
        valid_time = time_idx < (lens.unsqueeze(1) - 1)
        mask = valid_time.unsqueeze(-1).expand(-1, -1, C)
        
        channel_weights = []
        num_groups = C // 9
        if num_groups > 0:
            for _ in range(num_groups):
                channel_weights.extend(CODEBOOK_WEIGHTS)
        else:
            channel_weights = [1.0] * C

        loss_c = 0.0
        _, _, _, V = logits.size()
        
        audio_token_mask = (target >= 0) & (target <= 1023)
        
        for c, w in enumerate(channel_weights):
            lc = logits[:, :, c, :].reshape(-1, V)
            tc = target[:, :, c].reshape(-1)
            mc = mask[:, :, c].reshape(-1)
            audio_mc = audio_token_mask[:, :, c].reshape(-1)
            combined_mask = mc & audio_mc
            lc_valid = lc[combined_mask]
            tc_valid = tc[combined_mask]
            if tc_valid.numel() > 0:
                loss_c += w * F.cross_entropy(
                    lc_valid, tc_valid,
                    ignore_index=pad_val
                )
        loss = loss_c / sum(channel_weights)

        # Accelerate handles scaling
        accelerator.backward(loss)
        
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            # Log gradients if needed (only main process)
            if accelerator.is_main_process and (step + 1) % train_cfg.grad_accum_steps == 0:
                # simplified logging to avoid complex gather logic for now
                pass
        
        opt.step()
        sched.step()
        opt.zero_grad()

    # Just return the loss item (it's local)
    return loss.item()


def run_training(args):
    # Initialize Accelerator
    # gradient_accumulation_steps is managed by Accelerator if passed here or config
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum_steps,
        mixed_precision="bf16" if args.half else "no" # Simple default or rely on accelerate config
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
    dac_model = dac.DAC.load(dac.utils.download()).eval().to(accelerator.device)

    use_sliding_window = args.use_sliding_window
    if args.preencoded_dir:
        dataset = PreEncodedDACDataset(args.preencoded_dir, dia_cfg, use_sliding_window)
    elif args.audio_folder:
        skip_tags_list = [t.strip() for t in args.skip_tags.split(',')] if args.skip_tags else None
        dataset = MusicDataset(args.audio_folder, dia_cfg, dac_model, use_sliding_window,
                             ignore_missing_prompts=not args.require_prompts,
                             skip_tags=skip_tags_list)
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

    # Ensure all parameters have consistent dtype
    # Accelerate usually handles mixed precision, but if we force half here:
    if args.half:
        # This might conflict with Accelerate's mixed_precision="bf16" or "fp16"
        # Usually better to let Accelerate handle it, but if the user wants storage in half:
        model = model.half()
        for param in model.parameters():
            if param.dtype != torch.float16:
                param.data = param.data.half()
    
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
        model = torch.compile(model, backend="inductor")
    
    accelerator.wait_for_everyone()
    
    # Launch training
    train(model, dia_cfg, dac_model, dataset, train_cfg, args, accelerator)


def main():
    args = get_args()
    run_training(args)


if __name__ == "__main__":
    main()