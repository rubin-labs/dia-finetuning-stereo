"""
Dia Audio Model Fine-tuning with Accelerate.

Supports single and multi-GPU training via HuggingFace Accelerate.
Run with: accelerate launch -m dia.finetune_acc --args...
"""


import argparse
import gc
import glob
import logging
import math
import json
import os
import random
import re
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import dac
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from huggingface_hub import hf_hub_download
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers import get_scheduler

from .audio import apply_audio_delay, build_delay_indices
from .config import DiaConfig
from .dataset import PreEncodedDACDataset, TestingDataset
from .layers import DiaModel
from .model import Dia

warnings.filterwarnings("ignore", message="`torch.nn.utils.weight_norm` is deprecated")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

torch.backends.cudnn.benchmark = True

# =============================================================================
# CONSTANTS & CONFIG
# =============================================================================

GRAD_CLIP_MAX_NORM = 5.0
ENTROPY_LOG_INTERVAL = 50

TEST_PROMPTS = {
    "piano_ambient": "piano, pads, ambient, cinematic, melancholic, peaceful, reflective, instrumental",
    "dark": "cinematic, suspenseful, dark, energetic, mysterious, strings, bells, bass"
}
EVAL_CFG_SCALE = 4.0
EVAL_TEMPERATURE = 1.0
EVAL_TOP_P = 0.95
EVAL_CFG_SCALE_UNCOND = 0.0
EVAL_TEMPERATURE_UNCOND = 1.0
EVAL_TEMPERATURES = [0.0, 0.5, 1.0]
EVAL_SAMPLE_RATE = 44100
EVAL_AUDIO_DIR = "./audio_demos"


@dataclass
class TrainConfig:
    # Training hyperparameters
    epochs: int = 500
    batch_size: int = 4
    grad_accum_steps: int = 1
    learning_rate: float = 1e-5
    weight_decay: float = 0.1
    warmup_steps: int = 100
    unconditional_frac: float = 0.15
    eval_step: int = 100
    save_step: int = 2000
    split_ratio: float = 0.997

    # Reproducibility
    seed: int = 786
    
    # Output paths
    runs_dir: Path = field(default_factory=lambda: Path("runs"))
    run_name: str = "dia_finetune_cv"
    output_dir: Optional[Path] = None
    
    # Tag augmentation
    tag_shuffle: bool = True
    tag_dropout: float = 0.0
    tag_limit: Optional[int] = None

# =============================================================================
# UTILS
# =============================================================================

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _augment_tags(text: str, train_cfg: "TrainConfig") -> str:
    tags = [t.strip() for t in text.split(',') if t.strip()]
    if not tags: return text
    if train_cfg.tag_dropout > 0.0:
        kept = [t for t in tags if random.random() > train_cfg.tag_dropout]
        tags = kept if kept else [random.choice(tags)]
    if train_cfg.tag_shuffle and len(tags) > 1:
        random.shuffle(tags)
    if train_cfg.tag_limit is not None and train_cfg.tag_limit > 0:
        tags = tags[:train_cfg.tag_limit]
    return ', '.join(tags)

@contextmanager
def preserve_rng_state():
    """Context manager to preserve and restore RNG state."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)

def _save_and_log_audio(audio, audio_path: Path, wandb_key: str, caption: str, audio_samples: dict):
    arr = audio
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[0] <= 8 and arr.shape[1] > arr.shape[0]:
        arr = arr.T
    sf.write(audio_path, arr, EVAL_SAMPLE_RATE)
    logger.info(f"Saved demo audio: {audio_path}")
    audio_samples[wandb_key] = wandb.Audio(arr, sample_rate=EVAL_SAMPLE_RATE, caption=caption)

def compute_output_entropy(logits):
    """Compute entropy of output probability distribution."""
    probs = F.softmax(logits.float(), dim=-1)
    log_probs = torch.log(probs + 1e-10)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy.mean().item()

def cleanup_old_checkpoints(output_dir: Path, keep_last_n: int):
    if keep_last_n is None: return
    
    for pattern in ["ckpt_step*.pth", "ckpt_epoch*.pth"]:
        files = sorted(glob.glob(str(output_dir / pattern)), 
                       key=lambda x: int(re.search(r'(\d+).pth', x).group(1)) if re.search(r'(\d+).pth', x) else 0)
        if len(files) > keep_last_n:
            for old_ckpt in files[:-keep_last_n]:
                try:
                    os.remove(old_ckpt)
                    logger.info(f"Removed old checkpoint: {old_ckpt}")
                except Exception as e:
                    logger.warning(f"Failed to remove {old_ckpt}: {e}")

# =============================================================================
# CONFIG & ARGS
# =============================================================================

def load_train_config(config_path: Path) -> dict:
    """Recursively load and flatten JSON config."""
    if not config_path.exists(): return {}
    with open(config_path) as f:
        cfg = json.load(f)
    
    def flatten(d):
        out = {}
        for k, v in d.items():
            if isinstance(v, dict):
                out.update(flatten(v))
            else:
                out[k] = v
        return out
    return flatten(cfg)

def get_args() -> argparse.Namespace:
    # 1. Parse config path only
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--train_config", type=Path, default=Path("configs/train_config.json"))
    pre_args, _ = pre_parser.parse_known_args()
    
    # 2. Load defaults from config file
    defaults = load_train_config(pre_args.train_config)
    
    # 3. Define full argument parser
    parser = argparse.ArgumentParser(description="Train the Dia audio model")
    
    # Paths & Configs
    parser.add_argument("--train_config", type=Path, default=Path("configs/train_config.json"))
    parser.add_argument("--config", type=Path, default=Path(defaults.get('config', 'configs/architecture/model.json')))
    parser.add_argument("--hub_model", type=str, default="nari-labs/Dia-1.6B")
    parser.add_argument("--local_ckpt", type=str, default=None)
    parser.add_argument("--audio_folder", type=Path, default=defaults.get('audio_folder'), help="Audio folder path")
    parser.add_argument("--preencoded_dir", type=Path, default=defaults.get('preencoded_dir'), help="Pre-encoded codes path")
    parser.add_argument("--run_name", type=str, default=defaults.get('run_name'))
    parser.add_argument("--output_dir", type=Path, default=defaults.get('output_dir'), help="Checkpoint output dir")
    
    # Training
    parser.add_argument("--epochs", type=int, default=defaults.get('epochs', 500))
    parser.add_argument("--batch_size", type=int, default=defaults.get('batch_size', 4))
    parser.add_argument("--grad_accum_steps", type=int, default=defaults.get('grad_accum_steps', 1))
    parser.add_argument("--learning_rate", type=float, default=defaults.get('learning_rate', 1e-5))
    parser.add_argument("--weight_decay", type=float, default=defaults.get('weight_decay', 0.1))
    parser.add_argument("--warmup_steps", type=int, default=defaults.get('warmup_steps', 500))
    parser.add_argument("--unconditional_frac", type=float, default=defaults.get('unconditional_frac'), required=defaults.get('unconditional_frac') is None)
    parser.add_argument("--seed", type=int, default=defaults.get('seed', 42))
    
    # Evaluation & Saving
    parser.add_argument("--eval_step", type=int, default=defaults.get('eval_step', 200))
    parser.add_argument("--demo_every", type=int, default=defaults.get('demo_every'))
    parser.add_argument("--eval_every_epochs", type=int, default=defaults.get('eval_every_epochs'))
    parser.add_argument("--demo_every_epochs", type=int, default=defaults.get('demo_every_epochs'))
    parser.add_argument("--save_every", type=int, default=defaults.get('save_every'))
    parser.add_argument("--save_every_epochs", type=int, default=None)
    parser.add_argument("--save_after_epoch", type=int, default=defaults.get('save_after_epoch', 0))
    parser.add_argument("--demo_after_epoch", type=int, default=0)
    parser.add_argument("--save_last", type=int, default=None, help="Keep last N checkpoints")
    parser.add_argument("--early_stop_loss", type=float, default=None)
    parser.add_argument("--stop_on_overfit", action="store_true")
    
    # Flags
    parser.add_argument("--scratch", action=argparse.BooleanOptionalAction, default=defaults.get('scratch', False))
    parser.add_argument("--tag_no_shuffle", action=argparse.BooleanOptionalAction, default=defaults.get('tag_no_shuffle', False))
    parser.add_argument("--force_single_gpu", action=argparse.BooleanOptionalAction, default=defaults.get('force_single_gpu', False))
    parser.add_argument("--use_sliding_window", action=argparse.BooleanOptionalAction, default=defaults.get('use_sliding_window', False))
    parser.add_argument("--tag_shuffle", action=argparse.BooleanOptionalAction, default=True)

    # Standard boolean flags
    parser.add_argument("--half", action="store_true", help="Load model in fp16")
    parser.add_argument("--compile", action="store_true", help="Torch compile model")
    parser.add_argument("--no_decay_embed", action="store_true")
    parser.add_argument("--require_prompts", action="store_true")
    parser.add_argument("--skip_tags", type=str, default=None)
    
    # WandB
    parser.add_argument("--wandb_project", type=str, default="dia-music-finetuning")
    parser.add_argument("--wandb_entity", type=str, default=None)
    
    # Augmentation
    parser.add_argument("--tag_dropout", type=float, default=0.0)
    parser.add_argument("--tag_limit", type=int, default=None)

    args = parser.parse_args()
    
    if args.output_dir is None:
        parser.error("--output_dir is required")
        
    return args

# =============================================================================
# DATASET & MODEL
# =============================================================================

def collate_fn(batch, config: DiaConfig, train_cfg: TrainConfig, device: torch.device, use_sliding_window: bool = True):
    texts, encodings, waveforms = zip(*batch)

    # Audio Cropping
    window_size = config.data.audio_length
    cropped_encodings = []
    for e in encodings:
        if e.size(0) > window_size:
            start = random.randint(0, e.size(0) - window_size) if use_sliding_window else 0
            cropped_encodings.append(e[start : start + window_size])
        else:
            cropped_encodings.append(e)
    encodings = cropped_encodings

    # Text Input
    max_text = config.data.text_length
    pad_tok = config.data.text_pad_value
    text_ids = []
    for txt in texts:
        txt_aug = _augment_tags(txt, train_cfg)
        b_full = txt_aug.encode('utf-8')
        bts = b_full[:max_text]
        arr = list(bts) + [pad_tok] * (max_text - len(bts))
        text_ids.append(torch.tensor(arr, dtype=torch.long))
    src = torch.stack(text_ids).to(device)
    src_pos = torch.arange(max_text, device=device).unsqueeze(0).expand(src.size(0), -1)
    src_pad = src.ne(pad_tok)
    enc_self_attn_mask = (src_pad.unsqueeze(2) & src_pad.unsqueeze(1)).unsqueeze(1)

    # Audio Padding & Delays
    batch_max = max(e.size(0) for e in encodings)
    padded_encodings = [
        F.pad(e, (0, 0, 0, batch_max - e.size(0)), value=config.data.audio_pad_value) if e.size(0) < batch_max else e
        for e in encodings
    ]
    seq_lens = [e.size(0) for e in encodings]
    codes = torch.stack(padded_encodings).to(device)

    B, T, C = codes.shape
    t_idx, idxs = build_delay_indices(B, T, C, config.data.delay_pattern)
    delayed = apply_audio_delay(codes, config.data.audio_pad_value, config.data.audio_bos_value, (t_idx, idxs))

    # Targets
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
    causal = torch.tril(torch.ones((max_tgt_len, max_tgt_len), dtype=torch.bool, device=device))
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

def setup_loaders(dataset, dia_cfg: DiaConfig, train_cfg: TrainConfig, device, use_sliding_window=True):
    collate = lambda b: collate_fn(b, dia_cfg, train_cfg, device, use_sliding_window)
    ds_len = len(dataset)
    n_train = int(train_cfg.split_ratio * ds_len)
    n_val = ds_len - n_train
    
    if ds_len <= 1 or n_val == 0:
        logger.info(f"Dataset size {ds_len}: using all for training")
        train_ds, val_ds = dataset, None
    else:
        g = torch.Generator().manual_seed(train_cfg.seed)
        train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=g)
    
    train_loader = DataLoader(
        train_ds, batch_size=train_cfg.batch_size, shuffle=True,
        collate_fn=collate, num_workers=0, pin_memory=False, drop_last=True
    )
    if val_ds:
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate, num_workers=0)
    else:
        val_loader = None
        
    train_loader.steps_per_epoch = len(train_loader)
    return train_loader, val_loader

def setup_optimizer_and_scheduler(model, train_loader, train_cfg):
    # 1. Define distinct parameter groups (Logic from Function 2)
    decay, no_decay = [], []
    seen = set()
    
    # Auto-detect normalization layers for safety
    norm_types = (torch.nn.LayerNorm, torch.nn.GroupNorm, torch.nn.BatchNorm1d, 
                  torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)
    try:
        norm_types += (torch.nn.modules.normalization.RMSNorm,)
    except AttributeError: pass

    # Pass 1: Smart filtering via Modules (catches Norms correctly)
    for module in model.modules():
        for name, p in module.named_parameters(recurse=False):
            if not p.requires_grad or id(p) in seen: continue
            seen.add(id(p))
            
            is_bias = name.endswith("bias")
            is_norm = isinstance(module, norm_types)
            is_lora = "lora" in name.lower() and ("alpha" in name.lower() or "scale" in name.lower())
            
            if is_bias or is_norm or is_lora:
                no_decay.append(p)
            else:
                decay.append(p)

    # Pass 2: Catch-all for remaining parameters (e.g. embeddings not in above modules)
    for name, p in model.named_parameters():
        if p.requires_grad and id(p) not in seen:
            decay.append(p)
            seen.add(id(p))

    # 2. Setup Optimizer
    opt = optim.AdamW([
        {"params": decay, "weight_decay": train_cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=train_cfg.learning_rate)

    # 3. Setup Scheduler (with streaming support)
    try:
        steps_per_epoch = len(train_loader)
    except TypeError:
        steps_per_epoch = getattr(train_loader, 'steps_per_epoch', None)
        if not steps_per_epoch: raise RuntimeError("Loader has no length or steps_per_epoch")

    total_steps = steps_per_epoch * train_cfg.epochs
    sched = get_scheduler('cosine', opt, 
                          num_warmup_steps=train_cfg.warmup_steps // train_cfg.grad_accum_steps,
                          num_training_steps=total_steps // train_cfg.grad_accum_steps)
    
    return opt, sched


def eval_step(model, val_loader, dia_cfg, dac_model, global_step, device, train_cfg, accelerator, do_demo=True, current_train_loss=None, stop_on_overfit=False):
    """
    Evaluate model: calc loss, check overfit, generate demos.
    """
    eval_losses = []
    
    # 1. Validation Loop
    if val_loader:
        with torch.inference_mode():
            for eb in tqdm(val_loader, desc="eval"):
                with accelerator.autocast():
                    logits = model(
                        src_BxS=eb['src_tokens'],
                        tgt_BxTxC=eb['tgt_tokens'],
                        src_positions=eb['src_positions'],
                        tgt_positions=eb['tgt_positions'],
                        enc_self_attn_mask=eb['enc_self_attn_mask'],
                        dec_self_attn_mask=eb['dec_self_attn_mask'],
                        dec_cross_attn_mask=eb['dec_cross_attn_mask'],
                        enable_dropout=False,
                    )[:, :-1]

                # Explicit float cast for stability before loss calc
                logits = logits.float() 
                target = eb['tgt_tokens'][:, 1:]
                
                loss = 0.0
                channel_weights = [1.0] * target.shape[2]
                
                # Vectorized loss calculation (cleaner than Function 1)
                for c, w in enumerate(channel_weights):
                    loss += w * F.cross_entropy(
                        logits[:, :, c, :].flatten(0, 1), 
                        target[:, :, c].flatten(), 
                        ignore_index=dia_cfg.data.audio_pad_value
                    )
                eval_losses.append(loss / sum(channel_weights))

    # 2. Logging & Stop Check
    should_stop = False
    if eval_losses:
        avg_loss = sum(eval_losses) / len(eval_losses)
        if accelerator.is_main_process:
            wandb.log({'eval_loss': avg_loss.item()}, step=global_step)
            
            if stop_on_overfit and current_train_loss and avg_loss > current_train_loss:
                logger.info(f"Stop trigger: Eval {avg_loss:.4f} > Train {current_train_loss:.4f}")
                should_stop = True
                do_demo = True # Force demo generation on stop

    if not do_demo: 
        return should_stop

    # 3. Demo Generation
    if accelerator.is_main_process:
        logger.info(f"Generating eval demos at step {global_step}")
        unwrapped = accelerator.unwrap_model(model)
        orig_dtype = next(unwrapped.parameters()).dtype
        
        try:
            # Cast to float32 for generation quality
            unwrapped = unwrapped.float()
            dia_gen = Dia(dia_cfg, device)
            dia_gen.model, dia_gen.dac_model = unwrapped, dac_model
            
            with torch.inference_mode(), torch.cuda.amp.autocast(enabled=False):
                audio_samples = {}
                
                # Helper for safe filenames
                def safe_temp(t): return f"{t:.1f}".replace(".", "p")
                
                if train_cfg.unconditional_frac >= 1.0:
                    seeds = [int(train_cfg.seed), int(train_cfg.seed) + 1]
                    with preserve_rng_state():
                        for temp in EVAL_TEMPERATURES:
                            for s in seeds:
                                seed_everything(s)
                                audio = dia_gen.generate(text="", cfg_scale=EVAL_CFG_SCALE_UNCOND, temperature=temp)
                                path = Path(EVAL_AUDIO_DIR) / f"step_{global_step}_temp{safe_temp(temp)}_seed{s}.wav"
                                _save_and_log_audio(audio, path, f"eval_audio/temp{safe_temp(temp)}/seed{s}", 
                                                 f"temp={temp}", audio_samples)
                else:
                    cfg_s = EVAL_CFG_SCALE if train_cfg.unconditional_frac > 0 else None
                    for temp in EVAL_TEMPERATURES:
                        for name, prompt in TEST_PROMPTS.items():
                            audio = dia_gen.generate(text=prompt, cfg_scale=cfg_s, temperature=temp, top_p=EVAL_TOP_P)
                            path = Path(EVAL_AUDIO_DIR) / f"step_{global_step}_{name}_temp{safe_temp(temp)}.wav"
                            _save_and_log_audio(audio, path, f"eval_audio/temp{safe_temp(temp)}/{name}", 
                                             f"{prompt}", audio_samples)
                
                if audio_samples: 
                    wandb.log(audio_samples, step=global_step)
                    
        except Exception as e:
            logger.exception(f"Demo generation failed: {e}")
        finally:
            # Always restore original dtype
            if orig_dtype == torch.float16: unwrapped.half()
            elif orig_dtype == torch.bfloat16: unwrapped.bfloat16()
            
    accelerator.wait_for_everyone()
    return should_stop

def train_step(model, batch, dia_cfg, train_cfg, opt, sched, step, global_step, accelerator):
    gen_val = ((global_step * 997 + train_cfg.seed) % 10000) / 10000.0
    if gen_val < train_cfg.unconditional_frac:
        batch['src_tokens'].fill_(dia_cfg.data.text_pad_value)
        batch['enc_self_attn_mask'].zero_()
        batch['dec_cross_attn_mask'].zero_()

    with accelerator.autocast():
        logits = model(
            src_BxS=batch['src_tokens'],
            tgt_BxTxC=batch['tgt_tokens'],
            src_positions=batch['src_positions'],
            tgt_positions=batch['tgt_positions'],
            enc_self_attn_mask=batch['enc_self_attn_mask'],
            dec_self_attn_mask=batch['dec_self_attn_mask'],
            dec_cross_attn_mask=batch['dec_cross_attn_mask'],
            enable_dropout=True,
        )
        max_L = int(batch['tgt_lens'].max().item())
        logits = logits[:, :max_L-1]
        target = batch['tgt_tokens'][:, 1:max_L]
        
        # Masking padding in loss
        mask = torch.arange(target.shape[1], device=target.device).unsqueeze(0) < (batch['tgt_lens'].unsqueeze(1) - 1)
        mask = mask.unsqueeze(-1).expand_as(target)
        
        loss = 0.0
        channel_weights = [1.0] * target.shape[2]
        for c, w in enumerate(channel_weights):
            l_c = logits[:, :, c, :].reshape(-1, logits.size(-1))
            t_c = target[:, :, c].reshape(-1)
            m_c = mask[:, :, c].reshape(-1)
            loss += w * F.cross_entropy(l_c[m_c], t_c[m_c], ignore_index=dia_cfg.data.audio_pad_value)
        loss = loss / sum(channel_weights)

        if global_step % ENTROPY_LOG_INTERVAL == 0 and accelerator.is_main_process:
            wandb.log({'output_entropy': compute_output_entropy(logits.detach())}, step=global_step)

    loss = loss / train_cfg.grad_accum_steps
    accelerator.backward(loss)
    
    if (step + 1) % train_cfg.grad_accum_steps == 0:
        norm = accelerator.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
        if accelerator.is_main_process: wandb.log({'grad_norm': norm}, step=global_step)
        opt.step()
        sched.step()
        opt.zero_grad()
        if accelerator.is_main_process:
            wandb.log({'learning_rate': sched.get_last_lr()[0], 'train_loss': loss.item()*train_cfg.grad_accum_steps}, step=global_step)
            
    return loss.item() * train_cfg.grad_accum_steps


def train(model, dia_cfg, dac_model, dataset, train_cfg, args):
    """
    Robust training loop with Accelerate.
    Handles distributed synchronization, stopping logic, and precise checkpointing.
    """
    accelerator = Accelerator()
    device = accelerator.device
    
    # 1. Setup (Main Process Only)
    if accelerator.is_main_process:
        train_cfg.output_dir.mkdir(parents=True, exist_ok=True)
        Path(EVAL_AUDIO_DIR).mkdir(exist_ok=True)
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=train_cfg.run_name, config=vars(args))
    
    accelerator.wait_for_everyone()
    
    # 2. Prepare Data & Model
    train_loader, val_loader = setup_loaders(dataset, dia_cfg, train_cfg, device, args.use_sliding_window)
    opt, sched = setup_optimizer_and_scheduler(model, train_loader, train_cfg)
    
    model, opt, train_loader, sched = accelerator.prepare(model, opt, train_loader, sched)
    if val_loader: val_loader = accelerator.prepare(val_loader)
    
    # Handle infinite/streaming datasets safely
    steps_per_epoch = getattr(train_loader, 'steps_per_epoch', len(train_loader) if hasattr(train_loader, '__len__') else None)
    
    model.train()
    stop_training = False

    for epoch in range(train_cfg.epochs):
        if stop_training: break

        # Shuffle DistributedSampler safely
        if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
            
        loader_iter = tqdm(train_loader, desc=f"E{epoch+1}", disable=not accelerator.is_main_process)
        
        for step, batch in enumerate(loader_iter):
            global_step = epoch * (steps_per_epoch or 0) + step
            
            # --- Training Step ---
            loss = train_step(model, batch, dia_cfg, train_cfg, opt, sched, step, global_step, accelerator)
            
            if accelerator.is_main_process:
                loader_iter.set_postfix({'loss': f"{loss:.4f}"})

            # --- Evaluation / Demo Logic ---
            # Check if we should eval based on steps
            should_eval_step = (args.eval_every_epochs is None) and (global_step % train_cfg.eval_step == 0) and (global_step > 0)
            
            if should_eval_step:
                model.eval()
                # Determine if we should demo
                demo_interval = args.demo_every or train_cfg.eval_step
                do_demo = (global_step % demo_interval == 0) and ((epoch + 1) > args.demo_after_epoch)
                
                # Run Eval
                should_stop_local = False
                if val_loader or do_demo:
                    should_stop_local = eval_step(
                        model, val_loader, dia_cfg, dac_model, global_step, device, 
                        train_cfg, accelerator, do_demo=do_demo, 
                        current_train_loss=loss, stop_on_overfit=args.stop_on_overfit
                    )

                # CRITICAL: Sync stop decision across all GPUs
                if args.stop_on_overfit:
                    stop_tensor = torch.tensor(int(should_stop_local), device=device)
                    stop_training = accelerator.gather(stop_tensor).sum() > 0
                    if stop_training: break

                model.train()

            # --- Checkpointing (Step-based) ---
            if args.save_every_epochs is None and (epoch + 1) > args.save_after_epoch:
                save_interval = args.save_every or train_cfg.save_step
                if global_step > 0 and global_step % save_interval == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        save_checkpoint(model, accelerator, train_cfg.output_dir / f"ckpt_step{global_step}.pth")
                        cleanup_old_checkpoints(train_cfg.output_dir, args.save_last)

            # --- Early Stopping (Loss-based) ---
            if args.early_stop_loss and loss <= args.early_stop_loss:
                # Sync decision
                stop_tensor = torch.tensor(1, device=device)
                stop_training = accelerator.gather(stop_tensor).sum() > 0
                if stop_training:
                    if accelerator.is_main_process:
                         save_checkpoint(model, accelerator, train_cfg.output_dir / f"ckpt_early_stop.pth")
                    break

        # --- Epoch End Logic ---
        # 1. Epoch-based Evaluation
        if args.eval_every_epochs and (epoch + 1) % args.eval_every_epochs == 0:
            model.eval()
            eval_step(model, val_loader, dia_cfg, dac_model, global_step, device, train_cfg, accelerator, do_demo=True, current_train_loss=loss)
            model.train()

        # 2. Epoch-based Saving
        is_save_epoch = args.save_every_epochs and ((epoch + 1) % args.save_every_epochs == 0)
        is_last = (epoch + 1) == train_cfg.epochs
        
        if (is_save_epoch or is_last) and (epoch + 1) > args.save_after_epoch:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                save_checkpoint(model, accelerator, train_cfg.output_dir / f"ckpt_epoch{epoch+1}.pth")
                if is_last:
                    save_checkpoint(model, accelerator, train_cfg.output_dir / "latest.pth")
                cleanup_old_checkpoints(train_cfg.output_dir, args.save_last)

    accelerator.wait_for_everyone()

def save_checkpoint(model, accelerator, path):
    """Save unwrapped model state dict safely."""
    unwrapped = accelerator.unwrap_model(model)
    torch.save(unwrapped.state_dict(), path)
    logger.info(f"Saved checkpoint: {path}")

def main():
    """
    Main entry point. 
    Run with: accelerate launch finetune_acc.py --args...
    """
    args = get_args()
    
    # 1. Deterministic Setup
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    
    # 2. Config & Device
    dia_cfg = DiaConfig.load(args.config)
    
    # In 'accelerate launch', cuda:0 is automatically mapped to the local rank's GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 3. Load DAC Model (Handle distributed download race condition)
    # We use a simple try/except block to let rank 0 download first if needed
    try:
        dac_path = dac.utils.download()
    except Exception:
        # If multiple processes race, wait a random bit and retry (simple heuristic)
        import time, random
        time.sleep(random.random() * 2) 
        dac_path = dac.utils.download()
        
    dac_model = dac.DAC.load(dac_path).eval().to(device)
    
    # 4. Dataset Selection
    use_sliding_window = args.use_sliding_window
    if args.preencoded_dir:
        dataset = PreEncodedDACDataset(args.preencoded_dir, dia_cfg, use_sliding_window)
    elif args.audio_folder:
        skip_tags = [t.strip() for t in args.skip_tags.split(',')] if args.skip_tags else None
        dataset = TestingDataset(args.audio_folder, dia_cfg, dac_model, use_sliding_window, skip_tags=skip_tags)
    else:
        raise ValueError("Must specify either --audio_folder or --preencoded_dir")

    # 5. Build Training Config
    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        unconditional_frac=args.unconditional_frac,
        eval_step=args.eval_step,
        run_name=args.run_name or TrainConfig.run_name,
        output_dir=args.output_dir,
        seed=args.seed,
        tag_shuffle=(not args.tag_no_shuffle and args.tag_shuffle),
        tag_dropout=args.tag_dropout,
        tag_limit=args.tag_limit,
    )

    # 6. Initialize Model & Checkpoints
    model = DiaModel(dia_cfg)
    
    if args.scratch:
        logger.info("Initializing model from scratch...")
        if hasattr(model, '_init_weights'): model._init_weights()
    else:
        # Download or load local
        if args.local_ckpt:
            ckpt_path = args.local_ckpt
        else:
            ckpt_path = hf_hub_download(args.hub_model, filename="dia-v0_1.pth")
            
        state = torch.load(ckpt_path, map_location="cpu")
        
        # --- ROBUST STEREO EXPANSION LOGIC (From Function 1) ---
        key = "decoder.logits_dense.weight"
        expanded_stereo = False
        
        if key in state and dia_cfg.data.channels == 18:
            W_ckpt = state[key]
            # Check if checkpoint is Mono (9) and we need Stereo (18)
            if W_ckpt.shape[1] == 9:
                logger.info(f"Expanding checkpoint {key} from 9 to 18 channels...")
                state[key] = torch.cat([W_ckpt, W_ckpt], dim=1)
                expanded_stereo = True
        
        # Load weights
        missing, unexpected = model.load_state_dict(state, strict=False)
        logger.info(f"Loaded weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        
        # Warm-start embeddings if we just expanded to stereo
        if expanded_stereo:
            logger.info("Warm-starting stereo embeddings (copying Left -> Right)...")
            with torch.no_grad():
                for i in range(9, 18):
                    model.decoder.embeddings[i].weight.copy_(model.decoder.embeddings[i - 9].weight)
                # Also ensure logits weights are symmetric initially
                model.decoder.logits_dense.weight.data[:, 9:18, :].copy_(
                    model.decoder.logits_dense.weight.data[:, 0:9, :]
                )

    # 7. Optimization Tweaks
    if args.half:
        model = model.half()
        
    # Freezing heuristic for unconditional training
    if train_cfg.unconditional_frac >= 0.9 and hasattr(model, 'encoder'):
        logger.info("Freezing encoder for high-unconditional training")
        for p in model.encoder.parameters():
            p.requires_grad = False

    if args.compile:
        logger.info("Compiling model...")
        model = torch.compile(model, backend="inductor")
    
    # 8. Start Training
    # We pass the model and configs to the training loop
    # Note: Accelerator will be initialized inside 'train()'
    train(model, dia_cfg, dac_model, dataset, train_cfg, args)

if __name__ == "__main__":
    main()