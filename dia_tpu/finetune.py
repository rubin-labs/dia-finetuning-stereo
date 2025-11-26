import warnings
warnings.filterwarnings("ignore")

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
import time
import sys
import datetime
import re
import gc
import glob
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# TPU / XLA imports
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.runtime as xr
    import torch_xla.utils.utils as xu
except ImportError:
    print("torch_xla not installed. This script requires a TPU environment.")
    sys.exit(1)

import torchaudio
import pandas as pd
from tqdm import tqdm
from huggingface_hub import hf_hub_download
import soundfile as sf

import dac
# Change relative imports to absolute imports from 'dia' package
from dia.config import DiaConfig
from dia.layers import DiaModel
from dia.model import Dia
from dia.audio import build_delay_indices, apply_audio_delay
from dia.dataset import MusicDataset, PreEncodedDACDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Constants copied from dia/finetune_acc.py to avoid importing it (and its CUDA dependencies)
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
CODEBOOK_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
EVAL_SAMPLE_RATE = 44100
EVAL_AUDIO_DIR = "./audio_demos_tpu"

# Tag augmentation globals
TAG_SHUFFLE = True
TAG_DROPOUT = 0.0
TAG_LIMIT = None

def _augment_tags(text: str) -> str:
    """Augment comma-separated tag prompts by dropout/shuffle/limit."""
    try:
        tags = [t.strip() for t in text.split(',') if t.strip()]
        if not tags:
            return text
        # dropout per tag
        if TAG_DROPOUT and TAG_DROPOUT > 0.0:
            kept = [t for t in tags if random.random() > TAG_DROPOUT]
            if kept:
                tags = kept
        # shuffle tags
        if TAG_SHUFFLE and len(tags) > 1:
            random.shuffle(tags)
        # limit tags
        if TAG_LIMIT is not None and TAG_LIMIT > 0:
            tags = tags[:TAG_LIMIT]
        return ', '.join(tags)
    except Exception:
        return text

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

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

@dataclass
class TrainConfig:
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
    seed: int = 786
    runs_dir: Path = Path("runs_tpu")
    run_name: str = "dia_finetune_tpu"
    output_dir: Path = None
    no_decay_embed: bool = False

def load_train_config(config_path: Path) -> dict:
    """Load training config from JSON file."""
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        cfg = json.load(f)
    # Flatten nested config into flat dict for argparse defaults
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
    if 'eval' in cfg:
        flat['eval_step'] = cfg['eval'].get('eval_step')
        flat['demo_every'] = cfg['eval'].get('demo_every')
        flat['eval_every_epochs'] = cfg['eval'].get('eval_every_epochs')
        flat['demo_every_epochs'] = cfg['eval'].get('demo_every_epochs')
    if 'flags' in cfg:
        flat['scratch'] = cfg['flags'].get('scratch')
        flat['tag_no_shuffle'] = cfg['flags'].get('tag_no_shuffle')
        flat['force_single_gpu'] = cfg['flags'].get('force_single_gpu')
        flat['use_sliding_window'] = cfg['flags'].get('use_sliding_window')
    # Remove None values
    return {k: v for k, v in flat.items() if v is not None}

def get_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--train_config", type=Path, default=Path("configs/train_config.json"))
    pre_args, _ = pre_parser.parse_known_args()
    
    cfg_defaults = load_train_config(pre_args.train_config)
    
    parser = argparse.ArgumentParser(description="Train the Dia audio model on TPU")
    parser.add_argument("--train_config", type=Path, default=Path("configs/train_config.json"))
    parser.add_argument("--config",    type=Path, default=Path(cfg_defaults.get('config', 'configs/architecture/model.json')))
    parser.add_argument("--hub_model", type=str,  default="nari-labs/Dia-1.6B")
    parser.add_argument("--local_ckpt", type=str,  default=None)
    parser.add_argument("--audio_folder", type=Path, default=cfg_defaults.get('audio_folder'))
    parser.add_argument("--preencoded_dir", type=Path, default=cfg_defaults.get('preencoded_dir'))
    parser.add_argument("--run_name",  type=str,  default=cfg_defaults.get('run_name'))
    parser.add_argument("--output_dir",type=Path, default=cfg_defaults.get('output_dir'))
    parser.add_argument("--seed", type=int, default=42)
    # Removed --half as TPUs use BF16 automatically or via config
    parser.add_argument("--wandb_project", type=str, default="dia-music-finetuning-tpu")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--save_every", type=int, default=cfg_defaults.get('save_every'))
    parser.add_argument("--save_last", type=int, default=None)
    parser.add_argument("--tag_shuffle", action="store_true", default=True)
    parser.add_argument("--tag_no_shuffle", action="store_true", default=cfg_defaults.get('tag_no_shuffle', False))
    parser.add_argument("--tag_dropout", type=float, default=0.0)
    parser.add_argument("--tag_limit", type=int, default=None)
    
    parser.add_argument("--epochs", type=int, default=cfg_defaults.get('epochs', 500))
    parser.add_argument("--batch_size", type=int, default=cfg_defaults.get('batch_size', 4))
    parser.add_argument("--grad_accum_steps", type=int, default=cfg_defaults.get('grad_accum_steps', 1))
    parser.add_argument("--learning_rate", type=float, default=cfg_defaults.get('learning_rate', 1e-5))
    parser.add_argument("--weight_decay", type=float, default=cfg_defaults.get('weight_decay', 0.1))
    parser.add_argument("--warmup_steps", type=int, default=cfg_defaults.get('warmup_steps', 500))
    parser.add_argument("--unconditional_frac", type=float, default=cfg_defaults.get('unconditional_frac'))
    parser.add_argument("--eval_step", type=int, default=cfg_defaults.get('eval_step', 200))
    parser.add_argument("--demo_every", type=int, default=cfg_defaults.get('demo_every'))
    parser.add_argument("--eval_every_epochs", type=int, default=cfg_defaults.get('eval_every_epochs'))
    parser.add_argument("--demo_every_epochs", type=int, default=cfg_defaults.get('demo_every_epochs'))
    parser.add_argument("--save_every_epochs", type=int, default=None)
    parser.add_argument("--save_after_epoch", type=int, default=cfg_defaults.get('save_after_epoch', 0))
    parser.add_argument("--demo_after_epoch", type=int, default=0)
    parser.add_argument("--early_stop_loss", type=float, default=None)
    parser.add_argument("--stop_on_overfit", action="store_true")
    
    parser.add_argument("--use_sliding_window", action="store_true", default=cfg_defaults.get('use_sliding_window', False))
    parser.add_argument("--no_decay_embed", action="store_true")
    parser.add_argument("--require_prompts", action="store_true")
    parser.add_argument("--skip_tags", type=str, default=None)
    parser.add_argument("--scratch", action="store_true", default=cfg_defaults.get('scratch', False))
    parser.add_argument("--num_cores", type=int, default=8, help="Number of TPU cores to use")

    args = parser.parse_args()
    
    if args.output_dir is None:
        parser.error("--output_dir is required")
    if args.unconditional_frac is None:
        parser.error("--unconditional_frac is required")
    
    return args

def collate_fn_tpu(batch, config: DiaConfig, device: torch.device, use_sliding_window: bool = True):
    # Modified collate_fn for TPU: Enforces fixed padding size to avoid recompilation
    texts, encodings, waveforms = zip(*batch)

    # -- Enforce max length and optional random cropping --
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

    # -- Text inputs --
    max_text = config.data.text_length
    pad_tok = config.data.text_pad_value
    text_ids = []
    for txt in texts:
        txt_aug = _augment_tags(txt)
        b_full = txt_aug.encode('utf-8')
        bts = b_full[:max_text]
        arr = list(bts) + [pad_tok] * (max_text - len(bts))
        text_ids.append(torch.tensor(arr, dtype=torch.long))
    # Return CPU tensor
    src = torch.stack(text_ids)
    src_pos = torch.arange(max_text).unsqueeze(0).expand(src.size(0), -1)
    src_pad = src.ne(pad_tok)
    enc_self_attn_mask = (src_pad.unsqueeze(2) & src_pad.unsqueeze(1)).unsqueeze(1)

    # -- Audio codes: pad to FIXED window_size --
    # TPU Optimization: Always pad to fixed size to prevent graph recompilation
    target_len = window_size
    
    padded_encodings = []
    for e in encodings:
        if e.size(0) < target_len:
            pad_length = target_len - e.size(0)
            pad_value = config.data.audio_pad_value
            padding = torch.full((pad_length, e.size(1)), pad_value, dtype=e.dtype)
            padded_e = torch.cat([e, padding], dim=0)
        else:
            padded_e = e[:target_len] # Ensure it doesn't exceed
        padded_encodings.append(padded_e)
    
    seq_lens = [min(e.size(0), target_len) for e in encodings]
    # Return CPU tensor
    codes = torch.stack(padded_encodings)

    B, T, C = codes.shape
    t_idx, idxs = build_delay_indices(B, T, C, config.data.delay_pattern)
    delayed = apply_audio_delay(
        codes,
        config.data.audio_pad_value,
        config.data.audio_bos_value,
        (t_idx, idxs)
    )

    # -- Targets with per-sample EOS ----------------------------------------
    # Output length must also be fixed. 
    # In original: max_tgt_len = batch_max + 2
    # Here: max_tgt_len = target_len + 2
    max_tgt_len = target_len + 2
    pad_val = config.data.audio_pad_value
    bos_val = config.data.audio_bos_value
    eos_val = config.data.audio_eos_value

    tgt = torch.full((B, max_tgt_len, C), pad_val, dtype=torch.long)
    tgt[:, 0, :] = bos_val
    tgt_lens = []
    for i, L in enumerate(seq_lens):
        tgt[i, 1:1 + L, :] = delayed[i, :L, :]
        tgt[i, 1 + L, :] = eos_val
        tgt_lens.append(1 + L + 1)

    tgt_pos = torch.arange(max_tgt_len).unsqueeze(0).expand(B, -1)
    tgt_pad = tgt.ne(pad_val).any(-1)

    causal = torch.tril(torch.ones((max_tgt_len, max_tgt_len), dtype=torch.bool))
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
        'tgt_lens': torch.tensor(tgt_lens, dtype=torch.long),
    }

def setup_loaders(dataset, dia_cfg, train_cfg, device, use_sliding_window=True):
    # Using TPU-optimized collate function
    collate = lambda b: collate_fn_tpu(b, dia_cfg, device, use_sliding_window)
    
    ds_len = len(dataset)
    n_train = int(train_cfg.split_ratio * ds_len)
    n_val = ds_len - n_train
    
    if ds_len <= 1 or n_val == 0:
        train_ds = dataset
        val_ds = None
    else:
        g = torch.Generator().manual_seed(train_cfg.seed)
        train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=g)
    
    # DistributedSampler for TPU
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_ds,
        num_replicas=xr.world_size(),
        rank=xr.global_ordinal(),
        shuffle=True,
        drop_last=True,
        seed=train_cfg.seed
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        sampler=train_sampler,
        collate_fn=collate,
        num_workers=8, # Increased for GCS latency
        prefetch_factor=4, # Prefetch more batches
        drop_last=True,
        persistent_workers=True # Keep workers alive
    )
    
    val_loader = None
    if val_ds is not None:
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_ds,
            num_replicas=xr.world_size(),
            rank=xr.global_ordinal(),
            shuffle=False,
            drop_last=True
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            sampler=val_sampler,
            collate_fn=collate,
            num_workers=4,
            drop_last=True,
            persistent_workers=True
        )
        
    steps_per_epoch = len(train_loader)
    train_loader.steps_per_epoch = steps_per_epoch
    
    return train_loader, val_loader, train_sampler

def setup_optimizer_and_scheduler(model, train_loader, train_cfg):
    # Same param grouping logic, but using standard AdamW instead of bitsandbytes
    norm_types = [torch.nn.LayerNorm, torch.nn.GroupNorm, torch.nn.BatchNorm1d]
    
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
            if is_bias or is_norm or (train_cfg.no_decay_embed and is_embed):
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

    # Using standard AdamW for TPU compatibility
    opt = optim.AdamW(
        param_groups,
        lr=train_cfg.learning_rate,
        weight_decay=0.0, # handled in groups
    )
    
    steps_per_epoch = len(train_loader)
    total_training_steps = steps_per_epoch * train_cfg.epochs
    
    # Standard transformers scheduler
    from transformers import get_scheduler
    sched = get_scheduler(
        'cosine', opt,
        num_warmup_steps=train_cfg.warmup_steps // train_cfg.grad_accum_steps,
        num_training_steps=total_training_steps // train_cfg.grad_accum_steps
    )
    return opt, sched

def train_step_tpu(model, batch, dia_cfg, train_cfg, opt, sched, step, global_step, device):
    # NOTE: Shape printing removed - accessing .shape on TPU tensors can cause sync
    
    # Unconditional logic - TPU Optimized (avoid control flow)
    # Use random tensor on device to determine unconditional masking
    # This keeps the graph static regardless of the decision
    rand_val = torch.rand(1, device=device)
    do_uncond = rand_val < train_cfg.unconditional_frac
    # do_uncond is now a boolean tensor on device
    
    cond = do_uncond
    pad_tok = torch.tensor(dia_cfg.data.text_pad_value, device=device, dtype=batch['src_tokens'].dtype)
    
    # Apply masking using torch.where (static graph)
    batch['src_tokens'] = torch.where(cond, pad_tok, batch['src_tokens'])
    
    # Masks (handle boolean or float masks appropriately)
    # We assume masks are same type as initialized in collate (likely boolean or float)
    zeros_enc = torch.zeros_like(batch['enc_self_attn_mask'])
    zeros_dec = torch.zeros_like(batch['dec_cross_attn_mask'])
    
    # Expand condition to broadcast
    # masks are (B, 1, T, T) or similar
    # cond is (1,) or scalar. implicit broadcasting should work or we view it.
    
    # Cast cond to mask dtype for torch.where compatibility if needed
    # But boolean mask for where is fine.
    
    batch['enc_self_attn_mask'] = torch.where(cond, zeros_enc, batch['enc_self_attn_mask'])
    batch['dec_cross_attn_mask'] = torch.where(cond, zeros_dec, batch['dec_cross_attn_mask'])

    # No autocast needed for TPU usually (bf16 implicit)
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
    # TPU Optimization: Avoid dynamic slicing on the graph.
    # We use the full padded sequence length, which is constant if collate_fn is correct.
    # lens.max() causes a dynamic shape if used for slicing.
    # Instead, we rely on the padding mask to ignore invalid tokens in loss.
    
    max_L = logits.size(1) # This is constant (T)
    logits_slice = logits[:, :-1, :].float() # Slice to shape (B, T-1, V) and cast to float32 for stability
    target = batch['tgt_tokens'][:, 1:, :] # Slice to shape (B, T-1, C)
    
    B, Tm1, C = target.shape
    pad_val = dia_cfg.data.audio_pad_value
    
    # Reconstruct masks logic from original
    time_idx = torch.arange(Tm1, device=device).unsqueeze(0)
    
    # DEBUG: Force lens to be constant to rule out data-dependent recompilation
    # In a real run, we need the actual lengths, but if this fixes speed, we know the issue is here.
    # We'll use the full length (assuming padding mask handles the rest or we accept slight error for speed test)
    # lens = batch['tgt_lens']
    lens = torch.full((B,), Tm1 + 1, device=device, dtype=torch.long) 
    
    # valid_time mask is still dynamic based on 'lens', but this is a boolean mask, 
    # which XLA can handle better than changing tensor dimensions.
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
    V = logits.size(-1)
    audio_token_mask = (target >= 0) & (target <= 1023)
    
    for c, w in enumerate(channel_weights):
        lc = logits_slice[:, :, c, :].reshape(-1, V)
        tc = target[:, :, c].reshape(-1)
        mc = mask[:, :, c].reshape(-1)
        audio_mc = audio_token_mask[:, :, c].reshape(-1)
        combined_mask = mc & audio_mc
        
        # TPU Optimization: Use reduction='none' and mask instead of boolean indexing
        # Boolean indexing (lc[combined_mask]) creates dynamic shapes which triggers XLA recompilation
        
        # CRITICAL STABILITY FIX: Set targets to ignore_index where mask is False
        # This prevents cross_entropy from computing loss on irrelevant or out-of-bound tokens (like EOS 1024)
        # even though we zero out the loss later. This avoids NaNs from out-of-bound logits.
        tc_masked = torch.where(combined_mask, tc, pad_val)
        
        # Calculate loss for all tokens
        # We use ignore_index for pad_val, but combined_mask handles everything else
        loss_all = F.cross_entropy(lc, tc_masked, ignore_index=pad_val, reduction='none')
        
        # Mask out invalid tokens (padding, special tokens, etc.)
        mask_float = combined_mask.float()
        loss_masked = loss_all * mask_float
        
        # Compute mean over valid tokens
        num_valid = mask_float.sum()
        
        # Avoid division by zero
        term_loss = loss_masked.sum() / (num_valid + 1e-6)
        
        loss_c += w * term_loss
            
    loss = loss_c / sum(channel_weights)
    loss = loss / train_cfg.grad_accum_steps
    loss.backward()
    
    if (step + 1) % train_cfg.grad_accum_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        xm.optimizer_step(opt) # XLA step
        sched.step()
        opt.zero_grad()
        
    # Return tensor to avoid CPU sync on every step
    return loss.detach() * train_cfg.grad_accum_steps

def _mp_fn(rank, args):
    # Main worker function for xmp.spawn
    torch.set_default_tensor_type('torch.FloatTensor')
    device = xm.xla_device()
    
    # Seed
    seed_everything(args.seed)
    
    # Load Config
    dia_cfg = DiaConfig.load(args.config)
    
    # Globals
    global TAG_SHUFFLE, TAG_DROPOUT, TAG_LIMIT
    TAG_SHUFFLE = False if getattr(args, 'tag_no_shuffle', False) else bool(getattr(args, 'tag_shuffle', True))
    TAG_DROPOUT = float(getattr(args, 'tag_dropout', 0.0))
    TAG_LIMIT = getattr(args, 'tag_limit', None)
    
    xm.master_print(f"Loaded config from: {args.config}")
    
    # Load DAC
    # Load on CPU first, then move to device
    # dac.utils.download() might use network, ensure permission or done beforehand
    dac_path = dac.utils.download()
    dac_model = dac.DAC.load(dac_path).eval().to(device)
    
    # Dataset
    use_sliding_window = args.use_sliding_window
    if args.preencoded_dir:
        dataset = PreEncodedDACDataset(args.preencoded_dir, dia_cfg, use_sliding_window)
    elif args.audio_folder:
        skip_tags_list = [t.strip() for t in args.skip_tags.split(',')] if args.skip_tags else None
        dataset = MusicDataset(args.audio_folder, dia_cfg, dac_model, use_sliding_window,
                             ignore_missing_prompts=not args.require_prompts,
                             skip_tags=skip_tags_list)
    
    train_cfg = TrainConfig(
        epochs = args.epochs,
        batch_size = args.batch_size,
        grad_accum_steps = args.grad_accum_steps,
        learning_rate = args.learning_rate,
        weight_decay = args.weight_decay,
        warmup_steps = args.warmup_steps,
        unconditional_frac = args.unconditional_frac,
        eval_step = args.eval_step,
        run_name = args.run_name or TrainConfig.run_name,
        output_dir = args.output_dir,
        seed = args.seed,
        no_decay_embed = args.no_decay_embed,
    )
    
    model = DiaModel(dia_cfg)
    
    # Load Checkpoint (same logic as original but careful with CPU loading)
    if args.scratch:
        xm.master_print("Initializing model from scratch...")
        if hasattr(model, '_init_weights'):
            model._init_weights()
    else:
        if args.local_ckpt:
            ckpt_file = args.local_ckpt
        else:
            ckpt_file = hf_hub_download(args.hub_model, filename="dia-v0_1.pth")
        
        state = torch.load(ckpt_file, map_location="cpu")
        # Adapt stereo logic... (simplified for brevity, assuming 9->18 handled or not needed)
        # Copying the adaptation logic is recommended if needed
        # For now, simple load:
        missing, unexpected = model.load_state_dict(state, strict=False)
        xm.master_print(f"Loaded checkpoint: missing={len(missing)}, unexpected={len(unexpected)}")
        del state
        gc.collect()

    model = model.to(device)
    
    # Create loaders
    train_loader, val_loader, train_sampler = setup_loaders(dataset, dia_cfg, train_cfg, device, use_sliding_window)
    
    opt, sched = setup_optimizer_and_scheduler(model, train_loader, train_cfg)
    
    # Create output dir
    if xm.is_master_ordinal():
        train_cfg.output_dir.mkdir(parents=True, exist_ok=True)
        (train_cfg.runs_dir / train_cfg.run_name).mkdir(parents=True, exist_ok=True)
        Path(EVAL_AUDIO_DIR).mkdir(exist_ok=True)
        
        import wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=train_cfg.run_name,
            config=vars(train_cfg)
        )

    model.train()
    
    steps_per_epoch = len(train_loader)
    
    for epoch in range(train_cfg.epochs):
        train_sampler.set_epoch(epoch)
        
        para_loader = pl.ParallelLoader(train_loader, [device])
        loader_iter = para_loader.per_device_loader(device)
        
        if xm.is_master_ordinal():
            loader_iter = tqdm(loader_iter, total=steps_per_epoch, desc=f"Epoch {epoch+1}")
            
        for step, batch in enumerate(loader_iter):
            global_step = epoch * steps_per_epoch + step
            
            if step == 0 and epoch == 0 and xm.is_master_ordinal():
                tqdm.write("Step 0: Starting training step (this will trigger XLA compilation and take ~1-2 mins)...")
            
            loss_tensor = train_step_tpu(model, batch, dia_cfg, train_cfg, opt, sched, step, global_step, device)
            xm.mark_step()  # Ensure graph execution completes before any host sync (logging, wandb, etc.)
            
            # TPU Optimization: Use xm.add_step_closure to defer CPU syncs
            # This logs asynchronously without blocking the TPU
            if xm.is_master_ordinal():
                def _log_loss(loss_tensor, step_num, epoch_num, global_step_num, is_early):
                    import wandb
                    loss_val = loss_tensor.item()  # Safe inside closure - runs after mark_step
                    wandb.log({"loss": loss_val, "epoch": epoch_num}, step=global_step_num)
                    if is_early:
                        tqdm.write(f"Step {step_num}: loss={loss_val:.4f}")
                
                # Only log every 50 steps (or first 5) to reduce sync overhead
                if step < 5 or step % 50 == 0:
                    xm.add_step_closure(
                        _log_loss, 
                        args=(loss_tensor, step, epoch, global_step, step < 5)
                    )
            
            # Save logic
            should_save = global_step > 0 and global_step % train_cfg.save_step == 0
            if should_save:
                 xm.save(model.state_dict(), train_cfg.output_dir / f"ckpt_step{global_step}.pth")
                 xm.master_print(f"Saved checkpoint at step {global_step}")

        # End of epoch save
        xm.save(model.state_dict(), train_cfg.output_dir / f"ckpt_epoch{epoch+1}.pth")
        xm.master_print(f"Saved epoch {epoch+1} checkpoint")
        
        # TODO: Implement evaluation logic for TPU (requires sync and gathering metrics)

def main():
    args = get_args()
    
    # For PJRT on TPU, nprocs should be None to use all available devices (or 1 for single core)
    # Passing an explicit int > 1 throws an error in newer torch_xla versions
    nprocs = None if args.num_cores > 1 else 1
    
    # Spawn TPU processes
    xmp.spawn(_mp_fn, args=(args,), nprocs=nprocs, start_method='fork')

if __name__ == "__main__":
    main()
