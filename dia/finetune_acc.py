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
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from transformers import get_scheduler
import torch.nn.functional as F
import torch.optim as optim
import bitsandbytes as bnb
from tqdm import tqdm
from huggingface_hub import hf_hub_download
import math
import gc
import wandb
import soundfile as sf
import glob
import time
import signal
import sys
import datetime

import dac
from .config import DiaConfig
from .layers import DiaModel
from .model import Dia
from .audio import build_delay_indices, apply_audio_delay
from .dataset import MusicDataset, PreEncodedDACDataset
from torch.nn.functional import pad


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


# Tag augmentation controls (set from CLI in main/DDP worker)
TAG_SHUFFLE = True
TAG_DROPOUT = 0.0
TAG_LIMIT = None  # type: int | None


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


def seed_everything(seed: int, rank: int = 0):
    """Set seeds for reproducible training across all ranks."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _worker_init_fn(worker_id):
    """Initialize worker seeds for deterministic DataLoader workers."""
    ws = torch.initial_seed() % 2**32
    np.random.seed(ws)
    random.seed(ws)


# Music-specific processing can be added here if needed
test_prompts = {
    "travis_scott": "travis scott",
    "house": "house"
}


def cleanup_old_checkpoints(output_dir: Path, keep_last_n: int):
    """Keep only the last N checkpoints, delete older ones to save space."""
    if keep_last_n is None:
        return
    
    # Find all checkpoint files (both step and epoch checkpoints)
    step_checkpoints = sorted(glob.glob(str(output_dir / "ckpt_step*.pth")))
    epoch_checkpoints = sorted(glob.glob(str(output_dir / "ckpt_epoch*.pth")))
    
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


def setup_ddp(rank: int, world_size: int, port: str = "29500"):
    """Initialize the distributed environment."""
    init_method = os.environ.get("TORCH_DDP_INIT_METHOD")
    if not init_method:
        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        try:
            master_addr = socket.gethostbyname(master_addr)
        except socket.gaierror as exc:
            raise RuntimeError(f"Unable to resolve MASTER_ADDR='{master_addr}' to an IPv4 address") from exc
        os.environ["MASTER_ADDR"] = master_addr

        master_port = os.environ.get("MASTER_PORT", port)
        os.environ["MASTER_PORT"] = str(master_port)

        init_method = f"tcp://{master_addr}:{master_port}"
        if rank == 0:
            print(f"DDP rendezvous on {master_addr}:{master_port}")
    else:
        if rank == 0:
            print(f"DDP rendezvous via {init_method}")

    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("GLOO_SOCKET_FAMILY", "AF_INET")
    os.environ.setdefault("NCCL_SOCKET_FAMILY", "AF_INET")
    
    dist.init_process_group(
        backend="nccl",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=600)
    )
    torch.cuda.set_device(rank)
    def signal_handler(signum, frame):
        cleanup_ddp()
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def cleanup_ddp():
    """Clean up the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()

@dataclass
class TrainConfig:
    epochs: int = 500
    batch_size: int = 4  # batch size per GPU, effective batch size is batch_size * grad_accum_steps  
    grad_accum_steps: int = 1
    learning_rate: float = 1e-5
    weight_decay: float = 0.1
    warmup_steps: int = 100
    unconditional_frac: float = 0.15
    eval_step: int = 100
    save_step: int = 2000
    split_ratio: float = 0.997
    seed: int = 786                # seed for reproducibility
    runs_dir: Path = Path("runs")
    run_name: str = "dia_finetune_cv"
    output_dir: Path = None
    no_decay_embed: bool = False


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Dia audio model")
    parser.add_argument("--config",    type=Path, default=Path("dia/config.json"))
    parser.add_argument("--hub_model", type=str,  default="nari-labs/Dia-1.6B")
    parser.add_argument("--local_ckpt", type=str,  default=None)
    parser.add_argument("--audio_folder", type=Path, default=None,
                        help="Path to audio folder (expects audio_prompts folder at same level).")
    parser.add_argument("--preencoded_dir", type=Path, default=None,
                        help="Directory with pre-encoded DAC codes (encoded_audio/*.pt) and optional metadata.json.")
    parser.add_argument("--run_name",  type=str,  default=None)
    parser.add_argument("--output_dir",type=Path, required=True,
                        help="Output directory for checkpoints (required).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--half", action="store_true", help="load model in fp16")
    parser.add_argument("--compile", action="store_true", help="torch compile model")
    parser.add_argument("--wandb_project", type=str, default="dia-music-finetuning",
                        help="Weights & Biases project name.")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="Weights & Biases entity/team name.")
    parser.add_argument("--save_every", type=int, default=None,
                        help="Save checkpoint every N steps (overrides TrainConfig.save_step).")
    parser.add_argument("--save_last", type=int, default=None,
                        help="Keep only the last N checkpoints (e.g., --save_last 4). Saves disk space.")
    parser.add_argument("--force_single_gpu", action="store_true",
                        help="Force single GPU training even with multiple GPUs available")
    # Tag augmentation flags
    parser.add_argument("--tag_shuffle", action="store_true", default=True,
                        help="Shuffle comma-separated tags in prompts (default: on)")
    parser.add_argument("--tag_no_shuffle", action="store_true",
                        help="Disable tag shuffling (overrides --tag_shuffle)")
    parser.add_argument("--tag_dropout", type=float, default=0.0,
                        help="Per-tag dropout probability in [0,1] (default: 0.0)")
    parser.add_argument("--tag_limit", type=int, default=None,
                        help="Keep at most this many tags after shuffle/dropout (default: unlimited)")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=500,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size per GPU.")
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Gradient accumulation steps.")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.1,
                        help="AdamW weight decay coefficient.")
    parser.add_argument("--warmup_steps", type=int, default=500,
                        help="Number of warmup steps.")
    parser.add_argument("--unconditional_frac", type=float, default=1,
                        help="Fraction of unconditional training steps.")
    parser.add_argument("--eval_step", type=int, default=200,
                        help="Evaluate every N steps.")
    parser.add_argument("--eval_every_epochs", type=int, default=None,
                        help="Evaluate at the end of every N epochs (overrides step-based eval).")
    parser.add_argument("--save_every_epochs", type=int, default=None,
                        help="Save checkpoint at the end of every N epochs (overrides step-based save).")
    parser.add_argument("--early_stop_loss", type=float, default=None,
                        help="Early stop when training loss <= this value; saves final checkpoint then exits.")
    
    # Sliding window augmentation (default: enabled)
    parser.add_argument("--disable_sliding_window", action="store_true",
                        help="Disable sliding window: use original fixed-length loading instead of random cropping")
    # Optimizer param-group controls
    parser.add_argument("--no_decay_embed", action="store_true",
                        help="Exclude nn.Embedding parameters from weight decay")
    
    return parser.parse_args()



def collate_fn(batch, config: DiaConfig, device: torch.device):
    texts, encodings, waveforms = zip(*batch)

    # -- Text inputs ---------------------------------------------------------

    max_text = config.data.text_length
    pad_tok = config.data.text_pad_value
    text_ids = []
    for txt in texts:
        txt_aug = _augment_tags(txt)
        b_full = txt_aug.encode('utf-8')
        # Direct text encoding without language prefix processing
        bts = b_full[:max_text]
        arr = list(bts) + [pad_tok] * (max_text - len(bts))
        text_ids.append(torch.tensor(arr, dtype=torch.long))
    src = torch.stack(text_ids).to(device)
    src_pos = torch.arange(max_text, device=device).unsqueeze(0).expand(src.size(0), -1)
    src_pad = src.ne(pad_tok)
    enc_self_attn_mask = (src_pad.unsqueeze(2) & src_pad.unsqueeze(1)).unsqueeze(1)

    # -- Audio codes with batch-based padding (no windowing) ------------------

    # Find the maximum length in this batch
    batch_max = max(e.size(0) for e in encodings)
    
    # Pad all sequences to the batch maximum length
    padded_encodings = []
    for e in encodings:
        if e.size(0) < batch_max:
            # Pad shorter sequences to batch_max
            pad_length = batch_max - e.size(0)
            pad_value = config.data.audio_pad_value
            padding = torch.full((pad_length, e.size(1)), pad_value, dtype=e.dtype)
            padded_e = torch.cat([e, padding], dim=0)
        else:
            # Already at batch_max length
            padded_e = e
        padded_encodings.append(padded_e)
    
    # All sequences are now exactly batch_max length
    seq_lens = [e.size(0) for e in encodings]  # Original lengths before padding
    codes = torch.stack(padded_encodings).to(device)  # (B, T=batch_max, C)

    B, T, C = codes.shape
    t_idx, idxs = build_delay_indices(B, T, C, config.data.delay_pattern)
    delayed = apply_audio_delay(
        codes,
        config.data.audio_pad_value,
        config.data.audio_bos_value,
        (t_idx, idxs)
    )

    # -- Targets with per-sample EOS ----------------------------------------
    max_tgt_len = batch_max + 2  # Use dynamic batch max instead of fixed max_audio
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

def setup_loaders(dataset, dia_cfg: DiaConfig, train_cfg: TrainConfig, device, rank=0, world_size=1, use_ddp=False):
    collate = lambda b: collate_fn(b, dia_cfg, device)
    
    ds_len = len(dataset)
    n_train = int(train_cfg.split_ratio * ds_len)
    g = torch.Generator().manual_seed(train_cfg.seed)
    train_ds, val_ds = random_split(dataset, [n_train, ds_len - n_train], generator=g)
    
    # Create sampler for DDP
    if use_ddp:
        sampler = DistributedSampler(
            train_ds, 
            num_replicas=world_size, 
            rank=rank,
            shuffle=True,
            drop_last=True,
            seed=train_cfg.seed
        )
        shuffle = False  # DistributedSampler handles shuffling
    else:
        sampler = None
        shuffle = True
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=train_cfg.batch_size, 
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collate,
        num_workers=0,              # Avoid CUDA in forked workers; collate moves tensors to GPU
        pin_memory=False,           # Keep False since tensors are moved to GPU in collate
        drop_last=True,
        persistent_workers=False,   # Must be False when num_workers=0
        worker_init_fn=None
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=1, 
        shuffle=False, 
        collate_fn=collate,
        num_workers=0,  # Disable workers for DDP
        pin_memory=False  # Disabled since collate_fn puts data on GPU
    )
    
    # Set steps_per_epoch attribute for tqdm
    if use_ddp:
        steps_per_epoch = len(sampler) // train_cfg.batch_size  # Samples per GPU ÷ batch_size
    else:
        steps_per_epoch = len(train_ds) // train_cfg.batch_size
    
    train_loader.steps_per_epoch = steps_per_epoch
    
    return train_loader, val_loader



def setup_optimizer_and_scheduler(model, train_loader, train_cfg):
    # Build parameter groups according to decay policy
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

    # handle weight decay
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
    # catch any parameters not reached through modules() direct params
    for _, p in model.named_parameters():
        if p.requires_grad and id(p) not in seen:
            decay_params.append(p)
            seen.add(id(p))

    param_groups = [
        {"params": decay_params, "weight_decay": train_cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    opt = bnb.optim.AdamW8bit(
        param_groups,
        lr=train_cfg.learning_rate,
        weight_decay=0.0,
    )
    # Determine steps per epoch: prefer len(), else use attached attribute
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
        num_warmup_steps=train_cfg.warmup_steps / train_cfg.grad_accum_steps,
        num_training_steps=total_training_steps / train_cfg.grad_accum_steps
    )
    return opt, sched



def train_step(model, batch, dia_cfg, train_cfg, opt, sched, writer, step, global_step, accelerator):
    """
    Perform a single training step: forward, loss, backward, update, log.
    Now uses per‑sample tgt_lens to mask out padding after each EOS,
    and applies uniform loss across all channels.
    """
    # (optional) unconditional conditioning
    if random.random() < train_cfg.unconditional_frac:
        pad_tok = dia_cfg.data.text_pad_value
        batch['src_tokens'] = torch.zeros_like(batch['src_tokens'])
        batch['enc_self_attn_mask'] = torch.zeros_like(batch['enc_self_attn_mask'])
        batch['dec_cross_attn_mask'] = torch.zeros_like(batch['dec_cross_attn_mask'])

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        # forward pass
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
        # fetch per-sample target‑lengths (including BOS+frames+EOS)
        lens = batch['tgt_lens']                   # shape: (B,)
        max_L = int(lens.max().item())             # maximum over batch

        # keep only up through the last possible EOS slot
        # logits: (B, T, C, V) -> (B, max_L-1, C, V)
        logits = logits[:, : max_L - 1]

        # targets: shift off the BOS so 0..<max_L-1> align with logits
        # target: (B, T, C) -> (B, max_L-1, C)
        target = batch['tgt_tokens'][:, 1:max_L, :]

        B, Tm1, C = target.shape
        pad_val = dia_cfg.data.audio_pad_value

        # build a mask [B x (max_L-1)] that is True for t < (lens[i]-1)
        time_idx = torch.arange(Tm1, device=lens.device).unsqueeze(0)  # (1, Tm1)
        valid_time = time_idx < (lens.unsqueeze(1) - 1)                # (B, Tm1)
        mask = valid_time.unsqueeze(-1).expand(-1, -1, C)             # (B, Tm1, C)

        # use uniform channel weights
        channel_weights = [1.0] * C
        loss_c = 0.0
        _, _, _, V = logits.size()

        for c, w in enumerate(channel_weights):
            # flatten this channel
            lc = logits[:, :, c, :].reshape(-1, V)   # (B*Tm1, V)
            tc = target[:, :, c].reshape(-1)         # (B*Tm1,)
            mc = mask[:, :, c].reshape(-1)           # (B*Tm1,)

            # mask out padding and compute cross-entropy
            lc_valid = lc[mc]
            tc_valid = tc[mc]
            loss_c += w * F.cross_entropy(
                lc_valid, tc_valid,
                ignore_index=pad_val
            )

        # normalize by sum of weights
        loss = loss_c / sum(channel_weights)

    # scale + backward
    loss = loss / train_cfg.grad_accum_steps
    accelerator.backward(loss)

    # step & log
    if (step + 1) % train_cfg.grad_accum_steps == 0:
        # Clip once after accumulation, right before optimizer step
        pre_clip = clip_grad_norm_(model.parameters(), max_norm=5.0)
        writer.add_scalar('GradNorm/pre_clip', pre_clip, global_step)
        # Optional post-clip norm computation
        post_clip_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                g = p.grad.detach()
                post_clip_sq += g.float().norm(2).item() ** 2
        writer.add_scalar('GradNorm/post_clip', post_clip_sq ** 0.5, global_step)
        opt.step()
        sched.step()
        opt.zero_grad()
        true_loss = loss.item() * train_cfg.grad_accum_steps
        current_lr = sched.get_last_lr()[0]
        writer.add_scalar('LR', current_lr, global_step)
        writer.add_scalar('Loss/train', true_loss, global_step)

    return loss.item() * train_cfg.grad_accum_steps



def eval_step(model, val_loader, dia_cfg, dac_model, global_step, device, train_cfg, rank=0, use_ddp=False):
    """
    Run evaluation: generate audio demo samples
    """
    eval_losses = []
    last_batch = None
    with torch.inference_mode():
        for eb in tqdm(val_loader, desc="eval"):
            last_batch = eb

            # 1) do your forward in mixed precision
            with torch.amp.autocast('cuda'):
                logits16 = model(
                    src_BxS=eb['src_tokens'],
                    tgt_BxTxC=eb['tgt_tokens'],
                    src_positions=eb['src_positions'],
                    tgt_positions=eb['tgt_positions'],
                    enc_self_attn_mask=eb['enc_self_attn_mask'],
                    dec_self_attn_mask=eb['dec_self_attn_mask'],
                    dec_cross_attn_mask=eb['dec_cross_attn_mask'],
                    enable_dropout=False,
                )[:, :-1]

            logits = logits16.float()
            target = eb['tgt_tokens'][:, 1:]
            B_e, T_e, C_e = target.shape
            V_e = logits.size(-1)

            loss_e = 0.0
            weights_e = [1.0] * C_e
            for c, w in enumerate(weights_e):
                lc = logits[:, :, c, :].reshape(-1, V_e)
                tc = target[:, :, c].reshape(-1)
                loss_e += w * F.cross_entropy(
                    lc, tc, ignore_index=dia_cfg.data.audio_pad_value
                )
            loss_e = loss_e / sum(weights_e)

            eval_losses.append(loss_e)

    if len(eval_losses) > 0:
        avg_eval_loss = sum(eval_losses) / len(eval_losses)
        if rank == 0:
            wandb.log({'eval_loss': avg_eval_loss.item()}, step=global_step)
    else:
        if rank == 0:
            logger.warning("No validation samples available for evaluation - check split_ratio")

    # Only rank 0 does audio generation to avoid conflicts
    if rank == 0:
        logger.info(f"Starting eval demo generation at step {global_step}")
        # Unwrap DDP for evaluation
        unwrapped_model = model.module if hasattr(model, 'module') else model
        orig_dtype = next(unwrapped_model.parameters()).dtype

        try:
            unwrapped_model = unwrapped_model.float()
            dia_gen = Dia(dia_cfg, device)
            dia_gen.model, dia_gen.dac_model = unwrapped_model, dac_model

            with torch.inference_mode(), torch.cuda.amp.autocast(enabled=False):
                audio_samples = {}
                
                # Check if we're doing unconditional generation
                if train_cfg.unconditional_frac >= 1.0:
                    # Fully unconditional training - generate two samples with different seeds
                    seeds = [int(train_cfg.seed), int(train_cfg.seed) + 1]
                    logger.info(f"Generating {len(seeds)} unconditional demo samples with seeds: {seeds}")
                    # Save current RNG states to avoid impacting subsequent training randomness
                    prev_py_state = random.getstate()
                    prev_np_state = np.random.get_state()
                    prev_torch_state = torch.get_rng_state()
                    prev_cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                    try:
                        for s in seeds:
                            try:
                                seed_everything(s)
                                logger.info(f"Generating unconditional audio (seed={s})")
                                audio = dia_gen.generate(text="")
                                
                                # Save audio file to mono_demos directory
                                audio_filename = f"step_{global_step}_unconditional_seed{s}.wav"
                                audio_path = Path("./audio_demos") / audio_filename
                                arr = audio
                                if isinstance(arr, torch.Tensor):
                                    arr = arr.detach().cpu().numpy()
                                if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[0] <= 8 and arr.shape[1] > arr.shape[0]:
                                    arr = arr.T
                                sf.write(audio_path, arr, 44100)
                                logger.info(f"Saved demo audio: {audio_path}")
                                
                                # Convert to wandb Audio format
                                audio_samples[f"eval_audio/unconditional_seed{s}"] = wandb.Audio(arr, sample_rate=44100, caption=f"unconditional seed={s}")
                                logger.info(f"Added unconditional sample (seed={s}) to wandb log queue")
                            except Exception as e:
                                logger.exception(f"Error generating unconditional sample (seed={s}): {e}")
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
                    # Conditional training - use test prompts
                    logger.info(f"Generating {len(test_prompts)} demo samples...")
                    for test_name, prompt in test_prompts.items():
                        try:
                            logger.info(f"Generating audio for '{test_name}' with prompt: '{prompt}'")
                            audio = dia_gen.generate(text=prompt)
                            
                            # Save audio file to mono_demos directory
                            audio_filename = f"step_{global_step}_{test_name}.wav"
                            audio_path = Path("./audio_demos") / audio_filename
                            arr = audio
                            if isinstance(arr, torch.Tensor):
                                arr = arr.detach().cpu().numpy()
                            if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[0] <= 8 and arr.shape[1] > arr.shape[0]:
                                arr = arr.T
                            sf.write(audio_path, arr, 44100)
                            logger.info(f"Saved demo audio: {audio_path}")
                            
                            # Convert to wandb Audio format
                            audio_samples[f"eval_audio/{test_name}"] = wandb.Audio(arr, sample_rate=44100, caption=prompt)
                            logger.info(f"Added '{test_name}' to wandb log queue")
                        except Exception as e:
                             logger.exception(f"Error synthesizing test prompt for {test_name}: {e}")
                             continue
                
                # Log all audio samples at once
                if audio_samples:
                    logger.info(f"Logging {len(audio_samples)} audio samples to wandb")
                    wandb.log(audio_samples, step=global_step)
                    logger.info("Successfully logged demo audio samples to wandb")
                else:
                    logger.warning("No audio samples were generated for logging")
                
        except Exception as e:
            logger.exception(f"Eval demo generation failed: {e}")
        finally:
            # Restore training dtype
            if orig_dtype == torch.float16:
                logger.info("Restoring model to float16")
                unwrapped_model = unwrapped_model.half()
            elif orig_dtype == torch.bfloat16:
                logger.info("Restoring model to bfloat16")
                unwrapped_model = unwrapped_model.bfloat16()
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            
            logger.info(f"Completed eval demo generation at step {global_step}")
    
    # Synchronize all processes before returning to training
    if use_ddp:
        dist.barrier()


def train(model, dia_cfg: DiaConfig, dac_model: dac.DAC, dataset, train_cfg: TrainConfig, args, rank=0, world_size=1, use_ddp=False):
    """
    Run the full training loop over epochs with native PyTorch DDP.
    """
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    
    # Only rank 0 creates directories to avoid race conditions
    if rank == 0:
        train_cfg.output_dir.mkdir(parents=True, exist_ok=True)
        (train_cfg.runs_dir / train_cfg.run_name).mkdir(parents=True, exist_ok=True)
        # Create audio_demos directory for audio samples (supports mono/stereo)
        Path("./audio_demos").mkdir(exist_ok=True)
        
        # Initialize wandb
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
                "world_size": world_size,
            }
        )
    
    # Synchronize all processes
    if use_ddp:
        dist.barrier()
    
    # Move model to device BEFORE creating optimizer
    model = model.to(device)
    
    # Wrap model with DDP if using distributed training
    if use_ddp:
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    
    train_loader, val_loader = setup_loaders(dataset, dia_cfg, train_cfg, device, rank, world_size, use_ddp)
    opt, sched = setup_optimizer_and_scheduler(model, train_loader, train_cfg)

    model.train()

    steps_per_epoch = getattr(train_loader, 'steps_per_epoch', None)
    if steps_per_epoch is None:
        try:
            steps_per_epoch = len(train_loader)
        except Exception:
            steps_per_epoch = None

    stop_training = False
    for epoch in range(train_cfg.epochs):
        # Set epoch for DistributedSampler to ensure different shuffling across epochs
        if use_ddp and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        # Only show progress bar on rank 0
        if rank == 0:
            loader_iter = tqdm(
                train_loader,
                desc=f"E{epoch+1}",
                total=steps_per_epoch
            )
        else:
            loader_iter = train_loader
            
        for step, batch in enumerate(loader_iter):
            global_step = epoch * (steps_per_epoch or 0) + step
            
            # Add timing measurements for debugging
            batch_start = time.time()
            
            # training step
            loss = train_step_ddp(model, batch, dia_cfg, train_cfg, opt, sched, step, global_step, rank)
            
            total_step_time = time.time() - batch_start

            # Memory stats and progress update (only on rank 0)
            if rank == 0:
                cur_alloc = torch.cuda.memory_allocated()   # bytes currently allocated by tensors
                peak_alloc = torch.cuda.max_memory_allocated()  # bytes peak during program
                cur_gb  = cur_alloc  / 1024**3
                peak_gb = peak_alloc / 1024**3
                
                loader_iter.set_postfix({
                    'loss': f"{loss:.4f}",
                    'VRAM (GB)': f"{cur_gb:.2f}/{peak_gb:.2f}",
                    'step_time': f"{total_step_time:.1f}s"
                })
                
                torch.cuda.reset_peak_memory_stats()

            # evaluation during epoch (only if epoch-based eval is not requested)
            if args.eval_every_epochs is None and step > 0 and step % train_cfg.eval_step == 0:
                model.eval()
                with torch.no_grad():
                    eval_step(model, val_loader, dia_cfg, dac_model, global_step, device, train_cfg, rank, use_ddp)
                model.train()

            # checkpoint saving logic (only on rank 0) - step-based unless epoch-based save is requested
            should_save = False
            if args.save_every_epochs is None:
                if args.save_last is None:  # Normal saving behavior
                    if args.save_every is not None:
                        should_save = step > 0 and step % args.save_every == 0
                    else:
                        should_save = step > 0 and step % train_cfg.save_step == 0
                else:  # save_last is enabled, save every step/epoch but cleanup old ones
                    if args.save_every is not None:
                        should_save = step > 0 and step % args.save_every == 0
                    else:
                        should_save = step > 0 and step % train_cfg.save_step == 0
            
            if should_save and rank == 0:
                ckpt = train_cfg.output_dir / f"ckpt_step{global_step}.pth"
                state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                torch.save(state_dict, ckpt)
                logger.info(f"Saved checkpoint: {ckpt}")
                
                # Cleanup old checkpoints if save_last is specified
                cleanup_old_checkpoints(train_cfg.output_dir, args.save_last)

            # Early stopping check (DDP-safe, main process decides)
            if args.early_stop_loss is not None:
                trigger_local = loss <= args.early_stop_loss
                if use_ddp:
                    flag = torch.tensor(1 if (rank == 0 and trigger_local) else 0, device=device)
                    dist.broadcast(flag, src=0)
                    trigger = bool(flag.item())
                else:
                    trigger = trigger_local

                if trigger:
                    if rank == 0:
                        # Save final checkpoints
                        final_ckpt = train_cfg.output_dir / f"ckpt_final_step{global_step}.pth"
                        state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                        torch.save(state_dict, final_ckpt)
                        latest_ckpt = train_cfg.output_dir / "latest.pth"
                        torch.save(state_dict, latest_ckpt)
                        logger.info(f"Early stop triggered (loss <= {args.early_stop_loss}). Saved final checkpoints: {final_ckpt}, {latest_ckpt}")
                        # Cleanup old checkpoints if save_last is specified
                        cleanup_old_checkpoints(train_cfg.output_dir, args.save_last)
                    if use_ddp:
                        dist.barrier()
                    stop_training = True
                    break

        # End-of-epoch actions
        if stop_training:
            break

        # Epoch-end evaluation if requested
        if args.eval_every_epochs is not None and (epoch + 1) % args.eval_every_epochs == 0:
            model.eval()
            with torch.no_grad():
                gsp = ((epoch + 1) * (steps_per_epoch or 0)) - 1
                eval_step(model, val_loader, dia_cfg, dac_model, gsp if gsp >= 0 else 0, device, train_cfg, rank, use_ddp)
            model.train()

        # end of epoch checkpoint (only on rank 0)
        if rank == 0:
            is_last_epoch = (epoch + 1) == train_cfg.epochs
            
            if args.save_every_epochs is not None:
                # Save every N epochs and always on the final epoch
                if ((epoch + 1) % args.save_every_epochs == 0) or is_last_epoch:
                    ckpt_e = train_cfg.output_dir / f"ckpt_epoch{epoch+1}.pth"
                    state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                    torch.save(state_dict, ckpt_e)
                    logger.info(f"Saved end-of-epoch checkpoint: {ckpt_e}")
                    if is_last_epoch:
                        latest_ckpt = train_cfg.output_dir / "latest.pth"
                        torch.save(state_dict, latest_ckpt)
                        logger.info(f"Saved latest checkpoint: {latest_ckpt}")
                    cleanup_old_checkpoints(train_cfg.output_dir, args.save_last)
            else:
                # Default behavior: save epoch checkpoints if no step-based save_every provided, or always save final epoch
                if args.save_every is None or is_last_epoch:
                    ckpt_e = train_cfg.output_dir / f"ckpt_epoch{epoch+1}.pth"
                    state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                    torch.save(state_dict, ckpt_e)
                    logger.info(f"Saved end-of-epoch checkpoint: {ckpt_e}")
                    if is_last_epoch:
                        latest_ckpt = train_cfg.output_dir / "latest.pth"
                        torch.save(state_dict, latest_ckpt)
                        logger.info(f"Saved latest checkpoint: {latest_ckpt}")
                    cleanup_old_checkpoints(train_cfg.output_dir, args.save_last)
    
    # Synchronize all processes before ending
    if use_ddp:
        dist.barrier()


def train_step_ddp(model, batch, dia_cfg, train_cfg, opt, sched, step, global_step, rank):
    """
    Training step for DDP version.
    """
    if random.random() < train_cfg.unconditional_frac:
        pad_tok = dia_cfg.data.text_pad_value
        batch['src_tokens'] = torch.zeros_like(batch['src_tokens'])
        batch['enc_self_attn_mask'] = torch.zeros_like(batch['enc_self_attn_mask'])
        batch['dec_cross_attn_mask'] = torch.zeros_like(batch['dec_cross_attn_mask'])

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        # forward pass
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
        channel_weights = [1.0] * C
        loss_c = 0.0
        _, _, _, V = logits.size()
        for c, w in enumerate(channel_weights):
            lc = logits[:, :, c, :].reshape(-1, V)
            tc = target[:, :, c].reshape(-1)
            mc = mask[:, :, c].reshape(-1)
            lc_valid = lc[mc]
            tc_valid = tc[mc]
            loss_c += w * F.cross_entropy(
                lc_valid, tc_valid,
                ignore_index=pad_val
            )
        loss = loss_c / sum(channel_weights)

    loss = loss / train_cfg.grad_accum_steps
    loss.backward()
    
    if (step + 1) % train_cfg.grad_accum_steps == 0:
        # Clip once after accumulation, right before optimizer step
        pre_clip = clip_grad_norm_(model.parameters(), max_norm=5.0)
        if rank == 0:
            wandb.log({'grad_norm/pre_clip': pre_clip}, step=global_step)
            # Optional post-clip norm computation
            post_clip_sq = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    g = p.grad.detach()
                    post_clip_sq += g.float().norm(2).item() ** 2
            wandb.log({'grad_norm/post_clip': post_clip_sq ** 0.5}, step=global_step)
        opt.step()
        sched.step()
        opt.zero_grad()
        true_loss = loss.item() * train_cfg.grad_accum_steps
        current_lr = sched.get_last_lr()[0]
        if rank == 0:
            wandb.log({
                'learning_rate': current_lr,
                'train_loss': true_loss,
            }, step=global_step)
    
    return loss.item() * train_cfg.grad_accum_steps


def train_step_accelerate(model, batch, dia_cfg, train_cfg, opt, sched, step, global_step, accelerator):
    """
    Like train_step, but uses accelerator for backward and logging.
    """
    if random.random() < train_cfg.unconditional_frac:
        pad_tok = dia_cfg.data.text_pad_value
        batch['src_tokens'] = torch.zeros_like(batch['src_tokens'])
        batch['enc_self_attn_mask'] = torch.zeros_like(batch['enc_self_attn_mask'])
        batch['dec_cross_attn_mask'] = torch.zeros_like(batch['dec_cross_attn_mask'])

    # forward pass (autocast handled by caller)
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
    channel_weights = [1.0] * C
    loss_c = 0.0
    _, _, _, V = logits.size()
    for c, w in enumerate(channel_weights):
        lc = logits[:, :, c, :].reshape(-1, V)
        tc = target[:, :, c].reshape(-1)
        mc = mask[:, :, c].reshape(-1)
        lc_valid = lc[mc]
        tc_valid = tc[mc]
        loss_c += w * F.cross_entropy(
            lc_valid, tc_valid,
            ignore_index=pad_val
        )
    loss = loss_c / sum(channel_weights)
    loss = loss / train_cfg.grad_accum_steps
    accelerator.backward(loss)
    if (step + 1) % train_cfg.grad_accum_steps == 0:
        # Clip once after accumulation, right before optimizer step
        pre_clip = clip_grad_norm_(model.parameters(), max_norm=5.0)
        if accelerator.is_main_process:
            wandb.log({'grad_norm/pre_clip': pre_clip}, step=global_step)
            # Optional post-clip norm computation
            post_clip_sq = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    g = p.grad.detach()
                    post_clip_sq += g.float().norm(2).item() ** 2
            wandb.log({'grad_norm/post_clip': post_clip_sq ** 0.5}, step=global_step)
        opt.step()
        sched.step()
        opt.zero_grad()
        true_loss = loss.item() * train_cfg.grad_accum_steps
        current_lr = sched.get_last_lr()[0]
        if accelerator.is_main_process:
            wandb.log({
                'learning_rate': current_lr,
                'train_loss': true_loss,
            }, step=global_step)
    return loss.item() * train_cfg.grad_accum_steps



def run_ddp_worker(rank: int, world_size: int, args):
    """Worker function for DDP training."""
    try:
        # Setup DDP
        setup_ddp(rank, world_size)
        
        # Set deterministic seeds
        seed_everything(args.seed, rank=rank)
        
        # Optional: improve bf16 throughput on Ampere+ GPUs
        torch.set_float32_matmul_precision("high")
        
        device = torch.device(f"cuda:{rank}")
        
        # Load config
        dia_cfg = DiaConfig.load(args.config)
        
        # Configure tag augmentation for this process
        global TAG_SHUFFLE, TAG_DROPOUT, TAG_LIMIT
        TAG_SHUFFLE = False if getattr(args, 'tag_no_shuffle', False) else bool(getattr(args, 'tag_shuffle', True))
        TAG_DROPOUT = float(getattr(args, 'tag_dropout', 0.0))
        TAG_LIMIT = getattr(args, 'tag_limit', None)

        
        if rank == 0:
            print(f"Loaded config from: {args.config}")
        
        # Load DAC model
        if rank == 0:
            print("Loading DAC model...")
        dac_model = dac.DAC.load(dac.utils.download()).to(device)

        dataset=None


        # choose dataset
        if not dataset:
            use_sliding_window = not args.disable_sliding_window
            if args.preencoded_dir:
                dataset = PreEncodedDACDataset(args.preencoded_dir, dia_cfg, use_sliding_window)
            elif args.audio_folder:
                dataset = MusicDataset(args.audio_folder, dia_cfg, dac_model, use_sliding_window)
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
            eval_step = args.eval_step,
            run_name = args.run_name or TrainConfig.run_name,
            output_dir = args.output_dir,
            seed = args.seed,
            no_decay_embed = args.no_decay_embed,
        )

        # load model checkpoint
        if args.local_ckpt:
            ckpt_file = args.local_ckpt
        else:
            ckpt_file = hf_hub_download(args.hub_model, filename="dia-v0_1.pth")
        model = DiaModel(dia_cfg)
        state = torch.load(ckpt_file, map_location="cpu")
        # Adapt checkpoint for stereo (9 -> 18 channels) before loading to avoid size mismatch
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
                    # If checkpoint has 9 channels and model expects 18, duplicate along channel dim
                    if W_ckpt.shape[1] * 2 == expected_W.shape[1]:
                        state[key] = torch.cat([W_ckpt, W_ckpt], dim=1)
                        logger.info(
                            f"Expanded {key} from {tuple(W_ckpt.shape)} to {tuple(state[key].shape)} by duplication"
                        )
                    else:
                        # Fallback: drop the mismatched tensor so it's treated as missing
                        del state[key]
                        logger.warning(
                            f"Removed {key} from checkpoint due to incompatible shape {tuple(W_ckpt.shape)} -> expected {tuple(expected_W.shape)}"
                        )
        except Exception as e:
            logger.warning(f"While adapting checkpoint weights: {e}")
        missing, unexpected = model.load_state_dict(state, strict=False)
        logger.info(f"Loaded checkpoint with strict=False; missing={len(missing)}, unexpected={len(unexpected)}")
        # Warm-start stereo by duplicating left channel params into right, if expanding 9->18
        try:
            if dia_cfg.data.channels == 18:
                if hasattr(model.decoder, "embeddings") and len(model.decoder.embeddings) >= 18:
                    for i in range(9, 18):
                        model.decoder.embeddings[i].weight.data.copy_(model.decoder.embeddings[i - 9].weight.data)
                W = model.decoder.logits_dense.weight  # (E, C, V)
                if W.dim() == 3 and W.shape[1] >= 18:
                    W.data[:, 9:18, :].copy_(W.data[:, 0:9, :])
        except Exception as e:
            logger.warning(f"Stereo warm-start duplication skipped: {e}")
        # Release checkpoint tensors from CPU memory ASAP to reduce per-rank RSS
        del state
        gc.collect()

        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        logger.info(f"Rank {rank} after checkpoint load RSS ~{rss_mb:.1f} MB (PID {os.getpid()})")

        # Ensure all parameters have consistent dtype for DDP
        if args.half:
            model = model.half()
            # Force all parameters to half precision to avoid DDP dtype mismatch
            for param in model.parameters():
                if param.dtype != torch.float16:
                    param.data = param.data.half()
        else:
            # Force all parameters to float32 to ensure consistency
            for param in model.parameters():
                if param.dtype != torch.float32:
                    param.data = param.data.float()
            
        if args.compile:
            model = torch.compile(model, backend="inductor")
        
        # Synchronize before training
        dist.barrier()
        
        # start training with DDP
        train(model, dia_cfg, dac_model, dataset, train_cfg, args, rank, world_size, use_ddp=True)
        
    except Exception as e:
        print(f"Error in rank {rank}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup_ddp()


def main():
    args = get_args()
    
    # Determine world size for DDP
    world_size = torch.cuda.device_count()
    
    if world_size < 2 or args.force_single_gpu:
        if args.force_single_gpu:
            print("Forcing single GPU training as requested")
        else:
            print("WARNING: Multi-GPU training requested but only 1 GPU available, falling back to single GPU training")
        # Single GPU fallback
        # Set deterministic seeds
        seed_everything(args.seed, rank=0)
        
        # Optional: improve bf16 throughput on Ampere+ GPUs
        torch.set_float32_matmul_precision("high")
        
        dia_cfg = DiaConfig.load(args.config)
        # Configure tag augmentation for single process
        global TAG_SHUFFLE, TAG_DROPOUT, TAG_LIMIT
        TAG_SHUFFLE = False if getattr(args, 'tag_no_shuffle', False) else bool(getattr(args, 'tag_shuffle', True))
        TAG_DROPOUT = float(getattr(args, 'tag_dropout', 0.0))
        TAG_LIMIT = getattr(args, 'tag_limit', None)
        device = torch.device("cuda:0")
        dac_model = dac.DAC.load(dac.utils.download()).to(device)
        
        dataset=None
        if not dataset:
            use_sliding_window = not args.disable_sliding_window
            if args.preencoded_dir:
                dataset = PreEncodedDACDataset(args.preencoded_dir, dia_cfg, use_sliding_window)
            elif args.audio_folder:
                dataset = MusicDataset(args.audio_folder, dia_cfg, dac_model, use_sliding_window)
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
            eval_step = args.eval_step,
            run_name = args.run_name or TrainConfig.run_name,
            output_dir = args.output_dir,
            seed = args.seed,
            no_decay_embed = args.no_decay_embed,
        )

        if args.local_ckpt:
            ckpt_file = args.local_ckpt
        else:
            ckpt_file = hf_hub_download(args.hub_model, filename="dia-v0_1.pth")
        model = DiaModel(dia_cfg)
        state = torch.load(ckpt_file, map_location="cpu")
        # Adapt checkpoint for stereo (9 -> 18 channels) before loading to avoid size mismatch
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
                        logger.info(
                            f"Expanded {key} from {tuple(W_ckpt.shape)} to {tuple(state[key].shape)} by duplication"
                        )
                    else:
                        del state[key]
                        logger.warning(
                            f"Removed {key} from checkpoint due to incompatible shape {tuple(W_ckpt.shape)} -> expected {tuple(expected_W.shape)}"
                        )
        except Exception as e:
            logger.warning(f"While adapting checkpoint weights: {e}")
        missing, unexpected = model.load_state_dict(state, strict=False)
        logger.info(f"Loaded checkpoint with strict=False; missing={len(missing)}, unexpected={len(unexpected)}")
        try:
            if dia_cfg.data.channels == 18:
                if hasattr(model.decoder, "embeddings") and len(model.decoder.embeddings) >= 18:
                    for i in range(9, 18):
                        model.decoder.embeddings[i].weight.data.copy_(model.decoder.embeddings[i - 9].weight.data)
                W = model.decoder.logits_dense.weight
                if W.dim() == 3 and W.shape[1] >= 18:
                    W.data[:, 9:18, :].copy_(W.data[:, 0:9, :])
        except Exception as e:
            logger.warning(f"Stereo warm-start duplication skipped: {e}")
        
        # Ensure all parameters have consistent dtype
        if args.half:
            model = model.half()
            # Force all parameters to half precision for consistency
            for param in model.parameters():
                if param.dtype != torch.float16:
                    param.data = param.data.half()
        else:
            # Force all parameters to float32 to ensure consistency
            for param in model.parameters():
                if param.dtype != torch.float32:
                    param.data = param.data.float()
            
        # Optionally freeze encoder for primarily unconditional training (reduces drift)
        try:
            if train_cfg.unconditional_frac >= 0.9:
                for p in model.encoder.parameters():
                    p.requires_grad = False
                print("Frozen encoder parameters due to high unconditional_frac")
        except Exception:
            pass

        if args.compile:
            model = torch.compile(model, backend="inductor")
        
        # Single GPU training
        train(model, dia_cfg, dac_model, dataset, train_cfg, args, rank=0, world_size=1, use_ddp=False)
        return

    # Multi-GPU DDP training
    print(f"Launching DDP training with {world_size} processes...")

    # Configure rendezvous settings in the parent before spawning workers
    init_file_path = Path(tempfile.gettempdir()) / f"dia_ddp_init_{uuid.uuid4().hex}"
    os.environ["TORCH_DDP_INIT_METHOD"] = f"file://{init_file_path}"
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("GLOO_SOCKET_FAMILY", "AF_INET")
    os.environ.setdefault("NCCL_SOCKET_FAMILY", "AF_INET")
    
    # Set multiprocessing start method for DDP
    mp.set_start_method('spawn', force=True)
    
    return_code = None
    try:
        mp.spawn(
            run_ddp_worker,
            args=(world_size, args),
            nprocs=world_size,
            join=True
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return_code = 0
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return_code = 1
    finally:
        try:
            os.environ.pop("TORCH_DDP_INIT_METHOD", None)
        except Exception:
            pass
        try:
            if init_file_path.exists():
                init_file_path.unlink(missing_ok=True)
        except Exception:
            pass

    if return_code is not None:
        return return_code


if __name__ == "__main__":
    main()