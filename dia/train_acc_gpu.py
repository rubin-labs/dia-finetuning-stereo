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
import re

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

# -- Codebook Weighting (used in loss calculation) --
# Weights for each of the 9 DAC codebooks. First codebooks capture coarse features,
# later codebooks capture fine details. Adjust to emphasize different aspects.
CODEBOOK_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# -- Output Settings --
EVAL_SAMPLE_RATE = 44100       # Sample rate for saved audio files
EVAL_AUDIO_DIR = "./audio_demos"  # Directory to save demo audio files

# =============================================================================


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
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)


def _worker_init_fn(worker_id):
    """Initialize worker seeds for deterministic DataLoader workers."""
    ws = torch.initial_seed() % 2**32
    np.random.seed(ws)
    random.seed(ws)


# Music-specific processing can be added here if needed
# NOTE: Test prompts moved to top of file (see TEST_PROMPTS in config section)


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


def load_train_config(config_path: Path) -> dict:
    """Load training config from JSON file."""
    import json
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
    # First pass: just get train_config path
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--train_config", type=Path, default=Path("configs/train_config.json"),
                            help="Path to training config JSON (defaults loaded from here, CLI overrides)")
    pre_args, _ = pre_parser.parse_known_args()
    
    # Load defaults from train config
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
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--half", action="store_true", help="load model in fp16")
    parser.add_argument("--compile", action="store_true", help="torch compile model")
    parser.add_argument("--wandb_project", type=str, default="dia-music-finetuning",
                        help="Weights & Biases project name.")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="Weights & Biases entity/team name.")
    parser.add_argument("--save_every", type=int, default=cfg_defaults.get('save_every'),
                        help="Save checkpoint every N steps (overrides TrainConfig.save_step).")
    parser.add_argument("--save_last", type=int, default=None,
                        help="Keep only the last N checkpoints (e.g., --save_last 4). Saves disk space.")
    parser.add_argument("--force_single_gpu", action="store_true", default=cfg_defaults.get('force_single_gpu', False),
                        help="Force single GPU training even with multiple GPUs available")
    # Tag augmentation flags
    parser.add_argument("--tag_shuffle", action="store_true", default=True,
                        help="Shuffle comma-separated tags in prompts (default: on)")
    parser.add_argument("--tag_no_shuffle", action="store_true", default=cfg_defaults.get('tag_no_shuffle', False),
                        help="Disable tag shuffling (overrides --tag_shuffle)")
    parser.add_argument("--tag_dropout", type=float, default=0.0,
                        help="Per-tag dropout probability in [0,1] (default: 0.0)")
    parser.add_argument("--tag_limit", type=int, default=None,
                        help="Keep at most this many tags after shuffle/dropout (default: unlimited)")
    
    # Training hyperparameters
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
                        help="Calculate validation loss every N steps.")
    parser.add_argument("--demo_every", type=int, default=cfg_defaults.get('demo_every'),
                        help="Generate audio demos every N steps (if None, defaults to same as --eval_step).")
    parser.add_argument("--eval_every_epochs", type=int, default=cfg_defaults.get('eval_every_epochs'),
                        help="Evaluate at the end of every N epochs (overrides step-based eval).")
    parser.add_argument("--demo_every_epochs", type=int, default=cfg_defaults.get('demo_every_epochs'),
                        help="Generate audio demos every N epochs (use with --eval_every_epochs).")
    parser.add_argument("--save_every_epochs", type=int, default=None,
                        help="Save checkpoint at the end of every N epochs (overrides step-based save).")
    parser.add_argument("--save_after_epoch", type=int, default=cfg_defaults.get('save_after_epoch', 0),
                        help="Only start saving checkpoints after this epoch (default: 0, save from start).")
    parser.add_argument("--demo_after_epoch", type=int, default=0,
                        help="Only start generating demos after this epoch (default: 0, demo from start).")
    parser.add_argument("--early_stop_loss", type=float, default=None,
                        help="Early stop when training loss <= this value; saves final checkpoint then exits.")
    parser.add_argument("--stop_on_overfit", action="store_true",
                        help="Stop training and generate demo when eval loss > train loss")
    
    # Sliding window augmentation (default: disabled for deterministic training)
    parser.add_argument("--use_sliding_window", action="store_true", default=cfg_defaults.get('use_sliding_window', False),
                        help="Enable sliding window: random cropping for data augmentation (default: off for deterministic training)")
    # Optimizer param-group controls
    parser.add_argument("--no_decay_embed", action="store_true",
                        help="Exclude nn.Embedding parameters from weight decay")
    
    # Dataset filtering
    parser.add_argument("--require_prompts", action="store_true",
                        help="Fail if audio file is missing a corresponding prompt file (default: skip missing)")
    parser.add_argument("--skip_tags", type=str, default=None,
                        help="Comma-separated list of tags to skip (e.g. 'vocals,speech')")
    parser.add_argument("--scratch", action="store_true", default=cfg_defaults.get('scratch', False),
                        help="Train from scratch (random initialization) instead of loading a checkpoint.")

    args = parser.parse_args()
    
    # Validate required fields
    if args.output_dir is None:
        parser.error("--output_dir is required (set in train_config.json or via CLI)")
    if args.unconditional_frac is None:
        parser.error("--unconditional_frac is required (set in train_config.json or via CLI)")
    
    return args



def collate_fn(batch, config: DiaConfig, device: torch.device, use_sliding_window: bool = True):
    texts, encodings, waveforms = zip(*batch)

        # -- Enforce max length and optional random cropping --
    # ALWAYS enforce max length (safety net, even if dataset should have cropped)
    # use_sliding_window=True: random crop position (data augmentation)
    # use_sliding_window=False: fixed crop from start (deterministic)
    window_size = config.data.audio_length
    cropped_encodings = []
    for e in encodings:
        if e.size(0) > window_size:
            if use_sliding_window:
                # Random crop for data augmentation
                # Only use randomness if we have plenty of extra duration (> 5 seconds extra)
                # This prevents "jittering" around the same small clip which might confuse the model on exact timing
                # For overfitting 1 sample, we really want DETERMINISTIC behavior to memorize it.
                start = random.randint(0, e.size(0) - window_size)
            else:
                # Fixed crop from start for deterministic training
                start = 0
            cropped_encodings.append(e[start : start + window_size])
        else:
            cropped_encodings.append(e)
    encodings = cropped_encodings

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

    # -- Audio codes: pad to batch max and track original lengths --------------

    # Find the maximum length in this batch
    batch_max = max(e.size(0) for e in encodings)
    
    # Pad all sequences to the batch maximum length
    padded_encodings = []
    for e in encodings:
        if e.size(0) < batch_max:
            # Pad shorter sequences to batch_max
            pad_length = batch_max - e.size(0)
            pad_value = config.data.audio_pad_value
            padding = torch.full((pad_length, e.size(1)), pad_value, dtype=e.dtype, device=e.device)
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

def setup_loaders(dataset, dia_cfg: DiaConfig, train_cfg: TrainConfig, device, rank=0, world_size=1, use_ddp=False, use_sliding_window=True):
    collate = lambda b: collate_fn(b, dia_cfg, device, use_sliding_window)
    
    ds_len = len(dataset)
    n_train = int(train_cfg.split_ratio * ds_len)
    n_val = ds_len - n_train
    
    # If dataset has only 1 sample (or split would result in 0 val samples), skip validation
    if ds_len <= 1 or n_val == 0:
        if rank == 0:
            logger.info(f"Dataset has {ds_len} sample(s) - skipping validation split, using all data for training")
        train_ds = dataset
        val_ds = None
    else:
        g = torch.Generator().manual_seed(train_cfg.seed)
        train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=g)
    
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
    
    # Only create val_loader if we have validation data
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds, 
            batch_size=1, 
            shuffle=False, 
            collate_fn=collate,
            num_workers=0,  # Disable workers for DDP
            pin_memory=False  # Disabled since collate_fn puts data on GPU
        )
    else:
        val_loader = None
    
    # Set steps_per_epoch attribute for tqdm
    # DataLoader already handles batch_size division, so just use its length
    steps_per_epoch = len(train_loader)
    
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

    opt = optim.AdamW(
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
        num_warmup_steps=train_cfg.warmup_steps // train_cfg.grad_accum_steps,
        num_training_steps=total_training_steps // train_cfg.grad_accum_steps
    )
    return opt, sched



def eval_step(model, val_loader, dia_cfg, dac_model, global_step, device, train_cfg, rank=0, use_ddp=False, do_demo=True, current_train_loss=None, stop_on_overfit=False):
    """
    Run evaluation: calculate loss and optionally generate audio demos
    """
    eval_losses = []
    last_batch = None
    
    if val_loader is not None:
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
                # Custom weighting for eval as well to match training objective
                weights_e = []
                num_groups = C_e // 9
                if num_groups > 0:
                    for _ in range(num_groups):
                        weights_e.extend(CODEBOOK_WEIGHTS)
                else:
                    weights_e = [1.0] * C_e
                
                # CRITICAL: Only compute loss on actual audio tokens (0-1023)
                # Match training loss by excluding BOS/EOS/PAD tokens
                audio_token_mask_e = (target >= 0) & (target <= 1023)  # (B, T, C)
                
                for c, w in enumerate(weights_e):
                    lc = logits[:, :, c, :].reshape(-1, V_e)
                    tc = target[:, :, c].reshape(-1)
                    audio_mc = audio_token_mask_e[:, :, c].reshape(-1)
                    lc_valid = lc[audio_mc]
                    tc_valid = tc[audio_mc]
                    if tc_valid.numel() > 0:  # Only compute if we have valid tokens
                        loss_e += w * F.cross_entropy(
                            lc_valid, tc_valid, ignore_index=dia_cfg.data.audio_pad_value
                        )
                loss_e = loss_e / sum(weights_e)

                eval_losses.append(loss_e)

    should_stop = False
    if len(eval_losses) > 0:
        avg_eval_loss = sum(eval_losses) / len(eval_losses)
        if rank == 0:
            wandb.log({'eval_loss': avg_eval_loss.item()}, step=global_step)
            
            if stop_on_overfit and current_train_loss is not None:
                if avg_eval_loss > current_train_loss:
                    logger.info(f"Stop trigger: Eval loss {avg_eval_loss:.4f} > Train loss {current_train_loss:.4f}. Generating demo and stopping.")
                    should_stop = True
                    do_demo = True
    else:
        if rank == 0 and val_loader is not None:
            logger.warning("No validation samples available for evaluation - check split_ratio")

    # Only generate demos if requested
    if not do_demo:
        return should_stop

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
                    # Fully unconditional training - generate samples at multiple temperatures
                    # This helps track how embeddings are learning over time:
                    # - temp=0.0: deterministic, shows what model actually learned
                    # - temp=0.5: moderate sampling, tests confidence  
                    # - temp=1.0: full sampling, tests robustness
                    seeds = [int(train_cfg.seed), int(train_cfg.seed) + 1]
                    temperatures = EVAL_TEMPERATURES  # [0.0, 0.5, 1.0]
                    total_demos = len(seeds) * len(temperatures)
                    logger.info(f"Generating {total_demos} unconditional demos ({len(seeds)} seeds × {len(temperatures)} temperatures)")
                    
                    # Save current RNG states to avoid impacting subsequent training randomness
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
                                    
                                    # Save audio file to demo directory
                                    temp_str = f"{temp:.1f}".replace(".", "p")  # 0.5 -> "0p5"
                                    audio_filename = f"step_{global_step}_temp{temp_str}_seed{s}.wav"
                                    audio_path = Path(EVAL_AUDIO_DIR) / audio_filename
                                    arr = audio
                                    if isinstance(arr, torch.Tensor):
                                        arr = arr.detach().cpu().numpy()
                                    if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[0] <= 8 and arr.shape[1] > arr.shape[0]:
                                        arr = arr.T
                                    sf.write(audio_path, arr, EVAL_SAMPLE_RATE)
                                    logger.info(f"Saved demo audio: {audio_path}")
                                    
                                    # Convert to wandb Audio format - organize by temperature
                                    audio_samples[f"eval_audio/temp{temp_str}/seed{s}"] = wandb.Audio(
                                        arr, sample_rate=EVAL_SAMPLE_RATE, 
                                        caption=f"temp={temp}, seed={s}"
                                    )
                                    logger.info(f"Added sample (temp={temp}, seed={s}) to wandb log queue")
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
                    # Conditional training - use test prompts at multiple temperatures
                    # This helps track how embeddings are learning with conditioning:
                    # - temp=0.0: deterministic, shows what model learned for this prompt
                    # - temp=0.5: moderate sampling, tests confidence
                    # - temp=1.0: full sampling, tests robustness
                    temperatures = EVAL_TEMPERATURES  # [0.0, 0.5, 1.0]
                    total_demos = len(TEST_PROMPTS) * len(temperatures)
                    logger.info(f"Generating {total_demos} conditional demos ({len(TEST_PROMPTS)} prompts × {len(temperatures)} temperatures)")
                    
                    cfg_scale = EVAL_CFG_SCALE if train_cfg.unconditional_frac > 0 else None
                    for temp in temperatures:
                        for test_name, prompt in TEST_PROMPTS.items():
                            try:
                                logger.info(f"Generating audio for '{test_name}' (temp={temp}) with prompt: '{prompt}'")
                                audio = dia_gen.generate(
                                    text=prompt,
                                    cfg_scale=cfg_scale,
                                    temperature=temp,
                                    top_p=EVAL_TOP_P
                                )
                                
                                # Save audio file to demo directory
                                temp_str = f"{temp:.1f}".replace(".", "p")  # 0.5 -> "0p5"
                                audio_filename = f"step_{global_step}_{test_name}_temp{temp_str}.wav"
                                audio_path = Path(EVAL_AUDIO_DIR) / audio_filename
                                arr = audio
                                if isinstance(arr, torch.Tensor):
                                    arr = arr.detach().cpu().numpy()
                                if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[0] <= 8 and arr.shape[1] > arr.shape[0]:
                                    arr = arr.T
                                sf.write(audio_path, arr, EVAL_SAMPLE_RATE)
                                logger.info(f"Saved demo audio: {audio_path}")
                                
                                # Convert to wandb Audio format - organize by temperature
                                audio_samples[f"eval_audio/temp{temp_str}/{test_name}"] = wandb.Audio(
                                    arr, sample_rate=EVAL_SAMPLE_RATE, 
                                    caption=f"{prompt} (temp={temp})"
                                )
                                logger.info(f"Added '{test_name}' (temp={temp}) to wandb log queue")
                            except Exception as e:
                                 logger.exception(f"Error synthesizing '{test_name}' at temp={temp}: {e}")
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
    
    return should_stop


def train(model, dia_cfg: DiaConfig, dac_model: dac.DAC, dataset, train_cfg: TrainConfig, args, rank=0, world_size=1, use_ddp=False):
    """
    Run the full training loop over epochs with native PyTorch DDP.
    """
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    
    # Only rank 0 creates directories to avoid race conditions
    if rank == 0:
        train_cfg.output_dir.mkdir(parents=True, exist_ok=True)
        (train_cfg.runs_dir / train_cfg.run_name).mkdir(parents=True, exist_ok=True)
        # Create audio demos directory for audio samples (supports mono/stereo)
        Path(EVAL_AUDIO_DIR).mkdir(exist_ok=True)
        
        # Skip vocabulary coverage computation for large datasets (assume 100%)
        # vocab_stats = compute_vocabulary_coverage(dataset)
        # logger.info(f"Vocabulary coverage: {vocab_stats['num_unique']}/1024 tokens ({vocab_stats['coverage_pct']:.1f}%)")
        # logger.info(f"Total tokens in dataset: {vocab_stats['total_tokens']:,}")
        
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
    
    use_sliding_window = args.use_sliding_window
    train_loader, val_loader = setup_loaders(dataset, dia_cfg, train_cfg, device, rank, world_size, use_ddp, use_sliding_window)
    opt, sched = setup_optimizer_and_scheduler(model, train_loader, train_cfg)

    # --- Sanity Check REMOVED (was causing crashes due to BOS/EOS tokens) ---
    # Use scripts/check_preencoded.py instead.

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
            if args.eval_every_epochs is None and global_step > 0 and global_step % train_cfg.eval_step == 0:
                model.eval()
                # Determine if we should generate demos (respecting demo_after_epoch)
                demo_interval = args.demo_every if args.demo_every is not None else train_cfg.eval_step
                past_demo_threshold = (epoch + 1) > args.demo_after_epoch
                do_demo = (global_step % demo_interval == 0) and past_demo_threshold
                
                # Run eval step if we have val_loader OR if we need to generate demos
                if val_loader is not None or do_demo:
                    with torch.no_grad():
                        should_stop = eval_step(model, val_loader, dia_cfg, dac_model, global_step, device, train_cfg, rank, use_ddp, do_demo=do_demo, current_train_loss=loss, stop_on_overfit=args.stop_on_overfit)
                
                if args.stop_on_overfit and val_loader is not None:
                    if use_ddp:
                        flag = torch.tensor(1 if (rank == 0 and should_stop) else 0, device=device)
                        dist.broadcast(flag, src=0)
                        should_stop = bool(flag.item())
                    
                    if should_stop:
                        stop_training = True
                        if rank == 0:
                            ckpt_path = train_cfg.output_dir / f"ckpt_stop_overfit_{global_step}.pth"
                            state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                            torch.save(state_dict, ckpt_path)
                            logger.info(f"Saved overfit-stop checkpoint: {ckpt_path}")
                            
                model.train()

            # checkpoint saving logic (only on rank 0) - step-based unless epoch-based save is requested
            should_save = False
            past_save_threshold = (epoch + 1) > args.save_after_epoch
            if args.save_every_epochs is None and past_save_threshold:
                if args.save_last is None:  # Normal saving behavior
                    if args.save_every is not None:
                        should_save = global_step > 0 and global_step % args.save_every == 0
                    else:
                        should_save = global_step > 0 and global_step % train_cfg.save_step == 0
                else:  # save_last is enabled, save every step/epoch but cleanup old ones
                    if args.save_every is not None:
                        should_save = global_step > 0 and global_step % args.save_every == 0
                    else:
                        should_save = global_step > 0 and global_step % train_cfg.save_step == 0
            
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
        # We check for eval_every_epochs (for loss) OR demo_every_epochs (for demos)
        # If val_loader is None, we can't compute loss, but we can still do demos
        should_check_epoch = False
        if args.eval_every_epochs is not None and (epoch + 1) % args.eval_every_epochs == 0:
             should_check_epoch = True
        if args.demo_every_epochs is not None and (epoch + 1) % args.demo_every_epochs == 0:
             should_check_epoch = True
             
        if should_check_epoch:
            model.eval()
            with torch.no_grad():
                gsp = ((epoch + 1) * (steps_per_epoch or 0)) - 1
                # For epoch-based eval, check demo_every_epochs first, then fallback to demo_every (steps)
                # Also respect demo_after_epoch threshold
                past_demo_threshold = (epoch + 1) > args.demo_after_epoch
                if args.demo_every_epochs is not None:
                    do_demo = ((epoch + 1) % args.demo_every_epochs == 0) and past_demo_threshold
                elif args.demo_every is None:
                    do_demo = past_demo_threshold
                else:
                    do_demo = (gsp > 0 and gsp % args.demo_every == 0) and past_demo_threshold
                
                # Only run if we have validation data OR we want to demo
                if val_loader is not None or do_demo:
                    should_stop = eval_step(model, val_loader, dia_cfg, dac_model, gsp if gsp >= 0 else 0, device, train_cfg, rank, use_ddp, do_demo=do_demo, current_train_loss=loss, stop_on_overfit=args.stop_on_overfit)
                
                    if args.stop_on_overfit and val_loader is not None:
                        if use_ddp:
                            flag = torch.tensor(1 if (rank == 0 and should_stop) else 0, device=device)
                            dist.broadcast(flag, src=0)
                            should_stop = bool(flag.item())
                        
                        if should_stop:
                            stop_training = True
                            if rank == 0:
                                ckpt_path = train_cfg.output_dir / f"ckpt_stop_overfit_{gsp}.pth"
                                state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                                torch.save(state_dict, ckpt_path)
                                logger.info(f"Saved overfit-stop checkpoint: {ckpt_path}")

            model.train()

        # end of epoch checkpoint (only on rank 0)
        if rank == 0:
            is_last_epoch = (epoch + 1) == train_cfg.epochs
            past_save_threshold = (epoch + 1) > args.save_after_epoch
            
            if args.save_every_epochs is not None:
                # Save every N epochs and always on the final epoch (respecting save_after_epoch)
                should_save_epoch = ((epoch + 1) % args.save_every_epochs == 0) and past_save_threshold
                if should_save_epoch or is_last_epoch:
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
                # (respecting save_after_epoch threshold)
                should_save_epoch = (args.save_every is None and past_save_threshold) or is_last_epoch
                if should_save_epoch:
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
    # Deterministic unconditional decision to keep ranks in sync independent of local random state
    # (which might drift due to data loader differences)
    gen_val = ((global_step * 997 + train_cfg.seed) % 10000) / 10000.0
    if gen_val < train_cfg.unconditional_frac:
        pad_tok = dia_cfg.data.text_pad_value
        batch['src_tokens'].fill_(pad_tok)
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
        
        # Custom weighting: weights defined in CODEBOOK_WEIGHTS config at top of file
        channel_weights = []
        num_groups = C // 9
        if num_groups > 0:
            for _ in range(num_groups):
                channel_weights.extend(CODEBOOK_WEIGHTS)
        else:
            channel_weights = [1.0] * C

        loss_c = 0.0
        _, _, _, V = logits.size()
        
        # CRITICAL: Only compute loss on actual audio tokens (0-1023)
        # Mask out BOS (1026), EOS (1024), and PAD (1025) - these are trivial to predict
        # and cause artificially low loss if included (model "cheats" by learning delay pattern)
        audio_token_mask = (target >= 0) & (target <= 1023)  # (B, T, C)
        
        for c, w in enumerate(channel_weights):
            lc = logits[:, :, c, :].reshape(-1, V)
            tc = target[:, :, c].reshape(-1)
            mc = mask[:, :, c].reshape(-1)
            # Combine time mask with audio token mask
            audio_mc = audio_token_mask[:, :, c].reshape(-1)
            combined_mask = mc & audio_mc
            lc_valid = lc[combined_mask]
            tc_valid = tc[combined_mask]
            if tc_valid.numel() > 0:  # Only compute if we have valid tokens
                loss_c += w * F.cross_entropy(
                    lc_valid, tc_valid,
                    ignore_index=pad_val
                )
        loss = loss_c / sum(channel_weights)
        
        # Compute output entropy every 50 steps (measures model confidence)
        # Lower entropy = more confident predictions = embeddings are learning
        if global_step % 50 == 0:
            entropy = compute_output_entropy(logits.detach())
            if rank == 0:
                wandb.log({'output_entropy': entropy}, step=global_step)

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
        
        # Load DAC model (must be in eval mode for deterministic encoding)
        if rank == 0:
            print("Loading DAC model...")
        dac_model = dac.DAC.load(dac.utils.download()).eval().to(device)

        # Choose dataset
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
            eval_step = args.eval_step,
            run_name = args.run_name or TrainConfig.run_name,
            output_dir = args.output_dir,
            seed = args.seed,
            no_decay_embed = args.no_decay_embed,
        )

        # Initialize model
        model = DiaModel(dia_cfg)

        if args.scratch:
            if rank == 0:
                print("Initializing model from scratch (random weights)...")
            if hasattr(model, '_init_weights'):
                model._init_weights()
            expanded_stereo = False
        else:
            # load model checkpoint
            if args.local_ckpt:
                ckpt_file = args.local_ckpt
            else:
                ckpt_file = hf_hub_download(args.hub_model, filename="dia-v0_1.pth")
            
            state = torch.load(ckpt_file, map_location="cpu")
            # Track if we performed expansion to know if we should warm-start stereo
            expanded_stereo = False

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
                            expanded_stereo = True
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
            
            # Warm-start stereo by duplicating left channel params into right, ONLY if expanding 9->18
            # If we loaded an existing stereo checkpoint, expanded_stereo is False, so we skip this
            if expanded_stereo:
                try:
                    if dia_cfg.data.channels == 18:
                        logger.info("Warm-starting stereo channels (copying Left -> Right)...")
                        if hasattr(model.decoder, "embeddings") and len(model.decoder.embeddings) >= 18:
                            for i in range(9, 18):
                                model.decoder.embeddings[i].weight.data.copy_(model.decoder.embeddings[i - 9].weight.data)
                        W = model.decoder.logits_dense.weight  # (E, C, V)
                        if W.dim() == 3 and W.shape[1] >= 18:
                            W.data[:, 9:18, :].copy_(W.data[:, 0:9, :])
                except Exception as e:
                    logger.warning(f"Stereo warm-start duplication skipped: {e}")
            else:
                if dia_cfg.data.channels == 18:
                    logger.info("Skipping stereo warm-start duplication (loaded checkpoint appears to be stereo already)")
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
        dac_model = dac.DAC.load(dac.utils.download()).eval().to(device)
        
        # Choose dataset
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
            eval_step = args.eval_step,
            run_name = args.run_name or TrainConfig.run_name,
            output_dir = args.output_dir,
            seed = args.seed,
            no_decay_embed = args.no_decay_embed,
        )

        model = DiaModel(dia_cfg)
        
        if args.scratch:
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
            # Track if we performed expansion to know if we should warm-start stereo
            expanded_stereo = False

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
                            expanded_stereo = True
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
                if expanded_stereo and dia_cfg.data.channels == 18:
                    logger.info("Warm-starting stereo channels (copying Left -> Right)...")
                    if hasattr(model.decoder, "embeddings") and len(model.decoder.embeddings) >= 18:
                        for i in range(9, 18):
                            model.decoder.embeddings[i].weight.data.copy_(model.decoder.embeddings[i - 9].weight.data)
                    W = model.decoder.logits_dense.weight
                    if W.dim() == 3 and W.shape[1] >= 18:
                        W.data[:, 9:18, :].copy_(W.data[:, 0:9, :])
                elif dia_cfg.data.channels == 18:
                    logger.info("Skipping stereo warm-start duplication (loaded checkpoint appears to be stereo already)")
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