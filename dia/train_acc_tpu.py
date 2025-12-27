"""
Dia Audio Model Fine-tuning on TPU (FSDP Optimized).
"""

# ============================================================================
# CRITICAL: Set TPU environment BEFORE any torch_xla imports!
# ============================================================================
import os
import sys

# Standard TPU Env setup
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

import argparse
import glob
import logging
import shutil
import json
import random
import re
import warnings
import time
import threading
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
# hf_hub_download removed - always training from scratch
from torch.nn.utils import clip_grad_norm_, parametrize
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers import get_scheduler

from .audio import apply_audio_delay, build_delay_indices, codebook_to_audio
from .config import DiaConfig
from .dataset import PreEncodedDACDataset, HuggingFacePreEncodedDataset
from .layers import DiaModel, KVCache
from .model import Dia

# Conditional import for TestingDataset
try:
    from . import dataset as dataset_module
    TestingDataset = getattr(dataset_module, 'TestingDataset', None)
except (ImportError, AttributeError):
    TestingDataset = None

warnings.filterwarnings("ignore", message="`torch.nn.utils.weight_norm` is deprecated")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS & CONFIG
# =============================================================================

TEST_PROMPTS = {
    "celtic": "celtic",
    "poprock": "poprock, happy, positive",
    "jazz funk": "funk, jazz, percussions",
}
EVAL_CFG_SCALE = 4.0
EVAL_CFG_SCALE_UNCOND = 0.0
EVAL_TOP_P = 0.95
EVAL_TEMPERATURES = [1.0]
EVAL_SAMPLE_RATE = 44100
EVAL_AUDIO_DIR = "./audio_demos"
ENTROPY_LOG_INTERVAL = 50
CODEBOOK_WEIGHTS = [4.0] + [1.0] * 8  # 4× weight on first channel (semantic/content codebook)
_DAC_MODEL_CACHE = {}

@contextmanager
def preserve_rng_state():
    """Context manager to preserve and restore RNG state."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)

def _save_and_log_audio(audio, audio_path: Path, wandb_key: str, caption: str, audio_samples: dict):
    """Save audio to disk and prepare for wandb logging."""
    arr = audio
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[0] <= 8 and arr.shape[1] > arr.shape[0]:
        arr = arr.T
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(audio_path, arr, EVAL_SAMPLE_RATE)
    print(f"[DEMO] ✓ Saved audio: {audio_path} (shape={arr.shape})", flush=True)
    logger.info(f"Saved demo audio: {audio_path}")
    audio_samples[wandb_key] = wandb.Audio(arr, sample_rate=EVAL_SAMPLE_RATE, caption=caption)

@dataclass
class TrainConfig:
    epochs: int = 500
    batch_size: int = 4
    grad_accum_steps: int = 1
    learning_rate: float = 1e-5
    weight_decay: float = 0.1
    warmup_steps: int = 100
    unconditional_frac: float = 0.15
    eval_every: Optional[int] = None
    save_step: int = 2000
    demo_every: Optional[int] = None
    split_ratio: float = 0.997
    seed: int = 786
    runs_dir: Path = field(default_factory=lambda: Path("runs"))
    run_name: str = "dia_finetune_tpu"
    output_dir: Optional[Path] = None
    tag_shuffle: bool = True
    tag_dropout: float = 0.0
    tag_limit: Optional[int] = None
    no_decay_embed: bool = False
    keep_last_n: Optional[int] = None
    grad_clip_max_norm: float = 1.0

# =============================================================================
# UTILS
# =============================================================================

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def strip_weight_norms(module: torch.nn.Module) -> int:
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

def cleanup_old_checkpoints(output_dir: Path, keep_last_n: int):
    if keep_last_n is None:
        return
    # Look for checkpoint directories (not .pth files)
    dirs = sorted(
        [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("ckpt_step")],
        key=lambda x: int(re.search(r'(\d+)$', x.name).group(1)) if re.search(r'(\d+)$', x.name) else 0
    )
    if len(dirs) > keep_last_n:
        for old_ckpt_dir in dirs[:-keep_last_n]:
            try:
                shutil.rmtree(old_ckpt_dir)
                logger.info(f"Removed old checkpoint: {old_ckpt_dir}")
            except Exception as e:
                logger.warning(f"Failed to remove {old_ckpt_dir}: {e}")

def _extract_step_from_ckpt_name(name: str) -> Optional[int]:
    match = re.search(r"ckpt_step(\d+)", name)
    if match:
        return int(match.group(1))
    return None

def resolve_resume_path(resume_from: Optional[str], output_dir: Path):
    if not resume_from:
        return None, 0
    if resume_from == "latest":
        if not output_dir.exists():
            raise FileNotFoundError(f"Output dir does not exist: {output_dir}")
        candidates = []
        for p in output_dir.glob("ckpt_step*"):
            step = _extract_step_from_ckpt_name(p.name)
            if step is not None:
                candidates.append((step, p))
        if not candidates:
            raise FileNotFoundError(f"No checkpoints found in {output_dir}")
        candidates.sort(key=lambda x: x[0], reverse=True)
        resume_path = candidates[0][1]
    else:
        resume_path = Path(resume_from)
    if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {resume_path}")
    step = _extract_step_from_ckpt_name(resume_path.name) or 0
    return resume_path, step

def prepare_resume_checkpoint(resume_path: Path, accelerator):
    """Convert safetensors -> .bin for TPU load_state compatibility."""
    try:
        from accelerate.utils.constants import SAFE_MODEL_NAME, MODEL_NAME
    except Exception:
        return
    safe_files = list(resume_path.glob(f"{SAFE_MODEL_NAME}*.safetensors"))
    if not safe_files:
        return
    if accelerator.is_main_process:
        try:
            from safetensors.torch import load_file as safetensors_load_file
        except Exception as exc:
            raise RuntimeError("safetensors is required to convert checkpoints for resume") from exc
        for safe_path in safe_files:
            suffix = safe_path.stem[len(SAFE_MODEL_NAME):]
            bin_path = resume_path / f"{MODEL_NAME}{suffix}.bin"
            if not bin_path.exists():
                state = safetensors_load_file(safe_path, device="cpu")
                torch.save(state, bin_path)
    accelerator.wait_for_everyone()

@contextmanager
def disable_safetensors_load():
    try:
        import accelerate.checkpointing as ckpt
    except Exception:
        yield
        return
    original = ckpt.SAFE_MODEL_NAME
    ckpt.SAFE_MODEL_NAME = "model_disabled"
    try:
        yield
    finally:
        ckpt.SAFE_MODEL_NAME = original

@contextmanager
def patch_dataloaders_for_state(accelerator):
    patched = []
    for dl in getattr(accelerator, "_dataloaders", []):
        if not hasattr(dl, "dataset"):
            dl.dataset = None
            patched.append(dl)
    try:
        yield
    finally:
        for dl in patched:
            delattr(dl, "dataset")

def get_dac_model(device):
    """Load and cache DAC model per device for demo generation."""
    key = str(device)
    if key in _DAC_MODEL_CACHE:
        return _DAC_MODEL_CACHE[key]
    logger.info(f"[DEMO] Loading DAC model onto {device}")
    t0 = time.perf_counter()
    model = dac.DAC.load(dac.utils.download()).eval().to(device)
    strip_weight_norms(model)
    _DAC_MODEL_CACHE[key] = model
    logger.info(f"[DEMO] DAC model ready in {time.perf_counter() - t0:.1f}s")
    return model

# =============================================================================
# DATA
# =============================================================================

def collate_fn(batch, config: DiaConfig, train_cfg: TrainConfig, device: torch.device, use_sliding_window: bool = True):
    texts, encodings, waveforms = zip(*batch)
    window_size = config.data.audio_length
    
    # Pad to fixed size for TPU compilation stability
    batch_max = window_size 
    
    padded_encodings = []
    seq_lens = []

    for e in encodings:
        L = e.size(0)
        if L > window_size:
            start = random.randint(0, L - window_size) if use_sliding_window else 0
            e_cropped = e[start : start + window_size]
        else:
            e_cropped = e
        
        curr_L = e_cropped.size(0)
        seq_lens.append(curr_L)
        
        if curr_L < batch_max:
            pad_amt = batch_max - curr_L
            e_padded = F.pad(e_cropped, (0, 0, 0, pad_amt), value=config.data.audio_pad_value)
        else:
            e_padded = e_cropped
        padded_encodings.append(e_padded)

    # Note: Returning CPU tensors here. Accelerate moves them to TPU device automatically during iteration.
    codes = torch.stack(padded_encodings)

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
    
    src = torch.stack(text_ids)
    src_pos = torch.arange(max_text).unsqueeze(0).expand(src.size(0), -1)
    src_pad = src.ne(pad_tok)
    enc_self_attn_mask = (src_pad.unsqueeze(2) & src_pad.unsqueeze(1)).unsqueeze(1)

    # Audio Padding & Delays
    B, T, C = codes.shape
    t_idx, idxs = build_delay_indices(B, T, C, config.data.delay_pattern)
    # Move to CPU for delay op if needed, then back, but here we stay on CPU until dataloader yields
    delayed = apply_audio_delay(codes, config.data.audio_pad_value, config.data.audio_bos_value, (t_idx, idxs))

    # Targets
    max_tgt_len = batch_max + 2
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

def setup_loaders(dataset, dia_cfg: DiaConfig, train_cfg: TrainConfig, use_sliding_window=True):
    from functools import partial
    # Collate runs on CPU, Accelerate handles transfer
    collate = partial(collate_fn, config=dia_cfg, train_cfg=train_cfg, device=None, use_sliding_window=use_sliding_window)
    
    # Check if this is an IterableDataset (streaming mode)
    is_iterable = isinstance(dataset, torch.utils.data.IterableDataset)
    
    if is_iterable:
        # Streaming dataset: no random_split, no shuffle (shuffling happens via streaming)
        # For streaming, we skip validation to keep things simple
        train_loader = DataLoader(
            dataset, batch_size=train_cfg.batch_size, shuffle=False,
            collate_fn=collate, num_workers=0,  # num_workers=0 for streaming to avoid issues
            pin_memory=True, drop_last=True
        )
        # Attach length info for scheduler calculation
        ds_len = len(dataset)  # HuggingFacePreEncodedDataset provides __len__
        train_loader.steps_per_epoch = ds_len // train_cfg.batch_size
        return train_loader, None
    
    # Map-style dataset: standard handling
    ds_len = len(dataset)
    n_train = int(train_cfg.split_ratio * ds_len)
    n_val = ds_len - n_train
    
    if ds_len <= 1 or n_val == 0:
        train_ds, val_ds = dataset, None
    else:
        g = torch.Generator().manual_seed(train_cfg.seed)
        train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=g)
    
    train_loader = DataLoader(
        train_ds, batch_size=train_cfg.batch_size, shuffle=True,
        collate_fn=collate, num_workers=4, pin_memory=True, drop_last=True
    )
    if val_ds:
        val_loader = DataLoader(
            val_ds, batch_size=train_cfg.batch_size, shuffle=False,
            collate_fn=collate, num_workers=4, pin_memory=True, drop_last=True
        )
    else:
        val_loader = None
        
    return train_loader, val_loader


def setup_optimizer_and_scheduler(model, train_loader, train_cfg: TrainConfig):
    """Setup optimizer and learning rate scheduler.
    
    Uses standard AdamW (TPU-compatible) with cosine schedule.
    Scheduler steps are scaled by grad_accum_steps to match optimizer steps.
    """
    opt = optim.AdamW(model.parameters(), lr=train_cfg.learning_rate)
    
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
    return opt, sched, steps_per_epoch

# =============================================================================
# TRAIN STEP
# =============================================================================

def compute_grad_norm(model):
    """Compute total gradient norm across all parameters."""
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm_sq += p.grad.data.float().pow(2).sum()
    return total_norm_sq.sqrt()

def train_step(model, batch, dia_cfg, train_cfg, opt, sched, global_step, accelerator):
    
    # Unconditional dropout
    gen_val = ((global_step * 997 + train_cfg.seed) % 10000) / 10000.0
    if gen_val < train_cfg.unconditional_frac:
        batch['src_tokens'].fill_(dia_cfg.data.text_pad_value)
        batch['enc_self_attn_mask'].zero_()
        batch['dec_cross_attn_mask'].zero_()

    # Forward
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
        
        logits = logits[:, :-1]
        target = batch['tgt_tokens'][:, 1:]
        
        # Loss Masking (Safe for TPU/XLA) - time-based mask only, EOS included in loss
        mask = torch.arange(target.shape[1], device=target.device).unsqueeze(0) < (batch['tgt_lens'].unsqueeze(1) - 1)
        final_mask = mask.unsqueeze(-1).expand_as(target).float()
        
        loss = 0.0
        # equal weight on all channels (better for music - learn full spectrum)
        C = target.shape[2]
        channel_weights = [1.0] + [1.0] * (C - 1)
        
        for c, w in enumerate(channel_weights):
            # CHANGE: Cast logits to float() (FP32) before loss to prevent BF16 overflow
            l_c = logits[:, :, c, :].flatten(0, 1).float()
            t_c = target[:, :, c].flatten()
            m_c = final_mask[:, :, c].flatten()
            
            # Now this calculation happens in stable FP32
            ce_loss = F.cross_entropy(l_c, t_c, reduction='none', ignore_index=dia_cfg.data.audio_pad_value)
            masked_loss = (ce_loss * m_c).sum()
            mask_sum = m_c.sum() + 1e-9
            loss += w * (masked_loss / mask_sum)
        
        loss = loss / sum(channel_weights)

    # Backward & Step
    accelerator.backward(loss)
    
    # Initialize grad norm tracking
    pre_clip_grad_norm = None
    post_clip_grad_norm = None
    
    if (global_step + 1) % train_cfg.grad_accum_steps == 0:
        # Compute gradient norm BEFORE clipping
        pre_clip_grad_norm = compute_grad_norm(model)
        
        # Clip gradients
        accelerator.clip_grad_norm_(model.parameters(), train_cfg.grad_clip_max_norm)
        
        # Compute gradient norm AFTER clipping
        post_clip_grad_norm = compute_grad_norm(model)
        
        opt.step()
        sched.step()
        opt.zero_grad()
    
    return {
        'loss': loss.detach(),
        'pre_clip_grad_norm': pre_clip_grad_norm,
        'post_clip_grad_norm': post_clip_grad_norm,
    }

# =============================================================================
# DEMO GENERATION
# =============================================================================

def generate_demos(model, dia_cfg, train_cfg, global_step, accelerator):
    """Generate demo audio samples during training.
    
    IMPORTANT: All ranks must participate in generation to avoid FSDP deadlock.
    The model is sharded across TPU cores, so all processes must run the forward pass
    together to unshard the weights. Only rank 0 saves/logs the output.
    """
    # NOTE: Removed early return for non-main processes to avoid FSDP deadlock.
    # All ranks must participate in generation, but only rank 0 saves/logs.
     
    t_start = time.perf_counter()
    if accelerator.is_main_process:
        print(f"[DEMO] Generating audio samples at step {global_step}", flush=True)
        print(f"[DEMO DEBUG] Entered generate_demos function", flush=True)
        Path(EVAL_AUDIO_DIR).mkdir(parents=True, exist_ok=True)
    
    # ALL processes must unwrap the model (FSDP requirement)
    if accelerator.is_main_process:
        print("[DEMO DEBUG] Unwrapping model...", flush=True)
    unwrapped = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        print("[DEMO DEBUG] Model unwrapped successfully", flush=True)
    
    try:
        # Create Dia wrapper for generation (ALL processes)
        if accelerator.is_main_process:
            print("[DEMO DEBUG] Creating Dia wrapper...", flush=True)
        dia_gen = Dia(dia_cfg, accelerator.device)
        dia_gen.model = unwrapped
        if accelerator.is_main_process:
            print("[DEMO DEBUG] Dia wrapper created, loading DAC model...", flush=True)
        
        # Load DAC model for audio decoding (cached per device, ALL processes)
        dac_load_t0 = time.perf_counter()
        dac_model = get_dac_model(accelerator.device)
        dia_gen.dac_model = dac_model
        if accelerator.is_main_process:
            print(f"[DEMO] DAC model attached (ready in {time.perf_counter() - dac_load_t0:.1f}s)", flush=True)
            print("[DEMO DEBUG] About to enter generation loop...", flush=True)
        
        audio_samples = {}
        
        def safe_temp(t): 
            return f"{t:.1f}".replace(".", "p")
        
        with torch.no_grad():
            if train_cfg.unconditional_frac >= 1.0:
                # Unconditional generation
                seeds = [int(train_cfg.seed), int(train_cfg.seed) + 1]
                with preserve_rng_state():
                    for temp in EVAL_TEMPERATURES:
                        for s in seeds:
                            seed_everything(s)
                            demo_t0 = time.perf_counter()
                            if accelerator.is_main_process:
                                logger.info(f"[DEMO] Start unconditional temp={temp}, seed={s}")
                            try:
                                # ALL ranks run generation (FSDP requires all processes to participate)
                                # check_eos=False to avoid TPU sync-in-loop performance bug
                                audio = dia_gen.generate(text="", cfg_scale=EVAL_CFG_SCALE_UNCOND, temperature=temp, check_eos=False)
                                
                                # Only main process saves to disk and logs to wandb
                                if accelerator.is_main_process:
                                    path = Path(EVAL_AUDIO_DIR) / f"step_{global_step}_temp{safe_temp(temp)}_seed{s}.wav"
                                    _save_and_log_audio(audio, path, f"eval_audio/temp{safe_temp(temp)}/seed{s}", 
                                                     f"temp={temp}", audio_samples)
                            except Exception as e:
                                if accelerator.is_main_process:
                                    logger.warning(f"Demo generation failed for temp={temp}, seed={s}: {e}")
                            else:
                                if accelerator.is_main_process:
                                    logger.info(f"[DEMO] Finished temp={temp}, seed={s} in {time.perf_counter() - demo_t0:.1f}s")
            else:
                # Conditional generation with prompts
                cfg_s = EVAL_CFG_SCALE if train_cfg.unconditional_frac > 0 else None
                for temp in EVAL_TEMPERATURES:
                    for name, prompt in TEST_PROMPTS.items():
                        demo_t0 = time.perf_counter()
                        if accelerator.is_main_process:
                            logger.info(f"[DEMO] Start prompt={name}, temp={temp}")
                        try:
                            # ALL ranks run generation (FSDP requires all processes to participate)
                            # check_eos=False to avoid TPU sync-in-loop performance bug
                            audio = dia_gen.generate(text=prompt, cfg_scale=cfg_s, temperature=temp, top_p=EVAL_TOP_P, check_eos=False)
                            if accelerator.is_main_process:
                                print(f"[DEMO] Generation returned for {name}, temp={temp}, audio type={type(audio)}", flush=True)
                            
                            # Only main process saves to disk and logs to wandb
                            if accelerator.is_main_process:
                                path = Path(EVAL_AUDIO_DIR) / f"step_{global_step}_{name}_temp{safe_temp(temp)}.wav"
                                _save_and_log_audio(audio, path, f"eval_audio/temp{safe_temp(temp)}/{name}", 
                                                 prompt, audio_samples)
                        except Exception as e:
                            if accelerator.is_main_process:
                                print(f"[DEMO] ✗ Generation FAILED for {name}, temp={temp}: {e}", flush=True)
                                logger.warning(f"Demo generation failed for {name}, temp={temp}: {e}")
                        else:
                            if accelerator.is_main_process:
                                logger.info(f"[DEMO] Finished prompt={name}, temp={temp} in {time.perf_counter() - demo_t0:.1f}s")
            
            # Only main process logs to wandb
            if accelerator.is_main_process and audio_samples:
                wandb.log(audio_samples, step=global_step)
                logger.info(f"[DEMO] Logged {len(audio_samples)} audio samples to wandb")
                
    except Exception as e:
        if accelerator.is_main_process:
            logger.exception(f"[DEMO] Demo generation failed: {e}")
    finally:
        if accelerator.is_main_process:
            logger.info(f"[DEMO] Demo section finished in {time.perf_counter() - t_start:.1f}s")

# =============================================================================
# EVALUATION
# =============================================================================

def run_eval(model, val_loader, dia_cfg, global_step, accelerator):
    """Run evaluation on validation set and return average loss."""
    if val_loader is None:
        return None
    
    eval_losses = []
    model.eval()
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Eval", disable=not accelerator.is_main_process):
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
                )[:, :-1]
                
                # Cast to float for stable loss calculation
                logits = logits.float()
                target = batch['tgt_tokens'][:, 1:]
                
                # Masking (time-based only, EOS included - matches train_step)
                mask = torch.arange(target.shape[1], device=target.device).unsqueeze(0) < (batch['tgt_lens'].unsqueeze(1) - 1)
                final_mask = mask.unsqueeze(-1).expand_as(target).float()
                
                loss = 0.0
                # equal weight on all channels (better for music - learn full spectrum)
                C_e = target.shape[2]
                channel_weights = [1.0] + [1.0] * (C_e - 1)
                
                for c, w in enumerate(channel_weights):
                    l_c = logits[:, :, c, :].flatten(0, 1)
                    t_c = target[:, :, c].flatten()
                    m_c = final_mask[:, :, c].flatten()
                    
                    ce_loss = F.cross_entropy(l_c, t_c, reduction='none', ignore_index=dia_cfg.data.audio_pad_value)
                    masked_loss = (ce_loss * m_c).sum()
                    mask_sum = m_c.sum() + 1e-9
                    loss += w * (masked_loss / mask_sum)
                
                loss = loss / sum(channel_weights)
                
            # CRITICAL: Force XLA to materialize the loss before reading
            xm.mark_step()
            eval_losses.append(loss.detach().item())
    
    # Gather losses across all processes
    if eval_losses:
        # Convert to tensor for gathering (losses are already scalars now)
        local_avg = torch.tensor(sum(eval_losses) / len(eval_losses), device=accelerator.device)
        xm.mark_step()  # Ensure tensor is materialized before gather
        all_losses = accelerator.gather(local_avg)
        avg_loss = all_losses.mean().item()
        
        if accelerator.is_main_process:
            accelerator.log({'eval_loss': avg_loss}, step=global_step)
            logger.info(f"[EVAL] Step {global_step}: eval_loss = {avg_loss:.4f}")
        
        return avg_loss
    
    return None

# =============================================================================
# MAIN
# =============================================================================

def main():
    # Parse args inside main() so Accelerate's TPU launcher can call it without arguments
    print("[DEBUG] Python script starting...", flush=True)
    args = get_args()
    print(f"[DEBUG] Args parsed. hf_dataset={args.hf_dataset}, preencoded_dir={args.preencoded_dir}", flush=True)
    
    # Initialize Accelerator with BF16 and FSDP awareness
    print("[DEBUG] Initializing Accelerator...", flush=True)
    accelerator = Accelerator(mixed_precision="bf16", log_with="wandb")
    device = accelerator.device
    print(f"[DEBUG] Accelerator initialized. device={device}, is_main={accelerator.is_main_process}", flush=True)
    
    if accelerator.is_main_process:
        print(f"[INIT] Launching on {accelerator.num_processes} processes (FSDP enabled).")

    # Load Config
    print(f"[DEBUG] Loading config from {args.config}. is_main={accelerator.is_main_process}", flush=True)
    dia_cfg = DiaConfig.load(args.config)
    print(f"[DEBUG] Config loaded. is_main={accelerator.is_main_process}", flush=True)
    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        unconditional_frac=args.unconditional_frac,
        output_dir=args.output_dir,
        seed=args.seed,
        demo_every=args.demo_every,
        eval_every=args.eval_every,
        keep_last_n=args.keep_last_n,
        grad_clip_max_norm=args.grad_clip_max_norm,
        weight_decay=args.weight_decay,
        save_step=args.save_step,
    )
    
    # WandB Init (Accelerate handles main process check internally for loggers usually, but we are explicit)
    print(f"[DEBUG] About to init WandB. is_main={accelerator.is_main_process}", flush=True)
    if accelerator.is_main_process:
        print("[DEBUG] Main process: Initializing WandB...", flush=True)
        train_cfg.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[DEBUG] Main process: Output dir created: {train_cfg.output_dir}", flush=True)
        accelerator.init_trackers(
            project_name=args.wandb_project, 
            config=vars(args),
            init_kwargs={"wandb": {"name": train_cfg.run_name}}
        )
        print("[DEBUG] Main process: WandB initialized", flush=True)
    print(f"[DEBUG] WandB section complete. is_main={accelerator.is_main_process}", flush=True)

    # Dataset
    print(f"[DEBUG] About to load dataset. hf_dataset={args.hf_dataset}, preencoded_dir={args.preencoded_dir}, audio_folder={args.audio_folder}", flush=True)
    if accelerator.is_main_process:
        print("[DEBUG] Loading dataset...", flush=True)
    
    dataset = None
    if args.hf_dataset:
        # Load from HuggingFace datasets in STREAMING mode
        # Streaming avoids disk space issues (no Arrow cache needed)
        print(f"[DEBUG] Loading HuggingFace streaming dataset: {args.hf_dataset}", flush=True)
        try:
            dataset = HuggingFacePreEncodedDataset(
                dataset_name=args.hf_dataset,
                config=dia_cfg,
                split="train",
                use_sliding_window=args.use_sliding_window,
                dataset_length=args.hf_dataset_length,  # Known size for progress tracking
            )
            print(f"[DEBUG] Streaming dataset configured, length={len(dataset)}", flush=True)
        except Exception as e:
            print(f"[ERROR] Failed to configure streaming dataset: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise
            
    elif args.preencoded_dir:
        # Optimization: Load on main process first to generate cache and avoid GCS thundering herd
        if accelerator.is_main_process:
            print("[DEBUG] Main process scanning dataset...", flush=True)
            dataset = PreEncodedDACDataset(args.preencoded_dir, dia_cfg, args.use_sliding_window)
        
        # Wait for main process to finish scanning/caching
        accelerator.wait_for_everyone()
        
        if not accelerator.is_main_process:
            dataset = PreEncodedDACDataset(args.preencoded_dir, dia_cfg, args.use_sliding_window)
            
    elif args.audio_folder:
        # Load CPU DAC for encoding in dataloaders
        dac_model = dac.DAC.load(dac.utils.download()).eval().to('cpu')
        strip_weight_norms(dac_model)
        dataset = TestingDataset(args.audio_folder, dia_cfg, dac_model, args.use_sliding_window)
    
    if accelerator.is_main_process:
        print(f"[DEBUG] Dataset loaded with {len(dataset)} samples. Setting up data loaders...", flush=True)
    train_loader, val_loader = setup_loaders(dataset, dia_cfg, train_cfg, args.use_sliding_window)
    if accelerator.is_main_process:
        print(f"[DEBUG] Data loaders ready. train_loader has {len(train_loader)} batches.", flush=True)
    
    # Model
    if accelerator.is_main_process:
        print("[DEBUG] Creating DiaModel...", flush=True)
    model = DiaModel(dia_cfg)
    strip_weight_norms(model)
    if accelerator.is_main_process:
        print("[DEBUG] DiaModel created.", flush=True)
    
    if accelerator.is_main_process:
        if args.resume_from:
            print("[TRAIN] Resuming from checkpoint", flush=True)
        else:
            print("[TRAIN] Training from scratch (no pretrained weights)", flush=True)

    # Optimizer & Scheduler
    if accelerator.is_main_process:
        print("[DEBUG] Setting up optimizer...", flush=True)
    opt, sched, steps_per_epoch = setup_optimizer_and_scheduler(model, train_loader, train_cfg)
    total_steps = steps_per_epoch * train_cfg.epochs
    if accelerator.is_main_process:
        print(f"[DEBUG] Optimizer ready. steps_per_epoch={steps_per_epoch}, total_steps={total_steps}", flush=True)

    # CRITICAL FSDP STEP: Prepare everything together
    # FSDP will shard the model across TPUs here
    if accelerator.is_main_process:
        print("[DEBUG] Calling accelerator.prepare() - this may take a while for FSDP sharding...", flush=True)
    model, opt, train_loader, sched = accelerator.prepare(model, opt, train_loader, sched)
    if accelerator.is_main_process:
        print("[DEBUG] accelerator.prepare() complete for model, opt, train_loader, sched.", flush=True)
    if val_loader: 
        val_loader = accelerator.prepare(val_loader)
        if accelerator.is_main_process:
            print("[DEBUG] val_loader prepared.", flush=True)

    resume_path, resume_step = resolve_resume_path(args.resume_from, train_cfg.output_dir)
    if resume_path is not None:
        print(f"[RESUME] Worker {accelerator.process_index}: Found checkpoint at {resume_path}, step={resume_step}", flush=True)
        
        print(f"[RESUME] Worker {accelerator.process_index}: Calling prepare_resume_checkpoint...", flush=True)
        prepare_resume_checkpoint(resume_path, accelerator)
        print(f"[RESUME] Worker {accelerator.process_index}: prepare_resume_checkpoint done", flush=True)
        
        if accelerator.is_main_process:
            # List checkpoint contents for debugging
            import os
            ckpt_files = os.listdir(resume_path)
            total_size = sum(os.path.getsize(resume_path / f) for f in ckpt_files if os.path.isfile(resume_path / f))
            print(f"[RESUME] Checkpoint contains {len(ckpt_files)} files, total size: {total_size / 1e6:.1f} MB", flush=True)
            print(f"[RESUME] Files: {ckpt_files[:10]}{'...' if len(ckpt_files) > 10 else ''}", flush=True)
        
        print(f"[RESUME] Worker {accelerator.process_index}: Starting load_state...", flush=True)
        t_load_start = time.perf_counter()
        
        with disable_safetensors_load(), patch_dataloaders_for_state(accelerator):
            accelerator.load_state(resume_path)
        
        print(f"[RESUME] Worker {accelerator.process_index}: load_state done in {time.perf_counter() - t_load_start:.1f}s", flush=True)
        
        print(f"[RESUME] Worker {accelerator.process_index}: Waiting for everyone...", flush=True)
        accelerator.wait_for_everyone()
        print(f"[RESUME] Worker {accelerator.process_index}: All workers synced after load", flush=True)

    global_step = resume_step
    start_epoch = global_step // steps_per_epoch
    resume_step_in_epoch = global_step % steps_per_epoch
    log_interval = 10  # Log every N steps
    accumulated_loss = 0.0
    loss_count = 0
    
    if accelerator.is_main_process:
        print(f"[DEBUG] Setup complete. global_step={global_step}", flush=True)
        if resume_path is not None:
            print(
                f"[RESUME] Resuming at epoch {start_epoch + 1} step {resume_step_in_epoch}.",
                flush=True,
            )
        print("[DEBUG] STARTING TRAINING LOOP - VERIFYING LOGGING", flush=True)

    # Track gradient norms for logging
    accumulated_pre_clip_norm = 0.0
    accumulated_post_clip_norm = 0.0
    grad_norm_count = 0

    for epoch in range(start_epoch, train_cfg.epochs):
        model.train()
        loader_iter = tqdm(train_loader, disable=not accelerator.is_main_process, desc=f"E{epoch+1}")
        if resume_step_in_epoch and epoch == start_epoch:
            if accelerator.is_main_process:
                print(
                    f"[RESUME] Skipping {resume_step_in_epoch} steps to align with checkpoint.",
                    flush=True,
                )
            for _ in range(resume_step_in_epoch):
                try:
                    next(loader_iter)
                except StopIteration:
                    break
        for batch in loader_iter:
            # Debug: Unconditional print for first 20 steps to trace execution
            if accelerator.is_main_process and global_step < 20:
                 print(f"[DEBUG] Step {global_step}: Batch loaded. Starting training step...", flush=True)

            step_output = train_step(model, batch, dia_cfg, train_cfg, opt, sched, global_step, accelerator)
            loss = step_output['loss']
            
            if accelerator.is_main_process and global_step < 20:
                 print(f"[DEBUG] Step {global_step}: train_step finished. Waiting for XLA mark_step...", flush=True)

            # CRITICAL: Force XLA to execute pending ops BEFORE reading the loss value
            # This ensures the loss tensor is materialized and not a stale cached value
            xm.mark_step()
            
            # Accumulate loss (now safe to read after mark_step)
            accumulated_loss += loss.item()
            loss_count += 1
            
            # Accumulate gradient norms (only when available - on grad accum boundaries)
            if step_output['pre_clip_grad_norm'] is not None:
                accumulated_pre_clip_norm += step_output['pre_clip_grad_norm'].item()
                accumulated_post_clip_norm += step_output['post_clip_grad_norm'].item()
                grad_norm_count += 1

            # Log averaged loss every N steps for smoother, more meaningful metrics
            if (global_step + 1) % log_interval == 0:
                avg_loss = accumulated_loss / loss_count
                log_dict = {
                    "train_loss": avg_loss,
                    "lr": sched.get_last_lr()[0]
                }
                
                # Add gradient norm metrics if we have any
                if grad_norm_count > 0:
                    avg_pre_clip = accumulated_pre_clip_norm / grad_norm_count
                    avg_post_clip = accumulated_post_clip_norm / grad_norm_count
                    log_dict["grad_norm_pre_clip"] = avg_pre_clip
                    log_dict["grad_norm_post_clip"] = avg_post_clip
                    # Reset grad norm accumulators
                    accumulated_pre_clip_norm = 0.0
                    accumulated_post_clip_norm = 0.0
                    grad_norm_count = 0
                
                if accelerator.is_main_process:
                    accelerator.log(log_dict, step=global_step)
                # Reset loss accumulators
                accumulated_loss = 0.0
                loss_count = 0
            
            global_step += 1
            
            if accelerator.is_main_process and global_step < 20:
                 print(f"[DEBUG] Step {global_step}: Computation done. Checking save condition...", flush=True)

            # Save
            if global_step % train_cfg.save_step == 0:
                if accelerator.is_main_process:
                    print(f"[DEBUG] Step {global_step}: Entering save block. Waiting for everyone...", flush=True)
                t_wait_start = time.perf_counter()
                accelerator.wait_for_everyone()

                # FSDP saving requires all processes to participate in save_state
                save_path = train_cfg.output_dir / f"ckpt_step{global_step}"
                
                if accelerator.is_main_process:
                    print(f"[DEBUG] Step {global_step}: All processes synced. Wait time: {time.perf_counter() - t_wait_start:.2f}s", flush=True)
                    print(f"[DEBUG] Step {global_step}: Starting save_state to {save_path}...", flush=True)
                    
                    # Monitor progress in background
                    stop_event = threading.Event()
                    def _monitor():
                        while not stop_event.wait(5):
                            if not save_path.exists(): continue
                            try:
                                size = sum(p.stat().st_size for p in save_path.rglob('*') if p.is_file())
                                print(f"[DEBUG] Saving... {size/1e6:.1f} MB written", flush=True)
                            except: pass
                    t_mon = threading.Thread(target=_monitor)
                    t_mon.start()
                    
                    t_save_start = time.perf_counter()
                
                # WORKAROUND: MpDeviceLoaderWrapper doesn't have .dataset attribute
                # Patch it temporarily so accelerator.save_state() doesn't crash
                patched_loaders = []
                for dl in accelerator._dataloaders:
                    if not hasattr(dl, 'dataset'):
                        dl.dataset = None  # Add fake attribute
                        patched_loaders.append(dl)
                
                try:
                    accelerator.save_state(output_dir=save_path, safe_serialization=False)
                finally:
                    # Clean up patches
                    for dl in patched_loaders:
                        delattr(dl, 'dataset')

                # Ensure all ranks have finished writing before marking the checkpoint complete
                accelerator.wait_for_everyone()
                
                if accelerator.is_main_process:
                    # Mark checkpoint complete (used by AUTO_RESUME selection in run_dia_training.sh)
                    try:
                        (save_path / ".complete").write_text("ok\n")
                    except Exception as e:
                        print(f"[DEBUG] Step {global_step}: Warning: failed to write .complete marker: {e}", flush=True)

                    stop_event.set()
                    t_mon.join()
                    
                    save_duration = time.perf_counter() - t_save_start
                    logger.info(f"Saved checkpoint to {save_path}")
                    print(f"[DEBUG] Step {global_step}: save_state completed in {save_duration:.2f}s", flush=True)
                    logger.info(f"Saved checkpoint to {save_path} in {save_duration:.2f}s")

                    # Cleanup old checkpoints if configured
                    if train_cfg.keep_last_n is not None:
                        cleanup_old_checkpoints(train_cfg.output_dir, train_cfg.keep_last_n)
            
            # Demo generation
            if train_cfg.demo_every and global_step % train_cfg.demo_every == 0 and global_step > 0:
                accelerator.wait_for_everyone()
                model.eval()
                generate_demos(model, dia_cfg, train_cfg, global_step, accelerator)
                model.train()
                accelerator.wait_for_everyone()
            
            # Evaluation
            if train_cfg.eval_every and global_step % train_cfg.eval_every == 0 and global_step > 0:
                accelerator.wait_for_everyone()
                run_eval(model, val_loader, dia_cfg, global_step, accelerator)
                model.train()
                accelerator.wait_for_everyone()
        if epoch == start_epoch:
            resume_step_in_epoch = 0

    accelerator.end_training()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--audio_folder", type=Path)
    parser.add_argument("--preencoded_dir", type=Path)
    parser.add_argument("--hf_dataset", type=str, default=None, 
                       help="HuggingFace dataset name (e.g., 'oliver-camp/open-source-dataset')")
    parser.add_argument("--hf_cache_dir", type=str, default=None,
                       help="Cache directory for HuggingFace datasets (unused in streaming mode)")
    parser.add_argument("--hf_dataset_length", type=int, default=31333,
                       help="Known dataset size for progress tracking (default: 31333 for oliver-camp/open-source-dataset)")
    parser.add_argument("--hf_parquet_dir", type=Path, default=None,
                       help="Directory containing HuggingFace parquet files (from hf_to_gcs_fast.py)")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Checkpoint dir (ckpt_stepXXXX) or 'latest' to auto-resume from output_dir",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=100, help="Number of warmup steps for LR scheduler")
    parser.add_argument("--unconditional_frac", type=float, default=0.15, help="Fraction of batches to train unconditionally (for CFG)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default="dia-fsdp")
    parser.add_argument("--use_sliding_window", action="store_true")
    parser.add_argument("--demo_every", type=int, default=None, help="Generate demo audio every N steps")
    parser.add_argument("--eval_every", type=int, default=None, help="Run evaluation every N steps")
    parser.add_argument("--keep_last_n", type=int, default=None, help="Keep only the last N checkpoints, delete older ones")
    parser.add_argument("--grad_clip_max_norm", type=float, default=1.0, help="Maximum gradient norm for clipping")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay for optimizer")
    parser.add_argument("--save_step", type=int, default=2000, help="Save checkpoint every N steps")
    return parser.parse_args()

if __name__ == "__main__":
    main()
