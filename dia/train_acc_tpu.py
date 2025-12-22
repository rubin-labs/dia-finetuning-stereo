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
import json
import random
import re
import warnings
import time
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
from huggingface_hub import hf_hub_download
from torch.nn.utils import clip_grad_norm_, parametrize
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers import get_scheduler

from .audio import apply_audio_delay, build_delay_indices, codebook_to_audio
from .config import DiaConfig
from .dataset import PreEncodedDACDataset
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
    "piano_ambient": "piano, pads, ambient, cinematic, melancholic, peaceful, reflective, instrumental",
    "dark": "cinematic, suspenseful, dark, energetic, mysterious, strings, bells, bass"
}
EVAL_CFG_SCALE = 4.0
EVAL_CFG_SCALE_UNCOND = 0.0
EVAL_TOP_P = 0.95
EVAL_TEMPERATURES = [0.5, 1.0]
EVAL_SAMPLE_RATE = 44100
EVAL_AUDIO_DIR = "./audio_demos"
GRAD_CLIP_MAX_NORM = 1.0 # Lowered slightly for FSDP stability
ENTROPY_LOG_INTERVAL = 50
CODEBOOK_WEIGHTS = [1.0] * 9
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
    if keep_last_n is None: return
    for pattern in ["ckpt_step*.pth", "ckpt_epoch*.pth"]:
        files = sorted(glob.glob(str(output_dir / pattern)), 
                       key=lambda x: int(re.search(r'(\d+).pth', x).group(1)) if re.search(r'(\d+).pth', x) else 0)
        if len(files) > keep_last_n:
            for old_ckpt in files[:-keep_last_n]:
                try:
                    os.remove(old_ckpt)
                    logger.info(f"Removed old checkpoint: {old_ckpt}")
                except Exception:
                    pass

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

# =============================================================================
# TRAIN STEP
# =============================================================================

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
        
        # Loss Masking (Safe for TPU/XLA)
        mask = torch.arange(target.shape[1], device=target.device).unsqueeze(0) < (batch['tgt_lens'].unsqueeze(1) - 1)
        mask = mask.unsqueeze(-1).expand_as(target)
        audio_token_mask = (target >= 0) & (target <= 1023)
        final_mask = (mask & audio_token_mask).float()
        
        loss = 0.0
        # Assuming simplified weights for stability
        channel_weights = [1.0] * target.shape[2]
        
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
    
    if (global_step + 1) % train_cfg.grad_accum_steps == 0:
        accelerator.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
        opt.step()
        sched.step()
        opt.zero_grad()
    
    return loss.detach()

# =============================================================================
# DEMO GENERATION
# =============================================================================

def generate_demos(model, dia_cfg, train_cfg, global_step, accelerator):
    """Generate demo audio samples during training."""
    if not accelerator.is_main_process:
        return
    
    t_start = time.perf_counter()
    logger.info(f"[DEMO] Generating audio samples at step {global_step}")
    Path(EVAL_AUDIO_DIR).mkdir(parents=True, exist_ok=True)
    
    unwrapped = accelerator.unwrap_model(model)
    
    try:
        # Create Dia wrapper for generation
        dia_gen = Dia(dia_cfg, accelerator.device)
        dia_gen.model = unwrapped
        
        # Load DAC model for audio decoding (cached per device)
        dac_load_t0 = time.perf_counter()
        dac_model = get_dac_model(accelerator.device)
        dia_gen.dac_model = dac_model
        logger.info(f"[DEMO] DAC model attached (ready in {time.perf_counter() - dac_load_t0:.1f}s)")
        
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
                            logger.info(f"[DEMO] Start unconditional temp={temp}, seed={s}")
                            try:
                                audio = dia_gen.generate(text="", cfg_scale=EVAL_CFG_SCALE_UNCOND, temperature=temp)
                                path = Path(EVAL_AUDIO_DIR) / f"step_{global_step}_temp{safe_temp(temp)}_seed{s}.wav"
                                _save_and_log_audio(audio, path, f"eval_audio/temp{safe_temp(temp)}/seed{s}", 
                                                 f"temp={temp}", audio_samples)
                            except Exception as e:
                                logger.warning(f"Demo generation failed for temp={temp}, seed={s}: {e}")
                            else:
                                logger.info(f"[DEMO] Finished temp={temp}, seed={s} in {time.perf_counter() - demo_t0:.1f}s")
            else:
                # Conditional generation with prompts
                cfg_s = EVAL_CFG_SCALE if train_cfg.unconditional_frac > 0 else None
                for temp in EVAL_TEMPERATURES:
                    for name, prompt in TEST_PROMPTS.items():
                        demo_t0 = time.perf_counter()
                        logger.info(f"[DEMO] Start prompt={name}, temp={temp}")
                        try:
                            audio = dia_gen.generate(text=prompt, cfg_scale=cfg_s, temperature=temp, top_p=EVAL_TOP_P)
                            path = Path(EVAL_AUDIO_DIR) / f"step_{global_step}_{name}_temp{safe_temp(temp)}.wav"
                            _save_and_log_audio(audio, path, f"eval_audio/temp{safe_temp(temp)}/{name}", 
                                             prompt, audio_samples)
                        except Exception as e:
                            logger.warning(f"Demo generation failed for {name}, temp={temp}: {e}")
                        else:
                            logger.info(f"[DEMO] Finished prompt={name}, temp={temp} in {time.perf_counter() - demo_t0:.1f}s")
            
            if audio_samples:
                wandb.log(audio_samples, step=global_step)
                logger.info(f"[DEMO] Logged {len(audio_samples)} audio samples to wandb")
                
    except Exception as e:
        logger.exception(f"[DEMO] Demo generation failed: {e}")
    finally:
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
                
                # Masking
                mask = torch.arange(target.shape[1], device=target.device).unsqueeze(0) < (batch['tgt_lens'].unsqueeze(1) - 1)
                mask = mask.unsqueeze(-1).expand_as(target)
                audio_token_mask = (target >= 0) & (target <= 1023)
                final_mask = (mask & audio_token_mask).float()
                
                loss = 0.0
                channel_weights = [1.0] * target.shape[2]
                
                for c, w in enumerate(channel_weights):
                    l_c = logits[:, :, c, :].flatten(0, 1)
                    t_c = target[:, :, c].flatten()
                    m_c = final_mask[:, :, c].flatten()
                    
                    ce_loss = F.cross_entropy(l_c, t_c, reduction='none', ignore_index=dia_cfg.data.audio_pad_value)
                    masked_loss = (ce_loss * m_c).sum()
                    mask_sum = m_c.sum() + 1e-9
                    loss += w * (masked_loss / mask_sum)
                
                loss = loss / sum(channel_weights)
                eval_losses.append(loss.detach())
    
    # Gather losses across all processes
    if eval_losses:
        local_avg = torch.stack(eval_losses).mean()
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
    args = get_args()
    
    # Initialize Accelerator with BF16 and FSDP awareness
    accelerator = Accelerator(mixed_precision="bf16", log_with="wandb")
    device = accelerator.device
    
    if accelerator.is_main_process:
        print(f"[INIT] Launching on {accelerator.num_processes} processes (FSDP enabled).")

    # Load Config
    dia_cfg = DiaConfig.load(args.config)
    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        seed=args.seed,
        demo_every=args.demo_every,
        eval_every=args.eval_every,
    )
    
    # WandB Init (Accelerate handles main process check internally for loggers usually, but we are explicit)
    if accelerator.is_main_process:
        train_cfg.output_dir.mkdir(parents=True, exist_ok=True)
        accelerator.init_trackers(
            project_name=args.wandb_project, 
            config=vars(args),
            init_kwargs={"wandb": {"name": train_cfg.run_name}}
        )

    # Dataset
    if args.preencoded_dir:
        dataset = PreEncodedDACDataset(args.preencoded_dir, dia_cfg, args.use_sliding_window)
    elif args.audio_folder:
        # Load CPU DAC for encoding in dataloaders
        dac_model = dac.DAC.load(dac.utils.download()).eval().to('cpu')
        strip_weight_norms(dac_model)
        dataset = TestingDataset(args.audio_folder, dia_cfg, dac_model, args.use_sliding_window)
    
    train_loader, val_loader = setup_loaders(dataset, dia_cfg, train_cfg, args.use_sliding_window)
    
    # Model
    model = DiaModel(dia_cfg)
    strip_weight_norms(model)
    
    if args.hub_model and not args.scratch:
        ckpt_path = hf_hub_download(args.hub_model, filename="dia-v0_1.pth")
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state, strict=False)
    elif args.scratch:
        if accelerator.is_main_process:
            print("[TRAIN] Training from scratch (no pretrained weights)")

    # Optimizer
    # Separate params for decay
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        if "bias" in name or "norm" in name:
            no_decay.append(p)
        else:
            decay.append(p)
            
    opt = optim.AdamW([
        {"params": decay, "weight_decay": train_cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=train_cfg.learning_rate)
    
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * train_cfg.epochs
    sched = get_scheduler('cosine', opt, num_warmup_steps=train_cfg.warmup_steps, num_training_steps=total_steps)

    # CRITICAL FSDP STEP: Prepare everything together
    # FSDP will shard the model across TPUs here
    model, opt, train_loader, sched = accelerator.prepare(model, opt, train_loader, sched)
    if val_loader: val_loader = accelerator.prepare(val_loader)

    global_step = 0
    
    for epoch in range(train_cfg.epochs):
        model.train()
        for batch in tqdm(train_loader, disable=not accelerator.is_main_process, desc=f"E{epoch+1}"):
            loss = train_step(model, batch, dia_cfg, train_cfg, opt, sched, global_step, accelerator)
            
            # CRITICAL: Force XLA to execute pending ops after each step
            xm.mark_step()

            # Log every 10 steps to reduce sync overhead while debugging
            if global_step % 10 == 0:
                if accelerator.is_main_process:
                    accelerator.log({"train_loss": loss.item(), "lr": sched.get_last_lr()[0]}, step=global_step)
            
            global_step += 1
            
            # Save
            if global_step % train_cfg.save_step == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    # FSDP saving requires unwrapping or special state dict policy
                    # Accelerate handles this via save_state or unwrap_model depending on config
                    # Standard compliant way for FSDP:
                    save_path = train_cfg.output_dir / f"ckpt_step{global_step}"
                    accelerator.save_state(output_dir=save_path) 
                    logger.info(f"Saved checkpoint to {save_path}")
            
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

    accelerator.end_training()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--hub_model", type=str, default="nari-labs/Dia-1.6B")
    parser.add_argument("--audio_folder", type=Path)
    parser.add_argument("--preencoded_dir", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default="dia-fsdp")
    parser.add_argument("--use_sliding_window", action="store_true")
    parser.add_argument("--scratch", action="store_true", help="Train from scratch (skip loading pretrained weights)")
    parser.add_argument("--demo_every", type=int, default=None, help="Generate demo audio every N steps")
    parser.add_argument("--eval_every", type=int, default=None, help="Run evaluation every N steps")
    return parser.parse_args()

if __name__ == "__main__":
    main()
