"""
Dia Audio Model Fine-tuning on TPU (Full Model).

Adapts dia/train_acc_gpu.py for TPU/XLA compatibility.
Run with: accelerate launch dia/train_acc_tpu.py --args...
"""

import argparse
import glob
import logging
import json
import os
import random
import re
import sys
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

# Conditional import for TestingDataset (only needed when using --audio_folder)
try:
    from . import dataset as dataset_module
    TestingDataset = getattr(dataset_module, 'TestingDataset', None)
except (ImportError, AttributeError):
    TestingDataset = None

# TPU Imports
try:
    import torch_xla.core.xla_model as xm
    HAS_XLA = True
except ImportError:
    HAS_XLA = False

warnings.filterwarnings("ignore", message="`torch.nn.utils.weight_norm` is deprecated")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS & CONFIG (Synced with GPU)
# =============================================================================

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
GRAD_CLIP_MAX_NORM = 5.0
ENTROPY_LOG_INTERVAL = 50

# Codebook weighting for loss calculation
CODEBOOK_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
_nonfinite_hits = 0

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
    run_name: str = "dia_finetune_tpu"
    output_dir: Optional[Path] = None
    
    # Tag augmentation
    tag_shuffle: bool = True
    tag_dropout: float = 0.0
    tag_limit: Optional[int] = None
    
    # Optimizer
    no_decay_embed: bool = False

# =============================================================================
# UTILS
# =============================================================================

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def strip_weight_norms(module: torch.nn.Module) -> int:
    """Remove weight_norm parametrizations so XLA doesn't fall back to CPU."""
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
# DATA & COLLATE (TPU OPTIMIZED)
# =============================================================================

def collate_fn(batch, config: DiaConfig, train_cfg: TrainConfig, device: torch.device, use_sliding_window: bool = True):
    """
    TPU-optimized collate function with FIXED shapes.
    Pads everything to config.data.audio_length (window_size) to avoid XLA recompilation.
    """
    texts, encodings, waveforms = zip(*batch)
    window_size = config.data.audio_length
    
    # TPU: Force fixed batch length = window_size to avoid recompilation
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

    codes = torch.stack(padded_encodings).to(device)

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
    
    # We record actual lengths for masking in loss, but tensor shapes are fixed
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
    # For TPU, we MUST use fixed batch sizes.
    # collate_fn returns CPU tensors which Accelerate moves to device later.
    collate = lambda b: collate_fn(b, dia_cfg, train_cfg, torch.device('cpu'), use_sliding_window)
    
    ds_len = len(dataset)
    n_train = int(train_cfg.split_ratio * ds_len)
    n_val = ds_len - n_train
    
    if ds_len <= 1 or n_val == 0:
        logger.info(f"Dataset size {ds_len}: using all for training")
        train_ds, val_ds = dataset, None
    else:
        g = torch.Generator().manual_seed(train_cfg.seed)
        train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=g)
    
    # Drop last to keep batch sizes consistent (critical for TPU)
    train_loader = DataLoader(
        train_ds, batch_size=train_cfg.batch_size, shuffle=True,
        collate_fn=collate, num_workers=0, pin_memory=False, drop_last=True
    )
    if val_ds:
        val_loader = DataLoader(val_ds, batch_size=train_cfg.batch_size, shuffle=False, collate_fn=collate, num_workers=0, drop_last=True)
    else:
        val_loader = None
        
    train_loader.steps_per_epoch = len(train_loader)
    return train_loader, val_loader

# =============================================================================
# OPTIMIZER (Synced with GPU)
# =============================================================================

def setup_optimizer_and_scheduler(model, train_loader, train_cfg):
    # 1. Define distinct parameter groups
    decay, no_decay = [], []
    seen = set()
    
    norm_types = (torch.nn.LayerNorm, torch.nn.GroupNorm, torch.nn.BatchNorm1d, 
                  torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)
    try:
        norm_types += (torch.nn.modules.normalization.RMSNorm,)
    except AttributeError: pass

    for module in model.modules():
        for name, p in module.named_parameters(recurse=False):
            if not p.requires_grad or id(p) in seen: continue
            seen.add(id(p))
            
            is_bias = name.endswith("bias")
            is_norm = isinstance(module, norm_types)
            is_embed = isinstance(module, torch.nn.Embedding)
            is_lora = "lora" in name.lower() and ("alpha" in name.lower() or "scale" in name.lower())
            
            if is_bias or is_norm or is_lora or (train_cfg.no_decay_embed and is_embed):
                no_decay.append(p)
            else:
                decay.append(p)

    # Catch-all
    for name, p in model.named_parameters():
        if p.requires_grad and id(p) not in seen:
            decay.append(p)
            seen.add(id(p))

    opt = optim.AdamW([
        {"params": decay, "weight_decay": train_cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=train_cfg.learning_rate)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * train_cfg.epochs
    sched = get_scheduler('cosine', opt, 
                          num_warmup_steps=train_cfg.warmup_steps // train_cfg.grad_accum_steps,
                          num_training_steps=total_steps // train_cfg.grad_accum_steps)
    
    return opt, sched

# =============================================================================
# TPU GENERATION (Static Cache Optimized)
# =============================================================================

def generate_demo_sample_tpu(model, dia_cfg, device, max_tokens, text, temp, cfg_scale, out_path):
    """
    Inner generation loop for a single sample on TPU.
    Uses static KV caches and manual loop to avoid XLA graph recompilation.
    """
    dia_gen = Dia(dia_cfg, device)
    # Reuse preparation logic
    (src_BxS, src_pos, src_pad, enc_mask) = dia_gen._prepare_text_input(text)
    
    # Encoder
    encoder_out = model.encoder(src_BxS, src_pos, deterministic=True, attn_mask=enc_mask)
    
    # Static Caches
    cross_cache = model.decoder.precompute_cross_attention_kv(max_tokens, encoder_out, src_pos)
    self_cache = [KVCache(dia_cfg.model.decoder.gqa_query_heads, max_tokens, dia_cfg.model.decoder.gqa_head_dim, device, batch_size=1) for _ in range(model.decoder.num_layers)]
    
    generated = torch.full((1, 1, dia_cfg.data.channels), dia_cfg.data.audio_bos_value, dtype=torch.long, device=device)
    
    # Pre-allocate static mask buffer
    static_indices = torch.arange(max_tokens, device=device).view(1, 1, 1, max_tokens)

    # Decode Loop
    for step in range(max_tokens):
        if step % 20 == 0 and HAS_XLA: xm.mark_step()
        
        tgt_ids = generated[:, step].unsqueeze(1)
        tgt_pos = torch.full((1, 1), step, dtype=torch.long, device=device)
        
        # 1. Cross Mask
        dec_cross_mask = dia_gen._create_attn_mask(torch.ones((1, 1), dtype=torch.bool, device=device), src_pad, is_causal=False)
        
        # 2. Self Mask (Static logic)
        current_step_t = torch.tensor(step, device=device)
        self_attn_mask = torch.where(
            static_indices <= current_step_t,
            torch.tensor(0.0, device=device),
            torch.tensor(float('-inf'), device=device)
        )
        
        logits, _ = model.decoder.decode_step(tgt_ids, tgt_pos, encoder_out, self_attn_mask, dec_cross_mask, self_cache, cross_cache)

        # Sampling
        logits = logits[:, -1, :, :] # (1, C, V)
        
        # Classifier Free Guidance (Unconditional part skipped for simplicity in TPU demo, unless implemented)
        # Note: Implementing CFG on TPU efficiently requires batching conditional+unconditional together.
        # For this script, we stick to simple sampling or conditional only if cfg_scale is not handled.
        
        if temp > 0:
            probs = F.softmax(logits / temp, dim=-1)
            next_toks = torch.multinomial(probs.view(-1, logits.size(-1)), 1).view(1, -1)
        else:
            next_toks = torch.argmax(logits, dim=-1)
            
        generated = torch.cat([generated, next_toks.unsqueeze(1)], dim=1)
        if HAS_XLA: xm.mark_step()
    
    # Save
    dac_model = dac.DAC.load(dac.utils.download()).eval().cpu()
    codes = generated[0, 1:].cpu() # (T, C)
    audio = codebook_to_audio(codes.T.unsqueeze(0), dac_model, dia_cfg.data.delay_pattern, B=1, T=max_tokens, C=dia_cfg.data.channels)
    sf.write(out_path, audio.squeeze().detach().numpy().T, EVAL_SAMPLE_RATE)
    logger.info(f"Saved TPU demo: {out_path}")

# =============================================================================
# EVAL STEP
# =============================================================================

def eval_step(model, val_loader, dia_cfg, dac_model, global_step, device, train_cfg, accelerator, do_demo=True, current_train_loss=None, stop_on_overfit=False):
    """TPU Evaluation Step."""
    eval_losses = []
    
    if val_loader:
        model.eval()
        with torch.no_grad():
            for eb in tqdm(val_loader, desc="eval", disable=not accelerator.is_main_process):
                # Ensure device placement (Accelerate usually handles this, but collate returns CPU)
                for k, v in eb.items():
                    if isinstance(v, torch.Tensor): eb[k] = v.to(device)

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

                # Explicit float cast for stability
                logits = logits.float() 
                target = eb['tgt_tokens'][:, 1:]
                
                # --- LOSS MASKING (TPU OPTIMIZED) ---
                # GPU script uses boolean masking: tensor[mask]. This is fatal for TPU (dynamic shapes).
                # TPU solution: Use reduction='none' and mathematical masking.
                
                # 1. Length mask
                mask = torch.arange(target.shape[1], device=target.device).unsqueeze(0) < (eb['tgt_lens'].unsqueeze(1) - 1)
                mask = mask.unsqueeze(-1).expand_as(target) # (B, T, C)
                
                # 2. Audio value mask (ignore BOS/EOS/Special)
                audio_token_mask = (target >= 0) & (target <= 1023)
                
                # 3. Combine
                final_mask = (mask & audio_token_mask).float()
                
                loss = 0.0
                channel_weights = []
                num_groups = target.shape[2] // 9
                if num_groups > 0:
                    for _ in range(num_groups):
                        channel_weights.extend(CODEBOOK_WEIGHTS)
                else:
                    channel_weights = [1.0] * target.shape[2]
                
                for c, w in enumerate(channel_weights):
                    l_c = logits[:, :, c, :].flatten(0, 1) # (B*T, V)
                    t_c = target[:, :, c].flatten()      # (B*T)
                    m_c = final_mask[:, :, c].flatten()  # (B*T)
                    
                    # Compute unreduced loss
                    ce_loss = F.cross_entropy(l_c, t_c, reduction='none', ignore_index=dia_cfg.data.audio_pad_value)
                    
                    # Apply mask mathematically
                    masked_loss = (ce_loss * m_c).sum()
                    mask_sum = m_c.sum() + 1e-9
                    
                    loss += w * (masked_loss / mask_sum)
                
                eval_losses.append(loss / sum(channel_weights))
                if HAS_XLA: xm.mark_step()
        model.train()

    # Logging & Stop Check
    should_stop = False
    if eval_losses:
        local_avg = sum(eval_losses) / len(eval_losses)
        avg_loss = accelerator.gather(local_avg.unsqueeze(0)).mean().item()
        
        if accelerator.is_main_process:
            wandb.log({'eval_loss': avg_loss}, step=global_step)
            if stop_on_overfit and current_train_loss and avg_loss > current_train_loss:
                logger.info(f"Stop trigger: Eval {avg_loss:.4f} > Train {current_train_loss:.4f}")
                should_stop = True
                do_demo = True

    if not do_demo: 
        return should_stop

    # Demo Generation
    if accelerator.is_main_process:
        logger.info(f"Generating eval demos at step {global_step}")
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.eval()
        max_tokens = min(200, dia_cfg.data.audio_length)

        try:
            with torch.no_grad():
                def safe_temp(t): return f"{t:.1f}".replace(".", "p")
                
                if train_cfg.unconditional_frac >= 1.0:
                    seeds = [int(train_cfg.seed), int(train_cfg.seed) + 1]
                    for temp in EVAL_TEMPERATURES:
                        for s in seeds:
                            seed_everything(s)
                            generate_demo_sample_tpu(
                                unwrapped, dia_cfg, device, max_tokens, 
                                text="", temp=temp, cfg_scale=EVAL_CFG_SCALE_UNCOND,
                                out_path=Path(EVAL_AUDIO_DIR) / f"step_{global_step}_temp{safe_temp(temp)}_seed{s}.wav"
                            )
                else:
                    cfg_s = EVAL_CFG_SCALE if train_cfg.unconditional_frac > 0 else None
                    for temp in EVAL_TEMPERATURES:
                        for name, prompt in TEST_PROMPTS.items():
                            generate_demo_sample_tpu(
                                unwrapped, dia_cfg, device, max_tokens, 
                                text=prompt, temp=temp, cfg_scale=cfg_s,
                                out_path=Path(EVAL_AUDIO_DIR) / f"step_{global_step}_{name}_temp{safe_temp(temp)}.wav"
                            )

        except Exception as e:
            logger.exception(f"TPU Demo Failed: {e}")
        finally:
            unwrapped.train()
            
    accelerator.wait_for_everyone()
    return should_stop

# =============================================================================
# TRAIN STEP
# =============================================================================

def train_step(model, batch, dia_cfg, train_cfg, opt, sched, step, global_step, accelerator):
    global _nonfinite_hits
    
    # Unconditional dropout
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
        
        # We need to slice based on fixed shapes
        # batch['tgt_tokens'] has shape (B, window_size+2, C)
        # We want to predict from index 1 to end
        logits = logits[:, :-1]
        target = batch['tgt_tokens'][:, 1:]
        
        # --- LOSS MASKING (TPU OPTIMIZED) ---
        # 1. Length Mask
        mask = torch.arange(target.shape[1], device=target.device).unsqueeze(0) < (batch['tgt_lens'].unsqueeze(1) - 1)
        mask = mask.unsqueeze(-1).expand_as(target)
        
        # 2. Audio Value Mask
        audio_token_mask = (target >= 0) & (target <= 1023)
        
        # 3. Combine
        final_mask = (mask & audio_token_mask).float()
        
        loss = 0.0
        channel_weights = []
        num_groups = target.shape[2] // 9
        if num_groups > 0:
            for _ in range(num_groups):
                channel_weights.extend(CODEBOOK_WEIGHTS)
        else:
            channel_weights = [1.0] * target.shape[2]
        
        for c, w in enumerate(channel_weights):
            l_c = logits[:, :, c, :].flatten(0, 1)
            t_c = target[:, :, c].flatten()
            m_c = final_mask[:, :, c].flatten()
            
            # Unreduced CE
            ce_loss = F.cross_entropy(l_c, t_c, reduction='none', ignore_index=dia_cfg.data.audio_pad_value)
            
            # Masked Mean
            masked_loss = (ce_loss * m_c).sum()
            mask_sum = m_c.sum() + 1e-9
            
            loss += w * (masked_loss / mask_sum)
        
        loss = loss / sum(channel_weights)

        if global_step % ENTROPY_LOG_INTERVAL == 0 and accelerator.is_main_process:
            wandb.log({'output_entropy': compute_output_entropy(logits.detach())}, step=global_step)

    if not torch.isfinite(loss):
        _nonfinite_hits += 1
        if HAS_XLA: xm.mark_step()
        return float('nan')

    loss = loss / train_cfg.grad_accum_steps
    accelerator.backward(loss)
    
    if (step + 1) % train_cfg.grad_accum_steps == 0:
        norm = accelerator.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
        if accelerator.is_main_process:
             wandb.log({'grad_norm': norm}, step=global_step)
        
        opt.step()
        sched.step()
        opt.zero_grad()
    
    if HAS_XLA: xm.mark_step()
    return loss.item() * train_cfg.grad_accum_steps

# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def train(args):
    """TPU Training Loop (Robust)."""
    # Use bf16 by default for TPU
    accelerator = Accelerator(mixed_precision="bf16" if args.half else "no")
    device = accelerator.device
    
    dia_cfg = DiaConfig.load(args.config)
    
    # Setup Config
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
        no_decay_embed=args.no_decay_embed,
    )

    if accelerator.is_main_process:
        train_cfg.output_dir.mkdir(parents=True, exist_ok=True)
        Path(EVAL_AUDIO_DIR).mkdir(exist_ok=True)
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=train_cfg.run_name, config=vars(args))
    
    accelerator.wait_for_everyone()
    seed_everything(args.seed)

    # Dataset
    if args.preencoded_dir:
        dataset = PreEncodedDACDataset(args.preencoded_dir, dia_cfg, args.use_sliding_window)
    elif args.audio_folder:
        if TestingDataset is None:
            raise ImportError("TestingDataset is not available. Please ensure dia/dataset.py contains the TestingDataset class.")
        # Load CPU DAC for encoding in dataloaders
        dac_model = dac.DAC.load(dac.utils.download()).eval().to('cpu')
        strip_weight_norms(dac_model)
        skip_tags = [t.strip() for t in args.skip_tags.split(',')] if args.skip_tags else None
        dataset = TestingDataset(args.audio_folder, dia_cfg, dac_model, args.use_sliding_window, skip_tags=skip_tags)
    else:
        raise ValueError("Must specify --audio_folder or --preencoded_dir")

    train_loader, val_loader = setup_loaders(dataset, dia_cfg, train_cfg, device, args.use_sliding_window)
    
    # Model Setup
    model = DiaModel(dia_cfg)
    removed = strip_weight_norms(model)
    if accelerator.is_main_process: logger.info(f"Stripped {removed} weight_norms from DiaModel")

    # Checkpoint Loading with Stereo Expansion (Robustness)
    if args.local_ckpt:
        ckpt_path = args.local_ckpt
    else:
        ckpt_path = hf_hub_download(args.hub_model, filename="dia-v0_1.pth")
    
    state = torch.load(ckpt_path, map_location="cpu")
    
    # Stereo Logic (Ported from GPU script)
    key = "decoder.logits_dense.weight"
    expanded_stereo = False
    if key in state and dia_cfg.data.channels == 18:
        W_ckpt = state[key]
        if W_ckpt.shape[1] == 9:
            logger.info(f"Expanding checkpoint {key} from 9 to 18 channels...")
            state[key] = torch.cat([W_ckpt, W_ckpt], dim=1)
            expanded_stereo = True
            
    model.load_state_dict(state, strict=False)
    
    if expanded_stereo:
        logger.info("Warm-starting stereo embeddings...")
        with torch.no_grad():
            for i in range(9, 18):
                model.decoder.embeddings[i].weight.copy_(model.decoder.embeddings[i - 9].weight)
            model.decoder.logits_dense.weight.data[:, 9:18, :].copy_(
                model.decoder.logits_dense.weight.data[:, 0:9, :]
            )

    # Freezing Heuristic
    if train_cfg.unconditional_frac >= 0.9 and hasattr(model, 'encoder'):
        logger.info("Freezing encoder for high-unconditional training")
        for p in model.encoder.parameters():
            p.requires_grad = False

    # Optimizer
    opt, sched = setup_optimizer_and_scheduler(model, train_loader, train_cfg)
    
    model, opt, train_loader, sched = accelerator.prepare(model, opt, train_loader, sched)
    if val_loader: val_loader = accelerator.prepare(val_loader)
    
    # XLA Optimizer State Init
    if HAS_XLA:
        for g in opt.param_groups:
             for p in g['params']:
                 if p.requires_grad and len(opt.state[p]) == 0:
                     opt.state[p] = {'step': torch.zeros((), device=p.device), 'exp_avg': torch.zeros_like(p), 'exp_avg_sq': torch.zeros_like(p)}

    model.train()
    stop_training = False
    steps_per_epoch = len(train_loader)

    for epoch in range(train_cfg.epochs):
        if stop_training: break
        
        loader_iter = tqdm(train_loader, desc=f"E{epoch+1}", disable=not accelerator.is_main_process)
        
        for step, batch in enumerate(loader_iter):
            # Move batch items to device (needed because collate returns CPU tensors)
            for k, v in batch.items():
                if isinstance(v, torch.Tensor): batch[k] = v.to(device)

            global_step = epoch * steps_per_epoch + step
            loss = train_step(model, batch, dia_cfg, train_cfg, opt, sched, step, global_step, accelerator)
            
            if accelerator.is_main_process:
                loader_iter.set_postfix({'loss': f"{loss:.4f}"})
                wandb.log({'train_loss': loss, 'lr': sched.get_last_lr()[0]}, step=global_step)

            # Evaluation
            should_eval_step = (args.eval_every_epochs is None) and (global_step % train_cfg.eval_step == 0) and (global_step > 0)
            
            if should_eval_step:
                demo_interval = args.demo_every or train_cfg.eval_step
                do_demo = (global_step % demo_interval == 0) and ((epoch + 1) > args.demo_after_epoch)
                
                should_stop_local = eval_step(
                    model, val_loader, dia_cfg, None, global_step, device, 
                    train_cfg, accelerator, do_demo=do_demo, 
                    current_train_loss=loss, stop_on_overfit=args.stop_on_overfit
                )
                
                if args.stop_on_overfit:
                    stop_tensor = torch.tensor(int(should_stop_local), device=device)
                    stop_training = accelerator.gather(stop_tensor).sum() > 0
                    if stop_training: break

            # Step Checkpointing
            if args.save_every_epochs is None and (epoch + 1) > args.save_after_epoch:
                save_interval = args.save_every or train_cfg.save_step
                if global_step > 0 and global_step % save_interval == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        accelerator.save(accelerator.unwrap_model(model).state_dict(), train_cfg.output_dir / f"ckpt_step{global_step}.pth")
                        cleanup_old_checkpoints(train_cfg.output_dir, args.save_last)

            # Early Stopping (Loss)
            if args.early_stop_loss and loss <= args.early_stop_loss and not np.isnan(loss):
                stop_tensor = torch.tensor(1, device=device)
                stop_training = accelerator.gather(stop_tensor).sum() > 0
                if stop_training:
                    if accelerator.is_main_process:
                        accelerator.save(accelerator.unwrap_model(model).state_dict(), train_cfg.output_dir / f"ckpt_early_stop.pth")
                    break

        # Epoch End Logic
        if args.eval_every_epochs and (epoch + 1) % args.eval_every_epochs == 0:
             eval_step(model, val_loader, dia_cfg, None, global_step, device, train_cfg, accelerator, do_demo=True, current_train_loss=loss)
             model.train()

        is_save_epoch = args.save_every_epochs and ((epoch + 1) % args.save_every_epochs == 0)
        is_last = (epoch + 1) == train_cfg.epochs
        
        if (is_save_epoch or is_last) and (epoch + 1) > args.save_after_epoch:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                accelerator.save(accelerator.unwrap_model(model).state_dict(), train_cfg.output_dir / f"ckpt_epoch{epoch+1}.pth")
                if is_last:
                    accelerator.save(accelerator.unwrap_model(model).state_dict(), train_cfg.output_dir / "latest.pth")
                cleanup_old_checkpoints(train_cfg.output_dir, args.save_last)

    accelerator.wait_for_everyone()

# =============================================================================
# ARGS & UTILS COPY
# =============================================================================

def load_train_config(config_path: Path) -> dict:
    if not config_path.exists(): return {}
    with open(config_path) as f: cfg = json.load(f)
    def flatten(d):
        out = {}
        for k, v in d.items():
            if isinstance(v, dict): out.update(flatten(v))
            else: out[k] = v
        return out
    return flatten(cfg)

def get_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--train_config", type=Path, default=Path("configs/train_config.json"))
    pre_args, _ = pre_parser.parse_known_args()
    defaults = load_train_config(pre_args.train_config)
    
    parser = argparse.ArgumentParser(description="Train the Dia audio model (TPU)")
    
    # Paths & Configs
    parser.add_argument("--train_config", type=Path, default=Path("configs/train_config.json"))
    parser.add_argument("--config", type=Path, default=Path(defaults.get('config', 'configs/architecture/model.json')))
    parser.add_argument("--hub_model", type=str, default="nari-labs/Dia-1.6B")
    parser.add_argument("--local_ckpt", type=str, default=None)
    parser.add_argument("--audio_folder", type=Path, default=defaults.get('audio_folder'))
    parser.add_argument("--preencoded_dir", type=Path, default=defaults.get('preencoded_dir'))
    parser.add_argument("--run_name", type=str, default=defaults.get('run_name', "dia_tpu"))
    parser.add_argument("--output_dir", type=Path, default=defaults.get('output_dir'))
    
    # Training
    parser.add_argument("--epochs", type=int, default=defaults.get('epochs', 500))
    parser.add_argument("--batch_size", type=int, default=defaults.get('batch_size', 4))
    parser.add_argument("--grad_accum_steps", type=int, default=defaults.get('grad_accum_steps', 1))
    parser.add_argument("--learning_rate", type=float, default=defaults.get('learning_rate', 1e-5))
    parser.add_argument("--weight_decay", type=float, default=defaults.get('weight_decay', 0.1))
    parser.add_argument("--warmup_steps", type=int, default=defaults.get('warmup_steps', 500))
    parser.add_argument("--unconditional_frac", type=float, default=defaults.get('unconditional_frac', 0.15))
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
    parser.add_argument("--save_last", type=int, default=None)
    parser.add_argument("--early_stop_loss", type=float, default=None)
    parser.add_argument("--stop_on_overfit", action="store_true")
    
    # Flags
    parser.add_argument("--scratch", action=argparse.BooleanOptionalAction, default=defaults.get('scratch', False))
    parser.add_argument("--tag_no_shuffle", action=argparse.BooleanOptionalAction, default=defaults.get('tag_no_shuffle', False))
    parser.add_argument("--force_single_gpu", action=argparse.BooleanOptionalAction, default=defaults.get('force_single_gpu', False))
    parser.add_argument("--use_sliding_window", action=argparse.BooleanOptionalAction, default=defaults.get('use_sliding_window', True))
    parser.add_argument("--tag_shuffle", action=argparse.BooleanOptionalAction, default=True)
    
    # Optimizer
    parser.add_argument("--half", action="store_true", help="BF16 mixed precision")
    parser.add_argument("--compile", action="store_true") # No-op on TPU usually but kept for compat
    parser.add_argument("--no_decay_embed", action="store_true", help="Exclude nn.Embedding parameters from weight decay")
    parser.add_argument("--skip_tags", type=str, default=None)
    
    # WandB
    parser.add_argument("--wandb_project", type=str, default="dia-tpu")
    parser.add_argument("--wandb_entity", type=str, default=None)
    
    # Augmentation
    parser.add_argument("--tag_dropout", type=float, default=0.0)
    parser.add_argument("--tag_limit", type=int, default=None)

    args = parser.parse_args()
    
    if args.output_dir is None:
        parser.error("--output_dir is required")
        
    return args

def main():
    """Entry point for accelerate launch --module."""
    args = get_args()
    train(args)

if __name__ == "__main__":
    main()