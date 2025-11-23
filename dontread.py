#!/usr/bin/env python3
"""Simple training from scratch for preprocessed .pkl data without dynamic cropping."""

import argparse
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import dac
import soundfile as sf
import wandb
import pickle
import json
import os
import glob
import random
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import logging
import time
import signal
import sys
import datetime

from dia.config import DiaConfig
from dia.layers import DiaModel
from dia.finetune import TrainConfig, train, setup_loaders, setup_optimizer_and_scheduler, train_step
from dia.audio import codebook_to_audio_stereo, audio_to_codebook_stereo
from dia.model import _sample_next_token

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def manual_stereo_generation_like_working_script(dia_model, text_prompt: str, config: DiaConfig, device: torch.device, max_tokens: int = 430):
    """Manual stereo generation exactly like test_stereo_inference_fixed.py."""
    
    # Prepare text input (identical to working script)
    byte_text = text_prompt.encode('utf-8')
    text_tokens = list(byte_text)
    
    # Pad to config length
    max_len = config.data.text_length
    if len(text_tokens) > max_len:
        text_tokens = text_tokens[:max_len]
    else:
        text_tokens = text_tokens + [config.data.text_pad_value] * (max_len - len(text_tokens))
    
    src_tokens = torch.tensor(text_tokens, dtype=torch.long).unsqueeze(0).to(device)  # (1, 512)
    src_positions = torch.arange(max_len, device=device).unsqueeze(0)  # (1, 512)
    
    # Create source attention mask (identical to working script)
    src_padding_mask = (src_tokens != config.data.text_pad_value).to(device)
    enc_self_attn_mask = src_padding_mask.unsqueeze(1) & src_padding_mask.unsqueeze(2)  # (1, 512, 512)
    
    # Pre-allocate generated tokens tensor for speed (identical to working script)
    max_seq_len = max_tokens + 1
    generated_tokens = torch.full(
        (1, max_seq_len, 18), 
        config.data.audio_pad_value,  # Fill with pad tokens
        dtype=torch.long, 
        device=device
    )
    
    # Set first token as BOS (identical to working script)
    generated_tokens[:, 0, :] = config.data.audio_bos_value
    current_len = 1
    
    # Pre-allocate attention masks for max length (identical to working script)
    max_dec_len = max_seq_len
    causal_mask_full = torch.tril(torch.ones(max_dec_len, max_dec_len, dtype=torch.bool, device=device))
    cross_attn_mask_full = torch.ones(1, max_dec_len, max_len, dtype=torch.bool, device=device)
    
    dia_model.model.eval()
    
    with torch.no_grad():
        for step in range(max_tokens):
            # Get current sequence (only up to current_len) - identical to working script
            current_tokens = generated_tokens[:, :current_len, :]  # (1, current_len, 18)
            
            # Create target positions (identical to working script)
            tgt_positions = torch.arange(current_len, device=device).unsqueeze(0)  # (1, current_len)
            
            # Use pre-computed attention masks (slice to current length) - identical to working script
            dec_self_attn_mask = causal_mask_full[:current_len, :current_len].unsqueeze(0)  # (1, current_len, current_len)
            dec_cross_attn_mask = cross_attn_mask_full[:, :current_len, :]  # (1, current_len, max_len)
            
            try:
                # Use autocast like working script (professional optimization)
                with torch.autocast("cuda", dtype=torch.float32):
                    logits = dia_model.model(
                        src_BxS=src_tokens,
                        tgt_BxTxC=current_tokens,
                        src_positions=src_positions,
                        tgt_positions=tgt_positions,
                        enc_self_attn_mask=enc_self_attn_mask,
                        dec_self_attn_mask=dec_self_attn_mask,
                        dec_cross_attn_mask=dec_cross_attn_mask,
                        enable_dropout=False
                    )
                
                # Get next token logits: (1, current_len, 18, vocab_size) - identical to working script
                next_token_logits = logits[:, -1, :, :]  # (1, 18, vocab_size)
                
                # Use exact working implementation from stlohrey repo
                temperature = 1.2
                top_p = 0.95
                top_k = 45
                audio_eos_value = config.data.audio_eos_value
                
                # Flatten logits for proper sampling: (1, 18, vocab_size) -> (18, vocab_size)
                flat_logits_CxV = next_token_logits.squeeze(0)  # Remove batch dimension
                
                # Constrain logits to valid range like working repo
                flat_logits_CxV[:, 1027:] = -torch.inf  # Only allow tokens 0-1026
                
                # Apply working sampling (exact copy from stlohrey repo)
                next_tokens = _sample_next_token(
                    flat_logits_CxV,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    audio_eos_value=audio_eos_value,
                )
                
                # Reshape back to (1, 18)
                next_tokens = next_tokens.unsqueeze(0)
                
                # Add next tokens to pre-allocated tensor - identical to working script
                generated_tokens[:, current_len, :] = next_tokens
                current_len += 1
                
            except Exception as e:
                logger.error(f"Generation failed at step {step}: {e}")
                break
    
    # Extract only the generated tokens (up to current_len) and transpose: (1, current_len, 18) -> (18, current_len)
    # Identical to working script
    generated_codes = generated_tokens[:, :current_len, :].squeeze(0).transpose(0, 1)
    
    return generated_codes


class PreprocessedFolderDataset(Dataset):
    """Dataset for preprocessed audio files saved as .pkl files."""
    
    def __init__(self, preprocessed_dir: Path, target_length: int = 2584):
        """
        Args:
            preprocessed_dir: Path to directory containing .pkl files from folder_preprocess.py
            target_length: Target length for random chunks (default 2584 tokens = ~30 seconds)
        """
        self.target_length = target_length
        self.preprocessed_dir = Path(preprocessed_dir)
        
        if not self.preprocessed_dir.exists():
            raise FileNotFoundError(f"Preprocessed directory not found: {self.preprocessed_dir}")
        
        # Check for metadata file
        metadata_file = self.preprocessed_dir / "preprocessing_metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Preprocessing metadata not found: {metadata_file}. Run folder_preprocess.py first.")
        
        # Load metadata
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        print(f"Loading preprocessed files from: {self.preprocessed_dir}")
        print(f"Original base folder: {self.metadata.get('base_folder', 'unknown')}")
        print(f"Preprocessing completed: {self.metadata.get('preprocessing_complete', False)}")
        
        # Load all preprocessed files
        self._load_preprocessed_files()
        
    def _load_preprocessed_files(self):
        """Load all preprocessed .pkl files."""
        # Find all .pkl files
        pkl_files = list(self.preprocessed_dir.glob('*.pkl'))
        
        # Exclude metadata file
        pkl_files = [f for f in pkl_files if f.name != 'preprocessing_metadata.pkl']
        
        if not pkl_files:
            raise FileNotFoundError(f"No .pkl files found in {self.preprocessed_dir}")
        
        print(f"Found {len(pkl_files)} preprocessed files")
        
        self.samples = []
        
        for pkl_file in tqdm(pkl_files, desc="Loading preprocessed files"):
            try:
                with open(pkl_file, 'rb') as f:
                    preprocessed_data = pickle.load(f)
                
                # Extract relevant data
                sample = {
                    'audio_file': Path(preprocessed_data['audio_file']),
                    'prompt_text': preprocessed_data['prompt_text'],
                    'encoded_audio': preprocessed_data['encoded_audio']
                }
                
                self.samples.append(sample)
                
            except Exception as e:
                logger.error(f"Error loading {pkl_file}: {e}, skipping")
                continue
        
        if not self.samples:
            raise RuntimeError("No valid preprocessed files found")
        
        print(f"Successfully loaded {len(self.samples)} preprocessed audio-prompt pairs")
        
        # Print some stats
        total_tokens = sum(sample['encoded_audio'].shape[0] for sample in self.samples)
        avg_tokens = total_tokens / len(self.samples)
        print(f"Average tokens per sample: {avg_tokens:.1f}")
        print(f"Total audio tokens: {total_tokens:,}")
        
    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, idx: int):
        """Return audio sample for training with random chunking."""
        # Get actual sample
        sample = self.samples[idx]
        full_audio = sample['encoded_audio']  # (full_length, 18)
        
        # Get target chunk size from metadata or use config default
        target_length = getattr(self, 'target_length', 2584)  # Default 2584 tokens (~30 seconds)
        
        # Always use first chunk (no random chunking)
        if full_audio.shape[0] > target_length:
            # Always start from beginning of the song
            audio_chunk = full_audio[:target_length]
        else:
            # If shorter than target, pad with audio_pad_value
            audio_chunk = full_audio
            if audio_chunk.shape[0] < target_length:
                pad_length = target_length - audio_chunk.shape[0] 
                pad_value = 1025  # audio_pad_value from config
                padding = torch.full((pad_length, 18), pad_value, dtype=audio_chunk.dtype)
                audio_chunk = torch.cat([audio_chunk, padding], dim=0)
        
        # Return format expected by collate_fn: (text, encoded_audio, waveforms)
        return sample['prompt_text'], audio_chunk, None


def test_stereo_generation(model, config, dac_model, prompt_text, output_dir, step, device, train_name="simple", run_demos_dir=None, config_path=None):
    """Test stereo generation quality during training using demo.py functions directly."""
    try:
        # Use provided run-specific demos directory or create step-based one
        if run_demos_dir is not None:
            demos_dir = run_demos_dir
        else:
            demos_dir = Path("./demos") / f"{train_name}_{step}"
        demos_dir.mkdir(parents=True, exist_ok=True)
        
        # Save current model checkpoint for demo.py to use
        checkpoint_path = demos_dir / f"temp_checkpoint_step{step}.pth"
        
        # Handle DataParallel wrapped model
        if hasattr(model, 'module'):
            unwrapped_model = model.module
        else:
            unwrapped_model = model
        
        logger.info(f"Running demo inference directly using training model at step {step}...")
        
        # Use the training model directly (no checkpoint saving/loading needed)
        try:
            # Create a Dia wrapper using the training model and DAC model
            from dia.model import Dia
            
            # Create temporary Dia instance with the training model
            temp_dia = Dia(config, device)
            temp_dia.model = unwrapped_model  # Use the actual training model
            temp_dia.dac_model = dac_model   # Use the loaded DAC model
            
            # Generate audio using the training model directly
            with torch.inference_mode():
                output_audio_np = temp_dia.generate(
                    prompt_text,
                    cfg_scale=3.0,
                    temperature=1.3,
                    top_p=0.95,
                    use_cfg_filter=True,
                    cfg_filter_top_k=35,
                    use_torch_compile=False,
                    audio_prompt_path=None,
                )
            
            if output_audio_np is not None:
                # Save test audio
                test_file = demos_dir / f"{train_name}_step{step}.wav"
                
                logger.info(f"Generated audio type: {type(output_audio_np)}, shape: {output_audio_np.shape if hasattr(output_audio_np, 'shape') else 'no shape'}")
                
                # Handle stereo output
                if output_audio_np.ndim == 1:
                    # Mono
                    sf.write(test_file, output_audio_np, 44100)
                    audio_shape = output_audio_np.shape
                    duration = len(output_audio_np) / 44100
                else:
                    # Stereo - transpose to (samples, channels)
                    sf.write(test_file, output_audio_np.T, 44100)
                    audio_shape = output_audio_np.T.shape
                    duration = output_audio_np.shape[1] / 44100
                
                logger.info(f"Generated demo audio: {test_file} (shape: {audio_shape}, duration: {float(duration):.2f}s)")
                success = True
            else:
                logger.warning(f"Step {step}: Demo generation produced no output")
                success = False
                
            # Clean up temp reference
            del temp_dia
            
        except Exception as e:
            logger.error(f"Demo generation failed at step {step}: {e}")
            success = False
            
        return success
                
    except Exception as e:
        logger.error(f"Stereo generation test failed at step {step}: {e}")
        return False
    finally:
        model.train()


def manage_checkpoints(output_dir, max_keep=2):
    """Keep only the last N checkpoints to save disk space."""
    checkpoint_pattern = output_dir / "ckpt_step*.pth"
    checkpoints = glob.glob(str(checkpoint_pattern))
    
    if len(checkpoints) > max_keep:
        def extract_step_number(filename):
            import re
            match = re.search(r'ckpt_step(\d+)\.pth', filename)
            return int(match.group(1)) if match else 0
        
        checkpoints_sorted = sorted(checkpoints, key=extract_step_number)
        
        for old_ckpt in checkpoints_sorted[:-max_keep]:
            try:
                os.remove(old_ckpt)
                logger.info(f"Removed old checkpoint: {old_ckpt}")
            except Exception as e:
                logger.warning(f"Failed to remove {old_ckpt}: {e}")


def setup_loaders_ddp(dataset, dia_cfg, train_cfg, device, rank=0, world_size=1, use_ddp=False):
    """Setup data loaders with optional DistributedSampler for DDP."""
    from torch.utils.data import DataLoader
    from dia.finetune import collate_fn
    
    # Define a proper collate function that can be pickled
    def collate_batch(batch):
        result = collate_fn(batch, dia_cfg, device, train_cfg)
        
        # DEBUG: Log what the training data looks like (only on rank 0)
        if rank == 0 and hasattr(collate_batch, '_debug_count'):
            collate_batch._debug_count += 1
        elif rank == 0:
            collate_batch._debug_count = 1
            
        if rank == 0 and collate_batch._debug_count <= 3:  # Log first 3 batches
            tgt = result['tgt_tokens'][0]  # First item in batch
            logger.info(f"DEBUG BATCH {collate_batch._debug_count}: Target shape: {tgt.shape}")
            logger.info(f"DEBUG BATCH {collate_batch._debug_count}: First 10 timesteps:")
            for t in range(min(10, tgt.shape[0])):
                tokens = tgt[t].tolist()
                bos_count = tokens.count(1026)
                eos_count = tokens.count(1024) 
                pad_count = tokens.count(1025)
                valid_count = sum(1 for x in tokens if 0 <= x <= 1023)
                logger.info(f"  t={t}: BOS={bos_count}, EOS={eos_count}, PAD={pad_count}, VALID={valid_count}, sample={tokens[:6]}")
        
        return result
    
    # Create sampler
    if use_ddp:
        sampler = DistributedSampler(
            dataset, 
            num_replicas=world_size, 
            rank=rank,
            shuffle=True,
            drop_last=True
        )
        shuffle = False  # DistributedSampler handles shuffling
    else:
        sampler = None
        shuffle = True
    
    # Create data loader
    train_loader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collate_batch,  # Add the collate function!
        num_workers=0,  # Disable multiprocessing to avoid pickle issues
        pin_memory=False,  # Disabled because collate_fn puts tensors on GPU
        drop_last=True,
        persistent_workers=False  # Disabled when num_workers=0
    )
    
    # Set steps_per_epoch attribute for tqdm
    if use_ddp:
        steps_per_epoch = len(sampler) // train_cfg.batch_size  # Samples per GPU ÷ batch_size
    else:
        steps_per_epoch = len(dataset) // train_cfg.batch_size
    
    train_loader.steps_per_epoch = steps_per_epoch
    
    # No validation loader for this training setup
    val_loader = None
    
    return train_loader, val_loader


def train_with_generation_tests(model, dia_cfg, dac_model, dataset, train_cfg, prompt_text, demo_every=0, rank=0, world_size=1, use_ddp=False, resume_from=None, save_last=False, save_every=0, config_path=None):
    """Training loop with periodic generation testing."""
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    
    # Only rank 0 creates directories to avoid race conditions
    if rank == 0:
        train_cfg.output_dir.mkdir(parents=True, exist_ok=True)
        (train_cfg.runs_dir / train_cfg.run_name).mkdir(parents=True, exist_ok=True)
        
        # Create run-specific demos directory
        import time
        timestamp = int(time.time())
        run_demos_dir = Path("./demos") / f"{train_cfg.run_name}_{timestamp}"
        run_demos_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_demos_dir = None
    
    # Synchronize all processes
    if use_ddp:
        dist.barrier()
    
    # Move model to device BEFORE creating optimizer
    model = model.to(device)
    
    # Wrap model with DDP if using distributed training
    if use_ddp:
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    
    train_loader, val_loader = setup_loaders_ddp(dataset, dia_cfg, train_cfg, device, rank, world_size, use_ddp)
    opt, sched = setup_optimizer_and_scheduler(model, train_loader, train_cfg)
    
    # Load checkpoint if resuming
    start_epoch = 0
    global_step = 0
    last_loss = 0.0  # Initialize loss for checkpoint saving
    best_loss = float('inf')  # Track best loss for smart checkpoint saving
    best_checkpoint_path = None
    
    if resume_from and Path(resume_from).exists():
        start_epoch, global_step = load_checkpoint(model, opt, sched, resume_from, device, rank)
        # Load last loss from checkpoint if available
        checkpoint = torch.load(resume_from, map_location=device)
        last_loss = checkpoint.get('loss', 0.0)
        best_loss = last_loss  # Initialize best loss with resumed loss
        if rank == 0:
            print(f"Resuming from epoch {start_epoch + 1}, global step {global_step}, last loss: {last_loss}")
    elif resume_from and rank == 0:
        print(f"WARNING: Checkpoint file not found: {resume_from}, starting from scratch")
    
    # Only rank 0 creates tensorboard writer
    writer = None
    if rank == 0:
        writer = SummaryWriter(train_cfg.runs_dir / train_cfg.run_name)
    
    model.train()
    
    # Use the corrected steps_per_epoch from setup_loaders_ddp
    steps_per_epoch = getattr(train_loader, 'steps_per_epoch', None)
    if steps_per_epoch is None:
        # Fallback calculation for DDP
        if use_ddp:
            # For DDP: samples per GPU = len(sampler), steps = samples / batch_size
            dataset_size = len(train_loader.sampler) if hasattr(train_loader, 'sampler') else len(train_loader.dataset)
            steps_per_epoch = dataset_size // train_cfg.batch_size
        else:
            # For single GPU: use len(train_loader) 
            try:
                steps_per_epoch = len(train_loader)
            except Exception:
                steps_per_epoch = None
    
    for epoch in range(start_epoch, train_cfg.epochs):
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
            
            # Training step
            loss = train_step(model, batch, dia_cfg, train_cfg, opt, sched, writer if rank == 0 else None, step, global_step)
            last_loss = loss  # Update last_loss for checkpoint saving
            
            # Track best loss across all ranks
            if loss < best_loss:
                best_loss = loss
            
            # Memory stats and progress update (only on rank 0)
            if rank == 0:
                cur_alloc = torch.cuda.memory_allocated()
                peak_alloc = torch.cuda.max_memory_allocated()
                cur_gb = cur_alloc / 1024**3
                peak_gb = peak_alloc / 1024**3
                
                loader_iter.set_postfix({
                    'loss': f"{loss:.4f}",
                    'VRAM (GB)': f"{cur_gb:.2f}/{peak_gb:.2f}"
                })
                torch.cuda.reset_peak_memory_stats()
            
            # Generation testing based on demo_every parameter (only on rank 0)
            if demo_every > 0 and global_step > 0 and global_step % demo_every == 0 and rank == 0:
                model.eval()
                
                # Check if using unconditional training
                if train_cfg.unconditional_frac >= 1.0:
                    logger.info(f"\n🎵 Generating unconditional demo audio at step {global_step}...")
                    # Generate 3 random samples to show unconditional diversity
                    success_count = 0
                    for i in range(3):
                        success = test_stereo_generation(
                            model, dia_cfg, dac_model, f"unconditional_sample_{i}", 
                            train_cfg.output_dir, f"{global_step}_sample{i}", device, 
                            train_name="unconditional", run_demos_dir=run_demos_dir, config_path=config_path
                        )
                        if success:
                            success_count += 1
                    
                    if success_count > 0:
                        logger.info(f"✅ Generated {success_count}/3 unconditional samples at step {global_step}")
                    else:
                        logger.warning(f"❌ All unconditional demo generation failed at step {global_step}")
                else:
                    logger.info(f"\n🎵 Generating conditional demo audio at step {global_step}...")
                    success = test_stereo_generation(
                        model, dia_cfg, dac_model, prompt_text, 
                        train_cfg.output_dir, global_step, device, 
                        train_name="folder_training", run_demos_dir=run_demos_dir, config_path=config_path
                    )
                    if success:
                        logger.info(f"✅ Demo generation successful at step {global_step}")
                    else:
                        logger.warning(f"❌ Demo generation failed at step {global_step}")
                
                model.train()
            
            # No intermediate checkpoints during batch loop
        
        # Smart checkpoint saving (only save on rank 0)
        if save_every > 0 and (epoch + 1) % save_every == 0 and rank == 0:
            print(f"\n💾 Saving checkpoint at epoch {epoch + 1}...")
            
            # Save new periodic checkpoint
            periodic_path = train_cfg.output_dir / f"ckpt_epoch{epoch + 1}.pth"
            checkpoint = {
                'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'scheduler_state_dict': sched.state_dict(),
                'epoch': epoch,
                'global_step': global_step,
                'loss': last_loss,
                'config': dia_cfg.model_dump(),
                'training_args': {
                    'learning_rate': train_cfg.learning_rate,
                    'batch_size': train_cfg.batch_size,
                    'epochs': train_cfg.epochs,
                    'num_audio_files': len(dataset.samples),
                    'preprocessed_dir': str(dataset.preprocessed_dir),
                    'original_base_folder': dataset.metadata.get('base_folder', 'unknown')
                },
                'timestamp': time.time(),
                'periodic_save': True
            }
            torch.save(checkpoint, periodic_path)
            logger.info(f"Saved periodic checkpoint: {periodic_path}")
            
            # Check if this is the best checkpoint so far
            is_best = last_loss < best_loss
            if is_best:
                best_loss = last_loss
                # Remove old best checkpoint if it exists
                if best_checkpoint_path and os.path.exists(best_checkpoint_path):
                    try:
                        os.remove(best_checkpoint_path)
                        logger.info(f"Removed old best checkpoint: {best_checkpoint_path}")
                    except Exception as e:
                        logger.warning(f"Failed to remove old best checkpoint: {e}")
                
                # Save new best checkpoint
                best_checkpoint_path = train_cfg.output_dir / f"ckpt_best_epoch{epoch + 1}_loss{last_loss:.4f}.pth"
                torch.save(checkpoint, best_checkpoint_path)
                logger.info(f"🌟 New best checkpoint saved: {best_checkpoint_path}")
            
            # Smart cleanup: Keep 2 most recent + best checkpoint
            import glob
            import re
            periodic_pattern = train_cfg.output_dir / "ckpt_epoch*.pth"
            existing_checkpoints = glob.glob(str(periodic_pattern))
            
            # Filter out best checkpoint from cleanup
            checkpoints_to_consider = []
            for ckpt in existing_checkpoints:
                if not ckpt.startswith(str(train_cfg.output_dir / "ckpt_best_")):
                    checkpoints_to_consider.append(ckpt)
            
            # Extract epoch numbers and sort by epoch (newest first)
            def extract_epoch_number(filename):
                match = re.search(r'ckpt_epoch(\d+)\.pth', filename)
                return int(match.group(1)) if match else 0
                
            checkpoints_to_consider.sort(key=extract_epoch_number, reverse=True)
            
            # Remove checkpoints beyond the 2 most recent (but never the best)
            checkpoints_to_remove = checkpoints_to_consider[2:]  # Keep 2 most recent
            for old_ckpt in checkpoints_to_remove:
                try:
                    os.remove(old_ckpt)
                    logger.info(f"Removed old checkpoint: {old_ckpt}")
                except Exception as e:
                    logger.warning(f"Failed to remove {old_ckpt}: {e}")
    
    # Final checkpoint - Professional format (only save on rank 0)
    if save_last and rank == 0:
        ckpt_final_path = train_cfg.output_dir / f"ckpt_final_epoch{train_cfg.epochs}.pth"
        final_checkpoint = {
            'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'scheduler_state_dict': sched.state_dict(),
            'epoch': train_cfg.epochs,
            'global_step': global_step,
            'loss': last_loss,
            'config': dia_cfg.model_dump(),
            'training_args': {
                'learning_rate': train_cfg.learning_rate,
                'batch_size': train_cfg.batch_size,
                'epochs': train_cfg.epochs,
                'num_audio_files': len(dataset.samples),
                'preprocessed_dir': str(dataset.preprocessed_dir),
                'original_base_folder': dataset.metadata.get('base_folder', 'unknown')
            },
            'timestamp': time.time(),
            'training_complete': True
        }
        torch.save(final_checkpoint, ckpt_final_path)
        logger.info(f"Saved final professional checkpoint: {ckpt_final_path}")
    
    # Synchronize all processes before ending
    if use_ddp:
        dist.barrier()
    
    # Return the demos directory for printing
    return run_demos_dir if rank == 0 else None


def setup_ddp(rank: int, world_size: int, port: str = "29500"):
    """Initialize the distributed environment."""
    # Force IPv4 and avoid IPv6 issues
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = port
    os.environ['NCCL_SOCKET_IFNAME'] = 'lo'  # Use loopback interface
    os.environ['GLOO_SOCKET_IFNAME'] = 'lo'
    
    # Initialize the process group with timeout
    dist.init_process_group(
        "nccl", 
        rank=rank, 
        world_size=world_size,
        timeout=datetime.timedelta(seconds=60)
    )
    
    # Set the device for this process
    torch.cuda.set_device(rank)
    
    # Setup signal handlers for clean shutdown
    def signal_handler(signum, frame):
        cleanup_ddp()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def cleanup_ddp():
    """Clean up the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device, rank=0):
    """Load checkpoint and return start epoch and global step."""
    if rank == 0:
        print(f"Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer and scheduler state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  # Commented out to allow new LR
    
    start_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
    global_step = checkpoint.get('global_step', 0)
    
    if rank == 0:
        print(f"Resuming from epoch {start_epoch}, global step {global_step}")
        print(f"Previous loss: {checkpoint.get('loss', 'unknown')}")
    
    return start_epoch, global_step


def get_training_config(epochs: int = 100, learning_rate: float = 1e-4, output_dir: str = "/nfs/turbo/smtd-hwdong/ocamp/simple_checkpoints", batch_size: int = 1, use_ddp: bool = False, dataset_size: int = None, world_size: int = 1, no_warmup: bool = False):
    """Get training configuration for simple training from scratch."""
    
    grad_accum_steps = 1
    
    warmup_steps = 0 if no_warmup else 2000
    
    run_suffix = f"_bs{batch_size}_ddp{world_size}" if use_ddp else f"_bs{batch_size}"
    
    return TrainConfig(
        epochs=epochs,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        eval_step=50,
        save_step=250,
        split_ratio=0.0,
        run_name=f"folder_overfit_{Path(output_dir).name}{run_suffix}",
        output_dir=Path(output_dir),
        seed=42
    )


def run_ddp_worker(rank: int, world_size: int, args):
    """Worker function for DDP training."""
    try:
        # Setup DDP
        setup_ddp(rank, world_size)
        
        device = torch.device(f"cuda:{rank}")
        
        # Load config
        config_path = Path(args.config)
        config = DiaConfig.load(config_path)
        
        if rank == 0:
            print(f"Loaded config from: {config_path}")
        
        # Load DAC model
        if rank == 0:
            print("Loading DAC model...")
        dac_model = dac.DAC.load(dac.utils.download()).to(device)
        
        # Check preprocessed directory exists
        preprocessed_dir = Path(args.preprocessed_dir)
        if not preprocessed_dir.exists() and rank == 0:
            print(f"ERROR: Preprocessed directory not found: {preprocessed_dir}")
            return 1
        
        # Create preprocessed audio dataset
        if rank == 0:
            print("Creating preprocessed audio dataset...")
        dataset = PreprocessedFolderDataset(preprocessed_dir=preprocessed_dir, target_length=config.data.audio_length)
        
        if rank == 0:
            print(f"Preprocessed audio dataset: {len(dataset.samples)} files ready for training")
        
        # Load model WITHOUT pretrained weights
        if rank == 0:
            print("Creating Dia model from scratch...")
        model = DiaModel(config)
        
        if rank == 0:
            print("Model initialized with random weights")
        
        # Get training config
        train_config = get_training_config(
            epochs=args.epochs,
            learning_rate=args.lr,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            use_ddp=True,
            dataset_size=len(dataset),
            world_size=world_size,
            no_warmup=args.no_warmup
        )
        
        if rank == 0:
            print(f"\nTraining configuration:")
            print(f"  Dataset size: {len(dataset)}")
            print(f"  Batch size: {train_config.batch_size}")
            print(f"  Epochs: {train_config.epochs}")
            print(f"  Learning rate: {train_config.learning_rate}")
            print(f"  Warmup steps: {train_config.warmup_steps}")
            print(f"  Audio length in config: {config.data.audio_length}")
            print(f"  World size (GPUs): {world_size}")
        
        # Create output directory (only rank 0)
        if rank == 0:
            train_config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize W&B (only rank 0)
        if rank == 0:
            avg_sequence_tokens = sum(sample['encoded_audio'].shape[0] for sample in dataset.samples) // len(dataset.samples)
            
            wandb.init(
                project="dia-stereo-foundation",
                name=f"folder-stereo-ddp{world_size}-lr{args.lr}-{len(dataset.samples)}files",
                config={
                    "learning_rate": args.lr,
                    "epochs": args.epochs,
                    "audio_length": config.data.audio_length,
                    "avg_sequence_tokens": avg_sequence_tokens,
                    "num_audio_files": len(dataset.samples),
                    "batch_size": train_config.batch_size,
                    "grad_accum_steps": train_config.grad_accum_steps,
                    "preprocessed_dir": args.preprocessed_dir,
                    "original_base_folder": dataset.metadata.get('base_folder', 'unknown'),
                    "training_type": "folder_from_scratch_ddp",
                    "world_size": world_size,
                    "preprocessed": True
                },
                sync_tensorboard=True
            )
        
        # Synchronize before training
        dist.barrier()
        
        # Start training
        if rank == 0:
            print(f"\nStarting DDP folder training FROM SCRATCH with {world_size} GPUs...")
        
        # Use first sample's prompt for generation testing
        test_prompt = dataset.samples[0]['prompt_text']
        demos_dir = train_with_generation_tests(
            model, config, dac_model, dataset, train_config, 
            test_prompt, args.demo_every, rank, world_size, use_ddp=True, resume_from=args.resume_from,
            save_last=args.save_last, save_every=args.save_every, config_path=args.config
        )
        
        if rank == 0:
            print(f"\nTraining completed!")
            if args.save_last:
                print(f"Final checkpoint saved to: {train_config.output_dir}/ckpt_final_epoch{train_config.epochs}.pth")
            else:
                print(f"Final checkpoint saving disabled (use --save_last to enable)")
            if demos_dir:
                print(f"Generated test audio files saved in: {demos_dir}")
            else:
                print(f"Demo generation was disabled (demo_every=0)")
        
    except Exception as e:
        print(f"Error in rank {rank}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup_ddp()


def main():
    parser = argparse.ArgumentParser(description="Training from scratch on preprocessed audio files with DDP support")
    parser.add_argument("preprocessed_dir", type=str, help="Path to directory containing preprocessed .pkl files from folder_preprocess.py")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="/nfs/turbo/smtd-hwdong/ocamp/simple_checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--config", type=str, default="dia/stereo_small_config.json", help="Path to model config")
    parser.add_argument("--use_ddp", action="store_true", help="Use DistributedDataParallel instead of single GPU")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per GPU")
    parser.add_argument("--world_size", type=int, default=None, help="Number of GPUs to use (auto-detected if not specified)")
    parser.add_argument("--demo_every", type=int, default=0, help="Generate demo every N steps (0 = disabled)")
    parser.add_argument("--port", type=str, default="29500", help="Master port for DDP communication")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--no_warmup", action="store_true", help="Disable learning rate warmup")
    parser.add_argument("--save_last", action="store_true", help="Save final checkpoint at end of training")
    parser.add_argument("--save_every", type=int, default=0, help="Save checkpoint every N epochs (0 = disabled)")
    
    args = parser.parse_args()
    
    # Determine world size
    if args.use_ddp:
        if args.world_size is None:
            world_size = torch.cuda.device_count()
        else:
            world_size = args.world_size
        
        if world_size < 2:
            print("WARNING: DDP requested but only 1 GPU available, falling back to single GPU training")
            args.use_ddp = False
            world_size = 1
    else:
        world_size = 1
    
    print(f"Preprocessed Dia model training FROM SCRATCH:")
    print(f"  Preprocessed directory: {args.preprocessed_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Batch size per GPU: {args.batch_size}")
    print(f"  Output: {args.output_dir}")
    print(f"  Demo generation: {'every ' + str(args.demo_every) + ' steps' if args.demo_every > 0 else 'disabled'}")
    print(f"  Resume from: {args.resume_from if args.resume_from else 'none (training from scratch)'}")
    if args.use_ddp:
        print(f"  DDP Mode: {world_size} GPUs")
        print(f"  Total effective batch size: {args.batch_size * world_size}")
    else:
        print(f"  Single GPU mode")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available, this script requires GPU")
        return 1
    
    # Set multiprocessing start method for DDP
    if args.use_ddp:
        mp.set_start_method('spawn', force=True)
        
        # Launch DDP training
        print(f"Launching DDP training with {world_size} processes...")
        os.environ['MASTER_PORT'] = args.port
        
        try:
            mp.spawn(
                run_ddp_worker,
                args=(world_size, args),
                nprocs=world_size,
                join=True
            )
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        except Exception as e:
            print(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            return 1
        
        return 0
    
    # Single GPU training (fallback)
    device = torch.device("cuda:0")
    print(f"Using single GPU: {device}")
    
    # Load config
    config_path = Path(args.config)
    config = DiaConfig.load(config_path)
    print(f"Loaded config from: {config_path}")
    
    # Load DAC model
    print("Loading DAC model...")
    dac_model = dac.DAC.load(dac.utils.download()).to(device)
    
    # Check preprocessed directory exists
    preprocessed_dir = Path(args.preprocessed_dir)
    if not preprocessed_dir.exists():
        print(f"ERROR: Preprocessed directory not found: {preprocessed_dir}")
        print(f"Run folder_preprocess.py first to create preprocessed files.")
        return 1
    
    # Create preprocessed audio dataset (no DAC model needed)
    print("Creating preprocessed audio dataset...")
    dataset = PreprocessedFolderDataset(
        preprocessed_dir=preprocessed_dir,
        target_length=config.data.audio_length
    )
    print(f"Preprocessed audio dataset: {len(dataset.samples)} files ready for training")
    
    # Load model WITHOUT pretrained weights
    print("Creating Dia model from scratch...")
    model = DiaModel(config)
    print("Model initialized with random weights - no pretrained loading!")
    
    # No DataParallel needed for single GPU mode
    
    # Get training config
    train_config = get_training_config(
        epochs=args.epochs,
        learning_rate=args.lr,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        use_ddp=False,
        dataset_size=len(dataset),
        no_warmup=args.no_warmup
    )
    
    print(f"\nTraining configuration:")
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Batch size: {train_config.batch_size}")
    print(f"  Epochs: {train_config.epochs}")
    print(f"  Learning rate: {train_config.learning_rate}")
    print(f"  Warmup steps: {train_config.warmup_steps}")
    print(f"  Audio length in config: {config.data.audio_length}")
    
    # Create output directory
    train_config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize W&B
    avg_sequence_tokens = sum(sample['encoded_audio'].shape[0] for sample in dataset.samples) // len(dataset.samples)
    
    wandb.init(
        project="dia-stereo-foundation",
        name=f"folder-stereo-lr{args.lr}-{len(dataset.samples)}files",
        config={
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "audio_length": config.data.audio_length,
            "avg_sequence_tokens": avg_sequence_tokens,
            "num_audio_files": len(dataset.samples),
            "batch_size": train_config.batch_size,
            "grad_accum_steps": train_config.grad_accum_steps,
            "preprocessed_dir": args.preprocessed_dir,
            "original_base_folder": dataset.metadata.get('base_folder', 'unknown'),
            "training_type": "folder_from_scratch",
            "preprocessed": True
        },
        sync_tensorboard=True
    )
    wandb.watch(model, log_freq=50)
    
    # Start training
    print(f"\nStarting folder training FROM SCRATCH...")
    # Use first sample's prompt for generation testing
    test_prompt = dataset.samples[0]['prompt_text']
    demos_dir = train_with_generation_tests(
        model, config, dac_model, dataset, train_config, 
        test_prompt, args.demo_every, rank=0, world_size=1, use_ddp=False, resume_from=args.resume_from,
        save_last=args.save_last, save_every=args.save_every, config_path=args.config
    )
    
    print(f"\nTraining completed!")
    if args.save_last:
        print(f"Final checkpoint saved to: {train_config.output_dir}/ckpt_final_epoch{train_config.epochs}.pth")
    else:
        print(f"Final checkpoint saving disabled (use --save_last to enable)")
    if demos_dir:
        print(f"Generated test audio files saved in: {demos_dir}")
    else:
        print(f"Demo generation was disabled (demo_every=0)")
    
    return 0


if __name__ == "__main__":
    exit(main())