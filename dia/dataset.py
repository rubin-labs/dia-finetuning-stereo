from pathlib import Path

import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset

import dac
from .config import DiaConfig
import random
import json
import logging

logger = logging.getLogger(__name__)


class MusicDataset(Dataset):
    """Load from audio folder and audio_prompts folder structure."""
    def __init__(self, audio_folder: Path, config: DiaConfig, dac_model: dac.DAC, use_sliding_window: bool = True, ignore_missing_prompts: bool = True, skip_tags: list = None):
        print("WARNING: MusicDataset performs on-the-fly DAC encoding which is very slow and blocks training. Consider using scripts/preencode_audio.py and --preencoded_dir for much faster training.")
        self.audio_folder = Path(audio_folder)
        self.prompts_folder = self.audio_folder.parent / "audio_prompts"
        self.config = config
        self.dac_model = dac_model
        self.use_sliding_window = use_sliding_window
        
        # Find all audio files
        audio_extensions = ['.mp3', '.wav', '.flac', '.m4a', '.ogg']
        self.audio_files = []
        
        for ext in audio_extensions:
            self.audio_files.extend(list(self.audio_folder.glob(f"*{ext}")))
        
        # Filter to only include files that have corresponding prompt files
        self.valid_files = []
        skipped_missing = 0
        skipped_tags = 0

        for audio_file in self.audio_files:
            prompt_file = self.prompts_folder / f"{audio_file.stem}_prompt.txt"
            
            if not prompt_file.exists():
                if ignore_missing_prompts:
                    skipped_missing += 1
                    continue
                else:
                    raise FileNotFoundError(f"Missing prompt file: {prompt_file}")
            
            # Check for skip tags if provided
            if skip_tags:
                try:
                    with open(prompt_file, 'r', encoding='utf-8') as f:
                        text = f.read().strip().lower()
                    
                    should_skip = False
                    for tag in skip_tags:
                        if tag.strip().lower() in text:
                            should_skip = True
                            break
                    
                    if should_skip:
                        skipped_tags += 1
                        continue
                except Exception as e:
                    print(f"Warning: error reading prompt {prompt_file}: {e}")
                    skipped_missing += 1
                    continue

            self.valid_files.append(audio_file)
        
        if not self.valid_files:
            raise ValueError(f"No valid audio-prompt pairs found in {self.audio_folder}. Skipped {skipped_missing} missing, {skipped_tags} by tags.")
        
        print(f"Found {len(self.valid_files)} audio-prompt pairs. Skipped: {skipped_missing} missing prompts, {skipped_tags} filtered by tags.")

    def __len__(self) -> int:
        return len(self.valid_files)
    
    def _encode_mono_channel(self, wav_1xS: torch.Tensor) -> torch.Tensor:
        """Encode a single mono channel with DAC.
        
        Args:
            wav_1xS: Audio tensor of shape (1, samples)
            
        Returns:
            Encoded tensor of shape (T, 9)
        """
        device = next(self.dac_model.parameters()).device
        audio_tensor = self.dac_model.preprocess(
            wav_1xS.unsqueeze(0),  # -> (1, 1, S)
            44100
        ).to(device)
        _, enc, *_ = self.dac_model.encode(audio_tensor, n_quantizers=None)  # (1, 9, T)
        enc = enc.squeeze(0).transpose(0, 1).to(torch.long)  # (T, 9)
        return enc

    def __getitem__(self, idx: int):
        audio_file = self.valid_files[idx]
        prompt_file = self.prompts_folder / f"{audio_file.stem}_prompt.txt"
        
        # Load text prompt
        with open(prompt_file, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        # Load and process audio
        waveform, sr = torchaudio.load(audio_file)
        
        # Resample to 44.1kHz
        if sr != 44100:
            waveform = torchaudio.functional.resample(waveform, sr, 44100)
        
        # Crop waveform before encoding for efficiency and to get target length
        # - use_sliding_window=True: random crop (data augmentation + DAC jitter regularization)
        # - use_sliding_window=False: fixed crop from start (deterministic for debugging/overfitting)
        target_length_tokens = self.config.data.audio_length
        target_length_samples = int(target_length_tokens * 512)  # DAC hop_length = 512
        total_samples = waveform.shape[1]
        
        if total_samples > target_length_samples:
            if self.use_sliding_window:
                # Random crop for data augmentation + DAC edge regularization
                max_start = total_samples - target_length_samples
                start_sample = random.randint(0, max_start)
            else:
                # Fixed crop from start for deterministic training
                start_sample = 0
            waveform = waveform[:, start_sample:start_sample + target_length_samples]
        
        # Encode with DAC: process mono per channel and build stereo codes (9->18)
        with torch.no_grad():
            num_channels = waveform.shape[0]
            if num_channels > 2:
                raise ValueError(f"Audio file {audio_file} has {num_channels} channels. Only mono (1) or stereo (2) are supported.")
            elif num_channels == 2:
                left = waveform[0:1, :]
                right = waveform[1:2, :]
                enc_L = self._encode_mono_channel(left)
                enc_R = self._encode_mono_channel(right)
            else:
                # Duplicate mono to both channels
                mono = waveform[0:1, :]
                enc_L = self._encode_mono_channel(mono)
                enc_R = enc_L.clone()

            encoded = torch.cat([enc_L, enc_R], dim=1)  # (T, 18)

        # Keep waveform (with batch dim) for potential logging; not used in training logic
        waveform = waveform.unsqueeze(0)
        
        return text, encoded, waveform









class PreEncodedDACDataset(Dataset):
    """Dataset for pre-encoded DAC files (stereo .pt files) with prompts."""
    
    def __init__(self, preprocessed_dir: Path, config: DiaConfig, use_sliding_window: bool = True):
        """
        Args:
            preprocessed_dir: Path to directory containing encoded_audio/ and metadata.json
            config: DiaConfig for audio length and other parameters
            use_sliding_window: If True, load full sequences and crop in collate_fn
        """
        self.preprocessed_dir = Path(preprocessed_dir)
        self.config = config
        self.use_sliding_window = use_sliding_window
        
        # Load metadata
        metadata_file = self.preprocessed_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
        
        # Find all .pt files in encoded_audio directory
        encoded_dir = self.preprocessed_dir / "encoded_audio"
        if not encoded_dir.exists():
            raise FileNotFoundError(f"encoded_audio directory not found: {encoded_dir}")
        
        self.encoded_files = list(encoded_dir.glob("*.pt"))
        if not self.encoded_files:
            raise FileNotFoundError(f"No .pt files found in {encoded_dir}")
        
        # For now, assume prompts are stored in metadata or use filename as prompt
        print(f"Found {len(self.encoded_files)} pre-encoded DAC files")
        
    def __len__(self) -> int:
        return len(self.encoded_files)
    
    def __getitem__(self, idx: int):
        """Return audio sample for training."""
        encoded_file = self.encoded_files[idx]
        
        # Load pre-encoded DAC tensor
        encoded = torch.load(encoded_file, map_location='cpu')  # Shape: (T, 18) for stereo or (T, 9) for mono
        
        # Cropping behavior:
        # - use_sliding_window=True: return full sequence, let collate_fn do random cropping
        # - use_sliding_window=False: fixed crop from start here (deterministic)
        # Note: Do NOT pad here — let collate_fn handle padding so it can track
        # the true sequence length for proper loss masking
        target_length = self.config.data.audio_length
        if not self.use_sliding_window:
            # Fixed crop from start for deterministic training
            if encoded.shape[0] > target_length:
                encoded = encoded[:target_length]
            # If shorter than target_length, return as-is. collate_fn will:
            # 1. Record the true length in seq_lens
            # 2. Pad to batch_max
            # 3. Use seq_lens to mask out PAD tokens from loss
        # If using sliding window, keep the full sequence - collate_fn does random cropping
        
        # Use filename as prompt (full filename, not just first few words)
        filename = encoded_file.stem
        # Check if metadata has a prompt for this file
        if encoded_file.name in self.metadata and 'text' in self.metadata[encoded_file.name]:
            text_prompt = self.metadata[encoded_file.name]['text']
        else:
            # Try finding a prompt file
            # 1. Same dir, .txt
            prompt_path = encoded_file.with_suffix('.txt')
            prompt_path_alt = self.preprocessed_dir / "prompts" / f"{encoded_file.stem}.txt"
            prompt_path_alt2 = self.preprocessed_dir / "prompts" / f"{encoded_file.stem}_prompt.txt"
            
            if prompt_path.exists():
                with open(prompt_path, 'r', encoding='utf-8') as f:
                    text_prompt = f.read().strip()
            elif prompt_path_alt.exists():
                with open(prompt_path_alt, 'r', encoding='utf-8') as f:
                    text_prompt = f.read().strip()
            elif prompt_path_alt2.exists():
                with open(prompt_path_alt2, 'r', encoding='utf-8') as f:
                    text_prompt = f.read().strip()
            else:
                # Fallback to empty string (unconditional training) if no prompt found
                # We do NOT fallback to filename to avoid polluting the model with bad prompts
                text_prompt = ""
        
        # Ensure integer token dtype expected by training
        encoded = encoded.to(torch.long)

        # Return format expected by collate_fn: (text, encoded_audio, waveforms)
        return text_prompt, encoded, None  # No waveform since we have encoded data
