from pathlib import Path

import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset

import dac
from .config import DiaConfig
from .dac_utils import encode_waveform_stereo
import random
import json
import logging

logger = logging.getLogger(__name__)


class TestingDataset(Dataset):
    """Load from audio folder and audio_prompts folder structure."""
    def __init__(
        self,
        audio_folder: Path,
        config: DiaConfig,
        dac_model: dac.DAC,
        use_sliding_window: bool = True,
        skip_tags: list = None,
        allow_empty_prompts: bool = True,
    ):
        self.audio_folder = Path(audio_folder)
        self.prompts_folder = self.audio_folder.parent / "audio_prompts"
        self.config = config
        self.dac_model = dac_model
        self.use_sliding_window = use_sliding_window
        self.allow_empty_prompts = allow_empty_prompts
        
        # Find all audio files
        audio_extensions = ['.mp3', '.wav', '.flac', '.m4a', '.ogg']
        self.audio_files = []
        
        for ext in audio_extensions:
            self.audio_files.extend(list(self.audio_folder.glob(f"*{ext}")))
        
        # Filter to only include files that pass skip_tags filter
        self.valid_files = []
        skipped_tags = 0

        for audio_file in self.audio_files:
            prompt_file = self.prompts_folder / f"{audio_file.stem}_prompt.txt"
            
            # Check for skip tags if prompt exists
            if prompt_file.exists() and skip_tags:
                try:
                    with open(prompt_file, 'r', encoding='utf-8') as f:
                        text = f.read().strip().lower()
                    if any(tag.strip().lower() in text for tag in skip_tags):
                        skipped_tags += 1
                        continue
                except Exception as e:
                    print(f"Warning: error reading prompt {prompt_file}: {e}")
            
            self.valid_files.append(audio_file)
        
        if not self.valid_files:
            raise ValueError(f"No valid audio files found in {self.audio_folder}. Skipped {skipped_tags} by tags.")
        
        print(f"Found {len(self.valid_files)} audio files. Skipped: {skipped_tags} filtered by tags.")

    def __len__(self) -> int:
        return len(self.valid_files)
    
    def __getitem__(self, idx: int):
        audio_file = self.valid_files[idx]
        prompt_file = self.prompts_folder / f"{audio_file.stem}_prompt.txt"
        
        # Load text prompt
        if prompt_file.exists():
            with open(prompt_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
        else:
            if not self.allow_empty_prompts:
                raise FileNotFoundError(f"Missing prompt file: {prompt_file}")
            text = ""
        
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
            encoded = encode_waveform_stereo(
                waveform,
                dac_model=self.dac_model,
                sample_rate=44100,
                dtype=torch.long,
            )  # (T, 18)

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
        
        # Find all .pt files in encoded_audio directory or root
        encoded_dir = self.preprocessed_dir / "encoded_audio"
        
        self.encoded_files = []
        target_dir = encoded_dir if encoded_dir.exists() else self.preprocessed_dir
        
        print(f"[DATASET] Scanning {target_dir} for .pt files...", flush=True)
        count = 0
        for p in target_dir.glob("*.pt"):
            self.encoded_files.append(p)
            count += 1
            if count % 1000 == 0:
                print(f"[DATASET] Found {count} files...", flush=True)
        
        if not self.encoded_files:
            raise FileNotFoundError(f"No .pt files found in {self.preprocessed_dir} or {self.preprocessed_dir}/encoded_audio")
        
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
        # - use_sliding_window=True: random crop (data augmentation)
        # - use_sliding_window=False: fixed crop from start (deterministic)
        target_length = self.config.data.audio_length
        
        if encoded.shape[0] > target_length:
            if self.use_sliding_window:
                # Random crop
                max_start = encoded.shape[0] - target_length
                start = random.randint(0, max_start)
                encoded = encoded[start : start + target_length]
            else:
                # Fixed crop
                encoded = encoded[:target_length]
        
        # If shorter than target_length, return as-is. collate_fn will:
        # 1. Record the true length in seq_lens
        # 2. Pad to batch_max
        # 3. Use seq_lens to mask out PAD tokens from loss
        
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
