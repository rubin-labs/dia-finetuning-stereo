from pathlib import Path

import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset

import dac
from .config import DiaConfig
import random
import json


class MusicDataset(Dataset):
    """Load from audio folder and audio_prompts folder structure."""
    def __init__(self, audio_folder: Path, config: DiaConfig, dac_model: dac.DAC, use_sliding_window: bool = True):
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
        for audio_file in self.audio_files:
            prompt_file = self.prompts_folder / f"{audio_file.stem}_prompt.txt"
            if prompt_file.exists():
                self.valid_files.append(audio_file)
        
        if not self.valid_files:
            raise ValueError(f"No valid audio-prompt pairs found in {self.audio_folder}")
        
        print(f"Found {len(self.valid_files)} audio-prompt pairs")

    def __len__(self) -> int:
        return len(self.valid_files)

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
        
        if not self.use_sliding_window:
            # Original behavior: crop to target length here
            # DAC compression ratio is ~86 tokens per second
            # Convert audio_length (tokens) to samples: tokens * (44100 samples/sec) / (86 tokens/sec)
            target_length_tokens = self.config.data.audio_length
            target_length_samples = int(target_length_tokens * 44100 / 86)
            total_samples = waveform.shape[1]
            
            if total_samples > target_length_samples:
                # Random start position for files longer than target length
                max_start = total_samples - target_length_samples
                start_sample = random.randint(0, max_start)
                waveform = waveform[:, start_sample:start_sample + target_length_samples]
        # If using sliding window, keep the full waveform - cropping happens in collate_fn
        
        # Encode with DAC: process mono per channel and build stereo codes (9->18)
        with torch.no_grad():
            device = next(self.dac_model.parameters()).device

            def encode_mono(wav_1xS: torch.Tensor) -> torch.Tensor:
                # wav_1xS: shape (1, samples)
                audio_tensor = self.dac_model.preprocess(
                    wav_1xS.unsqueeze(0),  # -> (1, 1, S)
                    44100
                ).to(device)
                _, enc, *_ = self.dac_model.encode(audio_tensor, n_quantizers=None)  # (1, 9, T)
                enc = enc.squeeze(0).transpose(0, 1).to(torch.long)  # (T, 9)
                return enc

            if waveform.shape[0] >= 2:
                left = waveform[0:1, :]
                right = waveform[1:2, :]
                enc_L = encode_mono(left)
                enc_R = encode_mono(right)
            else:
                # Duplicate mono to both channels
                mono = waveform[0:1, :]
                enc_L = encode_mono(mono)
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
        encoded_stereo = torch.load(encoded_file, map_location='cpu')  # Shape: (T, 18) for stereo
        
        # Keep stereo codes as-is (expect shape (T, 18) for stereo)
        encoded_mono = encoded_stereo
        
        if not self.use_sliding_window:
            # Original behavior: crop to target length here
            target_length = self.config.data.audio_length
            if encoded_mono.shape[0] > target_length:
                # Random crop for training variety
                max_start = encoded_mono.shape[0] - target_length
                start_idx = random.randint(0, max_start)
                encoded_mono = encoded_mono[start_idx:start_idx + target_length]
            elif encoded_mono.shape[0] < target_length:
                # Pad if too short
                pad_length = target_length - encoded_mono.shape[0]
                pad_value = self.config.data.audio_pad_value
                padding = torch.full((pad_length, encoded_mono.shape[1]), pad_value, dtype=encoded_mono.dtype)
                encoded_mono = torch.cat([encoded_mono, padding], dim=0)
        # If using sliding window, keep the full sequence - cropping happens in collate_fn
        
        # Use filename as prompt (full filename, not just first few words)
        filename = encoded_file.stem
        # Check if metadata has a prompt for this file
        if encoded_file.name in self.metadata and 'text' in self.metadata[encoded_file.name]:
            text_prompt = self.metadata[encoded_file.name]['text']
        else:
            # Use full filename as prompt, replacing underscores/hyphens with spaces
            text_prompt = filename.replace('_', ' ').replace('-', ' ')
        
        # Ensure integer token dtype expected by training
        encoded_mono = encoded_mono.to(torch.long)

        # Return format expected by collate_fn: (text, encoded_audio, waveforms)
        return text_prompt, encoded_mono, None  # No waveform since we have encoded data
