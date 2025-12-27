from pathlib import Path
from typing import Optional

import os
import hashlib
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
import numpy as np

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

        # Cache file list locally (dataset bucket is mounted read-only on TPU)
        # One cache per machine is fine; it avoids repeated expensive directory scans over gcsfuse.
        cache_root = Path(os.environ.get("DIA_CACHE_DIR", Path.home() / ".cache" / "dia"))
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_key = hashlib.sha1(str(self.preprocessed_dir.resolve()).encode("utf-8")).hexdigest()[:16]
        cache_file = cache_root / f"file_list_{cache_key}.json"
        loaded_from_cache = False
        
        # 1. Try loading from cache
        if cache_file.exists():
            try:
                print(f"[Dataset] Loading file list from cache: {cache_file}")
                with open(cache_file, 'r') as f:
                    rel_paths = json.load(f)
                    self.encoded_files = [self.preprocessed_dir / p for p in rel_paths]
                loaded_from_cache = True
            except Exception as e:
                print(f"[Dataset] Cache load failed ({e}), falling back to scan.")

        # 2. Scan if not cached
        if not loaded_from_cache:
            print(f"[Dataset] Scanning {target_dir} for .pt files (this may take a moment)...")
            for p in target_dir.glob("*.pt"):
                self.encoded_files.append(p)
            
            # 3. Save cache
            try:
                rel_paths = [str(p.relative_to(self.preprocessed_dir)) for p in self.encoded_files]
                with open(cache_file, 'w') as f:
                    json.dump(rel_paths, f)
                print(f"[Dataset] Saved file list cache to {cache_file}")
            except Exception as e:
                print(f"[Dataset] Warning: Failed to save cache: {e}")
        
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


class HuggingFacePreEncodedDataset(torch.utils.data.IterableDataset):
    """Dataset for loading pre-encoded DAC audio from HuggingFace datasets.
    
    Uses STREAMING mode to avoid disk space issues (no Arrow cache needed).
    
    Expected dataset columns:
        - tags: str (text prompt/tags)
        - tensor: array/tensor (pre-encoded DAC audio, shape [T, C])
        - id: str (optional, unique identifier)
        - filename: str (optional, original filename)
    """
    
    def __init__(
        self, 
        dataset_name: str,
        config: DiaConfig,
        split: str = "train",
        use_sliding_window: bool = True,
        dataset_length: int = 31333,  # Known dataset size for oliver-camp/open-source-dataset
        num_workers: int = 1,
        worker_id: int = 0,
    ):
        """
        Args:
            dataset_name: HuggingFace dataset name (e.g., 'oliver-camp/open-source-dataset')
            config: DiaConfig for audio length and other parameters
            split: Dataset split to use (default: 'train')
            use_sliding_window: If True, random crop; else fixed crop from start
            dataset_length: Known size of the dataset (for progress tracking)
            num_workers: Total number of data loading workers (for sharding)
            worker_id: This worker's ID (for sharding)
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install the 'datasets' library: pip install datasets")
        
        self.dataset_name = dataset_name
        self.split = split
        self.config = config
        self.use_sliding_window = use_sliding_window
        self._length = dataset_length
        self.num_workers = num_workers
        self.worker_id = worker_id
        
        # Don't load here - load lazily in __iter__ to support DataLoader workers
        self._dataset = None
        
        logger.info(f"HuggingFace streaming dataset configured: {dataset_name} (length={dataset_length})")
    
    def _load_dataset(self):
        """Lazily load the streaming dataset."""
        if self._dataset is None:
            from datasets import load_dataset
            logger.info(f"Loading streaming dataset: {self.dataset_name}")
            self._dataset = load_dataset(self.dataset_name, split=self.split, streaming=True)
        return self._dataset
    
    def _deserialize_bytes(self, raw_bytes: bytes) -> np.ndarray:
        """Try multiple strategies to deserialize bytes into a numpy array."""
        import pickle
        import io
        
        byte_len = len(raw_bytes)
        
        # Strategy 1: PyTorch tensor (pickle format, starts with \x80)
        if raw_bytes[:1] == b'\x80':
            try:
                tensor = torch.load(io.BytesIO(raw_bytes), map_location='cpu', weights_only=False)
                if isinstance(tensor, torch.Tensor):
                    return tensor.numpy()
                return np.array(tensor)
            except Exception:
                pass
        
        # Strategy 2: NumPy .npy format (starts with \x93NUMPY)
        if raw_bytes[:6] == b'\x93NUMPY':
            try:
                return np.load(io.BytesIO(raw_bytes), allow_pickle=True)
            except Exception:
                pass
        
        # Strategy 3: PyArrow IPC format (for arrays serialized by HuggingFace)
        try:
            import pyarrow as pa
            reader = pa.ipc.open_stream(raw_bytes)
            table = reader.read_all()
            # Get the first column as numpy
            arr = table.column(0).to_numpy()
            return arr
        except Exception:
            pass
        
        # Strategy 4: Try pickle deserialization
        try:
            obj = pickle.loads(raw_bytes)
            if isinstance(obj, torch.Tensor):
                return obj.numpy()
            elif isinstance(obj, np.ndarray):
                return obj
            elif isinstance(obj, (list, tuple)):
                return np.array(obj, dtype=np.int64)
            return np.array(obj)
        except Exception:
            pass
        
        # Strategy 5: Try numpy frombuffer with various dtypes
        # Check byte alignment for different dtypes
        for dtype, size in [(np.int16, 2), (np.int32, 4), (np.int64, 8), (np.float32, 4)]:
            if byte_len % size == 0:
                try:
                    arr = np.frombuffer(raw_bytes, dtype=dtype)
                    # Sanity check: should have reasonable values for audio codes
                    if dtype in (np.int16, np.int32, np.int64):
                        # DAC codes should be in range [0, 1023] or similar
                        if arr.min() >= -2000 and arr.max() <= 2000:
                            return arr
                except Exception:
                    pass
        
        # Strategy 6: Raw bytes as uint8 (fallback)
        return np.frombuffer(raw_bytes, dtype=np.uint8)
    
    def __len__(self) -> int:
        """Return known dataset length for progress tracking."""
        return self._length
    
    def _process_item(self, item):
        """Process a single item from the streaming dataset."""
        import pickle
        import base64
        import io
        
        # Extract text prompt from 'tags' column
        text_prompt = item.get('tags', item.get('tag', '')) or ''
        
        # Extract pre-encoded tensor
        tensor_data = item.get('tensor')
        if tensor_data is None:
            return None
        
        # Debug: log detailed type info on first few items
        if not hasattr(self, '_debug_count'):
            self._debug_count = 0
        if self._debug_count < 5:
            debug_info = f"[DEBUG] tensor_data type: {type(tensor_data).__name__}"
            if isinstance(tensor_data, bytes):
                # Show first bytes as hex for debugging
                debug_info += f", len={len(tensor_data)}, first_20_hex={tensor_data[:20].hex()}"
            elif isinstance(tensor_data, (list, tuple)):
                debug_info += f", len={len(tensor_data)}"
                if len(tensor_data) > 0:
                    debug_info += f", first_elem_type={type(tensor_data[0]).__name__}"
                    if isinstance(tensor_data[0], (list, tuple)) and len(tensor_data[0]) > 0:
                        debug_info += f", nested_type={type(tensor_data[0][0]).__name__}"
            elif isinstance(tensor_data, np.ndarray):
                debug_info += f", dtype={tensor_data.dtype}, shape={tensor_data.shape}"
            elif isinstance(tensor_data, dict):
                debug_info += f", keys={list(tensor_data.keys())}"
            logger.info(debug_info)
            self._debug_count += 1
        
        # Handle various formats
        try:
            # Case 1: Already a list/nested list (HuggingFace parquet often does this)
            if isinstance(tensor_data, (list, tuple)):
                tensor_data = np.array(tensor_data, dtype=np.int64)
            
            # Case 2: Raw bytes - try multiple deserialization strategies
            elif isinstance(tensor_data, bytes):
                tensor_data = self._deserialize_bytes(tensor_data)
            
            # Case 3: Numpy array with bytes/object dtype
            elif isinstance(tensor_data, np.ndarray):
                if tensor_data.dtype.kind == 'S':  # Byte string
                    raw_bytes = tensor_data.tobytes()
                    tensor_data = self._deserialize_bytes(raw_bytes)
                elif tensor_data.dtype == np.dtype('O'):  # Object array
                    if tensor_data.ndim == 0:
                        inner = tensor_data.item()
                        if isinstance(inner, bytes):
                            tensor_data = self._deserialize_bytes(inner)
                        elif isinstance(inner, (list, tuple)):
                            tensor_data = np.array(inner, dtype=np.int64)
                        else:
                            tensor_data = inner
                    else:
                        tensor_data = np.stack([np.array(x, dtype=np.int64) for x in tensor_data])
                # Already numeric - pass through
            
            # Case 4: String (base64 encoded)
            elif isinstance(tensor_data, str):
                decoded = base64.b64decode(tensor_data)
                tensor_data = self._deserialize_bytes(decoded)
            
            # Case 5: Dict with 'bytes' key (HuggingFace format)
            elif isinstance(tensor_data, dict):
                if 'bytes' in tensor_data:
                    tensor_data = self._deserialize_bytes(tensor_data['bytes'])
                elif 'array' in tensor_data:
                    tensor_data = np.array(tensor_data['array'], dtype=np.int64)
                else:
                    raise ValueError(f"Unknown dict format with keys: {list(tensor_data.keys())}")
                
        except Exception as e:
            logger.warning(f"Failed to deserialize tensor: {type(tensor_data).__name__}, {e}")
            return None
        
        # Convert to torch tensor
        try:
            if isinstance(tensor_data, torch.Tensor):
                encoded = tensor_data.to(torch.long)
            elif isinstance(tensor_data, np.ndarray):
                if tensor_data.dtype.kind not in ('i', 'u', 'f', 'b'):  # int, uint, float, bool
                    logger.warning(f"Non-numeric array dtype after deserialize: {tensor_data.dtype}")
                    return None
                encoded = torch.from_numpy(tensor_data).to(torch.long)
            elif isinstance(tensor_data, list):
                encoded = torch.tensor(tensor_data, dtype=torch.long)
            else:
                encoded = torch.tensor(np.array(tensor_data), dtype=torch.long)
        except Exception as e:
            logger.warning(f"Failed to convert to tensor: {e}")
            return None
        
        # Ensure shape is [T, C] (time, channels)
        if encoded.ndim == 1:
            encoded = encoded.unsqueeze(-1)
        elif encoded.ndim > 2:
            encoded = encoded.squeeze()
            if encoded.ndim > 2:
                return None
        
        # Cropping
        target_length = self.config.data.audio_length
        if encoded.shape[0] > target_length:
            if self.use_sliding_window:
                max_start = encoded.shape[0] - target_length
                start = random.randint(0, max_start)
                encoded = encoded[start : start + target_length]
            else:
                encoded = encoded[:target_length]
        
        return text_prompt, encoded, None
    
    def __iter__(self):
        """Iterate over the streaming dataset with worker sharding."""
        dataset = self._load_dataset()
        
        # Handle DataLoader worker sharding
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Running in a DataLoader worker process
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
        else:
            # Running in main process
            num_workers = self.num_workers
            worker_id = self.worker_id
        
        # Iterate with sharding: each worker gets every Nth sample
        for idx, item in enumerate(dataset):
            # Simple round-robin sharding across workers
            if idx % num_workers != worker_id:
                continue
            
            result = self._process_item(item)
            if result is not None:
                yield result
