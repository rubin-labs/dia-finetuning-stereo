#!/usr/bin/env python3
"""
Test script to verify the HuggingFace dataset format and tensor structure.

Run with:
    python scripts/test_hf_dataset.py

This will load a few samples from your HuggingFace dataset and print their structure.
"""

import sys
from pathlib import Path

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    print("=" * 60)
    print("Testing HuggingFace Dataset: oliver-camp/open-source-dataset")
    print("=" * 60)
    
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' library not installed.")
        print("Install with: pip install datasets")
        return
    
    # Use STREAMING mode to avoid downloading the entire 22GB dataset
    print("\n[1/4] Loading dataset in STREAMING mode...")
    dataset = load_dataset(
        "oliver-camp/open-source-dataset", 
        split="train",
        streaming=True,  # Stream to avoid downloading all parquet files
    )
    
    print(f"✓ Streaming dataset loaded")
    print(f"✓ Features: {dataset.features}")
    
    # Check first sample (streaming requires iteration)
    print("\n[2/4] Examining first sample...")
    sample = next(iter(dataset))
    print(f"  Columns: {list(sample.keys())}")
    
    for key, value in sample.items():
        if key == 'tensor':
            print(f"  {key}: type={type(value)}")
            if hasattr(value, 'shape'):
                print(f"       shape={value.shape}, dtype={value.dtype}")
            elif hasattr(value, '__len__'):
                print(f"       len={len(value)}")
                if len(value) > 0:
                    first_elem = value[0]
                    print(f"       first element type={type(first_elem)}")
                    if hasattr(first_elem, '__len__'):
                        print(f"       first element len={len(first_elem)}")
        else:
            print(f"  {key}: {repr(value)[:100]}...")
    
    # Try to convert tensor to torch
    print("\n[3/4] Converting tensor to torch...")
    import torch
    import numpy as np
    
    tensor_data = sample['tensor']
    
    # Try different conversion methods
    converted = None
    if isinstance(tensor_data, torch.Tensor):
        converted = tensor_data
        print("  Already a torch.Tensor")
    elif isinstance(tensor_data, np.ndarray):
        converted = torch.from_numpy(tensor_data)
        print("  Converted from numpy.ndarray")
    elif isinstance(tensor_data, list):
        converted = torch.tensor(tensor_data)
        print("  Converted from list")
    elif isinstance(tensor_data, bytes):
        # Could be raw bytes - try to decode as numpy
        import io
        try:
            arr = np.load(io.BytesIO(tensor_data), allow_pickle=True)
            converted = torch.from_numpy(arr)
            print("  Converted from serialized numpy bytes")
        except:
            print("  WARNING: Could not decode bytes")
    elif hasattr(tensor_data, 'numpy'):
        # HuggingFace Tensor object
        converted = torch.from_numpy(tensor_data.numpy())
        print("  Converted from HF Tensor object")
    else:
        # Last resort - try direct conversion
        try:
            converted = torch.tensor(np.array(tensor_data))
            print(f"  Converted via np.array (original type: {type(tensor_data)})")
        except Exception as e:
            print(f"  ERROR converting tensor: {e}")
    
    if converted is not None:
        print(f"  ✓ Torch tensor shape: {converted.shape}")
        print(f"  ✓ Torch tensor dtype: {converted.dtype}")
        print(f"  ✓ Min/Max values: {converted.min().item()}, {converted.max().item()}")
        
        # Check if shape matches expected DAC format
        expected_channels = 18  # Stereo DAC (9 channels * 2)
        if converted.ndim == 2:
            T, C = converted.shape
            if C == expected_channels:
                print(f"  ✓ Shape matches expected DAC stereo format: [T={T}, C={C}]")
            elif T == expected_channels:
                print(f"  ⚠ Tensor might be transposed: [C={T}, T={C}]")
                print(f"    Consider transposing in dataloader")
            else:
                print(f"  ⚠ Unexpected channel count: {converted.shape}")
                print(f"    Expected C={expected_channels} for stereo DAC")
        else:
            print(f"  ⚠ Unexpected tensor dimensions: {converted.ndim}")
    
    # Summary
    print("\n[4/4] Summary and recommendations...")
    print("=" * 60)
    
    if converted is not None and converted.ndim == 2:
        T, C = converted.shape
        print(f"✓ Dataset tensor format verified!")
        print(f"  - Shape: [{T}, {C}] (time steps, channels)")
        print(f"  - dtype: {converted.dtype}")
        print(f"  - Value range: [{converted.min().item()}, {converted.max().item()}]")
        
        if C == 18:
            print("\n✓ COMPATIBLE: Tensor has 18 channels (stereo DAC format)")
            print("  Ready for training with HuggingFacePreEncodedDataset!")
        elif C == 9:
            print("\n⚠ MONO FORMAT: Tensor has 9 channels (mono DAC format)")
            print("  May need to duplicate channels for stereo training.")
        else:
            print(f"\n⚠ UNEXPECTED: {C} channels (expected 9 mono or 18 stereo)")
        
        print("\n" + "=" * 60)
        print("TO USE FOR TRAINING:")
        print("=" * 60)
        print("""
# Option 1: Run on TPU with HuggingFace dataset
# Edit run_dia_training.sh to use:
#   --hf_dataset oliver-camp/open-source-dataset
#   --hf_cache_dir /path/to/cache  # Important for large datasets!

# Option 2: Download first, then train from local
# This downloads ~22GB once, then training is faster:
from datasets import load_dataset
ds = load_dataset("oliver-camp/open-source-dataset", cache_dir="/path/to/cache")

# The cache directory can be reused across training runs.
""")
    else:
        print("⚠ Could not fully verify dataset format")
        print("  Check the tensor structure manually")

if __name__ == "__main__":
    main()

