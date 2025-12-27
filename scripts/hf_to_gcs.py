#!/usr/bin/env python3
"""
Download HuggingFace dataset and upload to GCS bucket in PreEncodedDACDataset format.

Usage:
    python scripts/hf_to_gcs.py --bucket gs://your-bucket-name

This will:
1. Stream the HuggingFace dataset (no full download needed)
2. Convert each sample to .pt file format
3. Upload directly to GCS using gsutil
4. Create metadata.json

The resulting format is compatible with PreEncodedDACDataset.
"""

import argparse
import json
import io
import subprocess
import tempfile
from pathlib import Path

import torch
import numpy as np


def deserialize_tensor(tensor_data) -> torch.Tensor:
    """Convert HuggingFace tensor data to torch tensor."""
    import pickle
    
    # Case 1: Already a list (HuggingFace often returns arrays as nested lists)
    if isinstance(tensor_data, (list, tuple)):
        return torch.tensor(tensor_data, dtype=torch.long)
    
    # Case 2: Numpy array
    if isinstance(tensor_data, np.ndarray):
        return torch.from_numpy(tensor_data).to(torch.long)
    
    # Case 3: Bytes - try various formats
    if isinstance(tensor_data, bytes):
        # Try torch.load (pickle format)
        try:
            tensor = torch.load(io.BytesIO(tensor_data), map_location='cpu', weights_only=False)
            if isinstance(tensor, torch.Tensor):
                return tensor.to(torch.long)
        except Exception:
            pass
        
        # Try numpy .npy format
        try:
            arr = np.load(io.BytesIO(tensor_data), allow_pickle=True)
            return torch.from_numpy(arr).to(torch.long)
        except Exception:
            pass
        
        # Try pickle
        try:
            obj = pickle.loads(tensor_data)
            if isinstance(obj, torch.Tensor):
                return obj.to(torch.long)
            elif isinstance(obj, np.ndarray):
                return torch.from_numpy(obj).to(torch.long)
            elif isinstance(obj, list):
                return torch.tensor(obj, dtype=torch.long)
        except Exception:
            pass
        
        # Try raw frombuffer
        byte_len = len(tensor_data)
        for dtype in [np.int16, np.int32, np.int64]:
            if byte_len % dtype().itemsize == 0:
                try:
                    arr = np.frombuffer(tensor_data, dtype=dtype)
                    return torch.from_numpy(arr.copy()).to(torch.long)
                except Exception:
                    pass
    
    # Case 4: Dict format
    if isinstance(tensor_data, dict):
        if 'bytes' in tensor_data:
            return deserialize_tensor(tensor_data['bytes'])
        elif 'array' in tensor_data:
            return torch.tensor(tensor_data['array'], dtype=torch.long)
    
    raise ValueError(f"Could not deserialize tensor of type {type(tensor_data)}")


def upload_to_gcs(local_path: Path, gcs_path: str, timeout: int = 60):
    """Upload a file to GCS using gsutil with timeout."""
    try:
        result = subprocess.run(
            ['gsutil', '-q', 'cp', str(local_path), gcs_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        if result.returncode != 0:
            raise RuntimeError(f"gsutil upload failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"gsutil upload timed out after {timeout}s")


def main():
    parser = argparse.ArgumentParser(description="Download HuggingFace dataset and upload to GCS")
    parser.add_argument("--hf_dataset", type=str, default="oliver-camp/open-source-dataset",
                       help="HuggingFace dataset name")
    parser.add_argument("--bucket", type=str, required=True,
                       help="GCS bucket path (e.g., gs://my-bucket)")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum samples to process (for testing)")
    parser.add_argument("--skip", type=int, default=0,
                       help="Skip first N samples (to resume)")
    args = parser.parse_args()
    
    # Normalize bucket path
    bucket_path = args.bucket.rstrip('/')
    if not bucket_path.startswith('gs://'):
        bucket_path = f'gs://{bucket_path}'
    
    print(f"Loading HuggingFace dataset: {args.hf_dataset}")
    from datasets import load_dataset
    
    # Use streaming to avoid downloading entire dataset
    dataset = load_dataset(args.hf_dataset, split="train", streaming=True)
    
    print(f"Uploading to: {bucket_path}")
    
    metadata = {}
    processed = 0
    failed = 0
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        
        for idx, item in enumerate(dataset):
            if args.max_samples and idx >= args.max_samples:
                break
            
            # Skip already processed samples
            if idx < args.skip:
                if idx % 1000 == 0:
                    print(f"\rSkipping to {args.skip}... ({idx})", end='', flush=True)
                continue
            
            # Show progress every sample (flush immediately)
            print(f"\rProcessing sample {idx}...", end='', flush=True)
            
            try:
                # Get text/tags
                text = item.get('tags', item.get('tag', '')) or ''
                
                # Get tensor
                tensor_data = item.get('tensor')
                if tensor_data is None:
                    print(f"\nSample {idx}: missing tensor")
                    failed += 1
                    continue
                
                tensor = deserialize_tensor(tensor_data)
                
                # Ensure shape is [T, C]
                if tensor.ndim == 1:
                    tensor = tensor.unsqueeze(-1)
                
                # Save to temp file
                filename = f"sample_{idx:06d}.pt"
                temp_path = temp_dir / filename
                torch.save(tensor.to(torch.int16), temp_path)
                
                # Upload to GCS
                gcs_path = f"{bucket_path}/encoded_audio/{filename}"
                upload_to_gcs(temp_path, gcs_path)
                
                # Clean up temp file
                temp_path.unlink()
                
                metadata[filename] = {
                    "text": text,
                    "channels": tensor.shape[1] if tensor.ndim > 1 else 1,
                    "length": tensor.shape[0],
                }
                processed += 1
                
            except Exception as e:
                print(f"\nFailed sample {idx}: {e}")
                failed += 1
            
            if (idx + 1) % 100 == 0:
                print(f"\n>>> Processed {idx + 1} samples ({processed} success, {failed} failed)")
        
        # Upload metadata.json
        print("Uploading metadata.json...")
        metadata_path = temp_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        upload_to_gcs(metadata_path, f"{bucket_path}/metadata.json")
    
    print(f"\n{'='*50}")
    print(f"Done! Uploaded {processed} samples to {bucket_path}")
    print(f"Failed: {failed}")
    print(f"\nTo use with training, mount the bucket and set:")
    print(f"  --preencoded_dir /path/to/mounted/bucket")


if __name__ == "__main__":
    main()
