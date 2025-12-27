#!/usr/bin/env python3
"""
Convert parquet files to .pt files (run ON THE TPU).

IMPORTANT: Writes to LOCAL disk first, then copies to GCS with gsutil.
gcsfuse is unreliable for high-throughput writes and corrupts files!

Usage (on TPU):
    python scripts/convert_parquet_to_pt.py \
        --input_dir /home/olivercamp/dataset_bucket/data \
        --output_bucket gs://rubin-dia-dataset
"""

import argparse
import json
import os
import subprocess
import shutil
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm


# Write to local disk, NOT directly to bucket mount (gcsfuse corrupts files)
LOCAL_TEMP_DIR = Path("/tmp/converted_pt_dataset")


def deserialize_tensor(tensor_data) -> torch.Tensor:
    """Convert tensor data to torch tensor."""
    import pickle
    import io
    
    if isinstance(tensor_data, (list, tuple)):
        return torch.tensor(tensor_data, dtype=torch.long)
    
    if isinstance(tensor_data, np.ndarray):
        return torch.from_numpy(tensor_data).to(torch.long)
    
    if isinstance(tensor_data, bytes):
        # Try torch.load
        try:
            tensor = torch.load(io.BytesIO(tensor_data), map_location='cpu', weights_only=False)
            if isinstance(tensor, torch.Tensor):
                return tensor.to(torch.long)
        except:
            pass
        
        # Try numpy
        try:
            arr = np.load(io.BytesIO(tensor_data), allow_pickle=True)
            return torch.from_numpy(arr).to(torch.long)
        except:
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
        except:
            pass
        
        # Raw frombuffer
        for dtype in [np.int16, np.int32, np.int64]:
            if len(tensor_data) % dtype().itemsize == 0:
                try:
                    arr = np.frombuffer(tensor_data, dtype=dtype)
                    return torch.from_numpy(arr.copy()).to(torch.long)
                except:
                    pass
    
    if isinstance(tensor_data, dict):
        if 'bytes' in tensor_data:
            return deserialize_tensor(tensor_data['bytes'])
        elif 'array' in tensor_data:
            return torch.tensor(tensor_data['array'], dtype=torch.long)
    
    raise ValueError(f"Could not deserialize: {type(tensor_data)}")


def process_parquet(parquet_path: Path, output_dir: Path, start_idx: int):
    """Process one parquet file and save as .pt files to LOCAL disk."""
    import pyarrow.parquet as pq
    
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    
    results = []
    for i, row in df.iterrows():
        idx = start_idx + i
        try:
            tensor_data = row.get('tensor')
            if tensor_data is None:
                continue
            
            tensor = deserialize_tensor(tensor_data)
            
            # Ensure [T, C] shape
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(-1)
            
            # Save to LOCAL disk (not gcsfuse mount!)
            filename = f"sample_{idx:06d}.pt"
            output_path = output_dir / "encoded_audio" / filename
            torch.save(tensor.to(torch.int16), output_path)
            
            # Verify the file was written correctly
            try:
                _ = torch.load(output_path, map_location='cpu', weights_only=False)
            except Exception as e:
                print(f"Verification failed for {filename}: {e}")
                output_path.unlink(missing_ok=True)
                continue
            
            results.append({
                "filename": filename,
                "text": row.get('tags', row.get('tag', '')) or '',
                "channels": tensor.shape[1] if tensor.ndim > 1 else 1,
                "length": tensor.shape[0],
            })
        except Exception as e:
            print(f"Failed sample {idx}: {e}")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True,
                       help="Directory with parquet files (can be gcsfuse mount)")
    parser.add_argument("--output_bucket", type=str, required=True,
                       help="GCS bucket (e.g., gs://rubin-dia-dataset)")
    args = parser.parse_args()
    
    # Find all parquet files
    parquet_files = sorted(args.input_dir.glob("*.parquet"))
    if not parquet_files:
        # Try subdirectory
        parquet_files = sorted(args.input_dir.glob("*/*.parquet"))
    
    if not parquet_files:
        print(f"No parquet files found in {args.input_dir}")
        return 1
    
    print(f"Found {len(parquet_files)} parquet files")
    
    # Create LOCAL output directory (not on gcsfuse!)
    local_output = LOCAL_TEMP_DIR
    if local_output.exists():
        print(f"Cleaning existing temp dir: {local_output}")
        shutil.rmtree(local_output)
    (local_output / "encoded_audio").mkdir(parents=True, exist_ok=True)
    
    print(f"Writing to LOCAL disk: {local_output}")
    print("(gcsfuse corrupts .pt files, so we write locally first)\n")
    
    # Process parquet files
    all_metadata = {}
    global_idx = 0
    
    for parquet_path in tqdm(parquet_files, desc="Processing parquets"):
        results = process_parquet(parquet_path, local_output, global_idx)
        
        for r in results:
            all_metadata[r["filename"]] = {
                "text": r["text"],
                "channels": r["channels"],
                "length": r["length"],
            }
        
        global_idx += len(results)
    
    # Save metadata
    with open(local_output / "metadata.json", 'w') as f:
        json.dump(all_metadata, f, indent=2)
    
    print(f"\nConversion complete: {len(all_metadata)} samples")
    print(f"Local files at: {local_output}")
    
    # Copy to GCS with gsutil (reliable, unlike gcsfuse writes)
    print(f"\nUploading to GCS: {args.output_bucket}")
    print("This uses gsutil which is reliable (unlike gcsfuse writes)...")
    
    cmd = [
        "gsutil", "-m", "cp", "-r",
        str(local_output / "encoded_audio"),
        str(local_output / "metadata.json"),
        args.output_bucket + "/"
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    print(f"\n✓ Done! Dataset uploaded to {args.output_bucket}")
    print(f"  - {args.output_bucket}/encoded_audio/*.pt")
    print(f"  - {args.output_bucket}/metadata.json")
    
    # Cleanup local temp
    print(f"\nCleaning up local temp dir...")
    shutil.rmtree(local_output)
    print("✓ Cleanup complete")


if __name__ == "__main__":
    main()

