#!/usr/bin/env python3
"""
FAST: Download HuggingFace parquet files and bulk upload to GCS.

This is 100x faster than one-by-one processing:
1. Download all parquet files locally (parallel)
2. Bulk upload entire folder to GCS with gsutil -m (parallel)
3. Training script reads parquets directly

Usage:
    python scripts/hf_to_gcs_fast.py --bucket gs://rubin-dia-dataset
"""

import argparse
import subprocess
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_dataset", default="oliver-camp/open-source-dataset")
    parser.add_argument("--bucket", required=True, help="gs://bucket-name")
    parser.add_argument("--local_dir", default="/tmp/hf_dataset", help="Local download dir")
    args = parser.parse_args()
    
    bucket = args.bucket.rstrip('/')
    if not bucket.startswith('gs://'):
        bucket = f'gs://{bucket}'
    
    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Download dataset using huggingface-cli (FAST, parallel)
    print("="*60)
    print("STEP 1: Downloading parquet files from HuggingFace...")
    print("="*60)
    
    # Use huggingface_hub to download
    from huggingface_hub import snapshot_download
    
    downloaded_path = snapshot_download(
        repo_id=args.hf_dataset,
        repo_type="dataset",
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded to: {downloaded_path}")
    
    # Step 2: Bulk upload to GCS (FAST, parallel with -m flag)
    print("\n" + "="*60)
    print("STEP 2: Bulk uploading to GCS (parallel)...")
    print("="*60)
    
    # Use gsutil -m for parallel upload
    cmd = [
        'gsutil', '-m', 'cp', '-r',
        f'{downloaded_path}/*',
        f'{bucket}/'
    ]
    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n" + "="*60)
        print("SUCCESS!")
        print("="*60)
        print(f"Dataset uploaded to: {bucket}")
        print(f"\nThe bucket now contains the raw parquet files.")
        print(f"Update your training script to load directly from parquet.")
    else:
        print(f"Upload failed with code {result.returncode}")
        return 1
    
    # Step 3: Show what's in the bucket
    print("\n" + "="*60)
    print("Bucket contents:")
    print("="*60)
    subprocess.run(['gsutil', 'ls', '-l', f'{bucket}/'])
    
    return 0


if __name__ == "__main__":
    exit(main())

