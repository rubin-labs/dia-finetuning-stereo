#!/usr/bin/env python3
"""
Rebuild metadata.json for pre-encoded .pt files by searching multiple prompt directories.

Usage:
    python scripts/rebuild_metadata.py \
        --preencoded_dir /path/to/preencoded_dir \
        --prompts_dirs /path/to/audio_prompts1 /path/to/audio_prompts2 \
        [--dry_run]
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import torch
from tqdm import tqdm


def find_prompt(stem: str, prompts_dirs: list[Path]) -> tuple[str, Path | None]:
    """
    Search for a prompt file matching the stem in multiple directories.
    
    Tries patterns:
      1. {stem}_prompt.txt
      2. {stem}.txt
    
    Returns:
        (prompt_text, found_path) or ("", None) if not found
    """
    patterns = [
        f"{stem}_prompt.txt",
        f"{stem}.txt",
    ]
    
    for prompts_dir in prompts_dirs:
        for pattern in patterns:
            prompt_file = prompts_dir / pattern
            if prompt_file.exists():
                try:
                    text = prompt_file.read_text(encoding='utf-8').strip()
                    return text, prompt_file
                except Exception as e:
                    print(f"Warning: Could not read {prompt_file}: {e}")
    
    return "", None


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild metadata.json for pre-encoded DAC files"
    )
    parser.add_argument(
        "--preencoded_dir",
        type=Path,
        required=True,
        help="Path to directory containing encoded_audio/ folder"
    )
    parser.add_argument(
        "--prompts_dirs",
        type=Path,
        nargs='+',
        required=True,
        help="One or more directories to search for prompt files"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only report what would be done, don't write metadata.json"
    )
    parser.add_argument(
        "--skip_load",
        action="store_true",
        help="Skip loading .pt files to get shape (faster but less metadata)"
    )
    args = parser.parse_args()

    preencoded_dir = args.preencoded_dir
    prompts_dirs = args.prompts_dirs
    
    # Validate paths
    encoded_dir = preencoded_dir / "encoded_audio"
    if not encoded_dir.exists():
        raise FileNotFoundError(f"encoded_audio directory not found: {encoded_dir}")
    
    for pd in prompts_dirs:
        if not pd.exists():
            print(f"Warning: Prompts directory does not exist: {pd}")
    
    # Find all .pt files
    pt_files = list(encoded_dir.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {encoded_dir}")
    
    print(f"Found {len(pt_files)} .pt files in {encoded_dir}")
    print(f"Searching for prompts in: {[str(p) for p in prompts_dirs]}")
    
    # Build metadata
    metadata = {}
    stats = defaultdict(int)
    missing_prompts = []
    
    for pt_file in tqdm(pt_files, desc="Processing"):
        stem = pt_file.stem
        prompt_text, found_path = find_prompt(stem, prompts_dirs)
        
        if found_path:
            stats['found'] += 1
            stats[f'from_{found_path.parent.name}'] += 1
        else:
            stats['missing'] += 1
            missing_prompts.append(stem)
        
        # Get tensor shape if not skipping
        if not args.skip_load:
            try:
                codes = torch.load(pt_file, map_location='cpu', weights_only=True)
                length = int(codes.shape[0])
                channels = int(codes.shape[1])
            except Exception as e:
                print(f"Warning: Could not load {pt_file}: {e}")
                length = 0
                channels = 18  # assume stereo
        else:
            length = 0
            channels = 18
        
        metadata[pt_file.name] = {
            "text": prompt_text,
            "sr": 44100,
            "channels": channels,
            "length": length,
        }
    
    # Print stats
    print("\n" + "=" * 60)
    print("STATISTICS")
    print("=" * 60)
    print(f"Total .pt files:     {len(pt_files)}")
    print(f"Prompts found:       {stats['found']}")
    print(f"Prompts missing:     {stats['missing']}")
    
    print("\nPrompts found per directory:")
    for key, count in stats.items():
        if key.startswith('from_'):
            dir_name = key[5:]
            print(f"  {dir_name}: {count}")
    
    if missing_prompts:
        print(f"\nMissing prompts for {len(missing_prompts)} files:")
        # Show first 10
        for stem in missing_prompts[:10]:
            print(f"  - {stem}")
        if len(missing_prompts) > 10:
            print(f"  ... and {len(missing_prompts) - 10} more")
    
    # Write or dry run
    output_path = preencoded_dir / "metadata.json"
    
    if args.dry_run:
        print(f"\n[DRY RUN] Would write metadata.json to: {output_path}")
        print(f"[DRY RUN] Sample entries:")
        sample_items = list(metadata.items())[:3]
        for name, data in sample_items:
            print(f"  {name}:")
            print(f"    text: {data['text'][:80]}..." if len(data['text']) > 80 else f"    text: {data['text']}")
            print(f"    length: {data['length']}, channels: {data['channels']}")
    else:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Wrote metadata.json to: {output_path}")
        print(f"  Contains {len(metadata)} entries")


if __name__ == "__main__":
    main()

