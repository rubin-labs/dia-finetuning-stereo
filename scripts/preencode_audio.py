import argparse
import json
import re
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

import dac
from dia.dac_utils import encode_waveform_stereo


def load_prompt(prompts_dir: Path, stem: str) -> tuple[str, bool]:
    prompt_file = prompts_dir / f"{stem}_prompt.txt"
    default_prompt = stem.replace("_", " ").replace("-", " ")
    if prompt_file.exists():
        try:
            text = prompt_file.read_text(encoding="utf-8").strip()
        except Exception:
            return default_prompt, False
        has_vocals_tag = bool(re.search(r"\bvocals\b", text, flags=re.IGNORECASE))
        return text, has_vocals_tag
    return default_prompt, False


def main():
    parser = argparse.ArgumentParser(description="Pre-encode audio files into DAC code tokens (.pt)")
    parser.add_argument("--audio_dir", type=Path, required=True, help="Directory containing input audio files")
    parser.add_argument("--out_dir", type=Path, required=True, help="Output directory to write encoded_audio/ and metadata.json")
    parser.add_argument("--prompts_dir", type=Path, default=None, help="Directory with <stem>_prompt.txt files; defaults to ../audio_prompts")
    parser.add_argument("--sr", type=int, default=44100, help="Target sample rate for encoding (default: 44100)")
    # Always encode full tracks; no cropping performed in this script
    parser.add_argument("--device", type=str, default=None, help="Force device: cuda or cpu (default: auto)")
    parser.add_argument("--dtype", type=str, default="int16", choices=["int16", "long"], help="Saved tensor dtype for codes")
    parser.add_argument(
        "--skip_vocals",
        action="store_true",
        help="Skip encoding tracks when the corresponding prompt file contains the 'vocals' tag",
    )
    args = parser.parse_args()

    audio_dir: Path = args.audio_dir
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    encoded_dir: Path = out_dir / "encoded_audio"
    encoded_dir.mkdir(exist_ok=True)

    prompts_dir = args.prompts_dir or (audio_dir.parent / "audio_prompts")
    prompts_dir = Path(prompts_dir)

    device = (
        torch.device("cuda")
        if (args.device == "cuda" or (args.device is None and torch.cuda.is_available()))
        else torch.device("cpu")
    )

    # Load DAC
    print(f"Loading DAC model on {device}...")
    model = dac.DAC.load(dac.utils.download()).eval().to(device)

    # Find audio files
    exts = [".wav", ".mp3", ".flac", ".m4a", ".ogg"]
    audio_files = [p for ext in exts for p in audio_dir.glob(f"*{ext}")]
    if not audio_files:
        raise SystemExit(f"No audio files found in {audio_dir}")

    metadata = {}
    sr_target = int(args.sr)

    for wav_path in tqdm(audio_files, desc="Encoding"):
        stem = wav_path.stem
        prompt_text, has_vocals_tag = load_prompt(prompts_dir, stem)
        if args.skip_vocals and has_vocals_tag:
            print(f"Skipping {wav_path.name} because its prompt contains the 'vocals' tag.")
            continue
        try:
            waveform, sr_in = torchaudio.load(wav_path)
            # Keep original channels (support stereo)
            if sr_in != sr_target:
                waveform = torchaudio.functional.resample(waveform, sr_in, sr_target)
            # Do not crop: always encode the full track

            # Encode stereo as 9+9 codebooks by processing each channel separately
            with torch.no_grad():
                codes = encode_waveform_stereo(
                    waveform,
                    dac_model=model,
                    sample_rate=sr_target,
                    device=device,
                    dtype=None,  # convert below based on args.dtype
                )

            if args.dtype == "int16":
                save_tensor = codes.to(torch.int16).cpu()
            else:
                save_tensor = codes.to(torch.long).cpu()

            out_file = encoded_dir / f"{stem}.pt"
            torch.save(save_tensor, out_file)

            metadata[out_file.name] = {
                "text": prompt_text,
                "sr": sr_target,
                "channels": int(codes.shape[1]),  # expect 18
                "length": int(codes.shape[0]),
            }
        except Exception as e:
            print(f"Failed to encode {wav_path}: {e}")

    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(metadata)} items to {encoded_dir} and metadata.json")


if __name__ == "__main__":
    main()

