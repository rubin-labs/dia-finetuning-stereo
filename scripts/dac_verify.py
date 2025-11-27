#!/usr/bin/env python3
"""
Quick DAC reconstruction smoke test.

Loads a single audio file, runs DAC encode -> decode (per channel), writes the
reconstruction, and prints basic stats so you can validate the environment
without launching a full training job.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

import dac

logger = logging.getLogger("dac_verify")


def encode_decode_channel(dac_model, wav_1xS: torch.Tensor, sr: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a mono channel and immediately decode it back."""
    device = next(dac_model.parameters()).device
    audio_tensor = dac_model.preprocess(wav_1xS.unsqueeze(0), sr)
    if audio_tensor.dim() == 2:
        audio_tensor = audio_tensor.unsqueeze(1)
    if audio_tensor.shape[1] != 1:
        logger.warning("DAC preprocess returned %d channels; downmixing to mono", audio_tensor.shape[1])
        audio_tensor = audio_tensor[:, :1, :]
    audio_tensor = audio_tensor.to(device).contiguous()

    with torch.no_grad():
        _, codes, *_ = dac_model.encode(audio_tensor, n_quantizers=None)  # (1, 9, T)
        z, _, _ = dac_model.quantizer.from_codes(codes)
        recon = dac_model.decode(z)  # (1, 1, S)

    codes_st = codes.squeeze(0).transpose(0, 1).cpu()  # (T, 9)
    recon = recon.squeeze(0).squeeze(0).cpu()  # (S,)
    return codes_st, recon


def main():
    parser = argparse.ArgumentParser(description="Verify DAC encode/decode on a single file.")
    parser.add_argument("audio_file", type=Path, help="Path to an audio file (wav/mp3/flac/ogg/m4a)")
    parser.add_argument("--output", type=Path, default=Path("dac_verify_out.wav"), help="Where to write the reconstruction")
    parser.add_argument("--seconds", type=float, default=2.0, help="Max seconds to process (crop for speed)")
    parser.add_argument("--sr", type=int, default=44100, help="Target sample rate")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"], help="Force device (default: auto)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    device = (
        torch.device("cuda")
        if (args.device == "cuda" or (args.device is None and torch.cuda.is_available()))
        else torch.device("cpu")
    )
    logger.info("Loading DAC model on %s", device)
    dac_model = dac.DAC.load(dac.utils.download()).eval().to(device)

    logger.info("Loading audio: %s", args.audio_file)
    waveform, sr_in = torchaudio.load(args.audio_file)
    if waveform.shape[0] > 2:
        raise SystemExit(f"Only mono/stereo supported; got {waveform.shape[0]} channels.")

    if sr_in != args.sr:
        waveform = torchaudio.functional.resample(waveform, sr_in, args.sr)
    max_samples = int(args.seconds * args.sr)
    if waveform.shape[1] > max_samples:
        waveform = waveform[:, :max_samples]

    codes_all = []
    recon_channels = []
    for ch in range(waveform.shape[0]):
        codes, recon = encode_decode_channel(dac_model, waveform[ch : ch + 1, :], args.sr)
        codes_all.append(codes)
        recon_channels.append(recon)
        logger.info("Ch%d codes shape: %s, recon samples: %d", ch, tuple(codes.shape), recon.numel())

    # Stack reconstructions; if mono input, keep single channel
    recon_audio = torch.stack(recon_channels, dim=0)  # (C, S)
    recon_np = recon_audio.numpy().T  # (S, C)

    # Simple quality metrics vs original (crop to common length)
    min_len = min(waveform.shape[1], recon_audio.shape[1])
    orig = waveform[:, :min_len]
    recon_aligned = recon_audio[:, :min_len]
    mse = torch.mean((orig - recon_aligned) ** 2).item()
    l2 = torch.sqrt(torch.mean((orig - recon_aligned) ** 2)).item()
    mae = torch.mean(torch.abs(orig - recon_aligned)).item()
    logger.info("Reconstruction error (cropped to %d samples): mse=%.6f, rmse=%.6f, mae=%.6f", min_len, mse, l2, mae)

    sf.write(args.output, recon_np, args.sr)
    logger.info("Wrote reconstruction to %s", args.output)


if __name__ == "__main__":
    main()
