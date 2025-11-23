#!/usr/bin/env python3
"""Run Demucs to generate instrumental, drums, bass, and other stems (no vocals)."""

from __future__ import annotations

import argparse
import logging
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf


LOGGER = logging.getLogger("demucs_instrumental_split")
SUPPORTED_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Separate audio tracks with Demucs while omitting vocals. "
            "Creates an instrumental mix (drums + bass + other) and individual stems."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Audio files or directories containing audio to process.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("demucs_outputs"),
        help="Directory where the processed stems will be written (default: demucs_outputs).",
    )
    parser.add_argument(
        "--model",
        default="htdemucs",
        help="Name of the Demucs pretrained model to use (default: htdemucs).",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=list(SUPPORTED_EXTENSIONS),
        help="Audio file extensions to consider when scanning directories.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directories if they already exist.",
    )
    parser.add_argument(
        "--demucs-args",
        nargs=argparse.REMAINDER,
        default=[],
        help=(
            "Additional arguments forwarded to `python -m demucs.separate`. "
            "Use `-- demucs options` to pass values here."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args()


def ensure_demucs_available() -> None:
    try:
        __import__("demucs.separate")
    except ImportError as exc:  # pragma: no cover - runtime dependency hint
        raise SystemExit(
            "The `demucs` package is required to run this script. "
            "Install it with `python -m pip install demucs`."
        ) from exc


def iter_audio_files(paths: Iterable[Path], extensions: set[str]) -> list[Path]:
    discovered: list[Path] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {path}")
        if path.is_dir():
            for candidate in path.rglob("*"):
                if candidate.is_file() and candidate.suffix.lower() in extensions:
                    discovered.append(candidate)
        elif path.is_file():
            if path.suffix.lower() in extensions:
                discovered.append(path)
            else:
                LOGGER.debug("Skipping unsupported file extension: %s", path)
        else:
            LOGGER.debug("Skipping non-file path: %s", path)
    unique = sorted(set(discovered))
    if not unique:
        raise SystemExit("No audio files found to process.")
    return unique


def separate_drums_bass_other(
    audio_path: Path, model: str, demucs_args: list[str]
) -> tuple[int, dict[str, np.ndarray]]:
    with tempfile.TemporaryDirectory(prefix="demucs_tmp_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        command = [
            sys.executable,
            "-m",
            "demucs.separate",
            "-n",
            model,
            "-o",
            str(tmp_dir),
        ]
        if demucs_args:
            command.extend(demucs_args)
        command.append(str(audio_path))

        LOGGER.debug("Running Demucs command: %s", " ".join(shlex.quote(arg) for arg in command))
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Demucs separation failed for {audio_path}") from exc

        model_dir = tmp_dir / model
        if not model_dir.exists():
            raise RuntimeError(f"Demucs output directory not found: {model_dir}")

        stem_dirs = [p for p in model_dir.iterdir() if p.is_dir()]
        if not stem_dirs:
            raise RuntimeError(f"No stems directory produced for {audio_path}")

        stems_dir = next((p for p in stem_dirs if p.name == audio_path.stem), stem_dirs[0])

        sample_rate: int | None = None
        stems: dict[str, np.ndarray] = {}
        for stem_name in ("drums", "bass", "other"):
            stem_path = stems_dir / f"{stem_name}.wav"
            if not stem_path.exists():
                raise RuntimeError(f"Expected stem not found: {stem_path}")
            audio, sr = sf.read(stem_path, always_2d=False)
            if sample_rate is None:
                sample_rate = sr
            elif sr != sample_rate:
                raise RuntimeError("Inconsistent sample rates across stems.")
            stems[stem_name] = np.asarray(audio, dtype=np.float32)

        if sample_rate is None:
            raise RuntimeError("Could not determine sample rate from Demucs output.")

        return sample_rate, stems


def generate_instrumental_with_two_stems(
    audio_path: Path, model: str, demucs_args: list[str]
) -> tuple[int, np.ndarray]:
    with tempfile.TemporaryDirectory(prefix="demucs_tmp_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        command = [
            sys.executable,
            "-m",
            "demucs.separate",
            "-n",
            model,
            "--two-stems=vocals",
            "-o",
            str(tmp_dir),
        ]
        if demucs_args:
            command.extend(demucs_args)
        command.append(str(audio_path))

        LOGGER.debug("Running Demucs command for instrumental: %s", " ".join(shlex.quote(arg) for arg in command))
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Demucs two-stem separation failed for {audio_path}") from exc

        model_dir = tmp_dir / model
        if not model_dir.exists():
            raise RuntimeError(f"Demucs output directory not found: {model_dir}")

        stem_dirs = [p for p in model_dir.iterdir() if p.is_dir()]
        if not stem_dirs:
            raise RuntimeError(f"No stems directory produced for {audio_path}")

        stems_dir = next((p for p in stem_dirs if p.name == audio_path.stem), stem_dirs[0])

        instrumental_path = stems_dir / "no_vocals.wav"
        if not instrumental_path.exists():
            raise RuntimeError(f"Instrumental stem not found: {instrumental_path}")

        audio, sample_rate = sf.read(instrumental_path, always_2d=False)
        return sample_rate, np.asarray(audio, dtype=np.float32)


def write_stems(
    output_dir: Path,
    sample_rate: int,
    stems: dict[str, np.ndarray],
    instrumental: np.ndarray,
    overwrite: bool,
) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. Use --overwrite to replace it."
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    sf.write(output_dir / "instrumental.wav", instrumental, sample_rate)
    for stem_name, audio in stems.items():
        sf.write(output_dir / f"{stem_name}.wav", audio, sample_rate)


def process_audio_file(
    audio_path: Path, args: argparse.Namespace, extensions: set[str]
) -> None:
    if audio_path.suffix.lower() not in extensions:
        LOGGER.debug("Skipping unsupported file: %s", audio_path)
        return

    LOGGER.info("Processing %s", audio_path)
    sample_rate, stems = separate_drums_bass_other(audio_path, args.model, args.demucs_args)
    instrumental_sr, instrumental = generate_instrumental_with_two_stems(audio_path, args.model, args.demucs_args)

    if instrumental_sr != sample_rate:
        raise RuntimeError(
            "Sample rate mismatch between instrumental mix and individual stems "
            f"({instrumental_sr} vs {sample_rate})."
        )

    target_dir = args.output_dir / audio_path.stem
    write_stems(target_dir, sample_rate, stems, instrumental, args.overwrite)
    LOGGER.info("Finished %s", audio_path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    ensure_demucs_available()

    extensions = {ext.lower() if ext.startswith(".") else f".{ext.lower().lstrip('.')}" for ext in args.extensions}
    audio_files = iter_audio_files(args.inputs, extensions)

    LOGGER.info("Found %d audio file(s) to process.", len(audio_files))
    for audio_file in audio_files:
        try:
            process_audio_file(audio_file, args, extensions)
        except FileExistsError as exc:
            LOGGER.warning("%s", exc)
        except Exception as exc:  # pragma: no cover - runtime failures
            LOGGER.error("Failed to process %s: %s", audio_file, exc)


if __name__ == "__main__":
    main()


