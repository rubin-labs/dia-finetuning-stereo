#!/usr/bin/env python3
"""Download audio tracks listed in a text file and export them as MP3 files.

Usage example:
    python scripts/download_music.py --input music.txt --output ./downloads
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import sys
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

try:
    from yt_dlp import YoutubeDL  # type: ignore
except ImportError as exc:  # pragma: no cover - dependency hint
    print(
        "yt-dlp is required to run this script. Install it with "
        "`python -m pip install yt-dlp`.",
        file=sys.stderr,
    )
    raise


LOGGER = logging.getLogger("download_music")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download audio from URLs listed in a text file and save as MP3 files."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("music.txt"),
        help="Path to the text file containing one audio/video URL per line (default: music.txt).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("downloads/mp3"),
        help="Directory where MP3 files will be saved (default: downloads/mp3).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files when downloading.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel download workers to use (default: 1).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit on how many URLs to download.",
    )
    parser.add_argument(
        "--log-level",
        choices=("debug", "info", "warning", "error"),
        default="info",
        help="Logging verbosity (default: info).",
    )
    return parser.parse_args()


def read_urls(input_path: Path, limit: int | None = None) -> list[str]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    urls: list[str] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            urls.append(line)
            if limit is not None and len(urls) >= limit:
                break

    return urls


def download_mp3s(
    urls: Iterable[str], output_dir: Path, overwrite: bool, workers: int
) -> None:
    if not urls:
        LOGGER.warning("No URLs to download. Exiting.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    base_opts = {
        "format": "bestaudio/best",
        "paths": {"home": str(output_dir)},
        "noplaylist": True,
        "overwrites": overwrite,
        "ignoreerrors": True,
        "no_warnings": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            },
            {"key": "FFmpegMetadata"},
        ],
    }

    def infer_output_template(url: str) -> str:
        """Derive a unique, sanitized filename template per URL to avoid collisions."""
        parsed = urlparse(url)
        parts = [segment for segment in parsed.path.split("/") if segment]
        track_id = None
        for idx, segment in enumerate(parts):
            if segment == "track" and idx + 1 < len(parts):
                track_id = parts[idx + 1]
                break
        if track_id is None and parts:
            track_id = parts[-1]
        if track_id:
            safe_id = "".join(ch for ch in track_id if ch.isalnum() or ch in ("-", "_"))
            if safe_id:
                return f"{safe_id}.%(ext)s"
        # Fallback to default yt-dlp naming.
        return "%(title)s.%(ext)s"

    def download_single(url: str) -> None:
        LOGGER.info("Processing URL: %s", url)
        try:
            opts = base_opts.copy()
            opts["outtmpl"] = infer_output_template(url)
            with YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
                entries = info.get("entries")
                if entries:
                    LOGGER.warning(
                        "Skipping playlist %s (playlist downloads are disabled).", url
                    )
                    return

                target_path = Path(ydl.prepare_filename(info)).with_suffix(".mp3")
                if target_path.exists() and not overwrite:
                    LOGGER.info("Skipping existing file %s", target_path.name)
                    return

                ydl.process_ie_result(info, download=True)
        except Exception as exc:  # pragma: no cover - passthrough for runtime failures
            LOGGER.error("Failed to download %s: %s", url, exc)

    max_workers = max(1, workers)
    if max_workers == 1:
        for url in urls:
            download_single(url)
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers
        ) as executor:
            futures = [executor.submit(download_single, url) for url in urls]
            for future in concurrent.futures.as_completed(futures):
                future.result()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    try:
        urls = read_urls(args.input, args.max_files)
    except Exception as exc:
        LOGGER.error("Could not read URLs: %s", exc)
        sys.exit(1)

    download_mp3s(urls, args.output, args.overwrite, args.workers)


if __name__ == "__main__":
    main()

