#!/usr/bin/env python3
"""Generate comma-separated prompt tags for audio files using Gemini 2.5 Flash.

For each supported audio file in a directory the script uploads the clip to the
Gemini 2.5 Flash API, asks for structured genre/mood/instrument metadata, and
writes a ``.txt`` file containing comma-separated tags ready for finetuning
prompts. By default it expects audio in ``dataset/audio`` and writes prompts to
``dataset/audio_prompts`` using filenames like ``track_prompt.txt``.
"""

from __future__ import annotations

import argparse
import logging
import mimetypes
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import google.generativeai as genai


LOGGER = logging.getLogger("gemini_audio_prompts")
DEFAULT_EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a", ".aac", ".aiff", ".ogg")
_THREAD_LOCAL = threading.local()


@dataclass(slots=True)
class AudioTask:
    idx: int
    audio_path: Path
    output_path: Path


PROMPT_TEMPLATE = """You are a meticulous music tagging assistant.
Listen to the supplied audio clip and respond with a single line of comma-separated tags.
Rules:
- Include the primary genres, moods, and notable instruments.
- Use concise lowercase tags; no duplicates or extra text.
- Do not add explanations or formatting; only the comma-separated tags.
- if the clip has vocals, only include the tag "vocals"
Clip context: {filename}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create prompt tag text files for each audio clip via Gemini 2.5 Flash."
    )
    parser.add_argument(
        "audio_dir",
        nargs="?",
        default=Path("dataset/audio"),
        type=Path,
        help="Directory containing audio files to analyze (default: dataset/audio).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/audio_prompts"),
        help="Directory to store generated .txt prompt files (default: dataset/audio_prompts).",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Gemini model name to invoke (default: gemini-2.5-flash).",
    )
    parser.add_argument(
        "--api-key",
        help="Explicit Gemini API key. Falls back to GEMINI_API_KEY or GOOGLE_API_KEY env vars.",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=list(DEFAULT_EXTENSIONS),
        help="Audio file extensions to include (default: .wav .mp3 .flac .m4a .aac .aiff .ogg).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        help="Optional limit on how many files to process.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing prompt files instead of skipping them.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the tags that would be written without touching the filesystem.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Request timeout in seconds for each Gemini call (default: 120).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent tagging workers to use (default: 1).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args()


def resolve_api_key(explicit_key: str | None) -> str:
    api_key = explicit_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "No Gemini API key provided. Supply --api-key or set GEMINI_API_KEY/GOOGLE_API_KEY."
        )
    return api_key


def gather_audio_files(
    audio_dir: Path, extensions: Sequence[str], max_files: int | None
) -> list[Path]:
    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
    if not audio_dir.is_dir():
        raise NotADirectoryError(f"Audio path is not a directory: {audio_dir}")

    normalized_exts = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}
    candidates = sorted(
        path
        for path in audio_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in normalized_exts
    )
    if max_files is not None:
        candidates = candidates[: max(0, max_files)]
    return candidates


def read_audio_part(audio_path: Path) -> dict[str, object]:
    mime_type, _ = mimetypes.guess_type(audio_path.name)
    mime_type = mime_type or "audio/wav"
    with audio_path.open("rb") as handle:
        audio_bytes = handle.read()
    if not audio_bytes:
        raise ValueError(f"Audio file is empty: {audio_path}")
    return {"mime_type": mime_type, "data": audio_bytes}


def request_tags(model: genai.GenerativeModel, audio_path: Path, timeout: int) -> str:
    audio_part = read_audio_part(audio_path)
    prompt = PROMPT_TEMPLATE.format(filename=audio_path.name)
    request_options = {"timeout": timeout} if timeout else None

    response = model.generate_content(
        [audio_part, prompt],
        request_options=request_options,
    )
    text = (response.text or "").strip()
    if not text:
        raise ValueError("Gemini returned an empty response.")

    cleaned = cleanup_code_fence(text).replace("\n", " ").strip()
    if not cleaned:
        raise ValueError("Gemini returned only whitespace.")
    return cleaned


def cleanup_code_fence(text: str) -> str:
    if "```" not in text:
        return text
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            continue
        lines.append(line)
    return "\n".join(lines)


def get_thread_model(model_name: str) -> genai.GenerativeModel:
    model = getattr(_THREAD_LOCAL, "model", None)
    cached_name = getattr(_THREAD_LOCAL, "model_name", None)
    if model is None or cached_name != model_name:
        model = genai.GenerativeModel(model_name)
        _THREAD_LOCAL.model = model
        _THREAD_LOCAL.model_name = model_name
    return model


def tag_audio_task(
    task: AudioTask, total_files: int, model_name: str, timeout: int, dry_run: bool
) -> bool:
    LOGGER.info("(%s/%s) Tagging %s", task.idx, total_files, task.audio_path.name)
    try:
        model = get_thread_model(model_name)
        tag_line = request_tags(model, task.audio_path, timeout)
        if not tag_line or not tag_line.strip():
            LOGGER.warning(
                "(%s/%s) Gemini returned empty tags for %s.",
                task.idx,
                total_files,
                task.audio_path.name,
            )
            raise ValueError("Gemini returned empty tags")

        write_tags_file(task.output_path, tag_line, dry_run)
        return True
    except Exception as exc:
        # Re-raise so it can be caught and logged in the main loop
        LOGGER.debug("(%s/%s) Exception in tag_audio_task for %s: %s", task.idx, total_files, task.audio_path.name, exc)
        raise


def write_tags_file(path: Path, content: str, dry_run: bool) -> None:
    if dry_run:
        LOGGER.info("[dry-run] %s -> %s", path.name, content)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content + "\n", encoding="utf-8")
    LOGGER.info("Wrote %s", path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="[%(levelname)s] %(message)s",
        stream=sys.stdout,
    )

    try:
        api_key = resolve_api_key(args.api_key)
    except ValueError as exc:
        LOGGER.error("%s", exc)
        sys.exit(1)

    try:
        audio_files = gather_audio_files(args.audio_dir, args.extensions, args.max_files)
    except Exception as exc:
        LOGGER.error("Failed to collect audio files: %s", exc)
        sys.exit(1)

    total_files = len(audio_files)
    if not total_files:
        LOGGER.warning("No audio files found in %s", args.audio_dir)
        return

    genai.configure(api_key=api_key)

    workers = max(1, args.workers)
    processed = 0
    failures = 0
    skipped_existing = 0
    tasks: list[AudioTask] = []
    for idx, audio_path in enumerate(audio_files, start=1):
        output_path = args.output_dir / f"{audio_path.stem}_prompt.txt"
        if output_path.exists() and not args.overwrite and not args.dry_run:
            LOGGER.info("(%s/%s) Skipping existing prompt %s", idx, total_files, output_path)
            skipped_existing += 1
            continue
        tasks.append(AudioTask(idx=idx, audio_path=audio_path, output_path=output_path))

    total_tasks = len(tasks)

    if not tasks:
        LOGGER.info(
            "No files to process after skipping existing prompts (skipped: %s, written: %s, failures: %s).",
            skipped_existing,
            processed,
            failures,
        )
        return

    if workers == 1:
        for task in tasks:
            try:
                tag_audio_task(task, total_files, args.model, args.timeout, args.dry_run)
            except Exception as exc:
                LOGGER.error("(%s/%s) Failed to tag %s: %s", task.idx, total_files, task.audio_path.name, exc)
                failures += 1
                continue
            processed += 1
            LOGGER.info("Progress: %s/%s files tagged.", processed, total_tasks)
    else:
        # Thread-safe counters for multi-worker execution
        stats_lock = threading.Lock()
        processed_count = 0
        failures_count = 0
        
        max_workers = min(workers, total_tasks)
        LOGGER.info("Starting multi-worker processing with %s workers for %s tasks", max_workers, total_tasks)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(
                    tag_audio_task,
                    task,
                    total_files,
                    args.model,
                    args.timeout,
                    args.dry_run,
                ): task
                for task in tasks
            }
            completed_count = 0
            for future in as_completed(future_to_task):
                completed_count += 1
                task = future_to_task[future]
                try:
                    future.result()
                except Exception as exc:
                    LOGGER.error(
                        "(%s/%s) Failed to tag %s: %s (type: %s)",
                        task.idx,
                        total_files,
                        task.audio_path.name,
                        exc,
                        type(exc).__name__,
                    )
                    with stats_lock:
                        failures_count += 1
                        current_failures = failures_count
                        current_processed = processed_count
                    if completed_count % 100 == 0:
                        LOGGER.info(
                            "Completed %s/%s tasks (processed: %s, failed: %s)",
                            completed_count,
                            total_tasks,
                            current_processed,
                            current_failures,
                        )
                    continue

                with stats_lock:
                    processed_count += 1
                    current_processed = processed_count
                    current_failures = failures_count
                if current_processed % 10 == 0 or completed_count % 100 == 0:
                    LOGGER.info(
                        "Progress: %s/%s files tagged. Completed %s/%s tasks (failed: %s)",
                        current_processed,
                        total_tasks,
                        completed_count,
                        total_tasks,
                        current_failures,
                    )
        
        processed = processed_count
        failures = failures_count

    LOGGER.info(
        "Finished run: %s prompt(s) written, %s existing prompt(s) skipped, %s file(s) failed.",
        processed,
        skipped_existing,
        failures,
    )


if __name__ == "__main__":
    main()
