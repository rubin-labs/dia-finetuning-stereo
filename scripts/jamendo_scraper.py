#!/usr/bin/env python3
"""Scrape instrumental Jamendo metadata grouped by genre."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import requests
from requests import Response


LOGGER = logging.getLogger("jamendo_scraper")

BASE_URL = "https://api.jamendo.com/v3.0/tracks"
TAGS_URL = "https://api.jamendo.com/v3.0/tags/music"

# Derived from MTG-Jamendo genre taxonomy (94 tags covering Jamendo catalog).
DEFAULT_GENRES = [
    "60s",
    "70s",
    "80s",
    "90s",
    "acidjazz",
    "african",
    "alternative",
    "alternativerock",
    "ambient",
    "atmospheric",
    "blues",
    "bluesrock",
    "bossanova",
    "breakbeat",
    "celtic",
    "chanson",
    "chillout",
    "choir",
    "classical",
    "classicrock",
    "club",
    "contemporary",
    "country",
    "dance",
    "darkambient",
    "darkwave",
    "deephouse",
    "disco",
    "downtempo",
    "drumnbass",
    "dub",
    "dubstep",
    "easylistening",
    "edm",
    "electronic",
    "electronica",
    "electropop",
    "ethnicrock",
    "ethno",
    "eurodance",
    "experimental",
    "folk",
    "funk",
    "fusion",
    "gothic",
    "groove",
    "grunge",
    "hard",
    "hardrock",
    "heavymetal",
    "hiphop",
    "house",
    "idm",
    "improvisation",
    "indie",
    "industrial",
    "instrumentalpop",
    "instrumentalrock",
    "jazz",
    "jazzfunk",
    "jazzfusion",
    "latin",
    "lounge",
    "medieval",
    "metal",
    "minimal",
    "newage",
    "newwave",
    "orchestral",
    "oriental",
    "pop",
    "popfolk",
    "poprock",
    "postrock",
    "progressive",
    "psychedelic",
    "punkrock",
    "rap",
    "reggae",
    "rnb",
    "rock",
    "rocknroll",
    "singersongwriter",
    "ska",
    "soul",
    "soundtrack",
    "swing",
    "symphonic",
    "synthpop",
    "techno",
    "trance",
    "tribal",
    "triphop",
    "world",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Jamendo instrumental metadata grouped by genre."
    )
    parser.add_argument(
        "--client-id",
        default="f4f5aad4",
        help="Jamendo API client ID. Defaults to $JAMENDO_CLIENT_ID.",
    )
    parser.add_argument(
        "--genres",
        nargs="+",
        default=None,
        help="Space-separated list of genres/tags to query. "
        "If omitted the built-in canonical list is used.",
    )
    parser.add_argument(
        "--genres-file",
        type=Path,
        help="Optional text file with one genre/tag per line.",
    )
    parser.add_argument(
        "--per-genre-limit",
        type=int,
        default=None,
        help="Maximum tracks to fetch per genre. "
        "If omitted the scraper keeps paging until Jamendo stops returning results.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Jamendo page size (max 200).",
    )
    parser.add_argument(
        "--min-duration",
        type=int,
        default=30,
        help="Minimum track length in seconds.",
    )
    parser.add_argument(
        "--max-duration",
        type=int,
        default=None,
        help="Optional maximum track length in seconds.",
    )
    parser.add_argument(
        "--include",
        default="musicinfo+stats+licenses",
        help="Jamendo include string for extra metadata.",
    )
    parser.add_argument(
        "--order",
        default="popularity_total",
        help="Jamendo ordering field (e.g. popularity_total, releasedate).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/jamendo"),
        help="Where to store the JSONL files and summaries.",
    )
    parser.add_argument(
        "--urls-output",
        type=str,
        default="{genre}_urls.txt",
        help="Template for the per-genre download URL list.",
    )
    parser.add_argument(
        "--summary-name",
        type=str,
        default="summary.json",
        help="Filename for the scrape summary.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Base seconds to sleep between requests.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Retry attempts per request.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout per request in seconds.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--user-agent",
        default="dia-jamendo-scraper/1.0",
        help="Custom User-Agent header.",
    )
    parser.add_argument(
        "--skip-url-export",
        action="store_true",
        help="Do not emit per-genre audio URL txt files.",
    )
    parser.add_argument(
        "--instrumental-only",
        action="store_true",
        help="Restrict results to tracks Jamendo marks as instrumental.",
    )
    return parser.parse_args()


def load_genres(args: argparse.Namespace) -> list[str]:
    genres: list[str] = []
    if args.genres_file:
        if not args.genres_file.exists():
            raise FileNotFoundError(f"Genres file not found: {args.genres_file}")
        with args.genres_file.open("r", encoding="utf-8") as handle:
            genres.extend(line.strip() for line in handle if line.strip())
    if args.genres:
        genres.extend(args.genres)
    if not genres:
        try:
            genres = fetch_all_tags(
                client_id=args.client_id,
                user_agent=args.user_agent,
                timeout=args.timeout,
                max_retries=args.max_retries,
                sleep=args.sleep,
            )
        except Exception as exc:  # pragma: no cover - network variability
            LOGGER.warning(
                "Falling back to built-in genres because automatic tag fetch failed: %s",
                exc,
            )
            genres = DEFAULT_GENRES.copy()
    # Preserve order but drop duplicates.
    seen: set[str] = set()
    unique_genres = []
    for genre in genres:
        if genre not in seen:
            unique_genres.append(genre)
            seen.add(genre)
    return unique_genres


def fetch_all_tags(
    *,
    client_id: str,
    user_agent: str,
    timeout: float,
    max_retries: int,
    sleep: float,
) -> list[str]:
    LOGGER.info("No genres provided; fetching global Jamendo tag list.")
    params: dict[str, str | int] = {
        "client_id": client_id,
        "format": "json",
        "limit": 200,
        "offset": 0,
    }
    headers = {"User-Agent": user_agent}
    tags: list[str] = []

    while True:
        payload = request_with_retry(
            TAGS_URL,
            params=params,
            headers=headers,
            timeout=timeout,
            max_retries=max_retries,
            sleep=sleep,
        )

        results = payload.get("results", [])
        if not results:
            break
        for entry in results:
            name = entry.get("name")
            if name:
                tags.append(name)
        params["offset"] += len(results)
        if len(results) < params["limit"]:
            break

    if not tags:
        raise RuntimeError(
            "Couldn't retrieve Jamendo tags automatically. Supply --genres or --genres-file."
        )

    unique_tags = sorted(set(tags))
    LOGGER.info("Fetched %s global tags from Jamendo.", len(unique_tags))
    return unique_tags


def request_with_retry(
    url: str,
    params: dict[str, str | int],
    headers: dict[str, str],
    timeout: float,
    max_retries: int,
    sleep: float,
) -> dict:
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response: Response = requests.get(
                url, params=params, headers=headers, timeout=timeout
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:  # pragma: no cover - network issues at runtime
            last_exc = exc
            backoff = sleep * attempt
            LOGGER.warning(
                "Jamendo request failed (attempt %s/%s): %s", attempt, max_retries, exc
            )
            time.sleep(backoff)
            continue
        headers_block = payload.get("headers", {})
        status = headers_block.get("status")
        code = headers_block.get("code")
        if status and status.lower() != "success":
            error_message = headers_block.get("error_message") or payload
            raise RuntimeError(f"Jamendo API error {code}: {error_message}")
        return payload
    raise RuntimeError(f"Jamendo request failed after {max_retries} attempts: {last_exc}")


def fetch_genre_tracks(
    genre: str,
    *,
    client_id: str,
    include: str,
    order: str,
    per_genre_limit: int | None,
    batch_size: int,
    min_duration: int,
    max_duration: int | None,
    timeout: float,
    max_retries: int,
    sleep: float,
    user_agent: str,
    instrumental_only: bool,
) -> list[dict]:
    collected: list[dict] = []
    offset = 0
    headers = {"User-Agent": user_agent}

    while True:
        if per_genre_limit is not None and len(collected) >= per_genre_limit:
            break

        remaining = (
            per_genre_limit - len(collected)
            if per_genre_limit is not None
            else batch_size
        )
        limit = min(batch_size, remaining) if per_genre_limit is not None else batch_size

        params: dict[str, str | int] = {
            "client_id": client_id,
            "format": "json",
            "limit": limit,
            "offset": offset,
            "tags": genre,
            "order": order,
            "include": include,
            "audioformat": "mp32",
            "audiodlformat": "mp32",
            "groupby": "artist_id",
            "min_duration": min_duration,
        }
        if instrumental_only:
            params["instrumental"] = 1
        if max_duration:
            params["max_duration"] = max_duration

        payload = request_with_retry(
            BASE_URL,
            params=params,
            headers=headers,
            timeout=timeout,
            max_retries=max_retries,
            sleep=sleep,
        )

        results = payload.get("results", [])
        if not results:
            LOGGER.info("No more tracks for genre '%s'.", genre)
            break

        for track in results:
            track["requested_genre"] = genre
            collected.append(track)
        offset += len(results)
        LOGGER.info(
            "Fetched %s tracks for genre '%s' (total=%s).",
            len(results),
            genre,
            len(collected),
        )
        time.sleep(sleep)

    return collected


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_url_list(path: Path, tracks: Iterable[dict]) -> int:
    urls: list[str] = []
    for track in tracks:
        allowed = track.get("audiodownload_allowed")
        url = track.get("audiodownload") or track.get("audio")
        if allowed in (True, "true", 1, "1") and url:
            urls.append(url)
    if not urls:
        return 0
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(urls))
    return len(urls)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    if not args.client_id:
        LOGGER.error("Jamendo client ID missing. Pass --client-id or set $JAMENDO_CLIENT_ID.")
        sys.exit(1)

    genres = load_genres(args)
    LOGGER.info("Scraping %s genres.", len(genres))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, object]] = []
    for genre in genres:
        try:
            tracks = fetch_genre_tracks(
                genre=genre,
                client_id=args.client_id,
                include=args.include,
                order=args.order,
                per_genre_limit=args.per_genre_limit,
                batch_size=args.batch_size,
                min_duration=args.min_duration,
                max_duration=args.max_duration,
                timeout=args.timeout,
                max_retries=args.max_retries,
                sleep=args.sleep,
                user_agent=args.user_agent,
                instrumental_only=args.instrumental_only,
            )
        except Exception as exc:
            LOGGER.error("Failed to fetch genre '%s': %s", genre, exc)
            continue

        if not tracks:
            LOGGER.warning("No tracks persisted for genre '%s'.", genre)
            continue

        genre_slug = genre.lower().replace(" ", "_")
        metadata_path = args.output_dir / f"{genre_slug}.jsonl"
        write_jsonl(metadata_path, tracks)

        url_count = 0
        if not args.skip_url_export:
            urls_filename = args.urls_output.format(genre=genre_slug)
            url_count = write_url_list(args.output_dir / urls_filename, tracks)

        summary.append(
            {
                "genre": genre,
                "tracks": len(tracks),
                "metadata_path": str(metadata_path),
                "url_count": url_count,
            }
        )
        LOGGER.info(
            "Genre '%s' complete. %s tracks written to %s (urls=%s).",
            genre,
            len(tracks),
            metadata_path,
            url_count,
        )

    summary_path = args.output_dir / args.summary_name
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("Summary written to %s", summary_path)


if __name__ == "__main__":
    main()


