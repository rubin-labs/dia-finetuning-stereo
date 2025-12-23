#!/usr/bin/env python3
"""Filter Jamendo JSONL files by license compatibility for AI training."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

LOGGER = logging.getLogger("filter_licenses")

# Licenses suitable for AI training (commercial use + derivatives allowed)
# CC0: Public domain, no restrictions
# CC-BY: Attribution required, commercial use allowed
# CC-BY-SA: Attribution + ShareAlike, commercial use allowed
# CC-BY-NC: Non-commercial only (include if you're doing non-commercial research)
# CC-BY-NC-SA: Non-commercial + ShareAlike
VALID_LICENSE_PATTERNS = [
    "cc0",
    "cc-by",
    "cc-by-sa",
    "cc-by-nc",  # Include if non-commercial is acceptable
    "cc-by-nc-sa",  # Include if non-commercial is acceptable
]

# Jamendo-specific license identifiers
JAMENDO_VALID_LICENSES = [
    "cc0",
    "cc-by",
    "cc-by-sa",
    "cc-by-nc",
    "cc-by-nc-sa",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter Jamendo tracks by license compatibility for AI training."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing JSONL files from jamendo_scraper.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for filtered JSONL files. Defaults to input_dir/filtered.",
    )
    parser.add_argument(
        "--allow-non-commercial",
        action="store_true",
        help="Include CC-BY-NC and CC-BY-NC-SA licenses (non-commercial only).",
    )
    parser.add_argument(
        "--require-download",
        action="store_true",
        help="Only include tracks where audiodownload_allowed is True.",
    )
    parser.add_argument(
        "--urls-output",
        type=str,
        default="{genre}_filtered_urls.txt",
        help="Template for filtered URL list filenames.",
    )
    parser.add_argument(
        "--skip-url-export",
        action="store_true",
        help="Do not emit filtered URL txt files.",
    )
    parser.add_argument(
        "--summary-name",
        type=str,
        default="license_filter_summary.json",
        help="Filename for the filtering summary.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    return parser.parse_args()


def normalize_license(license_str: str | None) -> str | None:
    """Normalize license string for comparison."""
    if not license_str:
        return None
    return license_str.lower().strip().replace("_", "-").replace(" ", "-")


def is_valid_license(
    track: dict, allow_non_commercial: bool = False
) -> tuple[bool, str | None]:
    """
    Check if track has a valid license for AI training.
    Returns (is_valid, license_identifier).
    """
    # Check various possible license fields
    license_fields = [
        "license_ccurl",
        "license",
        "licenseccurl",
        "licenses",  # Sometimes it's a list
    ]

    license_value = None
    for field in license_fields:
        value = track.get(field)
        if value:
            if isinstance(value, list):
                # If it's a list, take the first one
                license_value = value[0] if value else None
            else:
                license_value = value
            break

    if not license_value:
        return False, None

    normalized = normalize_license(str(license_value))

    # Check against valid patterns
    valid_patterns = VALID_LICENSE_PATTERNS.copy()
    if not allow_non_commercial:
        valid_patterns = [p for p in valid_patterns if "nc" not in p]

    for pattern in valid_patterns:
        if pattern in normalized:
            return True, normalized

    # Also check Jamendo-specific license codes
    if normalized in JAMENDO_VALID_LICENSES:
        if not allow_non_commercial and "nc" in normalized:
            return False, normalized
        return True, normalized

    return False, normalized


def filter_tracks(
    tracks: Iterable[dict],
    allow_non_commercial: bool = False,
    require_download: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Filter tracks by license and download permission."""
    valid_tracks: list[dict] = []
    invalid_tracks: list[dict] = []

    for track in tracks:
        is_valid, license_id = is_valid_license(track, allow_non_commercial)

        if not is_valid:
            invalid_tracks.append(track)
            continue

        if require_download:
            allowed = track.get("audiodownload_allowed")
            if allowed not in (True, "true", 1, "1"):
                invalid_tracks.append(track)
                continue

        track["_filtered_license"] = license_id
        valid_tracks.append(track)

    return valid_tracks, invalid_tracks


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    """Write tracks to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_url_list(path: Path, tracks: Iterable[dict]) -> int:
    """Write download URLs to text file."""
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


def analyze_licenses(tracks: Iterable[dict]) -> dict[str, int]:
    """Analyze license distribution in tracks."""
    license_counts: dict[str, int] = {}
    for track in tracks:
        _, license_id = is_valid_license(track, allow_non_commercial=True)
        license_key = license_id or "unknown"
        license_counts[license_key] = license_counts.get(license_key, 0) + 1
    return license_counts


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    if not args.input_dir.exists():
        LOGGER.error("Input directory does not exist: %s", args.input_dir)
        return

    output_dir = args.output_dir or (args.input_dir / "filtered")
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_files = list(args.input_dir.glob("*.jsonl"))
    if not jsonl_files:
        LOGGER.warning("No JSONL files found in %s", args.input_dir)
        return

    LOGGER.info("Processing %s JSONL files from %s", len(jsonl_files), args.input_dir)

    summary: list[dict[str, object]] = []
    total_valid = 0
    total_invalid = 0

    for jsonl_path in jsonl_files:
        if jsonl_path.name == args.summary_name:
            continue

        LOGGER.info("Processing %s", jsonl_path.name)

        # Read tracks
        tracks: list[dict] = []
        try:
            with jsonl_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        tracks.append(json.loads(line))
        except Exception as exc:
            LOGGER.error("Failed to read %s: %s", jsonl_path, exc)
            continue

        if not tracks:
            LOGGER.warning("No tracks in %s", jsonl_path.name)
            continue

        # Analyze licenses before filtering
        license_analysis = analyze_licenses(tracks)
        LOGGER.debug("License distribution: %s", license_analysis)

        # Filter tracks
        valid_tracks, invalid_tracks = filter_tracks(
            tracks,
            allow_non_commercial=args.allow_non_commercial,
            require_download=args.require_download,
        )

        total_valid += len(valid_tracks)
        total_invalid += len(invalid_tracks)

        if not valid_tracks:
            LOGGER.warning("No valid tracks found in %s", jsonl_path.name)
            continue

        # Write filtered JSONL
        output_path = output_dir / jsonl_path.name
        write_jsonl(output_path, valid_tracks)

        # Write URL list if requested
        url_count = 0
        if not args.skip_url_export:
            genre_slug = jsonl_path.stem  # filename without .jsonl
            urls_filename = args.urls_output.format(genre=genre_slug)
            url_count = write_url_list(output_dir / urls_filename, valid_tracks)

        summary.append(
            {
                "file": jsonl_path.name,
                "total_tracks": len(tracks),
                "valid_tracks": len(valid_tracks),
                "invalid_tracks": len(invalid_tracks),
                "url_count": url_count,
                "license_distribution": license_analysis,
            }
        )

        LOGGER.info(
            "Filtered %s: %s valid / %s total (urls=%s)",
            jsonl_path.name,
            len(valid_tracks),
            len(tracks),
            url_count,
        )

    # Write summary
    summary_data = {
        "total_files": len(jsonl_files),
        "total_valid_tracks": total_valid,
        "total_invalid_tracks": total_invalid,
        "allow_non_commercial": args.allow_non_commercial,
        "require_download": args.require_download,
        "files": summary,
    }

    summary_path = output_dir / args.summary_name
    summary_path.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")
    LOGGER.info("Summary written to %s", summary_path)
    LOGGER.info(
        "Total: %s valid tracks / %s invalid tracks across %s files",
        total_valid,
        total_invalid,
        len(jsonl_files),
    )


if __name__ == "__main__":
    main()

