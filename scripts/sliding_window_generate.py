#!/usr/bin/env python3
"""
Batch generation helper for dia-finetuning-stereo.

Iterates through a sliding window of prompts and sampling parameters and
invokes `cli.py` for each configuration.
"""

from __future__ import annotations

import argparse
import itertools
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List


DEFAULT_PROMPTS: List[str] = [
    "Whole Lotta Love",
    "Megadeth",
    "TAME IMPALA",
    "house"
]

DEFAULT_CFG_VALUES = [2, 3]
DEFAULT_TEMPERATURES = [1.5, 1.6, 1.7]
DEFAULT_TOP_P_VALUES = [0.9, 0.97, 0.99]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sliding-window generations with cli.py.")
    parser.add_argument(
        "--cli",
        default="cli.py",
        type=Path,
        help="Path to cli.py entry point (default: cli.py in current working dir).",
    )
    parser.add_argument(
        "--config",
        default="configs/architecture/model_inference.json",
        type=Path,
        help="Path to inference config JSON.",
    )
    parser.add_argument(
        "--checkpoint",
        default="/nfs/turbo/smtd-hwdong/ocamp/basemodel2_checkpoints/ckpt_epoch7.pth",
        type=Path,
        help="Path to model checkpoint to load.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("outputs"),
        type=Path,
        help="Directory where generated wav files will be written.",
    )
    parser.add_argument(
        "--prompts",
        nargs="*",
        help="Override prompt list. If omitted, defaults to builtin list (including empty string).",
    )
    parser.add_argument(
        "--cfg-values",
        nargs="*",
        type=float,
        help="Override cfg-scale values.",
    )
    parser.add_argument(
        "--temperatures",
        nargs="*",
        type=float,
        help="Override temperature values.",
    )
    parser.add_argument(
        "--top-p-values",
        nargs="*",
        type=float,
        help="Override top-p values.",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=5,
        help="Number of random seeds to sample (default: 5).",
    )
    parser.add_argument(
        "--seed-min",
        type=int,
        default=0,
        help="Inclusive lower bound for sampled seeds (default: 0).",
    )
    parser.add_argument(
        "--seed-max",
        type=int,
        default=2**32 - 1,
        help="Inclusive upper bound for sampled seeds (default: 2**32 - 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        dest="rng_seed",
        help="Seed for the Python RNG used to sample seeds. If not set, system entropy is used.",
    )
    parser.add_argument(
        "--local-paths",
        dest="local_paths",
        action="store_true",
        help="Forward --local-paths to cli.py (default: enabled).",
    )
    parser.add_argument(
        "--no-local-paths",
        dest="local_paths",
        action="store_false",
        help="Disable forwarding --local-paths.",
    )
    parser.set_defaults(local_paths=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the commands that would be run.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start index (0-based) in the cartesian product to run (useful for resuming).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of combinations to execute starting from start-index.",
    )
    return parser.parse_args()


def sanitize_for_filename(value: str) -> str:
    if not value:
        return "blank"
    sanitized = re.sub(r"\s+", "_", value.strip())
    sanitized = re.sub(r"[^A-Za-z0-9_.-]", "_", sanitized)
    return sanitized or "blank"


def format_command_output_name(prompt: str, cfg: float, temperature: float, top_p: float, seed: int) -> str:
    prompt_part = sanitize_for_filename(prompt)
    return f"{prompt_part}_cfg{cfg:g}_temp{temperature:g}_top{top_p:g}_seed{seed}"


def iter_combinations(
    prompts: Iterable[str],
    cfg_values: Iterable[float],
    temperatures: Iterable[float],
    top_p_values: Iterable[float],
    seeds: Iterable[int],
) -> Iterable[tuple[str, float, float, float, int]]:
    return itertools.product(prompts, cfg_values, temperatures, top_p_values, seeds)


def main() -> None:
    args = parse_args()

    prompts = args.prompts if args.prompts is not None else DEFAULT_PROMPTS

    cfg_values = args.cfg_values if args.cfg_values is not None else DEFAULT_CFG_VALUES
    temperatures = args.temperatures if args.temperatures is not None else DEFAULT_TEMPERATURES
    top_p_values = args.top_p_values if args.top_p_values is not None else DEFAULT_TOP_P_VALUES

    if args.seed_min > args.seed_max:
        raise ValueError("seed-min must be less than or equal to seed-max.")
    if args.num_seeds < 1:
        raise ValueError("num-seeds must be a positive integer.")

    rng = random.Random(args.rng_seed)
    seed_population = range(args.seed_min, args.seed_max + 1)
    if args.num_seeds > len(seed_population):
        raise ValueError("num-seeds is larger than the available seed range.")
    seeds = rng.sample(seed_population, args.num_seeds)

    combinations = list(iter_combinations(prompts, cfg_values, temperatures, top_p_values, seeds))
    total = len(combinations)

    if args.start_index < 0 or args.start_index >= total:
        raise ValueError(f"start-index {args.start_index} out of range for {total} combinations.")

    end_index = total if args.limit is None else min(total, args.start_index + args.limit)

    cli_path = args.cli.resolve()
    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Prompts ({len(prompts)}): {prompts}")
    print(f"cfg-scale values ({len(cfg_values)}): {cfg_values}")
    print(f"Temperatures ({len(temperatures)}): {temperatures}")
    print(f"Top-p values ({len(top_p_values)}): {top_p_values}")
    print(f"Seeds ({len(seeds)}): {seeds}")
    print(f"Total combinations: {total}")
    print(f"Executing indices [{args.start_index}, {end_index})")

    combos_to_run = combinations[args.start_index:end_index]

    for index, (prompt, cfg, temperature, top_p, seed) in enumerate(combos_to_run, start=args.start_index):
        output_name = format_command_output_name(prompt, cfg, temperature, top_p, seed)
        output_path = output_dir / f"{output_name}.wav"

        command = [
            sys.executable,
            str(cli_path),
            prompt,
            "--output",
            str(output_path),
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint_path),
            "--top-p",
            f"{top_p}",
            "--cfg-scale",
            f"{cfg}",
            "--temperature",
            f"{temperature}",
            "--seed",
            f"{seed}",
        ]

        if args.local_paths:
            command.insert(3, "--local-paths")

        print(f"[{index + 1}/{total}] {' '.join(shlex_quote(arg) for arg in command)}")
        if args.dry_run:
            continue

        subprocess.run(command, check=True)


def shlex_quote(value: str) -> str:
    """
    Minimal shlex-like quoting to improve readability of logged commands.
    """
    if value == "":
        return '""'
    if re.fullmatch(r"[A-Za-z0-9_./:-]+", value):
        return value
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


if __name__ == "__main__":
    main()

