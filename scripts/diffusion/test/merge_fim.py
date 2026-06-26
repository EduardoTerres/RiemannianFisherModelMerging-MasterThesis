#!/usr/bin/env python
"""Subtract the cat2 FIM from the cat FIM and save it as cat_mixed."""

import argparse
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.diffusion.dataset_1 import get_entry  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--minuend", default="cat", help="Concept FIM to subtract from.")
    parser.add_argument("--subtrahend", default="cat2", help="Concept FIM to subtract.")
    parser.add_argument("--output-prefix", default="cat_mixed", help="Prefix for the output FIM.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory to write the output FIM.")
    return parser.parse_args()


def output_path_for(minuend_path, minuend_name, output_prefix, output_dir):
    filename = Path(minuend_path).name
    if filename.startswith(f"{minuend_name}_"):
        filename = f"{output_prefix}_{filename[len(minuend_name) + 1:]}"
    else:
        filename = f"{output_prefix}_{filename}"
    return (output_dir or Path(minuend_path).parent) / filename


def main():
    args = parse_args()
    minuend_entry = get_entry("concept", args.minuend)
    subtrahend_entry = get_entry("concept", args.subtrahend)

    minuend_path = Path(minuend_entry["fim_path"])
    subtrahend_path = Path(subtrahend_entry["fim_path"])
    output_path = output_path_for(minuend_path, args.minuend, args.output_prefix, args.output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    minuend = load_file(minuend_path)
    subtrahend = load_file(subtrahend_path)

    minuend_keys = set(minuend)
    subtrahend_keys = set(subtrahend)
    if minuend_keys != subtrahend_keys:
        missing = sorted(minuend_keys ^ subtrahend_keys)
        raise ValueError(f"FIM keys do not match: {missing[:20]}")

    mixed = {}
    for key in sorted(minuend):
        if minuend[key].shape != subtrahend[key].shape:
            raise ValueError(
                f"Shape mismatch for {key}: {tuple(minuend[key].shape)} != {tuple(subtrahend[key].shape)}"
            )
        mixed[key] = minuend[key] - subtrahend[key]

    save_file(mixed, output_path)
    print(f"Saved {args.minuend} - {args.subtrahend} FIM to {output_path}")


if __name__ == "__main__":
    main()
