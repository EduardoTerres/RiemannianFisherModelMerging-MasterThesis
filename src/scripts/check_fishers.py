#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from safetensors import safe_open

from src.paths import MODEL_FAMILIES


def default_paths() -> list[str]:
    return [p for family in MODEL_FAMILIES.values() for p in family.fisher_paths]


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Fisher safetensors integrity.")
    parser.add_argument("paths", nargs="*", help="Fisher .safetensors files to check.")
    args = parser.parse_args()

    bad = 0
    for path in args.paths or default_paths():
        try:
            with safe_open(path, framework="numpy"):
                pass
        except Exception as exc:
            bad += 1
            print(f"BAD {path} ({type(exc).__name__}: {exc})")
        else:
            print(f"OK  {path}")

    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
