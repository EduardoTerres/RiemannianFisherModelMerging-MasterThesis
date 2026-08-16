"""Visualize layerwise average diagonal Fisher values."""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

ROOTDIR = Path(__file__).resolve().parents[2]
MPLCONFIGDIR = ROOTDIR / "outputs" / ".matplotlib"
XDG_CACHE_HOME = ROOTDIR / "outputs" / ".cache"
for path in (MPLCONFIGDIR, XDG_CACHE_HOME):
    path.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_HOME))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

if str(ROOTDIR) not in sys.path:
    sys.path.insert(0, str(ROOTDIR))

from src.dataset.dataset_3 import DATASET_3_TRAIN

TASKS = [tag for tag, *_ in DATASET_3_TRAIN]
LAYER_RE = re.compile(r"layers?\.(\d+)\.")
FISHERS_DIR = Path("/path/to/fishers")
FAMILIES = {
    "llama3.1": "llama3-1_8b_finetune",
    "qwen2.5": "qwen2.5_3b_finetune",
}


def _task_name(task: str) -> str:
    return {"social_iqa": "socialiqa", "commonsense_qa": "commonsense", "science_qa": "scienceqa"}.get(task, task)


def _fisher_paths(family_name: str, state: str) -> list[str]:
    prefix = FAMILIES[family_name]
    return [
        str(FISHERS_DIR / family_name / task / f"{prefix}_{_task_name(task)}_{state}.safetensors")
        for task in TASKS
    ]


def _layer_means(path: str) -> dict[int, float]:
    if not Path(path).exists():
        raise FileNotFoundError(path)
    fisher = load_file(path, device="cpu")
    values: dict[int, list[torch.Tensor]] = {}
    for key, tensor in fisher.items():
        match = LAYER_RE.search(key)
        if match:
            values.setdefault(int(match.group(1)), []).append(tensor.float().mean())
    return {layer: torch.stack(means).mean().item() for layer, means in values.items()}


def _heatmap(paths: list[str]) -> tuple[np.ndarray, list[int]]:
    rows = [_layer_means(path) for path in paths]
    layers = sorted(set().union(*rows))
    data = np.array([[row.get(layer, np.nan) for layer in layers] for row in rows])
    return np.log10(np.clip(data, 1e-12, None)), layers


def _plot(data: np.ndarray, layers: list[int], title: str, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(7, len(layers) * 0.35), 3.8), constrained_layout=True)
    im = ax.imshow(data, aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Task")
    ax.set_xticks(range(len(layers)), layers, rotation=90)
    ax.set_yticks(range(len(TASKS)), TASKS)
    fig.colorbar(im, ax=ax, label="log10 mean FIM")
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=Path, default=ROOTDIR / "outputs" / "analysis" / "fim")
    args = parser.parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)

    for family_name in FAMILIES:
        for state, paths in {
            "finetuned": _fisher_paths(family_name, "finetuned"),
            "pretrained": _fisher_paths(family_name, "pretrained"),
        }.items():
            print(f"Plotting {family_name} {state}...")
            data, layers = _heatmap(paths)
            save_path = args.save_dir / f"{family_name}_{state}_fim_heatmap.png"
            _plot(data, layers, f"{family_name} {state} Fisher", save_path)
            print(f"Saved {save_path}")


if __name__ == "__main__":
    main()
