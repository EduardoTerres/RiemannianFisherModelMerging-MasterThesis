"""Compare subset merges with their constituent finetuned adapters."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib-{os.getuid()}-{os.getpid()}")
import matplotlib
import numpy as np
import torch
from safetensors.torch import load_file

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SIZES = (2, 4, 6, 8, 10, 12)
MODES = ("standard_rescaled", "diagonal_fisher")
ADAPTER_DIRS = {
    "llama3.1": ROOT / "data/models/Llama-3.1-8B_OFT_dataset2_adapters",
    "qwen2.5": ROOT / "data/models/Qwen-2.5-3B_OFT_dataset2_adapters",
}


def load_vector(path: Path) -> torch.Tensor:
    state = load_file(str(path / "adapter_model.safetensors"), device="cpu")
    return torch.cat([state[key].float().flatten() for key in sorted(state)])


def summarize(merged: torch.Tensor, tasks: list[torch.Tensor]) -> dict[str, float]:
    values = {"cosine": [], "relative_l2": [], "norm_ratio": []}
    for task in tasks:
        values["cosine"].append(torch.nn.functional.cosine_similarity(merged, task, dim=0).item())
        values["relative_l2"].append(((merged - task).norm() / task.norm()).item())
        values["norm_ratio"].append((merged.norm() / task.norm()).item())
    return {
        f"{metric}_{stat}": float(function(samples))
        for metric, samples in values.items()
        for stat, function in (
            ("mean", np.mean),
            ("q1", lambda x: np.percentile(x, 25)),
            ("q3", lambda x: np.percentile(x, 75)),
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-family", choices=ADAPTER_DIRS, default="llama3.1")
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--save-dir", type=Path)
    parser.add_argument("--plot-dir", type=Path)
    args = parser.parse_args()

    models_dir = args.models_dir or ROOT / "outputs/models/subsets" / args.model_family
    save_dir = args.save_dir or ROOT / "outputs/analysis/subsets" / args.model_family
    plot_dir = args.plot_dir or ROOT / "outputs/subsets" / args.model_family
    subsets = json.loads((models_dir / "subsets.json").read_text())
    task_vectors = {
        name: load_vector(ADAPTER_DIRS[args.model_family] / name)
        for names in subsets.values()
        for name in names
    }

    rows = []
    for size in SIZES:
        tasks = [task_vectors[name] for name in subsets[str(size)]]
        for mode in MODES:
            merged = load_vector(models_dir / f"{mode}_{size}" / "merged_adapter")
            rows.append({"models": size, "method": mode, **summarize(merged, tasks)})

    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / "subset_statistics.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    metrics = (("cosine", "Cosine similarity"), ("relative_l2", "Relative L2 distance"),
               ("norm_ratio", "Norm ratio"))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    for ax, (metric, title) in zip(axes, metrics):
        for mode in MODES:
            selected = [row for row in rows if row["method"] == mode]
            x = [row["models"] for row in selected]
            ax.plot(x, [row[f"{metric}_mean"] for row in selected], marker="o", label=mode)
            ax.fill_between(
                x,
                [row[f"{metric}_q1"] for row in selected],
                [row[f"{metric}_q3"] for row in selected],
                alpha=0.2,
            )
        ax.set(title=title, xlabel="Number of models")
        ax.grid(alpha=0.3)
    axes[0].legend()
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / "subset_statistics.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"Saved {csv_path}")
    print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
