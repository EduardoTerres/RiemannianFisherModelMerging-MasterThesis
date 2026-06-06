from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rcParams

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dataset.dataset_2 import DATASET_2_PLOT_LABELS, DATASET_2_PLOT_METRICS

rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.titlesize": 18,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 10,
    }
)


def latest_json(paths: list[Path]) -> Path | None:
    return max(paths, key=lambda path: path.stat().st_mtime) if paths else None


def read_metric(model_dir: Path, task: str, metric: str) -> float | None:
    result_file = latest_json(list((model_dir / task).glob("**/results_*.json")))
    metric_file = model_dir / task / "metrics.json"
    path = result_file or (metric_file if metric_file.exists() else None)
    if path is None:
        return None

    data = json.loads(path.read_text())
    values = data.get("results", {}).get(task, data)
    value = values.get(metric)
    if value is None and task == "humanevalplus":
        value = values.get("pass@1") or values.get("pass_at_1")
    if not isinstance(value, int | float):
        return None
    return value


def latex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def format_model_title(name: str) -> str:
    return name.replace("llama", "Llama")


def plot_model(model_dir: Path, output_dir: Path) -> None:
    labels, values = [], []
    for task, (metric, preprocess) in DATASET_2_PLOT_METRICS.items():
        value = read_metric(model_dir, task, metric)
        if value is not None:
            labels.append(task)
            values.append(preprocess(value) if preprocess else value)

    if len(values) < 3:
        print(f"Skipping {model_dir.name}: found {len(values)} metrics")
        return

    sorted_points = sorted(zip(labels, values, strict=True), key=lambda item: item[1], reverse=True)
    labels = [label for label, _ in sorted_points]
    values = [value for _, value in sorted_points]

    values += values[:1]
    angles = [2 * math.pi * idx / len(labels) for idx in range(len(labels))]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9.5, 9.5), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(1)
    ax.plot(angles, values, color="#2457a7", linewidth=2.4)
    ax.fill(angles, values, color="#5b8bd9", alpha=0.25)
    ax.scatter(angles[:-1], values[:-1], color="#123c7c", s=18, zorder=3)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels([r"0.2", r"0.4", r"0.6", r"0.8", r"1.0"], color="#555555")
    ax.set_title(rf"\textbf{{{latex_escape(format_model_title(model_dir.name))}}}", pad=60)
    ax.spines["polar"].set_color("#777777")
    ax.spines["polar"].set_alpha(0.55)
    ax.grid(True, color="#999999", alpha=0.35, linewidth=0.8)

    for angle, label in zip(angles[:-1], labels, strict=True):
        display_angle = (angle + math.pi / 2) % (2 * math.pi)
        if math.pi / 2 < display_angle < 3 * math.pi / 2:
            ha = "right"
        elif display_angle == math.pi / 2 or display_angle == 3 * math.pi / 2:
            ha = "center"
        else:
            ha = "left"
        ax.text(
            angle,
            1.14,
            DATASET_2_PLOT_LABELS.get(label, latex_escape(label)),
            ha=ha,
            va="center",
            fontsize=12,
            clip_on=False,
        )

    fig.subplots_adjust(left=0.12, right=0.88, top=0.86, bottom=0.12)

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{model_dir.name}_spiderweb.png"
    pdf_path = output_dir / f"{model_dir.name}_spiderweb.pdf"
    fig.savefig(png_path, dpi=200)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Saved spiderweb plot for {model_dir.name} to {png_path} and {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluation_type", help="Folder under outputs/evaluation, e.g. pretrained")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()

    eval_root = args.repo_root / "outputs" / "evaluation" / args.evaluation_type
    output_dir = eval_root / "plots"
    for model_dir in sorted(path for path in eval_root.iterdir() if path.is_dir() and path.name not in {"slurm", "plots"}):
        plot_model(model_dir, output_dir)


if __name__ == "__main__":
    main()
