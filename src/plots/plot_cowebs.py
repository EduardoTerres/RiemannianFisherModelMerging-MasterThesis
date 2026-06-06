from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib import rcParams

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dataset.dataset_2 import DATASET_2_PLOT_LABELS, DATASET_2_PLOT_METRICS

RESULT_TASK_ALIASES = {
    "cnn_dailymail": ("cnn_dailymail_abisee",),
    "math500": ("minerva_math500",),
}

PLOT_MODES = ("eval_performance", "eval_loss")

METRIC_ALIASES = {
    ("xsum", "rougeL,none"): ("rouge,none",),
}

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
    values = data.get("results", {}).get(task)
    for alias in RESULT_TASK_ALIASES.get(task, ()):
        if values is not None:
            break
        values = data.get("results", {}).get(alias)
    if values is None:
        values = data.get(task, data)
    value = values.get(metric)
    for alias in METRIC_ALIASES.get((task, metric), ()):
        if value is not None:
            break
        value = values.get(alias)
    if value is None and task == "humanevalplus":
        value = values.get("pass@1") or values.get("pass_at_1")
    if not isinstance(value, int | float):
        return None
    return value


def latex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def format_model_title(name: str) -> str:
    return name.replace("_", r"\_")


def filename_model_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def format_family_title(name: str) -> str:
    return name.replace("llama", "Llama").replace("qwen", "Qwen")


def collect_metrics(model_dir: Path, plot_mode: str) -> dict[str, float]:
    metrics = {}
    for task, (performance_metric, preprocess) in DATASET_2_PLOT_METRICS.items():
        metric = "eval_loss" if plot_mode == "eval_loss" else performance_metric
        value = read_metric(model_dir, task, metric)
        if value is not None:
            if plot_mode == "eval_loss":
                metrics[task] = value
            else:
                metrics[task] = preprocess(value) if preprocess else value
    return metrics


def select_runs(results_root: Path, family: str, models: Iterable[str] | None) -> list[tuple[str, Path]]:
    available = {
        path.name: path
        for path in results_root.iterdir()
        if path.is_dir() and path.name not in {"slurm", "plots"}
    }
    selected_names = list(models) if models else sorted(available)
    missing = sorted(set(selected_names) - set(available))
    if missing:
        raise SystemExit(f"Missing output(s) in {results_root}: {', '.join(missing)}")

    runs = []
    for name in selected_names:
        model_dirs = [
            path
            for path in available[name].iterdir()
            if path.is_dir() and family.lower() in path.name.lower()
        ]
        if not model_dirs:
            raise SystemExit(f"Missing {family} model directory in {available[name]}")
        if len(model_dirs) > 1:
            choices = ", ".join(path.name for path in model_dirs)
            raise SystemExit(f"Multiple {family} model directories in {available[name]}: {choices}")
        runs.append((name, model_dirs[0]))
    return runs


def plot_models(runs: list[tuple[str, Path]], output_dir: Path, family: str, plot_mode: str) -> None:
    model_metrics = {label: collect_metrics(model_dir, plot_mode) for label, model_dir in runs}
    first_metrics = model_metrics[runs[0][0]]
    labels = sorted(first_metrics, key=first_metrics.get, reverse=plot_mode == "eval_performance")
    labels += [
        task
        for task in DATASET_2_PLOT_METRICS
        if task not in first_metrics and any(task in metrics for metrics in model_metrics.values())
    ]
    if not labels:
        print("Skipping plot: found 0 metrics")
        return
    finite_values = [
        value
        for metrics in model_metrics.values()
        for value in metrics.values()
        if math.isfinite(value)
    ]

    angles = [2 * math.pi * idx / len(labels) for idx in range(len(labels))]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(11.5, 9.5), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(1)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for idx, (model_name, metrics) in enumerate(model_metrics.items()):
        values = [metrics.get(label, math.nan) for label in labels]
        values += values[:1]
        color = colors[idx % len(colors)]
        ax.plot(angles, values, color=color, linewidth=2.4, label=format_model_title(model_name))
        ax.fill(angles, values, color=color, alpha=0.12)
        ax.scatter(angles[:-1], values[:-1], color=color, s=18, zorder=3)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])
    if plot_mode == "eval_loss":
        ymax = max(finite_values) if finite_values else 1
        ymax = ymax * 1.08 if ymax else 1
        ticks = [ymax * frac for frac in (0.2, 0.4, 0.6, 0.8, 1.0)]
        tick_labels = [f"{tick:.2g}" for tick in ticks]
    else:
        ymax = 1
        ticks = [0.2, 0.4, 0.6, 0.8, 1.0]
        tick_labels = [r"0.2", r"0.4", r"0.6", r"0.8", r"1.0"]
    ax.set_ylim(0, ymax)
    ax.set_yticks(ticks)
    ax.set_rlabel_position(-28)
    ax.set_yticklabels(tick_labels, color="#555555", fontsize=16)
    ax.set_title(rf"\textbf{{{latex_escape(format_family_title(runs[0][1].name))}}}", pad=60)
    ax.spines["polar"].set_color("#777777")
    ax.spines["polar"].set_alpha(0.55)
    ax.grid(True, color="#999999", alpha=0.35, linewidth=0.8)
    ax.legend(loc="upper left", bbox_to_anchor=(1.06, 1.02), frameon=False)

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

    fig.subplots_adjust(left=0.08, right=0.76, top=0.86, bottom=0.12)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        "coweb_"
        + plot_mode
        + "_"
        + filename_model_name(family)
        + "_"
        + "_".join(filename_model_name(name) for name in model_metrics)
    )
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved spiderweb plot to {png_path} and {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--model-family", choices=["llama", "qwen"], required=True, help="Model family to plot")
    parser.add_argument("--models", nargs="+", help="Evaluation outputs to include, e.g. pretrained finetunes")
    parser.add_argument("--plot-mode", choices=PLOT_MODES, default="eval_performance")
    args = parser.parse_args()

    results_root = args.repo_root / "outputs" / (
        "evaluation" if args.plot_mode == "eval_performance" else "eval_loss"
    )
    output_dir = args.repo_root / "outputs" / "plots"
    runs = select_runs(results_root, args.model_family, args.models)
    plot_models(runs, output_dir, args.model_family, args.plot_mode)


if __name__ == "__main__":
    main()
