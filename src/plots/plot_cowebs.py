from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import patheffects
from matplotlib.colors import to_rgb

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


def format_plot_mode_title(plot_mode: str) -> str:
    return "Loss" if plot_mode == "eval_loss" else "Performance"


def format_decimal_power_of_ten(exponent: int) -> str:
    if exponent >= 0:
        return str(10**exponent)
    return "0." + "0" * (-exponent - 1) + "1"


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


def method_colors(model_metrics: dict[str, dict[str, float]]) -> dict[str, str]:
    preferred_colors = ["#3f455f", "#7db69f", "#ef7f5f", "#f7d488", "#fff8e8"]
    extra_colors = plt.get_cmap("tab20").colors
    return {
        model_name: (
            preferred_colors[idx]
            if idx < len(preferred_colors)
            else extra_colors[(idx - len(preferred_colors)) % len(extra_colors)]
        )
        for idx, model_name in enumerate(model_metrics)
    }


def is_light_color(color: str | tuple[float, ...]) -> bool:
    red, green, blue = to_rgb(color)
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return luminance > 0.82


def line_path_effects(color: str | tuple[float, ...]) -> list[patheffects.AbstractPathEffect]:
    if not is_light_color(color):
        return []
    return [
        patheffects.Stroke(linewidth=4.0, foreground="#111111"),
        patheffects.Normal(),
    ]


def add_legend_line_outlines(legend) -> None:
    handles = getattr(legend, "legend_handles", None)
    if handles is None:
        handles = legend.legendHandles
    for handle in handles:
        color = handle.get_color()
        if is_light_color(color):
            handle.set_path_effects(line_path_effects(color))


def select_runs(results_root: Path, family: str, models: Iterable[str] | None) -> list[tuple[str, Path]]:
    if not results_root.exists():
        raise SystemExit(f"Missing results root: {results_root}")

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
        model_dir = available[name] / family
        if not model_dir.is_dir():
            raise SystemExit(f"Missing model family directory: {model_dir}")
        runs.append((name, model_dir))
    return runs


def plot_models(
    runs: list[tuple[str, Path]],
    output_dir: Path,
    family: str,
    plot_mode: str,
    log_scale: bool = False,
    ordering: str = "first",
    only_well_finetuned: bool = False,
    well_finetuned_tolerance: float = 0.1,
    radial_max: float | None = None,
) -> None:
    model_metrics = {label: collect_metrics(model_dir, plot_mode) for label, model_dir in runs}
    if only_well_finetuned:
        pretrained = model_metrics.get("pretrained")
        finetunes = model_metrics.get("finetunes")
        if pretrained is None or finetunes is None:
            raise SystemExit("--only-well-finetuned requires runs named pretrained and finetunes")
        keep = {
            task
            for task in DATASET_2_PLOT_METRICS
            if task in pretrained
            and task in finetunes
            and (
                finetunes[task] <= pretrained[task] + well_finetuned_tolerance
                if plot_mode == "eval_loss"
                else finetunes[task] >= pretrained[task] - well_finetuned_tolerance
            )
        }
        model_metrics = {
            label: {task: value for task, value in metrics.items() if task in keep}
            for label, metrics in model_metrics.items()
        }
        print(f"Keeping {len(keep)} tasks where finetunes beat pretrained")
    first_metrics = model_metrics[runs[0][0]]
    all_labels = [
        task
        for task in DATASET_2_PLOT_METRICS
        if any(task in metrics for metrics in model_metrics.values())
    ]
    if ordering == "alphabet":
        labels = sorted(all_labels)
    else:
        labels = sorted(first_metrics, key=first_metrics.get, reverse=True)
        labels += sorted(task for task in all_labels if task not in first_metrics)
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
    colors = method_colors(model_metrics)
    for model_name, metrics in model_metrics.items():
        values = [metrics.get(label, math.nan) for label in labels]
        values += values[:1]
        color = colors[model_name]
        ax.plot(
            angles,
            values,
            color=color,
            linewidth=3.0,
            label=format_model_title(model_name),
        )
        if all(math.isfinite(value) for value in values):
            ax.fill(angles, values, color=color, alpha=0.12)
        present = [
            (angle, value)
            for angle, value in zip(angles[:-1], values[:-1], strict=True)
            if math.isfinite(value)
        ]
        if present:
            present_angles, present_values = zip(*present, strict=True)
            ax.scatter(present_angles, present_values, color=color, s=28, zorder=3)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])
    if log_scale:
        positive_values = [value for value in finite_values if value > 0]
        if not positive_values:
            print("Skipping plot: log scale requires positive metrics")
            return
        if radial_max is not None and radial_max <= 0:
            raise SystemExit("--radial-max must be positive when using --log-scale")
        min_exponent = math.floor(math.log10(min(positive_values) * 0.92))
        ymax = radial_max if radial_max is not None else max(positive_values) * 1.08
        max_exponent = math.ceil(math.log10(ymax))
        ymin = 10**min_exponent
        ymax = radial_max if radial_max is not None else 10**max_exponent
        ax.set_yscale("log")
        ticks = [10**exponent for exponent in range(min_exponent, max_exponent + 1)]
        if plot_mode == "eval_performance":
            tick_labels = [
                format_decimal_power_of_ten(exponent)
                for exponent in range(min_exponent, max_exponent + 1)
            ]
        else:
            tick_labels = [rf"$10^{{{exponent}}}$" for exponent in range(min_exponent, max_exponent + 1)]
    elif plot_mode == "eval_loss":
        ymin = 0
        ymax = max(finite_values) if finite_values else 1
        ymax = ymax * 1.08 if ymax else 1
        if radial_max is not None:
            ymax = radial_max
        ticks = [ymax * frac for frac in (0.2, 0.4, 0.6, 0.8, 1.0)]
        tick_labels = [f"{tick:.2g}" for tick in ticks]
    else:
        ymin = 0
        ymax = radial_max if radial_max is not None else max(finite_values) * 1.08 if finite_values else 1
        ticks = [ymax * frac for frac in (0.2, 0.4, 0.6, 0.8, 1.0)]
        tick_labels = [f"{tick:.2g}" for tick in ticks]
    ax.set_ylim(ymin, ymax)
    ax.set_yticks(ticks)
    if log_scale:
        minor_ticks = [
            mantissa * 10**exponent
            for exponent in range(min_exponent, max_exponent)
            for mantissa in range(2, 10)
            if ymin < mantissa * 10**exponent < ymax
        ]
    else:
        minor_ticks = [
            0.5 * (left + right)
            for left, right in zip(ticks, ticks[1:])
        ]
    ax.set_yticks(minor_ticks, minor=True)
    ax.set_rlabel_position(-15)
    ax.set_yticklabels(tick_labels, color="#555555", fontsize=16)
    # for label in ax.get_yticklabels():
    #     label.set_zorder(20)
    #     label.set_bbox(dict(facecolor="white", alpha=0.65, edgecolor="none", pad=0.35))
    ax.set_yticklabels([], minor=True)
    title = f"{format_family_title(family)} {format_plot_mode_title(plot_mode)}"
    ax.set_title(rf"\textbf{{{latex_escape(title)}}}", pad=60)
    ax.spines["polar"].set_color("#777777")
    ax.spines["polar"].set_alpha(0.55)
    ax.xaxis.grid(True, color="#999999", alpha=0.28, linewidth=0.75)
    ax.yaxis.grid(True, which="major", color="#666666", alpha=0.55, linewidth=1.0)
    ax.yaxis.grid(True, which="minor", color="#999999", alpha=0.42, linewidth=0.6)
    legend = ax.legend(loc="upper left", bbox_to_anchor=(1.06, 1.02), frameon=False)
    add_legend_line_outlines(legend)

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
            ymax * 1.14,
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
    parser.add_argument("--model-family", choices=["llama3.1", "qwen2.5"], required=True, help="Model family to plot")
    parser.add_argument(
        "--models",
        nargs="+",
        help="Evaluation output directories under outputs/evaluation or outputs/eval_loss.",
    )
    parser.add_argument("--plot-mode", choices=PLOT_MODES, default="eval_performance")
    parser.add_argument("--log-scale", "--log_scale", action="store_true", help="Use a log-scaled radial axis")
    parser.add_argument("--ordering", choices=["first", "alphabet"], default="first")
    parser.add_argument("--only-well-finetuned", action="store_true")
    parser.add_argument("--well-finetuned-tolerance", type=float, default=0.1)
    parser.add_argument("--radial-max", type=float, help="Optional maximum value for the radial axis")
    args = parser.parse_args()

    results_root = args.repo_root / "outputs" / (
        "evaluation" if args.plot_mode == "eval_performance" else "eval_loss"
    )
    output_dir = args.repo_root / "outputs" / "coweb_plots"
    runs = select_runs(results_root, args.model_family, args.models)
    plot_models(
        runs,
        output_dir,
        args.model_family,
        args.plot_mode,
        args.log_scale,
        args.ordering,
        args.only_well_finetuned,
        args.well_finetuned_tolerance,
        args.radial_max,
    )


if __name__ == "__main__":
    main()
