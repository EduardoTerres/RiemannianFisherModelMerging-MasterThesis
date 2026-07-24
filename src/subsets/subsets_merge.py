"""Merge nested task subsets and summarize their evaluation results."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib-{os.getuid()}-{os.getpid()}")
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from safetensors.torch import save_file

from src.dataset.dataset_3 import DATASET_3_PLOT_LABELS, DATASET_3_PLOT_METRICS, DATASET_3_TRAIN
from src.merging import OFTMerging
from src.paths import MODEL_FAMILIES_D3_FISHER_PRETRAINED as MODEL_FAMILIES
from src.utils import parse_device

SUBSET_SIZES = [2, 4, 6, 8, 10, 12]
MERGE_MODES = ("standard_rescaled", "diagonal_fisher")
TASK_ALIASES = {"cnn_dailymail": "cnn_dailymail_abisee", "math500": "minerva_math500"}
METRIC_ALIASES = {("xsum", "rougeL,none"): "rouge,none"}
TRAIN_TO_EVAL = {"numinamath": "math500"}
ROOT = Path(__file__).resolve().parents[2]

COWEB_PLOT_STYLE = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.weight": "normal",
    "axes.labelweight": "normal",
    "axes.titleweight": "normal",
    "axes.titlesize": 30,
    "axes.labelsize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("merge", "results"), required=True)
    parser.add_argument(
        "--model-family",
        choices=MODEL_FAMILIES,
        nargs="+",
        default=["qwen2.5"],
        help="One family, or multiple families when plotting results.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--evaluation-dir", type=Path)
    return parser.parse_args()


def default_dirs(args: argparse.Namespace, model_family: str) -> tuple[Path, Path]:
    models = args.models_dir or ROOT / "outputs/models/subsets" / model_family
    evaluation = args.evaluation_dir or ROOT / "outputs/evaluation_subsets" / model_family
    return models, evaluation


def nested_indices(count: int, seed: int) -> list[int]:
    indices = list(range(count))
    random.Random(seed).shuffle(indices)
    return indices


def merge_subsets(
    args: argparse.Namespace,
    models_dir: Path,
    model_family: str,
) -> None:
    family = MODEL_FAMILIES[model_family]
    order = nested_indices(len(family.adapter_paths), args.seed)
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "subsets.json").write_text(
        json.dumps(
            {
                str(size): [Path(family.adapter_paths[i]).name for i in order[:size]]
                for size in SUBSET_SIZES
            },
            indent=2,
        )
        + "\n"
    )

    for size in SUBSET_SIZES:
        selected = order[:size]
        adapters = [family.adapter_paths[i] for i in selected]
        fishers = [family.fisher_paths[i] for i in selected]
        for mode in MERGE_MODES:
            output = models_dir / f"{mode}_{size}" / "merged_adapter"
            output.mkdir(parents=True, exist_ok=True)
            merger = OFTMerging(lam=0.0, device=parse_device(args.device))
            weights = merger.merge(
                adapter_paths=adapters,
                fisher_paths=fishers if mode == "diagonal_fisher" else None,
                mode=mode,
            )
            save_file(weights, output / "adapter_model.safetensors")
            shutil.copy(Path(adapters[0]) / "adapter_config.json", output / "adapter_config.json")
            print(f"Saved {mode} merge of {size} adapters to {output}")


def latest_result(task_dir: Path) -> Path | None:
    paths = list(task_dir.glob("**/results_*.json"))
    metrics = task_dir / "metrics.json"
    if metrics.exists():
        paths.append(metrics)
    return max(paths, key=lambda path: path.stat().st_mtime) if paths else None


def read_score(task_dir: Path, task: str) -> float | None:
    path = latest_result(task_dir)
    if path is None:
        return None
    data = json.loads(path.read_text())
    values = data.get("results", {}).get(task)
    values = values or data.get("results", {}).get(TASK_ALIASES.get(task, ""))
    values = values or data.get(task, data)
    metric, preprocess = DATASET_3_PLOT_METRICS[task]
    value = values.get(metric)
    if value is None:
        value = values.get(METRIC_ALIASES.get((task, metric), ""))
    if not isinstance(value, int | float):
        return None
    return preprocess(value) if preprocess else float(value)


def collect_results(
    evaluation_dir: Path,
    tasks_by_size: dict[int, list[str]] | None = None,
) -> list[dict[str, float | int | str]]:
    rows = []
    for size in SUBSET_SIZES:
        for mode in MERGE_MODES:
            tasks = tasks_by_size.get(size, []) if tasks_by_size else DATASET_3_PLOT_METRICS
            task_scores = {task: [] for task in tasks}
            runs = 0
            for run_dir in sorted((evaluation_dir / str(size)).glob("run_*")):
                found = False
                for task in tasks:
                    score = read_score(run_dir / mode / task, task)
                    if score is not None:
                        task_scores[task].append(score)
                        found = True
                runs += found

            dataset_means = [
                sum(scores) / len(scores) for scores in task_scores.values() if scores
            ]
            if dataset_means:
                rows.append(
                    {
                        "models": size,
                        "method": mode,
                        "average_accuracy": sum(dataset_means) / len(dataset_means),
                        "lower_quartile": np.percentile(dataset_means, 25),
                        "upper_quartile": np.percentile(dataset_means, 75),
                        "runs": runs,
                    }
                )
    return rows


def collect_dataset_distributions(
    evaluation_dir: Path,
    tasks_by_size: dict[int, list[str]] | None = None,
) -> dict[str, dict[int, list[float]]]:
    distributions = {mode: {} for mode in MERGE_MODES}
    for size in SUBSET_SIZES:
        for mode in MERGE_MODES:
            tasks = tasks_by_size.get(size, []) if tasks_by_size else DATASET_3_PLOT_METRICS
            dataset_means = []
            for task in tasks:
                scores = [
                    score
                    for run_dir in sorted((evaluation_dir / str(size)).glob("run_*"))
                    if (score := read_score(run_dir / mode / task, task)) is not None
                ]
                if scores:
                    dataset_means.append(float(np.mean(scores)))
            if dataset_means:
                distributions[mode][size] = dataset_means
    return {
        mode: values_by_size
        for mode, values_by_size in distributions.items()
        if values_by_size
    }


def collect_raw_score_distributions(
    evaluation_dir: Path,
    tasks_by_size: dict[int, list[str]] | None = None,
) -> dict[str, list[float]]:
    distributions = {mode: [] for mode in MERGE_MODES}
    for size in SUBSET_SIZES:
        for mode in MERGE_MODES:
            tasks = tasks_by_size.get(size, []) if tasks_by_size else DATASET_3_PLOT_METRICS
            for run_dir in sorted((evaluation_dir / str(size)).glob("run_*")):
                for task in tasks:
                    score = read_score(run_dir / mode / task, task)
                    if score is not None:
                        distributions[mode].append(score)
    return {
        mode: scores
        for mode, scores in distributions.items()
        if scores
    }


def collect_dataset_std_distribution(
    evaluation_dir: Path,
    tasks_by_size: dict[int, list[str]] | None = None,
) -> dict[str, list[float]]:
    distributions = {mode: [] for mode in MERGE_MODES}
    tasks = (
        sorted({task for tasks in tasks_by_size.values() for task in tasks})
        if tasks_by_size
        else DATASET_3_PLOT_METRICS
    )
    for mode in MERGE_MODES:
        for task in tasks:
            subset_means = []
            for size in SUBSET_SIZES:
                if tasks_by_size and task not in tasks_by_size.get(size, []):
                    continue
                scores = [
                    score
                    for run_dir in sorted((evaluation_dir / str(size)).glob("run_*"))
                    if (score := read_score(run_dir / mode / task, task)) is not None
                ]
                if scores:
                    subset_means.append(float(np.mean(scores)))
            if len(subset_means) > 1:
                distributions[mode].append(float(np.std(subset_means, ddof=1)))
    return {
        mode: values
        for mode, values in distributions.items()
        if values
    }


def trained_tasks(models_dir: Path, model_family: str) -> dict[int, list[str]]:
    subsets_path = models_dir / "subsets.json"
    subsets = json.loads(subsets_path.read_text())
    return trained_tasks_from_subsets(subsets, model_family)


def trained_tasks_from_subsets(
    subsets: dict[str, list[str]],
    model_family: str,
) -> dict[int, list[str]]:
    family = MODEL_FAMILIES[model_family]
    adapter_tasks = {
        Path(path).name: TRAIN_TO_EVAL.get(task, task)
        for path, (task, *_) in zip(family.adapter_paths, DATASET_3_TRAIN, strict=True)
    }
    return {
        int(size): [adapter_tasks[adapter] for adapter in adapters]
        for size, adapters in subsets.items()
    }


def trained_tasks_from_seed(model_family: str, seed: int) -> dict[int, list[str]]:
    family = MODEL_FAMILIES[model_family]
    order = nested_indices(len(family.adapter_paths), seed)
    subsets = {
        str(size): [Path(family.adapter_paths[index]).name for index in order[:size]]
        for size in SUBSET_SIZES
    }
    return trained_tasks_from_subsets(subsets, model_family)


METHOD_DISPLAY = {
    "standard_rescaled": r"\textsc{OrthoMerge}",
    "diagonal_fisher": r"\textsc{Diagonal Fisher}",
}
METHOD_COLORS = {
    "standard_rescaled": "black",
    "diagonal_fisher": "#ef7f5f",
}
FAMILY_DISPLAY = {
    "qwen2.5": "Qwen 2.5",
    "llama3.1": "Llama 3.1",
}
FAMILY_LINESTYLES = {
    "qwen2.5": "-",
    "llama3.1": "--",
}
FAMILY_MARKERS = {
    "qwen2.5": "o",
    "llama3.1": "s",
}


def save_dataset_distribution_plot(
    rows: list[dict[str, float | int | str]],
    distributions: dict[str, dict[int, list[float]]],
    raw_distributions: dict[str, list[float]],
    evaluation_dir: Path,
    plot_dir: Path,
    csv_name: str,
    plot_name: str,
    ylabel: str,
) -> None:
    if not rows:
        raise FileNotFoundError(f"No evaluation results found under {evaluation_dir}")

    print(f"{'Models':>6}  {'Method':<20}  {'Average':>8}  {'Runs':>4}")
    for row in rows:
        print(
            f"{row['models']:>6}  {row['method']:<20}  "
            f"{row['average_accuracy']:>8.4f}  {row['runs']:>4}"
        )

    evaluation_dir.mkdir(parents=True, exist_ok=True)
    with (evaluation_dir / csv_name).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    with plt.rc_context(COWEB_PLOT_STYLE):
        fig, (ax, violin_ax) = plt.subplots(
            1,
            2,
            figsize=(11.8, 5.0),
            gridspec_kw={"width_ratios": [3.0, 1.0]},
            constrained_layout=True,
        )

        for mode in MERGE_MODES:
            selected = [row for row in rows if row["method"] == mode]
            x = [int(row["models"]) for row in selected]
            means = [float(row["average_accuracy"]) for row in selected]
            stds = [
                (
                    float(np.std(distributions.get(mode, {}).get(size, []), ddof=1))
                    if len(distributions.get(mode, {}).get(size, [])) > 1
                    else 0.0
                )
                for size in x
            ]
            if not x:
                continue

            ax.errorbar(
                x,
                means,
                yerr=stds,
                color=METHOD_COLORS[mode],
                marker="o",
                markersize=7.0,
                linewidth=3.0,
                elinewidth=2.0,
                capsize=4.5,
                capthick=2.0,
                label=METHOD_DISPLAY[mode],
            )

        violin_values = [
            raw_distributions.get(mode, [])
            for mode in MERGE_MODES
        ]
        non_empty_violin_values = [
            values
            for values in violin_values
            if values
        ]
        violin_positions = [
            index
            for index, values in enumerate(violin_values, start=1)
            if values
        ]
        if non_empty_violin_values:
            violin_modes = [
                mode
                for mode, values in zip(MERGE_MODES, violin_values, strict=True)
                if values
            ]
            parts = violin_ax.violinplot(
                non_empty_violin_values,
                positions=violin_positions,
                widths=0.74,
                showmeans=True,
                showmedians=False,
                showextrema=False,
            )
            for body, mode in zip(parts["bodies"], violin_modes, strict=True):
                color = METHOD_COLORS[mode]
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(0.42)
                body.set_linewidth(1.8)
            parts["cmeans"].set_color("black")
            parts["cmeans"].set_linewidth(2.4)

        ax.set_xlabel(r"Number of models merged")
        ax.set_ylabel(ylabel)
        ax.set_xticks(SUBSET_SIZES)
        ax.set_xticklabels([rf"{size}" for size in SUBSET_SIZES])
        ax.set_ylim(-0.03, 1.05)
        yticks = np.linspace(0.0, 1.0, 6)
        ax.set_yticks(yticks)
        ax.set_yticklabels([rf"{tick:.1f}" for tick in yticks])
        ax.grid(axis="y", color="#666666", alpha=0.32, linewidth=0.85)
        ax.grid(axis="x", color="#999999", alpha=0.18, linewidth=0.65)
        legend = ax.legend(
            loc="lower right",
            frameon=True,
            fancybox=False,
            framealpha=0.78,
            facecolor="white",
            edgecolor="0.4",
            handlelength=1.2,
            borderpad=0.4,
            prop={"weight": "normal", "size": 18},
        )
        legend_handles = getattr(legend, "legend_handles", None)
        if legend_handles is None:
            legend_handles = legend.legendHandles
        for handle in legend_handles:
            handle.set_linewidth(4.0)

        violin_ax.set_xticks(
            [1, 2],
            [METHOD_DISPLAY[mode] for mode in MERGE_MODES],
            rotation=18,
            ha="right",
        )
        violin_ax.set_title(r"All accuracies")
        violin_ax.set_ylim(-0.03, 1.05)
        violin_ax.set_yticks(yticks)
        violin_ax.grid(axis="y", color="#666666", alpha=0.32, linewidth=0.85)
        violin_ax.tick_params(axis="y", labelleft=False)

        plot_dir.mkdir(parents=True, exist_ok=True)
        save_path = plot_dir / plot_name
        pdf_path = save_path.with_suffix(".pdf")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved {save_path} and {pdf_path}")


def save_combined_subset_plot(
    evaluation_dirs_by_family: dict[str, Path],
) -> None:
    rows_by_family = {
        family: collect_results(evaluation_dir)
        for family, evaluation_dir in evaluation_dirs_by_family.items()
    }
    stds_by_method = {mode: [] for mode in MERGE_MODES}
    for evaluation_dir in evaluation_dirs_by_family.values():
        family_stds = collect_dataset_std_distribution(evaluation_dir)
        for mode, values in family_stds.items():
            stds_by_method[mode].extend(values)
    stds_by_method = {
        mode: values
        for mode, values in stds_by_method.items()
        if values
    }

    if not any(rows_by_family.values()):
        print("Skipping combined subset plot: no subset results available.")
        return

    with plt.rc_context(COWEB_PLOT_STYLE):
        fig, (ax, violin_ax) = plt.subplots(
            1,
            2,
            figsize=(13.0, 6.5),
            gridspec_kw={"width_ratios": [3.0, 1.0]},
            constrained_layout=True,
        )

        curve_values = []
        for family, rows in rows_by_family.items():
            for mode in MERGE_MODES:
                selected = [row for row in rows if row["method"] == mode]
                if not selected:
                    continue
                x = [int(row["models"]) for row in selected]
                means = [float(row["average_accuracy"]) for row in selected]
                curve_values.extend(means)
                ax.plot(
                    x,
                    means,
                    color=METHOD_COLORS[mode],
                    linestyle=FAMILY_LINESTYLES.get(family, "-"),
                    marker=FAMILY_MARKERS.get(family, "o"),
                    markersize=7.0,
                    linewidth=3.0,
                    label=rf"{FAMILY_DISPLAY.get(family, family)} {METHOD_DISPLAY[mode]}",
                )

        violin_values = [
            stds_by_method.get(mode, [])
            for mode in MERGE_MODES
        ]
        non_empty_violin_values = [values for values in violin_values if values]
        violin_positions = [
            index
            for index, values in enumerate(violin_values, start=1)
            if values
        ]
        if non_empty_violin_values:
            violin_modes = [
                mode
                for mode, values in zip(MERGE_MODES, violin_values, strict=True)
                if values
            ]
            parts = violin_ax.violinplot(
                non_empty_violin_values,
                positions=violin_positions,
                widths=0.74,
                showmeans=True,
                showmedians=False,
                showextrema=False,
            )
            for body, mode in zip(parts["bodies"], violin_modes, strict=True):
                color = METHOD_COLORS[mode]
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(0.42)
                body.set_linewidth(1.8)
            parts["cmeans"].set_color("black")
            parts["cmeans"].set_linewidth(2.4)

        ax.set_title(r"Accuracy", fontsize=27, pad=10)
        ax.set_xlabel(r"Number of models merged")
        ax.set_ylabel(r"Average accuracy")
        ax.set_xticks(SUBSET_SIZES)
        ax.set_xticklabels([rf"{size}" for size in SUBSET_SIZES])
        ymin = min(curve_values) if curve_values else 0.0
        ymax = max(curve_values) if curve_values else 1.0
        padding = max(0.025, (ymax - ymin) * 0.18)
        ymin = max(0.0, ymin - padding)
        ymax = min(1.0, ymax + padding)
        ax.set_ylim(ymin, ymax)
        yticks = np.linspace(ymin, ymax, 5)
        ax.set_yticks(yticks)
        ax.set_yticklabels([rf"{tick:.2f}" for tick in yticks])
        ax.grid(axis="y", color="#666666", alpha=0.32, linewidth=0.85)
        ax.grid(axis="x", color="#999999", alpha=0.18, linewidth=0.65)
        legend = ax.legend(
            loc="lower right",
            frameon=True,
            fancybox=False,
            framealpha=0.78,
            facecolor="white",
            edgecolor="0.4",
            handlelength=1.4,
            borderpad=0.4,
            prop={"weight": "normal", "size": 17},
        )
        legend_handles = getattr(legend, "legend_handles", None)
        if legend_handles is None:
            legend_handles = legend.legendHandles
        for handle in legend_handles:
            handle.set_linewidth(4.0)

        violin_ax.set_xticks(
            [1, 2],
            [METHOD_DISPLAY[mode] for mode in MERGE_MODES],
            rotation=18,
            ha="right",
        )
        violin_ax.set_title(r"Accuracy change ($\downarrow$)", fontsize=27, pad=10)
        violin_ax.set_ylabel(r"Std. accuracy across subset sizes")
        std_ymax = max(
            0.12,
            max((max(values) for values in stds_by_method.values()), default=0.1) * 1.12,
        )
        violin_ax.set_ylim(-0.01, std_ymax)
        std_ticks = np.linspace(0.0, std_ymax, 6)
        violin_ax.set_yticks(std_ticks)
        violin_ax.set_yticklabels([rf"{tick:.2f}" for tick in std_ticks])
        violin_ax.grid(axis="y", color="#666666", alpha=0.32, linewidth=0.85)

        plot_dir = ROOT / "outputs/subsets/combined"
        plot_dir.mkdir(parents=True, exist_ok=True)
        png_path = plot_dir / "subset_combined.png"
        pdf_path = plot_dir / "subset_combined.pdf"
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved {png_path} and {pdf_path}")


def save_combined_only_included_plot(
    rows_by_family: dict[str, list[dict[str, float | int | str]]],
    evaluation_dirs_by_family: dict[str, Path],
    tasks_by_size_by_family: dict[str, dict[int, list[str]]],
) -> None:
    rows_by_family = {
        family: rows
        for family, rows in rows_by_family.items()
        if rows
    }
    if not rows_by_family:
        print("Skipping combined included-dataset plot: no included-dataset rows available.")
        return

    stds_by_method = {mode: [] for mode in MERGE_MODES}
    for family, evaluation_dir in evaluation_dirs_by_family.items():
        family_stds = collect_dataset_std_distribution(
            evaluation_dir,
            tasks_by_size_by_family[family],
        )
        for mode, values in family_stds.items():
            stds_by_method[mode].extend(values)
    stds_by_method = {
        mode: values
        for mode, values in stds_by_method.items()
        if values
    }

    with plt.rc_context(COWEB_PLOT_STYLE):
        fig, (ax, violin_ax) = plt.subplots(
            1,
            2,
            figsize=(13.0, 6.5),
            gridspec_kw={"width_ratios": [3.0, 1.0]},
            constrained_layout=True,
        )

        curve_values = []
        for family, rows in rows_by_family.items():
            for mode in MERGE_MODES:
                selected = [row for row in rows if row["method"] == mode]
                if not selected:
                    continue
                x = [int(row["models"]) for row in selected]
                means = [float(row["average_accuracy"]) for row in selected]
                curve_values.extend(means)
                ax.plot(
                    x,
                    means,
                    color=METHOD_COLORS[mode],
                    linestyle=FAMILY_LINESTYLES.get(family, "-"),
                    marker=FAMILY_MARKERS.get(family, "o"),
                    markersize=7.0,
                    linewidth=3.0,
                    label=rf"{FAMILY_DISPLAY.get(family, family)} {METHOD_DISPLAY[mode]}",
                )

        violin_values = [
            stds_by_method.get(mode, [])
            for mode in MERGE_MODES
        ]
        non_empty_violin_values = [values for values in violin_values if values]
        violin_positions = [
            index
            for index, values in enumerate(violin_values, start=1)
            if values
        ]
        if non_empty_violin_values:
            violin_modes = [
                mode
                for mode, values in zip(MERGE_MODES, violin_values, strict=True)
                if values
            ]
            parts = violin_ax.violinplot(
                non_empty_violin_values,
                positions=violin_positions,
                widths=0.74,
                showmeans=True,
                showmedians=False,
                showextrema=False,
            )
            for body, mode in zip(parts["bodies"], violin_modes, strict=True):
                color = METHOD_COLORS[mode]
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(0.42)
                body.set_linewidth(1.8)
            parts["cmeans"].set_color("black")
            parts["cmeans"].set_linewidth(2.4)

        ax.set_title(r"Included-task accuracy", fontsize=27, pad=10)
        ax.set_xlabel(r"Number of models merged")
        ax.set_ylabel(r"Average accuracy on included datasets")
        ax.set_xticks(SUBSET_SIZES)
        ax.set_xticklabels([rf"{size}" for size in SUBSET_SIZES])
        ymin = min(curve_values) if curve_values else 0.0
        ymax = max(curve_values) if curve_values else 1.0
        padding = max(0.025, (ymax - ymin) * 0.18)
        ymin = max(0.0, ymin - padding)
        ymax = min(1.0, ymax + padding)
        ax.set_ylim(ymin, ymax)
        yticks = np.linspace(ymin, ymax, 5)
        ax.set_yticks(yticks)
        ax.set_yticklabels([rf"{tick:.2f}" for tick in yticks])
        ax.grid(axis="y", color="#666666", alpha=0.32, linewidth=0.85)
        ax.grid(axis="x", color="#999999", alpha=0.18, linewidth=0.65)
        legend = ax.legend(
            loc="lower right",
            frameon=True,
            fancybox=False,
            framealpha=0.78,
            facecolor="white",
            edgecolor="0.4",
            handlelength=1.4,
            borderpad=0.4,
            prop={"weight": "normal", "size": 17},
        )
        legend_handles = getattr(legend, "legend_handles", None)
        if legend_handles is None:
            legend_handles = legend.legendHandles
        for handle in legend_handles:
            handle.set_linewidth(4.0)

        violin_ax.set_xticks(
            [1, 2],
            [METHOD_DISPLAY[mode] for mode in MERGE_MODES],
            rotation=18,
            ha="right",
        )
        violin_ax.set_title(r"Accuracy change ($\downarrow$)", fontsize=27, pad=10)
        violin_ax.set_ylabel(r"Std. included-task accuracy")
        std_ymax = max(
            0.12,
            max((max(values) for values in stds_by_method.values()), default=0.1) * 1.12,
        )
        violin_ax.set_ylim(-0.01, std_ymax)
        std_ticks = np.linspace(0.0, std_ymax, 6)
        violin_ax.set_yticks(std_ticks)
        violin_ax.set_yticklabels([rf"{tick:.2f}" for tick in std_ticks])
        violin_ax.grid(axis="y", color="#666666", alpha=0.32, linewidth=0.85)

        plot_dir = ROOT / "outputs/subsets/combined"
        plot_dir.mkdir(parents=True, exist_ok=True)
        png_path = plot_dir / "subset_combined_only_included.png"
        pdf_path = plot_dir / "subset_combined_only_included.pdf"
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved {png_path} and {pdf_path}")


def save_report(
    rows: list[dict[str, float | int | str]],
    evaluation_dir: Path,
    plot_dir: Path,
    csv_name: str,
    plot_name: str,
    ylabel: str,
) -> None:
    if not rows:
        raise FileNotFoundError(f"No evaluation results found under {evaluation_dir}")

    print(f"{'Models':>6}  {'Method':<20}  {'Average':>8}  {'Runs':>4}")
    for row in rows:
        print(
            f"{row['models']:>6}  {row['method']:<20}  "
            f"{row['average_accuracy']:>8.4f}  {row['runs']:>4}"
        )

    evaluation_dir.mkdir(parents=True, exist_ok=True)
    with (evaluation_dir / csv_name).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    for mode in MERGE_MODES:
        selected = [row for row in rows if row["method"] == mode]
        x = [row["models"] for row in selected]
        mean = [row["average_accuracy"] for row in selected]
        lower = [row["lower_quartile"] for row in selected]
        upper = [row["upper_quartile"] for row in selected]
        plt.plot(
            x,
            mean,
            marker="o",
            label=mode,
        )
        plt.fill_between(
            x,
            lower,
            upper,
            alpha=0.2,
        )
    plt.xlabel("Number of models merged")
    plt.ylabel(ylabel)
    plt.xticks(list(SUBSET_SIZES))
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plot_dir.mkdir(parents=True, exist_ok=True)
    save_path = plot_dir / plot_name
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved {save_path}")


def save_dataset_metrics_plot(
    evaluation_dir: Path,
    plot_dir: Path,
    tasks_by_size: dict[int, list[str]],
    model_family: str,
) -> None:
    with plt.rc_context(COWEB_PLOT_STYLE):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(17.0, 6.5),
            sharey=True,
            constrained_layout=True,
        )
        colors = plt.cm.tab20(np.linspace(0, 1, len(DATASET_3_PLOT_METRICS)))

        for ax, mode in zip(axes, MERGE_MODES, strict=True):
            for color, task in zip(colors, DATASET_3_PLOT_METRICS, strict=True):
                metrics = []
                for size in SUBSET_SIZES:
                    scores = [
                        score
                        for run_dir in sorted((evaluation_dir / str(size)).glob("run_*"))
                        if (score := read_score(run_dir / mode / task, task)) is not None
                    ]
                    metrics.append(np.mean(scores) if scores else np.nan)

                ax.plot(
                    SUBSET_SIZES,
                    metrics,
                    color=color,
                    linewidth=2.4,
                    label=DATASET_3_PLOT_LABELS[task],
                )
                for size, metric in zip(SUBSET_SIZES, metrics, strict=True):
                    marker = "^" if task in tasks_by_size[size] else "o"
                    ax.scatter(
                        size,
                        metric,
                        marker=marker,
                        color=color,
                        edgecolor="black",
                        linewidth=0.45,
                        s=54,
                        zorder=3,
                    )

            ax.set_title(METHOD_DISPLAY[mode])
            ax.set_xlabel(r"Number of models merged")
            ax.set_xticks(SUBSET_SIZES)
            ax.set_xticklabels([rf"{size}" for size in SUBSET_SIZES])
            ax.set_ylim(-0.03, 1.05)
            yticks = np.linspace(0.0, 1.0, 6)
            ax.set_yticks(yticks)
            ax.set_yticklabels([rf"{tick:.1f}" for tick in yticks])
            ax.grid(axis="y", color="#666666", alpha=0.32, linewidth=0.85)
            ax.grid(axis="x", color="#999999", alpha=0.18, linewidth=0.65)

        axes[0].set_ylabel(r"Evaluation metric")
        dataset_handles, dataset_labels = axes[1].get_legend_handles_labels()
        marker_handles = [
            Line2D(
                [],
                [],
                color="black",
                marker="^",
                linestyle="None",
                markersize=8,
                label=r"Included",
            ),
            Line2D(
                [],
                [],
                color="black",
                marker="o",
                linestyle="None",
                markersize=8,
                label=r"Not included",
            ),
        ]
        legend = fig.legend(
            dataset_handles + marker_handles,
            dataset_labels + [r"Included", r"Not included"],
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            frameon=False,
            handlelength=1.4,
            borderpad=0.4,
            prop={"weight": "normal", "size": 17},
        )
        legend_handles = getattr(legend, "legend_handles", None)
        if legend_handles is None:
            legend_handles = legend.legendHandles
        for handle in legend_handles:
            if hasattr(handle, "set_linewidth"):
                handle.set_linewidth(3.0)

        plot_dir.mkdir(parents=True, exist_ok=True)
        family_suffix = "llama" if model_family.startswith("llama") else "qwen"
        save_path = plot_dir / f"subset_dataset_metrics_{family_suffix}.png"
        pdf_path = save_path.with_suffix(".pdf")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved {save_path} and {pdf_path}")


def report_results(
    evaluation_dir: Path,
    models_dir: Path,
    model_family: str,
    seed: int,
) -> tuple[list[dict[str, float | int | str]], dict[int, list[str]]]:
    plot_dir = ROOT / "outputs/subsets" / model_family
    rows = collect_results(evaluation_dir)
    save_dataset_distribution_plot(
        rows,
        collect_dataset_distributions(evaluation_dir),
        collect_raw_score_distributions(evaluation_dir),
        evaluation_dir,
        plot_dir,
        "subset_results.csv",
        "subset_accuracy.png",
        "Average accuracy",
    )
    subsets_path = models_dir / "subsets.json"
    if subsets_path.exists():
        tasks_by_size = trained_tasks(models_dir, model_family)
    else:
        print(
            f"Missing subset metadata at {subsets_path}; "
            f"reconstructing trained-task subsets from seed {seed}."
        )
        tasks_by_size = trained_tasks_from_seed(model_family, seed)

    trained_rows = collect_results(evaluation_dir, tasks_by_size)
    save_report(
        trained_rows,
        evaluation_dir,
        plot_dir,
        "subset_trained_results.csv",
        "subset_trained_accuracy.png",
        "Average accuracy on trained datasets",
    )
    save_dataset_metrics_plot(evaluation_dir, plot_dir, tasks_by_size, model_family)
    return trained_rows, tasks_by_size


def save_combined_trained_plot(
    rows_by_family: dict[str, list[dict[str, float | int | str]]],
) -> None:
    rows_by_family = {
        family: rows
        for family, rows in rows_by_family.items()
        if rows
    }
    if not rows_by_family:
        print("Skipping combined trained-dataset plot: no trained-dataset rows available.")
        return

    mode_colors = dict(zip(MERGE_MODES, ("tab:blue", "tab:orange"), strict=True))
    markers = ("o", "^", "s", "D")

    for family_index, (family, rows) in enumerate(rows_by_family.items()):
        for mode in MERGE_MODES:
            selected = [row for row in rows if row["method"] == mode]
            x = [row["models"] for row in selected]
            mean = [row["average_accuracy"] for row in selected]
            lower = [row["lower_quartile"] for row in selected]
            upper = [row["upper_quartile"] for row in selected]
            color = mode_colors[mode]
            plt.plot(
                x,
                mean,
                color=color,
                marker=markers[family_index],
                label=f"{family} — {mode}",
            )
            plt.fill_between(x, lower, upper, color=color, alpha=0.1)

    plt.xlabel("Number of models merged")
    plt.ylabel("Average accuracy on trained datasets")
    plt.xticks(list(SUBSET_SIZES))
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plot_dir = ROOT / "outputs/subsets/combined"
    plot_dir.mkdir(parents=True, exist_ok=True)
    save_path = plot_dir / "subset_trained_accuracy.png"
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved {save_path}")


def main() -> None:
    args = parse_args()
    if args.phase == "merge":
        if len(args.model_family) != 1:
            raise ValueError("The merge phase accepts exactly one --model-family.")
        model_family = args.model_family[0]
        models_dir, _ = default_dirs(args, model_family)
        merge_subsets(args, models_dir, model_family)
    else:
        if len(args.model_family) > 1 and (args.models_dir or args.evaluation_dir):
            raise ValueError(
                "--models-dir and --evaluation-dir cannot be used with multiple families."
            )
        rows_by_family = {}
        evaluation_dirs_by_family = {}
        tasks_by_size_by_family = {}
        for model_family in args.model_family:
            models_dir, evaluation_dir = default_dirs(args, model_family)
            evaluation_dirs_by_family[model_family] = evaluation_dir
            rows_by_family[model_family], tasks_by_size_by_family[model_family] = report_results(
                evaluation_dir,
                models_dir,
                model_family,
                args.seed,
            )
        if len(rows_by_family) > 1:
            save_combined_subset_plot(evaluation_dirs_by_family)
            save_combined_only_included_plot(
                rows_by_family,
                evaluation_dirs_by_family,
                tasks_by_size_by_family,
            )
            save_combined_trained_plot(rows_by_family)


if __name__ == "__main__":
    main()
