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


def trained_tasks(models_dir: Path, model_family: str) -> dict[int, list[str]]:
    subsets = json.loads((models_dir / "subsets.json").read_text())
    family = MODEL_FAMILIES[model_family]
    adapter_tasks = {
        Path(path).name: TRAIN_TO_EVAL.get(task, task)
        for path, (task, *_) in zip(family.adapter_paths, DATASET_3_TRAIN, strict=True)
    }
    return {
        int(size): [adapter_tasks[adapter] for adapter in adapters]
        for size, adapters in subsets.items()
    }


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
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True, constrained_layout=True)
    colors = plt.cm.tab20(np.linspace(0, 1, len(DATASET_3_PLOT_METRICS)))

    for ax, mode in zip(axes, MERGE_MODES):
        for color, task in zip(colors, DATASET_3_PLOT_METRICS):
            metrics = []
            for size in SUBSET_SIZES:
                scores = [
                    score
                    for run_dir in sorted((evaluation_dir / str(size)).glob("run_*"))
                    if (score := read_score(run_dir / mode / task, task)) is not None
                ]
                metrics.append(np.mean(scores) if scores else np.nan)

            ax.plot(SUBSET_SIZES, metrics, color=color, label=DATASET_3_PLOT_LABELS[task])
            for size, metric in zip(SUBSET_SIZES, metrics):
                marker = "^" if task in tasks_by_size[size] else "o"
                ax.scatter(size, metric, marker=marker, color=color, zorder=3)

        ax.set(title=mode, xlabel="Number of models merged", xticks=SUBSET_SIZES)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Normalized evaluation metric")
    dataset_handles, dataset_labels = axes[1].get_legend_handles_labels()
    marker_handles = [
        Line2D([], [], color="black", marker="^", linestyle="None", label="Included"),
        Line2D([], [], color="black", marker="o", linestyle="None", label="Not included"),
    ]
    fig.legend(
        dataset_handles + marker_handles,
        dataset_labels + ["Included", "Not included"],
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )
    plot_dir.mkdir(parents=True, exist_ok=True)
    save_path = plot_dir / "subset_dataset_metrics.png"
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def report_results(
    evaluation_dir: Path,
    models_dir: Path,
    model_family: str,
) -> list[dict[str, float | int | str]]:
    plot_dir = ROOT / "outputs/subsets" / model_family
    save_report(
        collect_results(evaluation_dir),
        evaluation_dir,
        plot_dir,
        "subset_results.csv",
        "subset_accuracy.png",
        "Average accuracy",
    )
    trained_rows = collect_results(evaluation_dir, trained_tasks(models_dir, model_family))
    save_report(
        trained_rows,
        evaluation_dir,
        plot_dir,
        "subset_trained_results.csv",
        "subset_trained_accuracy.png",
        "Average accuracy on trained datasets",
    )
    save_dataset_metrics_plot(evaluation_dir, plot_dir, trained_tasks(models_dir, model_family))
    return trained_rows


def save_combined_trained_plot(
    rows_by_family: dict[str, list[dict[str, float | int | str]]],
) -> None:
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
        for model_family in args.model_family:
            models_dir, evaluation_dir = default_dirs(args, model_family)
            rows_by_family[model_family] = report_results(
                evaluation_dir,
                models_dir,
                model_family,
            )
        if len(rows_by_family) > 1:
            save_combined_trained_plot(rows_by_family)


if __name__ == "__main__":
    main()
