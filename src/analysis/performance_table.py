from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

TMPDIR = Path(os.environ.get("TMPDIR", "/tmp"))
os.environ.setdefault("MPLCONFIGDIR", str(TMPDIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(TMPDIR / "xdg-cache"))

from src.plots.plot_cowebs import collect_metrics, read_metric
from dataset.dataset_3 import DATASET_3_PLOT_LABELS, DATASET_3_PLOT_METRICS


FAMILIES = ("llama3.1", "qwen2.5")
PLOT_MODE_ROOTS = {
    "eval_performance": "evaluation",
    "eval_loss": "eval_loss",
    "perplexity": "perplexity",
}
TASKS = tuple(DATASET_3_PLOT_METRICS)


@dataclass(frozen=True)
class Summary:
    mean: float
    std: float
    n: int


def summarize(values: list[float]) -> Summary:
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return Summary(math.nan, math.nan, 0)
    return Summary(
        statistics.fmean(finite_values),
        statistics.pstdev(finite_values),
        len(finite_values),
    )


def collect_summary(repo_root: Path, family: str, method: str, plot_mode: str) -> Summary:
    model_dir = repo_root / "outputs" / PLOT_MODE_ROOTS[plot_mode] / method / family
    metrics = collect_metrics(model_dir, plot_mode)
    return summarize(list(metrics.values()))


def collect_metric_summary(
    repo_root: Path, family: str, method: str, output_key: str, metric: str
) -> Summary:
    model_dir = repo_root / "outputs" / PLOT_MODE_ROOTS[output_key] / method / family
    values = [
        read_metric(model_dir, task, metric)
        for task in TASKS
    ]
    return summarize_optional(values)


def collect_perplexity_summary(repo_root: Path, family: str, method: str) -> Summary:
    summary = collect_metric_summary(repo_root, family, method, "perplexity", "perplexity")
    if summary.n > 0:
        return summary
    return collect_metric_summary(repo_root, family, method, "eval_loss", "perplexity")


def collect_performance(repo_root: Path, family: str, method: str) -> dict[str, float]:
    model_dir = repo_root / "outputs" / PLOT_MODE_ROOTS["eval_performance"] / method / family
    return collect_metrics(model_dir, "eval_performance")


def format_summary(summary: Summary, decimals: int) -> str:
    if summary.n == 0:
        return "n/a"
    return f"{summary.mean:.{decimals}f} ± {summary.std:.{decimals}f}"


def format_value(value: float | None, decimals: int) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.{decimals}f}"


def summarize_optional(values: list[float | None]) -> Summary:
    return summarize(
        [value for value in values if value is not None and math.isfinite(value)]
    )


def family_label(family: str) -> str:
    return {"llama3.1": "llama", "qwen2.5": "Qwen"}.get(family, family)


def parse_method_spec(spec: str) -> tuple[str, str]:
    method, separator, label = spec.partition("=")
    method = method.strip()
    label = label.strip()
    if not method:
        raise argparse.ArgumentTypeError("method specs must start with a method name")
    return method, label if separator else method


def build_rows(
    repo_root: Path, methods: list[tuple[str, str]]
) -> list[tuple[str, Summary, Summary, Summary]]:
    rows = []
    for family in FAMILIES:
        for method, method_label in methods:
            label = f"{family_label(family)} {method_label}"
            performance = collect_summary(repo_root, family, method, "eval_performance")
            loss = collect_summary(repo_root, family, method, "eval_loss")
            perplexity = collect_perplexity_summary(repo_root, family, method)
            rows.append((label, performance, loss, perplexity))
    return rows


def build_performance_rows(
    repo_root: Path, methods: list[tuple[str, str]]
) -> list[tuple[str, dict[str, float]]]:
    rows = []
    for family in FAMILIES:
        for method, method_label in methods:
            label = f"{family_label(family)} {method_label}"
            rows.append((label, collect_performance(repo_root, family, method)))
    return rows


def print_markdown(rows: list[tuple[str, Summary, Summary, Summary]], decimals: int) -> None:
    print("| method | mean ± std performance | mean ± std eval loss | mean ± std perplexity |")
    print("|---|---:|---:|---:|")
    for label, performance, loss, perplexity in rows:
        print(
            f"| {label} | {format_summary(performance, decimals)} | "
            f"{format_summary(loss, decimals)} | "
            f"{format_summary(perplexity, decimals)} |"
        )


def print_performance_markdown(
    rows: list[tuple[str, dict[str, float]]], decimals: int
) -> None:
    columns = [f"({idx})" for idx in range(1, len(TASKS) + 1)]
    print("| method | " + " | ".join(columns) + " | avg |")
    print("|---" + "|---:" * (len(TASKS) + 1) + "|")
    for label, metrics in rows:
        task_values = [metrics.get(task) for task in TASKS]
        values = [format_value(value, decimals) for value in task_values]
        values.append(format_summary(summarize_optional(task_values), decimals))
        print(f"| {label} | " + " | ".join(values) + " |")
    print()
    print(
        "Legend: "
        + "; ".join(
            f"({idx}) {DATASET_3_PLOT_LABELS.get(task, task)}"
            for idx, task in enumerate(TASKS, start=1)
        )
    )


def latex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def print_latex(rows: list[tuple[str, Summary, Summary, Summary]], decimals: int) -> None:
    print(r"\begin{tabular}{lccc}")
    print(r"\toprule")
    print(r"Method & Performance & Eval loss & Perplexity \\")
    print(r"\midrule")
    for label, performance, loss, perplexity in rows:
        print(
            f"{latex_escape(label)} & "
            f"{format_summary(performance, decimals)} & "
            f"{format_summary(loss, decimals)} & "
            f"{format_summary(perplexity, decimals)} \\\\"
        )
    print(r"\bottomrule")
    print(r"\end{tabular}")


def print_performance_latex(
    rows: list[tuple[str, dict[str, float]]], decimals: int
) -> None:
    print(r"\begin{tabular}{l" + "c" * (len(TASKS) + 1) + "}")
    print(r"\toprule")
    print(
        "Method & "
        + " & ".join(f"({idx})" for idx in range(1, len(TASKS) + 1))
        + r" & avg \\"
    )
    print(r"\midrule")
    for label, metrics in rows:
        task_values = [metrics.get(task) for task in TASKS]
        values = [format_value(value, decimals) for value in task_values]
        values.append(format_summary(summarize_optional(task_values), decimals))
        print(f"{latex_escape(label)} & " + " & ".join(values) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print()
    print(
        "Legend: "
        + "; ".join(
            f"({idx}) {latex_escape(DATASET_3_PLOT_LABELS.get(task, task))}"
            for idx, task in enumerate(TASKS, start=1)
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Print the mean ± std performance and eval-loss table using the same "
            "metrics collected by src.plots.plot_cowebs."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--decimals", type=int, default=3)
    parser.add_argument("--format", choices=["markdown", "latex"], default="markdown")
    parser.add_argument(
        "--methods",
        nargs="+",
        type=parse_method_spec,
        required=True,
        metavar="METHOD[=LABEL]",
        help="Methods to include, optionally with display labels.",
    )
    args = parser.parse_args()

    rows = build_rows(args.repo_root, args.methods)
    performance_rows = build_performance_rows(args.repo_root, args.methods)
    if args.format == "latex":
        print_latex(rows, args.decimals)
        print()
        print_performance_latex(performance_rows, args.decimals)
    else:
        print_markdown(rows, args.decimals)
        print()
        print_performance_markdown(performance_rows, args.decimals)


if __name__ == "__main__":
    main()
