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

TMPDIR = Path(os.environ.get("TMPDIR", "/tmp"))
os.environ.setdefault("MPLCONFIGDIR", str(TMPDIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(TMPDIR / "xdg-cache"))

from src.plots.plot_cowebs import collect_metrics


FAMILIES = ("llama3.1", "qwen2.5")
METHODS = (
    ("standard_rescaled", "OrthoMerge"),
    ("diagonal_fisher", "diagonal fisher"),
)
PLOT_MODE_ROOTS = {
    "eval_performance": "evaluation",
    "eval_loss": "eval_loss",
}


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


def format_summary(summary: Summary, decimals: int) -> str:
    if summary.n == 0:
        return "n/a"
    return f"{summary.mean:.{decimals}f} ± {summary.std:.{decimals}f}"


def family_label(family: str) -> str:
    return {"llama3.1": "llama", "qwen2.5": "Qwen"}.get(family, family)


def build_rows(repo_root: Path) -> list[tuple[str, Summary, Summary]]:
    rows = []
    for family in FAMILIES:
        for method, method_label in METHODS:
            label = f"{family_label(family)} {method_label}"
            performance = collect_summary(repo_root, family, method, "eval_performance")
            loss = collect_summary(repo_root, family, method, "eval_loss")
            rows.append((label, performance, loss))
    return rows


def print_markdown(rows: list[tuple[str, Summary, Summary]], decimals: int) -> None:
    print("| method | mean ± std performance | mean ± std eval loss |")
    print("|---|---:|---:|")
    for label, performance, loss in rows:
        print(
            f"| {label} | {format_summary(performance, decimals)} | "
            f"{format_summary(loss, decimals)} |"
        )


def latex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def print_latex(rows: list[tuple[str, Summary, Summary]], decimals: int) -> None:
    print(r"\begin{tabular}{lcc}")
    print(r"\toprule")
    print(r"Method & Performance & Eval loss \\")
    print(r"\midrule")
    for label, performance, loss in rows:
        print(
            f"{latex_escape(label)} & "
            f"{format_summary(performance, decimals)} & "
            f"{format_summary(loss, decimals)} \\\\"
        )
    print(r"\bottomrule")
    print(r"\end{tabular}")


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
    args = parser.parse_args()

    rows = build_rows(args.repo_root)
    if args.format == "latex":
        print_latex(rows, args.decimals)
    else:
        print_markdown(rows, args.decimals)


if __name__ == "__main__":
    main()
