"""Compare merged adapters under transported-Fisher KL and geodesic distance.

Usage:
    python -m src.analysis.compare_kl
    python src/analysis/compare_kl.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
from safetensors.torch import load_file

ROOTDIR = Path(__file__).resolve().parents[2]
if str(ROOTDIR) not in sys.path:
    sys.path.insert(0, str(ROOTDIR))

from src.dataset.dataset_3 import DATASET_3_PLOT_LABELS, DATASET_3_TRAIN
from src.geometry import SOnManifold
from src.paths import MODEL_FAMILIES
from src.plots.plot_cowebs import collect_metrics, latex_bold


MERGE_MODES = ("diagonal_fisher", "standard_rescaled")
XY_PLOT_MODES = ("standard_rescaled", "diagonal_fisher")
TASKS = [tag for tag, *_ in DATASET_3_TRAIN]
FAMILIES = tuple(MODEL_FAMILIES)
MANIFOLD = SOnManifold()
VIOLIN_COLORS = ("black", "#7B2CBF")


def _task_adapter_paths(family_name: str) -> list[Path]:
    return [Path(path) for path in MODEL_FAMILIES[family_name].adapter_paths]


def _finetuned_fisher_paths(family_name: str) -> list[Path]:
    return [Path(path) for path in MODEL_FAMILIES[family_name].fisher_finetuned_paths]


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(path)
    state = load_file(str(path), device="cpu")
    return {key.replace(".default", ""): value.float() for key, value in state.items()}


def _merged_adapter_path(
    mode: str,
    family_name: str,
    merged_models_dir: Path,
) -> Path:
    return merged_models_dir / mode / family_name / "merged_model" / "merged_adapter"


def fisher_kl(
    merged: dict[str, torch.Tensor],
    task: dict[str, torch.Tensor],
    fisher: dict[str, torch.Tensor],
) -> float:
    """Diagonal approximation to the transported-Fisher quadratic form.

    The stored Fisher is the transported diagonal at the pretrained tangent
    space, so the vector entering the quadratic form must use the relative
    task-to-merged displacement in the same upper-triangular skew coordinates.
    """
    total = torch.zeros((), dtype=torch.float64)
    seen = 0
    for key, fisher_value in fisher.items():
        if key not in merged or key not in task:
            continue
        relative_value = _relative_oft_params(
            task_oft_params=task[key].double(),
            merged_oft_params=merged[key].double(),
        )
        fisher_diag = fisher_value.double()
        if fisher_diag.shape != relative_value.shape:
            raise ValueError(
                f"Shape mismatch for {key}: Fisher has {tuple(fisher_diag.shape)}, "
                f"relative vector has {tuple(relative_value.shape)}."
            )
        total += 0.5 * (fisher_diag * relative_value.square()).sum()
        seen += 1
    if seen == 0:
        raise ValueError("No matching keys between task, merged adapter, and Fisher.")
    return total.item()


def _oft_keys(*states: dict[str, torch.Tensor]) -> list[str]:
    return sorted(
        key
        for key in states[0]
        if ("oft_r" in key or "oft_" in key.lower())
        and all(key in state for state in states[1:])
    )


def _oft_params_to_so(oft_params: torch.Tensor) -> torch.Tensor:
    skew = _oft_params_to_skew(oft_params)
    return MANIFOLD.cayley_exp(skew)


def _oft_params_to_skew(oft_params: torch.Tensor) -> torch.Tensor:
    num_blocks, son_dim = oft_params.shape
    block_size = int((1 + (1 + 8 * son_dim) ** 0.5) / 2)
    if block_size * (block_size - 1) // 2 != son_dim:
        raise ValueError(f"Cannot infer SO(n) block size from {son_dim} OFT parameters.")

    indices = torch.triu_indices(block_size, block_size, offset=1, device=oft_params.device)
    skew = torch.zeros(
        num_blocks,
        block_size,
        block_size,
        dtype=oft_params.dtype,
        device=oft_params.device,
    )
    skew[:, indices[0], indices[1]] = oft_params
    skew = skew - skew.transpose(-1, -2)
    return skew


def _skew_to_oft_params(skew: torch.Tensor) -> torch.Tensor:
    block_size = skew.shape[-1]
    indices = torch.triu_indices(block_size, block_size, offset=1, device=skew.device)
    return skew[:, indices[0], indices[1]]


def _relative_oft_params(
    task_oft_params: torch.Tensor,
    merged_oft_params: torch.Tensor,
) -> torch.Tensor:
    task_so = _oft_params_to_so(task_oft_params)
    merged_so = _oft_params_to_so(merged_oft_params)
    relative_so = task_so.transpose(-1, -2) @ merged_so
    relative_skew = MANIFOLD.cayley_inverse_log(relative_so)
    return _skew_to_oft_params(relative_skew)


def geodesic_distance(
    merged: dict[str, torch.Tensor],
    task: dict[str, torch.Tensor],
) -> float:
    distances = []
    for key in _oft_keys(merged, task):
        merged_so = _oft_params_to_so(merged[key].float())
        task_so = _oft_params_to_so(task[key].float())
        dist = MANIFOLD.dist_sq(merged_so, task_so).clamp_min(0).sqrt().mean()
        distances.append(dist)
    if not distances:
        raise ValueError("No matching OFT keys between merged adapter and task adapter.")
    return torch.stack(distances).mean().item()


def markdown_table(rows: dict[str, list[float]]) -> str:
    rows = dict(rows)
    if "standard_rescaled" in rows and "diagonal_fisher" in rows:
        rows["standard_rescaled/diagonal_fisher"] = [
            standard / diagonal if diagonal != 0 else float("inf")
            for standard, diagonal in zip(
                rows["standard_rescaled"],
                rows["diagonal_fisher"],
                strict=True,
            )
        ]

    header = ["merge"] + TASKS
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for mode, values in rows.items():
        lines.append("| " + " | ".join([mode] + [f"{value:.6g}" for value in values]) + " |")
    return "\n".join(lines)


def _latex_label(mode: str) -> str:
    return {
        "diagonal_fisher": r"\textsc{Diagonal\\Fisher (Ours)}",
        "standard_rescaled": r"\textsc{OrthoMerge}",
    }[mode]


def save_violin_plot(rows: dict[str, list[float]], save_path: Path, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    _configure_latex_plot(plt)

    positions = [1.0, 1.55]
    values_by_mode = [rows[mode] for mode in MERGE_MODES]

    fig, ax = plt.subplots(figsize=(2.45, 2.7))
    parts = ax.violinplot(
        values_by_mode,
        positions=positions,
        widths=0.42,
        showmeans=True,
        showextrema=False,
    )
    for body, color in zip(parts["bodies"], VIOLIN_COLORS, strict=True):
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(0.75)
    parts["cmeans"].set_color(VIOLIN_COLORS)
    parts["cmeans"].set_linewidth(1.8)

    ax.set_xticks(positions, [_latex_label(mode) for mode in MERGE_MODES])
    ax.set_xlim(0.72, 1.83)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _configure_latex_plot(plt) -> None:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "axes.titlesize": 10,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 14,
    })


def _clip_violin_halves(parts, positions: list[float], side: str) -> None:
    for body, position in zip(parts["bodies"], positions, strict=True):
        vertices = body.get_paths()[0].vertices
        if side == "left":
            vertices[:, 0] = vertices[:, 0].clip(max=position)
        elif side == "right":
            vertices[:, 0] = vertices[:, 0].clip(min=position)
        else:
            raise ValueError(f"Unsupported violin side: {side}")


def save_mixed_violin_plot(
    kl_rows: dict[str, list[float]],
    geodesic_rows: dict[str, list[float]],
    save_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    _configure_latex_plot(plt)

    positions = [1.0, 1.55]
    kl_values = [kl_rows[mode] for mode in MERGE_MODES]
    geodesic_values = [geodesic_rows[mode] for mode in MERGE_MODES]

    fig, ax_kl = plt.subplots(figsize=(3.7, 3.7))
    ax_geo = ax_kl.twinx()

    kl_parts = ax_kl.violinplot(
        kl_values,
        positions=positions,
        widths=0.44,
        showmeans=True,
        showextrema=False,
    )
    geo_parts = ax_geo.violinplot(
        geodesic_values,
        positions=positions,
        widths=0.44,
        showmeans=True,
        showextrema=False,
    )
    _clip_violin_halves(kl_parts, positions, "left")
    _clip_violin_halves(geo_parts, positions, "right")

    for body in kl_parts["bodies"]:
        body.set_facecolor(VIOLIN_COLORS[0])
        body.set_edgecolor("black")
        body.set_alpha(0.75)
    for body in geo_parts["bodies"]:
        body.set_facecolor(VIOLIN_COLORS[1])
        body.set_edgecolor("black")
        body.set_alpha(0.75)

    kl_parts["cmeans"].set_color(VIOLIN_COLORS[0])
    geo_parts["cmeans"].set_color(VIOLIN_COLORS[1])
    kl_parts["cmeans"].set_linewidth(1.8)
    geo_parts["cmeans"].set_linewidth(1.8)

    ax_kl.set_xticks(positions, [_latex_label(mode) for mode in MERGE_MODES])
    ax_kl.set_xlim(0.72, 1.83)
    # ax_kl.set_ylabel("KL Divergence")
    # ax_geo.set_ylabel("Geodesic Distance")
    ax_kl.grid(axis="y", alpha=0.22)
    ax_kl.legend(
        handles=[
            Patch(facecolor=VIOLIN_COLORS[0], edgecolor="black", label=r"$\approx$ KL divergence (left axis)"),
            Patch(facecolor=VIOLIN_COLORS[1], edgecolor="black", label="Geodesic Distance (right axis)"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=1,
        frameon=False,
        handlelength=1.0,
        columnspacing=1.0,
        fontsize=15,
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.23)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_xy_plot(
    kl_rows: dict[str, list[float]],
    geodesic_rows: dict[str, list[float]],
    save_path: Path,
) -> dict[str, dict[str, float]]:
    import matplotlib.pyplot as plt

    _configure_latex_plot(plt)
    stats = correlation_stats(kl_rows, geodesic_rows)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), sharey=True)
    for ax, mode in zip(axes, XY_PLOT_MODES, strict=True):
        x = np.asarray(kl_rows[mode], dtype=float)
        y = np.asarray(geodesic_rows[mode], dtype=float)
        line_x = np.linspace(x.min(), x.max(), 100)
        line_y = stats[mode]["slope"] * line_x + stats[mode]["intercept"]
        ax.scatter(
            x,
            y,
            s=34,
            color=VIOLIN_COLORS[MERGE_MODES.index(mode)],
            edgecolor="black",
            linewidth=0.6,
            alpha=0.85,
        )
        ax.plot(line_x, line_y, color="black", linewidth=1.2, alpha=0.75)
        ax.text(
            0.04,
            0.96,
            rf"$r={stats[mode]['r']:.2f},\ p={stats[mode]['p']:.2g}$",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
        )
        ax.set_title(_latex_label(mode))
        ax.set_xlabel(r"$\widehat{D}_{\mathrm{KL}}$")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(r"Geodesic Distance")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return stats


def _ratio_rows(rows: dict[str, list[float]]) -> np.ndarray:
    return np.asarray(rows["diagonal_fisher"], dtype=float) / np.asarray(
        rows["standard_rescaled"], dtype=float
    )


def save_ratio_xy_plot(
    kl_rows: dict[str, list[float]],
    geodesic_rows: dict[str, list[float]],
    save_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    _configure_latex_plot(plt)
    x = _ratio_rows(kl_rows)
    y = _ratio_rows(geodesic_rows)
    keep = x <= 10.0
    x = x[keep]
    y = y[keep]
    upper = max(float(np.max(x)), float(np.max(y)), 1.0)
    upper = float(np.ceil(upper * 5) / 5)
    ticks = np.linspace(0.0, upper, 6)
    fig, ax = plt.subplots(figsize=(3.8, 3.2))
    ax.scatter(
        x,
        y,
        s=36,
        color="#2A9D8F",
        edgecolor="black",
        linewidth=0.6,
        alpha=0.9,
    )
    ax.axvline(1.0, color="black", linestyle="--", linewidth=0.9, alpha=0.55)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.9, alpha=0.55)
    ax.set_xlim(0.0, upper)
    ax.set_ylim(0.0, upper)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel(r"$\widehat{D}_{\mathrm{KL}}(\mathrm{Ours}) / \widehat{D}_{\mathrm{KL}}(\mathrm{OrthoMerge})$")
    ax.set_ylabel(r"$d_{\mathrm{geo}}(\mathrm{Ours}) / d_{\mathrm{geo}}(\mathrm{OrthoMerge})$")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_paired_xy_plot(
    kl_rows: dict[str, list[float]],
    geodesic_rows: dict[str, list[float]],
    save_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    _configure_latex_plot(plt)
    ortho_x = np.asarray(kl_rows["standard_rescaled"], dtype=float)
    ortho_y = np.asarray(geodesic_rows["standard_rescaled"], dtype=float)
    fisher_x = np.asarray(kl_rows["diagonal_fisher"], dtype=float)
    fisher_y = np.asarray(geodesic_rows["diagonal_fisher"], dtype=float)

    fig, ax = plt.subplots(figsize=(4.8, 4.3))
    for x0, y0, x1, y1 in zip(ortho_x, ortho_y, fisher_x, fisher_y, strict=True):
        ax.plot([x0, x1], [y0, y1], color="0.55", linewidth=0.8, alpha=0.65, zorder=1)
    for task, x0, y0, x1, y1 in zip(
        TASKS, ortho_x, ortho_y, fisher_x, fisher_y, strict=True
    ):
        ax.annotate(
            task.replace("_", r"\_"),
            (x1, y1),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6,
            color=VIOLIN_COLORS[MERGE_MODES.index("diagonal_fisher")],
        )
    ax.scatter(
        ortho_x,
        ortho_y,
        s=34,
        marker="o",
        color=VIOLIN_COLORS[MERGE_MODES.index("standard_rescaled")],
        edgecolor="black",
        linewidth=0.6,
        alpha=0.85,
        label=r"\textsc{OrthoMerge}",
        zorder=2,
    )
    ax.scatter(
        fisher_x,
        fisher_y,
        s=44,
        marker="x",
        color=VIOLIN_COLORS[MERGE_MODES.index("diagonal_fisher")],
        linewidth=1.4,
        alpha=0.9,
        label=r"\textsc{Diagonal Fisher (Ours)}",
        zorder=3,
    )
    ax.set_xlabel(r"$\widehat{\mathrm{KL}}$")
    ax.set_ylabel(r"Geodesic Distance")
    ax.grid(alpha=0.25)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=2,
        frameon=False,
        fontsize=10,
    )
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_kl_geodesic_method_ratio_plot(
    kl_rows: dict[str, list[float]],
    geodesic_rows: dict[str, list[float]],
    save_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    _configure_latex_plot(plt)
    x = np.arange(len(TASKS))
    ortho = np.asarray(kl_rows["standard_rescaled"], dtype=float) / np.asarray(
        geodesic_rows["standard_rescaled"], dtype=float
    )
    fisher = np.asarray(kl_rows["diagonal_fisher"], dtype=float) / np.asarray(
        geodesic_rows["diagonal_fisher"], dtype=float
    )

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    for idx, y0, y1 in zip(x, ortho, fisher, strict=True):
        ax.plot([idx, idx], [y0, y1], color="0.55", linewidth=0.8, alpha=0.65, zorder=1)
    ax.scatter(
        x,
        ortho,
        s=34,
        marker="o",
        color=VIOLIN_COLORS[MERGE_MODES.index("standard_rescaled")],
        edgecolor="black",
        linewidth=0.6,
        alpha=0.85,
        label=r"\textsc{OrthoMerge}",
        zorder=2,
    )
    ax.scatter(
        x,
        fisher,
        s=44,
        marker="x",
        color=VIOLIN_COLORS[MERGE_MODES.index("diagonal_fisher")],
        linewidth=1.4,
        alpha=0.9,
        label=r"\textsc{Diagonal Fisher (Ours)}",
        zorder=3,
    )
    ax.set_xticks(x, [task.replace("_", r"\_") for task in TASKS], rotation=70, ha="right")
    ax.set_ylabel(r"$\widehat{D}_{\mathrm{KL}} / d_{\mathrm{geo}}$")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _performance_task_name(task: str) -> str:
    return {"numinamath": "math500"}.get(task, task)


def _performance_plot_label(task: str) -> str:
    task_name = _performance_task_name(task)
    return latex_bold(DATASET_3_PLOT_LABELS.get(task_name, task_name))


def _performance_metrics(
    family_name: str,
    eval_dir: Path,
) -> dict[str, dict[str, float]]:
    return {
        mode: collect_metrics(eval_dir / mode / family_name, "eval_performance")
        for mode in ("finetunes", "standard_rescaled", "diagonal_fisher")
    }


def save_performance_ratio_kl_plot(
    kl_rows: dict[str, list[float]],
    family_name: str,
    save_path: Path,
    eval_dir: Path,
) -> dict[str, float]:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogFormatterSciNotation, LogLocator
    from matplotlib.lines import Line2D

    _configure_latex_plot(plt)
    metrics = _performance_metrics(family_name, eval_dir)
    fig, ax = plt.subplots(figsize=(3.8, 3.3))
    xs, ys, labels = [], [], []
    for task, ortho_kl, fisher_kl_value in zip(
        TASKS,
        kl_rows["standard_rescaled"],
        kl_rows["diagonal_fisher"],
        strict=True,
    ):
        performance_task = _performance_task_name(task)
        ortho_performance = metrics["standard_rescaled"].get(performance_task)
        fisher_performance = metrics["diagonal_fisher"].get(performance_task)
        if fisher_performance in (None, 0) or ortho_performance is None or fisher_kl_value == 0:
            continue
        xs.append(ortho_performance / fisher_performance)
        ys.append(ortho_kl / fisher_kl_value)
        labels.append(task)
    x_values = np.asarray(xs, dtype=float)
    y_values = np.asarray(ys, dtype=float)
    keep = np.isfinite(x_values) & np.isfinite(y_values) & (y_values > 0)
    x_fit = x_values[keep]
    y_fit = y_values[keep]
    log_y_fit = np.log10(y_fit)
    stats = {
        "n": float(x_fit.size),
        "r": float("nan"),
        "p": float("nan"),
        "slope": float("nan"),
        "intercept": float("nan"),
    }
    markers = ("s", "^", "D", "P", "X", "v", "<", ">", "*", "h", "p", "8")
    colors = (
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#CC79A7",
        "#E69F00",
        "#56B4E9",
        "#C8B400",
        "#6A3D9A",
        "#4D4D4D",
        "#A6761D",
        "#E7298A",
        "#1B9E77",
    )
    legend_handles = []
    for label, x_value, y_value in zip(labels, x_values, y_values, strict=True):
        if not (np.isfinite(x_value) and np.isfinite(y_value) and y_value > 0):
            continue
        style_idx = len(legend_handles)
        marker = markers[style_idx % len(markers)]
        color = colors[style_idx % len(colors)]
        ax.scatter(
            x_value,
            y_value,
            s=54,
            marker=marker,
            color=color,
            edgecolor="black",
            linewidth=0.6,
            alpha=0.9,
            zorder=3,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                color="none",
                markerfacecolor=color,
                markeredgecolor="black",
                markeredgewidth=0.6,
                markersize=5,
                label=_performance_plot_label(label),
            )
        )
    if x_fit.size >= 2:
        try:
            from scipy.stats import pearsonr
        except ImportError:
            pearsonr = None

        slope, intercept = np.polyfit(x_fit, log_y_fit, deg=1)
        r = float(np.corrcoef(x_fit, log_y_fit)[0, 1])
        p = float(pearsonr(x_fit, log_y_fit).pvalue) if pearsonr is not None else float("nan")
        stats.update({
            "r": r,
            "p": p,
            "slope": float(slope),
            "intercept": float(intercept),
        })
        line_x = np.linspace(float(x_fit.min()), float(x_fit.max()), 100)
        line_y = 10 ** (slope * line_x + intercept)
        ax.plot(line_x, line_y, color="black", linewidth=1.2, alpha=0.75)
        ax.text(
            0.96,
            0.96,
            rf"$r={r:.2f},\ p={p:.2g}$",
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=9,
        )
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            frameon=True,
            fancybox=False,
            framealpha=0.78,
            facecolor="white",
            edgecolor="0.75",
            fontsize=6.5,
            loc="upper right",
            bbox_to_anchor=(0.98, 0.86),
            borderaxespad=0.0,
            handletextpad=0.35,
            labelspacing=0.18,
            columnspacing=0.7,
            ncol=2,
        )
    ax.set_xlabel(r"Performance ratio")
    ax.set_ylabel(r"KL ratio")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 1.5, 2.0, 3.0, 4.0, 5.0)))
    ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10, labelOnlyBase=False))
    ax.margins(x=0.08, y=0.18)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return stats


def save_performance_over_kl_by_method_plot(
    kl_rows: dict[str, list[float]],
    family_name: str,
    save_path: Path,
    eval_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    _configure_latex_plot(plt)
    metrics = _performance_metrics(family_name, eval_dir)
    xs, ys, labels = [], [], []

    for task_idx, task in enumerate(TASKS):
        performance_task = _performance_task_name(task)
        ortho_performance = metrics["standard_rescaled"].get(performance_task)
        fisher_performance = metrics["diagonal_fisher"].get(performance_task)
        ortho_kl = kl_rows["standard_rescaled"][task_idx]
        fisher_kl_value = kl_rows["diagonal_fisher"][task_idx]
        if (
            ortho_performance is None
            or fisher_performance is None
            or ortho_kl == 0
            or fisher_kl_value == 0
        ):
            continue
        x_value = fisher_performance / fisher_kl_value
        y_value = ortho_performance / ortho_kl
        if x_value <= 0 or y_value <= 0:
            continue
        xs.append(x_value)
        ys.append(y_value)
        labels.append(task)

    x_values = np.asarray(xs, dtype=float)
    y_values = np.asarray(ys, dtype=float)
    if x_values.size == 0:
        raise ValueError(f"No valid performance/KL pairs found for {family_name}.")
    lower = min(float(np.min(x_values)), float(np.min(y_values)))
    upper = max(float(np.max(x_values)), float(np.max(y_values)), 1.0)
    lower *= 0.8
    upper *= 1.2

    fig, ax = plt.subplots(figsize=(4.3, 3.3))
    ax.scatter(
        x_values,
        y_values,
        s=42,
        color="#2A9D8F",
        edgecolor="black",
        linewidth=0.6,
        alpha=0.9,
    )
    for label, x_value, y_value in zip(labels, x_values, y_values, strict=True):
        ax.annotate(
            label.replace("_", r"\_"),
            (x_value, y_value),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6,
        )
    ax.plot(
        [lower, upper],
        [lower, upper],
        color="black",
        linestyle="--",
        linewidth=0.9,
        alpha=0.55,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel(
        r"$\mathrm{Performance}(\textsc{Ours}) / \widehat{D}_{\mathrm{KL}}(\textsc{Ours})$"
    )
    ax.set_ylabel(
        r"$\mathrm{Performance}(\textsc{OrthoMerge}) / \widehat{D}_{\mathrm{KL}}(\textsc{OrthoMerge})$"
    )
    ax.grid(alpha=0.25)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_performance_kl_distribution_plot(
    kl_rows: dict[str, list[float]],
    family_name: str,
    save_path: Path,
    eval_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    _configure_latex_plot(plt)
    metrics = _performance_metrics(family_name, eval_dir)
    fig, ax = plt.subplots(figsize=(4.5, 3.4))

    style_by_mode = {
        "standard_rescaled": {
            "marker": "o",
            "color": VIOLIN_COLORS[MERGE_MODES.index("standard_rescaled")],
            "label": r"\textsc{OrthoMerge}",
        },
        "diagonal_fisher": {
            "marker": "x",
            "color": VIOLIN_COLORS[MERGE_MODES.index("diagonal_fisher")],
            "label": r"\textsc{Diagonal Fisher (Ours)}",
        },
    }

    for task_idx, task in enumerate(TASKS):
        ortho_performance = metrics["standard_rescaled"].get(_performance_task_name(task))
        fisher_performance = metrics["diagonal_fisher"].get(_performance_task_name(task))
        if ortho_performance is None or fisher_performance is None:
            continue
        ax.plot(
            [ortho_performance, fisher_performance],
            [
                kl_rows["standard_rescaled"][task_idx],
                kl_rows["diagonal_fisher"][task_idx],
            ],
            color="0.55",
            linewidth=0.8,
            alpha=0.65,
            zorder=1,
        )

    for mode, style in style_by_mode.items():
        xs, ys = [], []
        for task_idx, task in enumerate(TASKS):
            performance = metrics[mode].get(_performance_task_name(task))
            if performance is None:
                continue
            xs.append(performance)
            ys.append(kl_rows[mode][task_idx])
        ax.scatter(
            xs,
            ys,
            s=42,
            marker=style["marker"],
            color=style["color"],
            edgecolor="black" if style["marker"] != "x" else None,
            linewidth=0.6 if style["marker"] != "x" else 1.4,
            alpha=0.88,
            label=style["label"],
            zorder=2,
        )

    ax.set_xlabel(r"Performance")
    ax.set_ylabel(r"$\widehat{D}_{\mathrm{KL}}$")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def correlation_stats(
    kl_rows: dict[str, list[float]],
    geodesic_rows: dict[str, list[float]],
) -> dict[str, dict[str, float]]:
    try:
        from scipy.stats import pearsonr
    except ImportError:
        pearsonr = None

    stats = {}
    for mode in XY_PLOT_MODES:
        x = np.asarray(kl_rows[mode], dtype=float)
        y = np.asarray(geodesic_rows[mode], dtype=float)
        slope, intercept = np.polyfit(x, y, deg=1)
        r = float(np.corrcoef(x, y)[0, 1])
        p = float(pearsonr(x, y).pvalue) if pearsonr is not None else float("nan")
        stats[mode] = {
            "n": float(len(x)),
            "slope": float(slope),
            "intercept": float(intercept),
            "r": r,
            "r2": r * r,
            "p": p,
        }
    return stats


def print_correlation_stats(family_name: str, stats: dict[str, dict[str, float]]) -> None:
    print(f"\nApproximate KL/geodesic Pearson correlation ({family_name})")
    for mode in XY_PLOT_MODES:
        values = stats[mode]
        print(
            f"{mode}: n={values['n']:.0f}, slope={values['slope']:.6g}, "
            f"intercept={values['intercept']:.6g}, r={values['r']:.6g}, "
            f"r^2={values['r2']:.6g}, p={values['p']:.6g}"
        )


def compare_family(
    family_name: str,
    merged_models_dir: Path,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    task_states = [_load_state(path / "adapter_model.safetensors") for path in _task_adapter_paths(family_name)]
    finetuned_fishers = [_load_state(path) for path in _finetuned_fisher_paths(family_name)]

    kl_rows: dict[str, list[float]] = {}
    geodesic_rows: dict[str, list[float]] = {}
    for mode in MERGE_MODES:
        merged = _load_state(
            _merged_adapter_path(mode, family_name, merged_models_dir)
            / "adapter_model.safetensors"
        )
        kl_rows[mode] = [
            fisher_kl(merged=merged, task=task, fisher=fisher)
            for task, fisher in tqdm(zip(task_states, finetuned_fishers, strict=True), total=len(TASKS))
        ]
        geodesic_rows[mode] = [
            geodesic_distance(merged=merged, task=task)
            for task in tqdm(task_states)
        ]
    return kl_rows, geodesic_rows


def _cache_path(family_name: str, data_dir: Path) -> Path:
    return data_dir / f"compare_kl_geodist_relative_fisher_{family_name}.pt"


def load_or_compute_family(
    family_name: str,
    data_dir: Path,
    merged_models_dir: Path,
    force_compute: bool = False,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    cache_path = _cache_path(family_name, data_dir)
    if cache_path.exists() and not force_compute:
        print(f"Loading computed data from {cache_path}")
        payload = torch.load(cache_path, weights_only=False)
        return payload["kl_rows"], payload["geodesic_rows"]

    print(f"Computing data for {family_name}; cache path is {cache_path}")
    kl_rows, geodesic_rows = compare_family(family_name, merged_models_dir)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "family": family_name,
            "tasks": TASKS,
            "merge_modes": MERGE_MODES,
            "kl_rows": kl_rows,
            "geodesic_rows": geodesic_rows,
        },
        cache_path,
    )
    print(f"Saved computed data to {cache_path}")
    return kl_rows, geodesic_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--families",
        nargs="+",
        default=FAMILIES,
        choices=FAMILIES,
    )
    parser.add_argument("--plot_dir", type=Path, default=ROOTDIR / "outputs" / "analysis")
    parser.add_argument("--data-dir", type=Path, default=ROOTDIR / "outputs" / "analysis" / "data")
    parser.add_argument("--eval-dir", type=Path, default=ROOTDIR / "outputs" / "evaluation")
    parser.add_argument("--merged-models-dir", type=Path, default=ROOTDIR / "outputs" / "models")
    parser.add_argument("--force-compute", action="store_true")
    args = parser.parse_args()

    results = {
        family_name: load_or_compute_family(
            family_name,
            data_dir=args.data_dir,
            merged_models_dir=args.merged_models_dir,
            force_compute=args.force_compute,
        )
        for family_name in args.families
    }
    for family_name in args.families:
        print(f"\n --- {family_name} ---")
        kl_rows, geodesic_rows = results[family_name]
        print("\nKL")
        print(markdown_table(kl_rows))
        print("\nGeodesic distance")
        print(markdown_table(geodesic_rows))

    for family_name, (kl_rows, geodesic_rows) in results.items():
        kl_path = args.plot_dir / f"compare_kl_violin_{family_name}.png"
        ylabel="KL"
        save_violin_plot(kl_rows, kl_path, ylabel)
        save_violin_plot(kl_rows, kl_path.with_suffix(".pdf"), ylabel)
        print(f"\nSaved KL violin plot to {kl_path} and {kl_path.with_suffix('.pdf')}")

        geodesic_path = args.plot_dir / f"compare_geodesic_violin_{family_name}.png"
        ylabel=r"${\mathrm{Geodesic\_Distance}(\theta_{1:T}, \theta_t)}$"
        ylabel="Geodesic Distance"
        save_violin_plot(geodesic_rows, geodesic_path, ylabel)
        save_violin_plot(geodesic_rows, geodesic_path.with_suffix(".pdf"), ylabel)
        print(f"Saved geodesic violin plot to {geodesic_path} and {geodesic_path.with_suffix('.pdf')}")

        mixed_path = args.plot_dir / f"compare_mixed_violin_{family_name}.png"
        save_mixed_violin_plot(kl_rows, geodesic_rows, mixed_path)
        save_mixed_violin_plot(kl_rows, geodesic_rows, mixed_path.with_suffix(".pdf"))
        print(f"Saved mixed violin plot to {mixed_path} and {mixed_path.with_suffix('.pdf')}")

        xy_path = args.plot_dir / f"compare_kl_geodesic_xy_{family_name}.png"
        stats = save_xy_plot(kl_rows, geodesic_rows, xy_path)
        save_xy_plot(kl_rows, geodesic_rows, xy_path.with_suffix(".pdf"))
        print(f"Saved KL/geodesic xy plot to {xy_path} and {xy_path.with_suffix('.pdf')}")
        print_correlation_stats(family_name, stats)

        ratio_xy_path = args.plot_dir / f"compare_kl_geodesic_ratio_xy_{family_name}.png"
        save_ratio_xy_plot(kl_rows, geodesic_rows, ratio_xy_path)
        save_ratio_xy_plot(kl_rows, geodesic_rows, ratio_xy_path.with_suffix(".pdf"))
        print(
            f"Saved KL/geodesic ratio xy plot to {ratio_xy_path} "
            f"and {ratio_xy_path.with_suffix('.pdf')}"
        )

        paired_xy_path = args.plot_dir / f"compare_kl_geodesic_paired_xy_{family_name}.png"
        save_paired_xy_plot(kl_rows, geodesic_rows, paired_xy_path)
        save_paired_xy_plot(kl_rows, geodesic_rows, paired_xy_path.with_suffix(".pdf"))
        print(
            f"Saved paired KL/geodesic xy plot to {paired_xy_path} "
            f"and {paired_xy_path.with_suffix('.pdf')}"
        )

        method_ratio_path = args.plot_dir / f"compare_kl_over_geodesic_by_method_{family_name}.png"
        save_kl_geodesic_method_ratio_plot(kl_rows, geodesic_rows, method_ratio_path)
        save_kl_geodesic_method_ratio_plot(
            kl_rows,
            geodesic_rows,
            method_ratio_path.with_suffix(".pdf"),
        )
        print(
            f"Saved KL/geodesic method-ratio plot to {method_ratio_path} "
            f"and {method_ratio_path.with_suffix('.pdf')}"
        )

        performance_kl_path = args.plot_dir / f"compare_performance_ratio_kl_{family_name}.png"
        performance_ratio_stats = save_performance_ratio_kl_plot(
            kl_rows,
            family_name,
            performance_kl_path,
            args.eval_dir,
        )
        save_performance_ratio_kl_plot(
            kl_rows,
            family_name,
            performance_kl_path.with_suffix(".pdf"),
            args.eval_dir,
        )
        print(
            f"Saved performance-ratio/KL plot to {performance_kl_path} "
            f"and {performance_kl_path.with_suffix('.pdf')}"
        )
        print(
            f"Performance-ratio/KL stats ({family_name}): "
            f"n={performance_ratio_stats['n']:.0f}, "
            f"Pearson r={performance_ratio_stats['r']:.6g}, "
            f"p={performance_ratio_stats['p']:.6g}"
        )

        performance_over_kl_path = (
            args.plot_dir / f"compare_performance_over_kl_by_method_{family_name}.png"
        )
        save_performance_over_kl_by_method_plot(
            kl_rows,
            family_name,
            performance_over_kl_path,
            args.eval_dir,
        )
        save_performance_over_kl_by_method_plot(
            kl_rows,
            family_name,
            performance_over_kl_path.with_suffix(".pdf"),
            args.eval_dir,
        )
        print(
            f"Saved performance/KL by-method plot to {performance_over_kl_path} "
            f"and {performance_over_kl_path.with_suffix('.pdf')}"
        )

        performance_kl_distribution_path = (
            args.plot_dir / f"compare_performance_kl_distribution_{family_name}.png"
        )
        save_performance_kl_distribution_plot(
            kl_rows,
            family_name,
            performance_kl_distribution_path,
            args.eval_dir,
        )
        save_performance_kl_distribution_plot(
            kl_rows,
            family_name,
            performance_kl_distribution_path.with_suffix(".pdf"),
            args.eval_dir,
        )
        print(
            f"Saved performance/KL distribution plot to {performance_kl_distribution_path} "
            f"and {performance_kl_distribution_path.with_suffix('.pdf')}"
        )

if __name__ == "__main__":
    main()
