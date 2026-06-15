"""Compare merged adapters by dataset KL and OFT geodesic distance."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from tqdm import tqdm

ROOTDIR = Path(__file__).resolve().parents[2]
if str(ROOTDIR) not in sys.path:
    sys.path.insert(0, str(ROOTDIR))

from src.dataset.dataset_3 import DATASET_3_PLOT_LABELS, DATASET_3_TRAIN, build_loader
from src.geometry import SOnManifold
from src.plots.plot_cowebs import collect_metrics, latex_bold


MERGE_MODES = ("diagonal_fisher", "standard_rescaled")
XY_PLOT_MODES = ("standard_rescaled", "diagonal_fisher")
TASKS = [tag for tag, *_ in DATASET_3_TRAIN]
FAMILIES = ("llama3.1", "qwen2.5")
KL_NUM_SAMPLES = 256
MODELS_DIR = ROOTDIR / "data" / "models"
BASE_MODEL_PATHS = {
    "llama3.1": MODELS_DIR / "Llama-3.1-8B",
    "qwen2.5": MODELS_DIR / "Qwen-2.5-3B",
}
ANALYSIS_DIR = ROOTDIR / "outputs" / "analysis_exact"
DATA_DIR = ANALYSIS_DIR / "data"
EVAL_DIR = ROOTDIR / "outputs" / "evaluation"
MANIFOLD = SOnManifold()
VIOLIN_COLORS = ("black", "#7B2CBF")


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(path)
    state = load_file(str(path), device="cpu")
    return {key.replace(".default", ""): value.float() for key, value in state.items()}


def _merged_adapter_path(mode: str, family_name: str) -> Path:
    return ROOTDIR / "outputs" / "models" / mode / family_name / "merged_model" / "merged_adapter"


def _adapter_task_name(task: str) -> str:
    return {
        "social_iqa": "socialiqa",
        "commonsense_qa": "commonsense",
        "science_qa": "scienceqa",
    }.get(task, task)


def _task_adapter_paths(family_name: str) -> list[Path]:
    folder, prefix = {
        "llama3.1": ("Llama-3.1-8B_OFT_dataset2_adapters", "llama3-1_8b_finetune"),
        "qwen2.5": ("Qwen-2.5-3B_OFT_dataset2_adapters", "qwen2.5_3b_finetune"),
    }[family_name]
    return [MODELS_DIR / folder / f"{prefix}_{_adapter_task_name(task)}" for task in TASKS]


def _oft_keys(*states: dict[str, torch.Tensor]) -> list[str]:
    return sorted(
        key
        for key in states[0]
        if ("oft_r" in key or "oft_" in key.lower()) and all(key in state for state in states[1:])
    )


def _oft_params_to_so(oft_params: torch.Tensor) -> torch.Tensor:
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
    return MANIFOLD.cayley_exp(skew - skew.transpose(-1, -2))


def geodesic_distance(merged: dict[str, torch.Tensor], task: dict[str, torch.Tensor]) -> float:
    distances = []
    for key in _oft_keys(merged, task):
        merged_so = _oft_params_to_so(merged[key].float())
        task_so = _oft_params_to_so(task[key].float())
        distances.append(MANIFOLD.dist_sq(merged_so, task_so).clamp_min(0).sqrt().mean())
    if not distances:
        raise ValueError("No matching OFT keys between merged adapter and task adapter.")
    return torch.stack(distances).mean().item()


def _load_model(family_name: str, merged_adapter: Path, device: str):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_path = BASE_MODEL_PATHS[family_name]
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(
        base_model,
        str(merged_adapter),
        adapter_name="merged",
        is_trainable=False,
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def _target_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    mask = labels[..., 1:].ne(-100)
    return logits[..., :-1, :][mask].float()


@torch.inference_mode()
def kl_divergence(
    model,
    loader,
    device: str,
    finetuned_adapter: str,
    chunk_size: int,
) -> tuple[float, int]:
    total_kl = 0.0
    total_tokens = 0
    for batch in tqdm(loader, desc="KL", leave=False):
        batch = {key: value.to(device) for key, value in batch.items()}

        model.set_adapter(finetuned_adapter)
        p_logits = _target_logits(
            model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits,
            batch["labels"],
        )
        model.set_adapter("merged")
        q_logits = _target_logits(
            model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits,
            batch["labels"],
        )

        for start in range(0, p_logits.size(0), chunk_size):
            p_log_probs = p_logits[start : start + chunk_size].log_softmax(dim=-1)
            q_log_probs = q_logits[start : start + chunk_size].log_softmax(dim=-1)
            total_kl += (p_log_probs.exp() * (p_log_probs - q_log_probs)).sum().item()
        total_tokens += p_logits.size(0)

    return total_kl / max(total_tokens, 1), total_tokens


def compare_family(
    args: argparse.Namespace,
    family_name: str,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    task_adapter_paths = _task_adapter_paths(family_name)
    task_states = [_load_state(path / "adapter_model.safetensors") for path in task_adapter_paths]
    kl_rows: dict[str, list[float]] = {}
    geodesic_rows: dict[str, list[float]] = {}

    for mode in MERGE_MODES:
        merged_path = _merged_adapter_path(mode, family_name)
        merged_state = _load_state(merged_path / "adapter_model.safetensors")
        model, tokenizer = _load_model(family_name, merged_path, args.device)
        kl_rows[mode] = []
        geodesic_rows[mode] = [
            geodesic_distance(merged=merged_state, task=task_state)
            for task_state in tqdm(task_states, desc=f"{mode} geodesic")
        ]

        for task_id, task_spec in enumerate(DATASET_3_TRAIN):
            task, dataset_path, dataset_name, split, formatter = task_spec
            adapter_name = f"finetuned_{task}"
            if adapter_name not in model.peft_config:
                model.load_adapter(
                    str(task_adapter_paths[task_id]),
                    adapter_name=adapter_name,
                    is_trainable=False,
                )
            loader = build_loader(
                dataset_path=dataset_path,
                dataset_name=dataset_name,
                split=split,
                doc_to_text_fn=formatter,
                tokenizer=tokenizer,
                num_samples=KL_NUM_SAMPLES,
                batch_size=args.batch_size,
                max_length=args.max_length,
                task=task,
                cache_dir=str(args.dataset_cache_dir) if args.dataset_cache_dir else None,
            )
            kl, tokens = kl_divergence(model, loader, args.device, adapter_name, args.kl_chunk_size)
            print(f"{family_name}/{mode}/{task}: KL={kl:.6g}, tokens={tokens}")
            kl_rows[mode].append(kl)

        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    return kl_rows, geodesic_rows


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


def _configure_latex_plot(plt) -> None:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "axes.titlesize": 10,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 14,
    })


def save_violin_plot(rows: dict[str, list[float]], save_path: Path, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    _configure_latex_plot(plt)
    positions = [1.0, 1.55]
    parts = plt.violinplot(
        [rows[mode] for mode in MERGE_MODES],
        positions=positions,
        widths=0.42,
        showmeans=True,
        showextrema=False,
    )
    ax = plt.gca()
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
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


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
        ax.set_xlabel(r"$D_{\mathrm{KL}}$")
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
    ax.set_xlabel(r"$D_{\mathrm{KL}}(\mathrm{Ours}) / D_{\mathrm{KL}}(\mathrm{OrthoMerge})$")
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

    fig, ax = plt.subplots(figsize=(4.2, 3.3))
    for x0, y0, x1, y1 in zip(ortho_x, ortho_y, fisher_x, fisher_y, strict=True):
        ax.plot([x0, x1], [y0, y1], color="0.55", linewidth=0.8, alpha=0.65, zorder=1)
    for task, x0, y0, x1, y1 in zip(
        TASKS, ortho_x, ortho_y, fisher_x, fisher_y, strict=True
    ):
        ax.annotate(
            task.replace("_", r"\_"),
            (x0, y0),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6,
            color="black",
        )
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
    ax.set_xlabel(r"$D_{\mathrm{KL}}$")
    ax.set_ylabel(r"Geodesic Distance")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
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
    ax.set_ylabel(r"$D_{\mathrm{KL}} / d_{\mathrm{geo}}$")
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


def _performance_metrics(family_name: str) -> dict[str, dict[str, float]]:
    return {
        mode: collect_metrics(EVAL_DIR / mode / family_name, "eval_performance")
        for mode in ("finetunes", "standard_rescaled", "diagonal_fisher")
    }


def save_performance_ratio_kl_plot(
    kl_rows: dict[str, list[float]],
    family_name: str,
    save_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogFormatterSciNotation, LogLocator

    _configure_latex_plot(plt)
    metrics = _performance_metrics(family_name)
    fig, ax = plt.subplots(figsize=(3.8, 3.3))
    xs, ys, labels = [], [], []
    for task, ortho_kl, fisher_kl in zip(
        TASKS,
        kl_rows["standard_rescaled"],
        kl_rows["diagonal_fisher"],
        strict=True,
    ):
        performance_task = _performance_task_name(task)
        ortho_performance = metrics["standard_rescaled"].get(performance_task)
        fisher_performance = metrics["diagonal_fisher"].get(performance_task)
        if fisher_performance in (None, 0) or ortho_performance is None or fisher_kl == 0:
            continue
        xs.append(ortho_performance / fisher_performance)
        ys.append(ortho_kl / fisher_kl)
        labels.append(task)
    x_values = np.asarray(xs, dtype=float)
    y_values = np.asarray(ys, dtype=float)
    keep = np.isfinite(x_values) & np.isfinite(y_values) & (y_values > 0)
    x_fit = x_values[keep]
    y_fit = y_values[keep]
    ax.scatter(
        x_fit,
        y_fit,
        s=54,
        marker="^",
        color="#2A9D8F",
        edgecolor="black",
        linewidth=0.6,
        alpha=0.9,
    )
    for label, x_value, y_value in zip(labels, x_values, y_values, strict=True):
        if not (np.isfinite(x_value) and np.isfinite(y_value) and y_value > 0):
            continue
        ax.annotate(
            _performance_plot_label(label),
            (x_value, y_value),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold",
        )
    if x_fit.size >= 2:
        try:
            from scipy.stats import pearsonr
        except ImportError:
            pearsonr = None

        log_y_fit = np.log10(y_fit)
        slope, intercept = np.polyfit(x_fit, log_y_fit, deg=1)
        r = float(np.corrcoef(x_fit, log_y_fit)[0, 1])
        p = float(pearsonr(x_fit, log_y_fit).pvalue) if pearsonr is not None else float("nan")
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


def save_performance_over_kl_by_method_plot(
    kl_rows: dict[str, list[float]],
    family_name: str,
    save_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    _configure_latex_plot(plt)
    metrics = _performance_metrics(family_name)
    xs, ys, labels = [], [], []

    for task_idx, task in enumerate(TASKS):
        performance_task = _performance_task_name(task)
        ortho_performance = metrics["standard_rescaled"].get(performance_task)
        fisher_performance = metrics["diagonal_fisher"].get(performance_task)
        ortho_kl = kl_rows["standard_rescaled"][task_idx]
        fisher_kl = kl_rows["diagonal_fisher"][task_idx]
        if (
            ortho_performance is None
            or fisher_performance is None
            or ortho_kl == 0
            or fisher_kl == 0
        ):
            continue
        x_value = fisher_performance / fisher_kl
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
    lower = min(float(np.min(x_values)), float(np.min(y_values))) * 0.8
    upper = max(float(np.max(x_values)), float(np.max(y_values)), 1.0) * 1.2

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
    ax.set_xlabel(r"$\mathrm{Performance}(\textsc{Ours}) / D_{\mathrm{KL}}(\textsc{Ours})$")
    ax.set_ylabel(
        r"$\mathrm{Performance}(\textsc{OrthoMerge}) / D_{\mathrm{KL}}(\textsc{OrthoMerge})$"
    )
    ax.grid(alpha=0.25)
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
    print(f"\nKL/geodesic Pearson correlation ({family_name})")
    for mode in XY_PLOT_MODES:
        values = stats[mode]
        print(
            f"{mode}: n={values['n']:.0f}, slope={values['slope']:.6g}, "
            f"intercept={values['intercept']:.6g}, r={values['r']:.6g}, "
            f"r^2={values['r2']:.6g}, p={values['p']:.6g}"
        )


def _cache_path(family_name: str) -> Path:
    return DATA_DIR / f"compare_dataset_kl_geodist_{family_name}.pt"


def _cache_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        "tasks": TASKS,
        "merge_modes": MERGE_MODES,
        "num_samples": KL_NUM_SAMPLES,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
    }


def _cache_payload_matches(payload: dict, args: argparse.Namespace) -> bool:
    config = payload.get("config")
    if config is not None:
        return config == _cache_config(args)

    return (
        payload.get("tasks") == TASKS
        and tuple(payload.get("merge_modes", ())) == MERGE_MODES
    )


def load_or_compute_family(
    args: argparse.Namespace,
    family_name: str,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    cache_path = _cache_path(family_name)
    if cache_path.exists() and not args.force_compute:
        payload = torch.load(cache_path, weights_only=False)
        if _cache_payload_matches(payload, args):
            print(f"Loading computed data from {cache_path}")
            return payload["kl_rows"], payload["geodesic_rows"]
        print(f"Ignoring stale cache at {cache_path}")

    print(f"Computing data for {family_name}; cache path is {cache_path}")
    kl_rows, geodesic_rows = compare_family(args, family_name)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "family": family_name,
            "config": _cache_config(args),
            "kl_rows": kl_rows,
            "geodesic_rows": geodesic_rows,
        },
        cache_path,
    )
    print(f"Saved computed data to {cache_path}")
    return kl_rows, geodesic_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", nargs="+", default=FAMILIES, choices=FAMILIES)
    parser.add_argument("--plot_dir", type=Path, default=ANALYSIS_DIR)
    parser.add_argument("--dataset-cache-dir", type=Path, default=ROOTDIR / "data" / "hf_cache")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--kl-chunk-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force-compute", action="store_true")
    args = parser.parse_args()
    if args.kl_chunk_size < 1:
        raise ValueError("--kl-chunk-size must be positive")

    results = {
        family_name: load_or_compute_family(args, family_name)
        for family_name in args.families
    }
    for family_name, (kl_rows, geodesic_rows) in results.items():
        print(f"\n--- {family_name} ---")
        print("\nKL")
        print(markdown_table(kl_rows))
        print("\nGeodesic distance")
        print(markdown_table(geodesic_rows))

        kl_path = args.plot_dir / f"compare_kl_violin_{family_name}.png"
        save_violin_plot(kl_rows, kl_path, "KL")
        save_violin_plot(kl_rows, kl_path.with_suffix(".pdf"), "KL")
        print(f"\nSaved KL violin plot to {kl_path} and {kl_path.with_suffix('.pdf')}")

        geodesic_path = args.plot_dir / f"compare_geodesic_violin_{family_name}.png"
        save_violin_plot(geodesic_rows, geodesic_path, "Geodesic Distance")
        save_violin_plot(geodesic_rows, geodesic_path.with_suffix(".pdf"), "Geodesic Distance")
        print(
            f"Saved geodesic violin plot to {geodesic_path} "
            f"and {geodesic_path.with_suffix('.pdf')}"
        )

        xy_path = args.plot_dir / f"compare_kl_geodesic_xy_{family_name}.png"
        stats = save_xy_plot(kl_rows, geodesic_rows, xy_path)
        print(f"Saved KL/geodesic xy plot to {xy_path}")
        print_correlation_stats(family_name, stats)

        ratio_xy_path = args.plot_dir / f"compare_kl_geodesic_ratio_xy_{family_name}.png"
        save_ratio_xy_plot(kl_rows, geodesic_rows, ratio_xy_path)
        print(f"Saved KL/geodesic ratio xy plot to {ratio_xy_path}")

        paired_xy_path = args.plot_dir / f"compare_kl_geodesic_paired_xy_{family_name}.png"
        save_paired_xy_plot(kl_rows, geodesic_rows, paired_xy_path)
        print(f"Saved paired KL/geodesic xy plot to {paired_xy_path}")

        method_ratio_path = args.plot_dir / f"compare_kl_over_geodesic_by_method_{family_name}.png"
        save_kl_geodesic_method_ratio_plot(kl_rows, geodesic_rows, method_ratio_path)
        print(f"Saved KL/geodesic method-ratio plot to {method_ratio_path}")

        performance_kl_path = args.plot_dir / f"compare_performance_ratio_kl_{family_name}.png"
        save_performance_ratio_kl_plot(kl_rows, family_name, performance_kl_path)
        print(f"Saved performance-ratio/KL plot to {performance_kl_path}")

        performance_over_kl_path = (
            args.plot_dir / f"compare_performance_over_kl_by_method_{family_name}.png"
        )
        save_performance_over_kl_by_method_plot(kl_rows, family_name, performance_over_kl_path)
        print(f"Saved performance/KL by-method plot to {performance_over_kl_path}")


if __name__ == "__main__":
    main()
