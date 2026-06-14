"""Compare merged adapters under finetuned Fisher KL and geodesic distance.

Usage:
    python -m src.analysis.compare_kl
    python src/analysis/compare_kl.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from tqdm import tqdm

import torch
from safetensors.torch import load_file

ROOTDIR = Path(__file__).resolve().parents[2]
if str(ROOTDIR) not in sys.path:
    sys.path.insert(0, str(ROOTDIR))

from src.dataset.dataset_3 import DATASET_3_TRAIN
from src.geometry import SOnManifold


MERGE_MODES = ("diagonal_fisher", "standard_rescaled")
TASKS = [tag for tag, *_ in DATASET_3_TRAIN]
FAMILIES = ("llama3.1", "qwen2.5")
FISHERS_DIR = Path("/scratch-shared/eterres/fishers")
DATA_DIR = ROOTDIR / "outputs" / "analysis" / "data"
MANIFOLD = SOnManifold()
VIOLIN_COLORS = ("black", "#7B2CBF")


def _adapter_task_name(task: str) -> str:
    return {
        "social_iqa": "socialiqa",
        "commonsense_qa": "commonsense",
        "science_qa": "scienceqa",
    }.get(task, task)


def _adapter_prefix(family_name: str) -> str:
    return {
        "llama3.1": "llama3-1_8b_finetune",
        "qwen2.5": "qwen2.5_3b_finetune",
    }[family_name]


def _adapter_root(family_name: str) -> Path:
    folder = {
        "llama3.1": "Llama-3.1-8B_OFT_dataset2_adapters",
        "qwen2.5": "Qwen-2.5-3B_OFT_dataset2_adapters",
    }[family_name]
    return ROOTDIR / "data" / "models" / folder


def _task_adapter_paths(family_name: str) -> list[Path]:
    prefix = _adapter_prefix(family_name)
    return [
        _adapter_root(family_name) / f"{prefix}_{_adapter_task_name(task)}"
        for task in TASKS
    ]


def _finetuned_fisher_paths(family_name: str) -> list[Path]:
    prefix = _adapter_prefix(family_name)
    return [
        FISHERS_DIR
        / family_name
        / task
        / f"{prefix}_{_adapter_task_name(task)}_finetuned.safetensors"
        for task in TASKS
    ]


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(path)
    state = load_file(str(path), device="cpu")
    return {key.replace(".default", ""): value.float() for key, value in state.items()}


def _merged_adapter_path(mode: str, family_name: str) -> Path:
    return ROOTDIR / "outputs" / "models" / mode / family_name / "merged_model" / "merged_adapter"


def fisher_kl(
    merged: dict[str, torch.Tensor],
    fisher: dict[str, torch.Tensor],
) -> float:
    total = torch.zeros((), dtype=torch.float64)
    seen = 0
    for key, fisher_value in fisher.items():
        if key not in merged:
            continue
        merged_value = merged[key].float()
        total += 0.5 * (fisher_value.float() * merged_value * merged_value).double().sum()
        seen += 1
    if seen == 0:
        raise ValueError("No matching keys between merged adapter and Fisher.")
    return total.item()


def _oft_keys(*states: dict[str, torch.Tensor]) -> list[str]:
    return sorted(
        key
        for key in states[0]
        if ("oft_r" in key or "oft_" in key.lower())
        and all(key in state for state in states[1:])
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
    skew = skew - skew.transpose(-1, -2)
    return MANIFOLD.cayley_exp(skew)


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


def compare_family(family_name: str) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    task_states = [_load_state(path / "adapter_model.safetensors") for path in _task_adapter_paths(family_name)]
    finetuned_fishers = [_load_state(path) for path in _finetuned_fisher_paths(family_name)]

    kl_rows: dict[str, list[float]] = {}
    geodesic_rows: dict[str, list[float]] = {}
    for mode in MERGE_MODES:
        merged = _load_state(_merged_adapter_path(mode, family_name) / "adapter_model.safetensors")
        kl_rows[mode] = [
            fisher_kl(merged=merged, fisher=fisher)
            for fisher in tqdm(finetuned_fishers)
        ]
        geodesic_rows[mode] = [
            geodesic_distance(merged=merged, task=task)
            for task in tqdm(task_states)
        ]
    return kl_rows, geodesic_rows


def _cache_path(family_name: str) -> Path:
    return DATA_DIR / f"compare_kl_geodist_{family_name}.pt"


def load_or_compute_family(
    family_name: str,
    force_compute: bool = False,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    cache_path = _cache_path(family_name)
    if cache_path.exists() and not force_compute:
        print(f"Loading computed data from {cache_path}")
        payload = torch.load(cache_path, weights_only=False)
        return payload["kl_rows"], payload["geodesic_rows"]

    print(f"Computing data for {family_name}; cache path is {cache_path}")
    kl_rows, geodesic_rows = compare_family(family_name)
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
    parser.add_argument("--force-compute", action="store_true")
    args = parser.parse_args()

    results = {
        family_name: load_or_compute_family(
            family_name,
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

if __name__ == "__main__":
    main()
