import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

USER = os.environ.get("USER", "user")
os.environ.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib-{USER}")
os.environ.setdefault("XDG_CACHE_HOME", f"/tmp/cache-{USER}")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.diffusion.analysis.sdxl_merge_stats import (
    build_fisher_map,
    block_size_from_coords,
    fisher,
    fisher_key,
    is_oft_key,
    norm_rescale,
    selected_pairs,
    standard,
    standard_norm_rescale,
    to_coords,
)

METHODS = (
    "orthofuse__curve_over_id",
    "fisher__rescaled",
    "fisher__std_rescaled",
    "standard__rescaled",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept_name", default="cat")
    parser.add_argument("--style_name", default="01_08")
    parser.add_argument("--all_dataset", action="store_true")
    parser.add_argument("--output_dir", default="outputs/diffusion/analysis")
    parser.add_argument("--num_points", type=int, default=21)
    parser.add_argument("--fisher_min", type=float, default=1e-14)
    parser.add_argument("--fisher_rescale", type=float, default=1e10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    return parser.parse_args()


def cayley(data):
    skew = 0.5 * (data.float() - data.float().transpose(-1, -2))
    eye = torch.eye(data.shape[-1], device=data.device, dtype=skew.dtype)
    eye = eye.expand(*skew.shape[:-2], data.shape[-1], data.shape[-1])
    return torch.linalg.solve(eye - skew, eye + skew, left=False)


def inverse_cayley(matrix):
    matrix = matrix.float()
    eye = torch.eye(matrix.shape[-1], device=matrix.device, dtype=matrix.dtype)
    eye = eye.expand(*matrix.shape[:-2], matrix.shape[-1], matrix.shape[-1])
    skew = torch.linalg.solve(matrix + eye, matrix - eye, left=False)
    return 0.5 * (skew - skew.transpose(-1, -2))


def coords_to_skew(coords):
    coords = coords.float()
    block_size = block_size_from_coords(coords)
    rows, cols = torch.triu_indices(block_size, block_size, offset=1, device=coords.device)
    skew = torch.zeros(*coords.shape[:-1], block_size, block_size, dtype=coords.dtype, device=coords.device)
    skew[..., rows, cols] = coords
    return skew - skew.transpose(-1, -2)


def orthofuse(concept, style, alpha_concept, curve_over_id=False):
    t = 1.0 - alpha_concept
    merged = cayley(t * concept + (1.0 - t) * style)
    if not curve_over_id:
        return merged
    eta = 0.5 + 2.0 * t * (1.0 - t)
    return cayley(eta * merged)


def side_from_key(key):
    return key.rsplit(".", 1)[-1]


def norm_ratio_rows(base_by_key, corrected_by_key):
    rows = {}
    for side in ("L", "R"):
        keys = [key for key in base_by_key if side_from_key(key) == side]
        if not keys:
            rows[f"base_{side.lower()}_mean"] = 0.0
            rows[f"corrected_{side.lower()}_mean"] = 0.0
            rows[f"ratio_{side.lower()}_mean"] = 0.0
            continue
        base_norms = torch.tensor(
            [torch.linalg.vector_norm(base_by_key[key].float()).item() for key in keys]
        )
        corrected_norms = torch.tensor(
            [torch.linalg.vector_norm(corrected_by_key[key].float()).item() for key in keys]
        )
        rows[f"base_{side.lower()}_mean"] = base_norms.mean().item()
        rows[f"corrected_{side.lower()}_mean"] = corrected_norms.mean().item()
        rows[f"ratio_{side.lower()}_mean"] = (corrected_norms / base_norms.clamp_min(1e-8)).mean().item()

    base_total = torch.sqrt(
        sum(base.float().pow(2).sum() for base in base_by_key.values())
    )
    corrected_total = torch.sqrt(
        sum(corrected.float().pow(2).sum() for corrected in corrected_by_key.values())
    )
    rows["base_total"] = base_total.item()
    rows["corrected_total"] = corrected_total.item()
    rows["ratio_total"] = (corrected_total / base_total.clamp_min(1e-8)).item()
    return rows


def fisher_norm_rescale(weights, fishers, merged, alphas):
    stacked = torch.stack(weights).float().to(merged.device)
    fishers = torch.stack([f.float().to(merged.device).clamp_min(0.0) for f in fishers])
    a = torch.tensor(alphas, dtype=stacked.dtype, device=stacked.device)
    while a.dim() < stacked.dim():
        a = a.unsqueeze(-1)
    a_flat = a.reshape(a.shape[0], -1)[:, 0].abs()
    target = (
        a_flat
        * torch.sqrt((fishers * stacked.square()).flatten(1).sum(dim=1).clamp_min(0.0))
    ).sum()
    source = torch.sqrt(((a.abs() * fishers).sum(dim=0) * merged.float().square()).sum().clamp_min(0.0))
    return merged * target / source.clamp_min(1e-8)


def analyze_pair(pair, args):
    device = torch.device(args.device)
    concept_state = load_file(pair["concept"]["adapter_path"], device=str(device))
    style_state = load_file(pair["style"]["adapter_path"], device=str(device))
    concept_fisher = load_file(pair["concept"]["fim_path"], device=str(device))
    style_fisher = load_file(pair["style"]["fim_path"], device=str(device))

    keys = sorted(k for k in concept_state if is_oft_key(k))
    processor_to_fisher = build_fisher_map(concept_state, concept_fisher, keys)
    grid = torch.linspace(0.0, 1.0, args.num_points).tolist()
    rows = []

    for alpha_concept in grid:
        alphas = (1.0 - alpha_concept, alpha_concept)
        ortho_base, ortho_curve = {}, {}
        standard_base, standard_rescaled = {}, {}
        fisher_base, fisher_rescaled, fisher_std_rescaled = {}, {}, {}

        for key in keys:
            concept = concept_state[key]
            style = style_state[key]
            if "orthofuse__curve_over_id" in args.methods:
                ortho_base[key] = inverse_cayley(
                    orthofuse(concept, style, alpha_concept, curve_over_id=False)
                )
                ortho_curve[key] = inverse_cayley(
                    orthofuse(concept, style, alpha_concept, curve_over_id=True)
                )

            cf = concept_fisher[fisher_key(key, processor_to_fisher)]
            sf = style_fisher[fisher_key(key, processor_to_fisher)]
            fishers = [
                cf.clamp_min(args.fisher_min) * args.fisher_rescale,
                sf.clamp_min(args.fisher_min) * args.fisher_rescale,
            ]
            weights = [to_coords(concept), to_coords(style)]
            if "standard__rescaled" in args.methods:
                merged_standard = standard(weights, alphas)
                standard_base[key] = coords_to_skew(merged_standard)
                standard_rescaled[key] = coords_to_skew(norm_rescale(weights, merged_standard, alphas))
            if (
                "fisher__rescaled" in args.methods
                or "fisher__std_rescaled" in args.methods
            ):
                merged_fisher = fisher(weights, fishers, alphas)
                fisher_base[key] = coords_to_skew(merged_fisher)
                if "fisher__rescaled" in args.methods:
                    fisher_rescaled[key] = coords_to_skew(fisher_norm_rescale(weights, fishers, merged_fisher, alphas))
                if "fisher__std_rescaled" in args.methods:
                    fisher_std_rescaled[key] = coords_to_skew(standard_norm_rescale(weights, merged_fisher, alphas))

        comparisons = {
            "orthofuse__curve_over_id": (ortho_base, ortho_curve),
            "fisher__rescaled": (fisher_base, fisher_rescaled),
            "fisher__std_rescaled": (fisher_base, fisher_std_rescaled),
            "standard__rescaled": (standard_base, standard_rescaled),
        }
        for method in args.methods:
            base, corrected = comparisons[method]
            rows.append(
                {
                    "pair": pair["name"],
                    "method_pair": method,
                    "alpha_concept": alphas[0],
                    "alpha_style": alphas[1],
                    **norm_ratio_rows(base, corrected),
                }
            )
    return rows


def average_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method_pair"], row["alpha_concept"])].append(row)
    out = []
    for (method_pair, alpha), values in sorted(grouped.items()):
        numeric = {
            key: sum(float(row[key]) for row in values) / len(values)
            for key in values[0]
            if key not in {"pair", "method_pair"}
        }
        out.append({"pair": "average", "method_pair": method_pair, **numeric})
    return out


def plot(rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for method_pair in sorted({row["method_pair"] for row in rows}):
        series = sorted(
            [row for row in rows if row["method_pair"] == method_pair],
            key=lambda row: row["alpha_concept"],
        )
        x = [row["alpha_concept"] for row in series]
        axes[0].plot(x, [row["ratio_total"] for row in series], marker="o", label=method_pair)
        axes[1].plot(x, [row["ratio_l_mean"] for row in series], marker="o", label=f"{method_pair} L")
        axes[1].plot(x, [row["ratio_r_mean"] for row in series], marker="s", linestyle="--", label=f"{method_pair} R")
    axes[0].set_title("Correction norm ratio")
    axes[0].set_ylabel("global skew ||corrected|| / ||base||")
    axes[1].set_title("Average L/R skew norm ratios")
    axes[1].set_ylabel("mean per-tensor skew ||corrected|| / ||base||")
    for ax in axes:
        ax.set_xlabel("alpha_concept")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_rescaled_norm(rows, path):
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for method_pair in sorted({row["method_pair"] for row in rows}):
        series = sorted(
            [row for row in rows if row["method_pair"] == method_pair],
            key=lambda row: row["alpha_concept"],
        )
        ax.plot(
            [row["alpha_concept"] for row in series],
            [row["corrected_total"] for row in series],
            marker="o",
            label=method_pair,
        )
    ax.set_title("Rescaled merge norm")
    ax.set_xlabel("alpha_concept")
    ax.set_ylabel("global skew ||corrected||")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    pairs = selected_pairs(args)
    for pair in pairs:
        print(f"[analysis] {pair['name']}", flush=True)
        rows.extend(analyze_pair(pair, args))

    prefix = "all_pairs" if args.all_dataset else pairs[0]["name"]
    averaged = average_rows(rows) if args.all_dataset else rows
    plot(averaged, output_dir / f"{prefix}_rescaling_curve.png")
    plot_rescaled_norm(averaged, output_dir / f"{prefix}_rescaled_norm.png")
    print(f"Saved {prefix} rescaling curve to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
