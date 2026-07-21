import argparse
import os
import sys
from pathlib import Path

USER = os.environ.get("USER", "user")
os.environ.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib-{USER}")
os.environ.setdefault("XDG_CACHE_HOME", f"/tmp/cache-{USER}")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm
from safetensors.torch import load_file

plt.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.family": "serif",
    }
)

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.diffusion.analysis.sdxl_merge_stats import (
    build_fisher_map,
    fisher_key,
    is_oft_key,
    to_coords,
)
from src.diffusion.dataset_1 import get_pair

# Must stay in sync with src/diffusion/correction_hyperparam_search.py::mus(). Duplicated
# rather than imported so this script stays a lightweight, GPU-free safetensors/torch
# consumer instead of pulling in the diffusers/transformers pipeline stack.
DEFAULT_MUS = (-0.5, 0.0, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)

DEFAULT_PLOT_DIR = REPO_ROOT / "outputs/diffusion/analysis/correction_tangent_visualization"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize the OFT tangent-space (so(n) coordinate) geodesics traced by the "
            "diagonal-Fisher correction merge (see correction_hyperparam_search.py) for one "
            "concept/style pair, averaged per matrix block over all layers, with the Fisher-"
            "weighted loss landscape they cross rendered as a background heatmap."
        )
    )
    parser.add_argument("--concept_name", type=str, default="cat")
    parser.add_argument("--style_name", type=str, default="01_01")
    parser.add_argument("--plot_dir", type=Path, default=DEFAULT_PLOT_DIR)
    parser.add_argument("--fisher_min", type=float, default=1e-14)
    parser.add_argument("--fisher_rescale", type=float, default=1e10)
    parser.add_argument(
        "--fim_normalization",
        choices=["none", "trace", "frobenius", "layer-trace", "layer-frobenius"],
        default="frobenius",
        help="Matches the default OFTMerging uses when correction_hyperparam_search.py omits it.",
    )
    parser.add_argument(
        "--mus",
        type=float,
        nargs="+",
        default=list(DEFAULT_MUS),
        help="Correction mu values to trace; defaults to correction_hyperparam_search.py::mus().",
    )
    parser.add_argument(
        "--num_t",
        type=int,
        default=41,
        help=(
            "Points along each t in [0, 1] curve. This is also the 'generation' grid the "
            "background loss heatmap is averaged over (see loss_grid) -- the same t values at "
            "which correction_hyperparam_search.py actually renders an image for each mu."
        ),
    )
    parser.add_argument("--loss_resolution", type=int, default=90, help="Heatmap grid points per axis.")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def latex_text(text):
    escaped = str(text).replace("\\", r"\textbackslash{}").replace("_", r"\_")
    return r"\textrm{" + escaped + "}"


def normalize_fisher_diagonal(diagonal, mode):
    """Mirrors OFTMerging._normalize_fisher_diagonal (src/merging.py)."""
    diagonal = torch.nan_to_num(diagonal.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    if mode == "none":
        return diagonal
    if mode == "trace":
        denom = diagonal.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    elif mode == "frobenius":
        denom = torch.linalg.vector_norm(diagonal, dim=-1, keepdim=True).clamp_min(1e-8)
    elif mode == "layer-trace":
        denom = diagonal.sum().clamp_min(1e-8)
    elif mode == "layer-frobenius":
        denom = torch.linalg.vector_norm(diagonal).clamp_min(1e-8)
    else:
        raise ValueError(f"Unsupported fim_normalization: {mode!r}")
    return diagonal / denom


def prepared_fisher(raw, fisher_min, fisher_rescale, fim_normalization):
    """Fisher weights exactly as used by the merge: clamp -> rescale -> normalize (mirrors
    moft/inferencer_sdxl.py::_prepare_fisher_layer + OFTMerging._normalize_fisher_diagonal)."""
    return normalize_fisher_diagonal(raw.float().clamp_min(fisher_min) * fisher_rescale, fim_normalization)


def diagonal_fisher_merge(concept_coords, style_coords, cf, sf, t):
    """Mirrors OFTMerging._diagonal_fisher_merging for the "fisher" merge mode used by
    correction_hyperparam_search.py, with alphas=(t, 1 - t) (concept, style). `cf`/`sf` must
    already be prepared via `prepared_fisher`.

    This is also, by construction, the closed-form minimizer of the quadratic loss
    L(x) = t * sum_j cf_j (x_j - concept_j)^2 + (1 - t) * sum_j sf_j (x_j - style_j)^2,
    which is exactly the loss `loss_grid` below evaluates as a heatmap.
    """
    alpha_concept, alpha_style = t, 1.0 - t
    denom = alpha_concept * cf + alpha_style * sf
    numer = alpha_concept * cf * concept_coords.float() + alpha_style * sf * style_coords.float()
    return numer / denom.clamp_min(1e-8)


def diagonal_fisher_correction(merged, t, mu):
    """Mirrors moft/inferencer_sdxl.py::_diagonal_fisher_correction's post-merge scaling."""
    if mu is None:
        return merged
    correction = 1.0 + mu * t * (1.0 - t)
    return merged * correction


def block_size_groups(concept_state, keys):
    groups = {}
    for key in keys:
        block_size = concept_state[key].shape[-1]
        groups.setdefault(block_size, []).append(key)
    return dict(sorted(groups.items()))


def analyze_group(concept_state, style_state, concept_fisher, style_fisher, fisher_map, group_keys, args, t_grid):
    concept_blocks = torch.cat([to_coords(concept_state[key]) for key in group_keys], dim=0)
    style_blocks = torch.cat([to_coords(style_state[key]) for key in group_keys], dim=0)
    concept_fisher_blocks = torch.cat(
        [concept_fisher[fisher_key(key, fisher_map)].float() for key in group_keys], dim=0
    )
    style_fisher_blocks = torch.cat(
        [style_fisher[fisher_key(key, fisher_map)].float() for key in group_keys], dim=0
    )
    cf = prepared_fisher(concept_fisher_blocks, args.fisher_min, args.fisher_rescale, args.fim_normalization)
    sf = prepared_fisher(style_fisher_blocks, args.fisher_min, args.fisher_rescale, args.fim_normalization)

    # "Average across all blocks and layers": collapse the (num_blocks * num_layers, dim)
    # representation for this matrix-block type down to one representative vector. The Fisher
    # weights get the same treatment so the loss-grid heatmap (see loss_grid) is built from
    # exactly the concept/style/Fisher vectors the PCA basis below is fit on -- not some other,
    # per-block-precise quantity that would live in a different, unprojected space.
    concept_mean = concept_blocks.mean(dim=0)
    style_mean = style_blocks.mean(dim=0)
    concept_fisher_mean = cf.mean(dim=0)
    style_fisher_mean = sf.mean(dim=0)

    curve_means = {}
    for mu in args.mus:
        points = []
        for t in t_grid.tolist():
            merged = diagonal_fisher_merge(concept_blocks, style_blocks, cf, sf, t)
            merged = diagonal_fisher_correction(merged, t, mu)
            points.append(merged.mean(dim=0))
        curve_means[mu] = torch.stack(points)

    return concept_mean, style_mean, concept_fisher_mean, style_fisher_mean, curve_means


def pca_project(points):
    """Fit PCA (mean-center + SVD) on all rows of `points` at once and project them.

    Fitting a single transform on every point that will appear in a plot -- concept anchor,
    style anchor, and every mu-curve -- keeps axes 1 and 2 meaningful and comparable across
    the whole figure instead of re-fitting (and thus rotating/rescaling) per curve. Returns the
    fitted center and top-2 components too, so anything else (e.g. the loss-grid heatmap) can be
    mapped through the exact same transform.
    """
    center = points.mean(axis=0, keepdims=True)
    centered = points - center
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2]  # (2, D)
    coords = centered @ components.T
    explained = (s[:2] ** 2) / ((s**2).sum() + 1e-30)
    return coords, explained, center[0], components


def loss_grid(components, center, concept_vec, style_vec, concept_fisher_vec, style_fisher_vec, t_values, xlim, ylim, resolution):
    """Fisher-weighted quadratic loss, averaged over every 'generation' t -- the same t grid used
    to render the mu-curves (and, in correction_hyperparam_search.py, to actually generate one
    image per t): mean_t [ t*||x-concept||^2_{F_c} + (1-t)*||x-style||^2_{F_s} ], evaluated on a
    regular grid directly in the PC1-PC2 plane that the scatter/curve points were projected into.

    Choosing the grid points consistently with the PCA projection: a heatmap grid point (gx, gy)
    is *not* an independent sample -- it is mapped back into the same D-dimensional so(n)-
    coordinate space the PCA basis came from, via the same affine map used to fit that basis:
    x_full = center + gx*pc1 + gy*pc2. Because the PCA only keeps 2 of D components, this
    reconstruction is the orthogonal projection of the true D-dim point onto the 2-D plane spanned
    by (pc1, pc2) -- i.e. the heatmap is the true loss landscape restricted to exactly the plane
    the concept/style anchors and mu-curves already live on (up to whatever variance the discarded
    components hold), so a heatmap cell under a plotted point is always a faithful read of that
    point's loss, and grid extent/units match the scatter axes without any separate rescaling step.

    Averaging over t: L(x; t) is affine in t, so mean_t L(x; t) == L(x; mean(t_values)) exactly --
    no need to loop over t_values or re-reconstruct the grid per t. mean(t_values) is computed
    from the actual generation grid (not hard-coded) so this stays correct even if it is ever made
    non-uniform/asymmetric; for the default linspace(0, 1, num_t) it works out to 0.5, the point
    where the mu*t*(1-t) correction term (and thus deviation from this Fisher-optimal landscape)
    is largest.
    """
    xs = np.linspace(*xlim, resolution)
    ys = np.linspace(*ylim, resolution)
    gx, gy = np.meshgrid(xs, ys)
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1)  # (R*R, 2)
    reconstructed = center[None, :] + grid @ components  # (R*R, D)
    concept_term = ((reconstructed - concept_vec[None, :]) ** 2 * concept_fisher_vec[None, :]).sum(axis=1)
    style_term = ((reconstructed - style_vec[None, :]) ** 2 * style_fisher_vec[None, :]).sum(axis=1)
    t_bar = float(np.mean(t_values))
    loss = t_bar * concept_term + (1.0 - t_bar) * style_term
    return xs, ys, loss.reshape(resolution, resolution), t_bar


def draw_group(
    ax,
    block_size,
    concept_mean,
    style_mean,
    concept_fisher_mean,
    style_fisher_mean,
    curve_means,
    t_grid,
    n_layers,
    n_blocks_per_layer,
    loss_resolution,
):
    num_t = len(t_grid)
    mus_sorted = sorted(curve_means)

    stacked = np.concatenate(
        [
            concept_mean.numpy()[None, :],
            style_mean.numpy()[None, :],
            *(curve_means[mu].numpy() for mu in mus_sorted),
        ],
        axis=0,
    )
    coords, explained, center, components = pca_project(stacked)

    concept_xy = coords[0]
    style_xy = coords[1]
    curve_xy = {}
    offset = 2
    for mu in mus_sorted:
        curve_xy[mu] = coords[offset : offset + num_t]
        offset += num_t

    pad_x = 0.15 * max(np.ptp(coords[:, 0]), 1e-12)
    pad_y = 0.15 * max(np.ptp(coords[:, 1]), 1e-12)
    xlim = (coords[:, 0].min() - pad_x, coords[:, 0].max() + pad_x)
    ylim = (coords[:, 1].min() - pad_y, coords[:, 1].max() + pad_y)

    xs, ys, loss, t_bar = loss_grid(
        components,
        center,
        concept_mean.numpy(),
        style_mean.numpy(),
        concept_fisher_mean.numpy(),
        style_fisher_mean.numpy(),
        t_grid.tolist(),
        xlim,
        ylim,
        loss_resolution,
    )
    mesh = ax.pcolormesh(
        xs,
        ys,
        loss,
        cmap="cividis",
        norm=LogNorm(vmin=max(loss.min(), loss.max() * 1e-6), vmax=loss.max()),
        shading="gouraud",
        zorder=0,
        rasterized=True,
    )
    cbar = plt.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(rf"$\langle L(x;t)\rangle_{{t}},\ \bar t={t_bar:g}$", fontsize=9)

    white_halo = [patheffects.Stroke(linewidth=3.2, foreground="white", alpha=0.85), patheffects.Normal()]
    cmap = plt.get_cmap("plasma")
    norm = plt.Normalize(vmin=min(mus_sorted), vmax=max(mus_sorted))
    marker_every = max(1, num_t // 8)
    for mu in mus_sorted:
        xy = curve_xy[mu]
        color = cmap(norm(mu))
        ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=1.8, zorder=3, path_effects=white_halo, label=rf"$\mu={mu:g}$")
        ax.scatter(xy[::marker_every, 0], xy[::marker_every, 1], color=color, s=16, zorder=4, edgecolor="white", linewidth=0.4)
        ax.annotate(
            "$t=1$",
            xy[-1],
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=6,
            color=color,
        )

    ax.scatter(*concept_xy, marker="*", s=280, color="tab:blue", edgecolor="white", linewidth=0.8, zorder=5)
    ax.annotate(latex_text("concept"), concept_xy, textcoords="offset points", xytext=(6, 6), fontsize=9, fontweight="bold", color="white")
    ax.scatter(*style_xy, marker="D", s=140, color="tab:red", edgecolor="white", linewidth=0.8, zorder=5)
    ax.annotate(latex_text("style"), style_xy, textcoords="offset points", xytext=(6, 6), fontsize=9, fontweight="bold", color="white")

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(f"PC1 ({explained[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1%} var)")
    ax.set_title(
        latex_text(f"block size {block_size} ({n_layers} layers x {n_blocks_per_layer} blocks averaged)"),
        fontsize=10,
    )
    ax.legend(fontsize=7, ncol=2, loc="best", framealpha=0.75)


def plot_group_standalone(
    pair_name,
    block_size,
    concept_mean,
    style_mean,
    concept_fisher_mean,
    style_fisher_mean,
    curve_means,
    t_grid,
    n_layers,
    n_blocks_per_layer,
    loss_resolution,
    plot_dir,
):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    draw_group(
        ax,
        block_size,
        concept_mean,
        style_mean,
        concept_fisher_mean,
        style_fisher_mean,
        curve_means,
        t_grid,
        n_layers,
        n_blocks_per_layer,
        loss_resolution,
    )
    ax.set_title(latex_text(f"{pair_name}: OFT tangent space") + "\n" + ax.get_title(), fontsize=10)
    fig.tight_layout()
    plot_dir.mkdir(parents=True, exist_ok=True)
    save_path = plot_dir / f"{pair_name}_block{block_size}_tangent.png"
    fig.savefig(save_path, dpi=170)
    plt.close(fig)
    return save_path


def main():
    args = parse_args()
    pair = get_pair(args.concept_name, args.style_name)
    device = torch.device(args.device)

    concept_state = load_file(pair["concept"]["adapter_path"], device=str(device))
    style_state = load_file(pair["style"]["adapter_path"], device=str(device))
    concept_fisher = load_file(pair["concept"]["fim_path"], device=str(device))
    style_fisher = load_file(pair["style"]["fim_path"], device=str(device))

    keys = sorted(k for k in concept_state if is_oft_key(k) and k in style_state)
    if not keys:
        raise RuntimeError(f"No OFT keys found for pair {pair['name']!r}")
    fisher_map = build_fisher_map(concept_state, concept_fisher, keys)

    groups = block_size_groups(concept_state, keys)
    t_grid = torch.linspace(0.0, 1.0, args.num_t)

    plot_dir = args.plot_dir
    plot_dir.mkdir(parents=True, exist_ok=True)

    n_groups = len(groups)
    fig_combined, axes_combined = plt.subplots(1, n_groups, figsize=(7 * n_groups, 6))
    axes_combined = np.atleast_1d(axes_combined)

    for ax, (block_size, group_keys) in zip(axes_combined, groups.items()):
        n_blocks_per_layer = concept_state[group_keys[0]].shape[0]
        print(
            f"[correction-tangent] pair={pair['name']} block_size={block_size} "
            f"layers={len(group_keys)} blocks_per_layer={n_blocks_per_layer}",
            flush=True,
        )
        concept_mean, style_mean, concept_fisher_mean, style_fisher_mean, curve_means = analyze_group(
            concept_state, style_state, concept_fisher, style_fisher, fisher_map, group_keys, args, t_grid
        )

        save_path = plot_group_standalone(
            pair["name"],
            block_size,
            concept_mean,
            style_mean,
            concept_fisher_mean,
            style_fisher_mean,
            curve_means,
            t_grid,
            len(group_keys),
            n_blocks_per_layer,
            args.loss_resolution,
            plot_dir,
        )
        print(f"[correction-tangent] saved {save_path}", flush=True)

        draw_group(
            ax,
            block_size,
            concept_mean,
            style_mean,
            concept_fisher_mean,
            style_fisher_mean,
            curve_means,
            t_grid,
            len(group_keys),
            n_blocks_per_layer,
            args.loss_resolution,
        )

    fig_combined.suptitle(
        latex_text(f"{pair['name']}: OFT tangent-space geodesics under diagonal-Fisher correction"),
        fontsize=13,
    )
    fig_combined.tight_layout(rect=(0, 0, 1, 0.94))
    combined_path = plot_dir / f"{pair['name']}_all_blocks_tangent.png"
    fig_combined.savefig(combined_path, dpi=170)
    plt.close(fig_combined)
    print(f"[correction-tangent] saved {combined_path}", flush=True)


if __name__ == "__main__":
    main()
