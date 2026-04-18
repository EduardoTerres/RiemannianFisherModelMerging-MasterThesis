"""Visualization utilities for OFT task vectors: cosine similarity and norm distributions."""

from typing import Dict, List, Optional
import re
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _oft_keys(task_vectors: List[Dict[str, Tensor]]) -> List[str]:
    return sorted(task_vectors[0].keys(), key=lambda k: int(next(iter(re.findall(r"\d+", k)), 0)))


def _frobenius_norm(v: Tensor) -> float:
    """Frobenius norm of a (num_blocks, d) tensor, treating it as a flat matrix."""
    return v.norm(p="fro").item()


def _layerwise_cosine(vi: Tensor, vj: Tensor) -> float:
    """
    Frobenius cosine similarity between two so(n) task vectors at one layer.

    vi, vj: (num_blocks, d) — upper-triangle so(n) parameters per block.
    For skew-symmetric matrices <A,B>_F = 2(a·b) and ||A||_F = sqrt(2)||a||,
    so the cosine cancels the factor-of-2 and reduces to cosine on the flat param vector.
    """
    fi, fj = vi.flatten(), vj.flatten()
    return (fi @ fj / (fi.norm().clamp(min=1e-8) * fj.norm().clamp(min=1e-8))).item()


# ---------------------------------------------------------------------------
# Function 1 — Pairwise layerwise cosine-similarity matrix
# ---------------------------------------------------------------------------

def plot_cosine_similarity_matrix(
    task_vectors: List[Dict[str, Tensor]],
    task_names: Optional[List[str]] = None,
    title: str = "Avg. layerwise cosine similarity",
    ax: Optional[plt.Axes] = None,
) -> np.ndarray:
    """
    For T tasks, build a (T, T) matrix where entry (i, j) is the cosine similarity
    averaged over all OFT layers.

    task_vectors: T dicts, each mapping layer key -> (num_blocks, d)
    Returns the (T, T) similarity matrix.
    """
    T = len(task_vectors)
    keys = _oft_keys(task_vectors)
    names = task_names or [f"Task {i}" for i in range(T)]

    sim_matrix = np.zeros((T, T))
    for i in tqdm(range(T), desc="Cosine similarity"):
        for j in range(T):
            sims = [_layerwise_cosine(task_vectors[i][k], task_vectors[j][k]) for k in keys]
            sim_matrix[i, j] = float(np.mean(sims))

    if ax is None:
        _, ax = plt.subplots(figsize=(max(4, T), max(3, T - 1)))

    im = ax.imshow(sim_matrix, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(T))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(T))
    ax.set_yticklabels(names)
    for i in range(T):
        for j in range(T):
            ax.text(j, i, f"{sim_matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    return sim_matrix


# ---------------------------------------------------------------------------
# Function 2 — Distribution of task-vector Frobenius norms
# ---------------------------------------------------------------------------

def plot_norm_distributions(
    standard_task_vectors: List[Dict[str, Tensor]],
    fisher_task_vectors: List[Dict[str, Tensor]],
    task_names: Optional[List[str]] = None,
    axes: Optional[List[plt.Axes]] = None,
) -> None:
    """
    For each task, plot the distribution (histogram + KDE) of per-layer Frobenius norms,
    comparing standard vs. Fisher-rescaled task vectors side by side.

    standard_task_vectors: T dicts, each mapping layer key -> (num_blocks, d)
    fisher_task_vectors:   T dicts, each mapping layer key -> (num_blocks, d)
    """
    T = len(standard_task_vectors)
    names = task_names or [f"Task {i}" for i in range(T)]
    keys = _oft_keys(standard_task_vectors)

    if axes is None:
        _, axes = plt.subplots(1, T, figsize=(5 * T, 4), sharey=False)
        if T == 1:
            axes = [axes]

    for t, ax in enumerate(axes):
        # Per-layer Frobenius norms — one scalar per layer key
        std_norms = [_frobenius_norm(standard_task_vectors[t][k]) for k in keys]
        fsh_norms = [_frobenius_norm(fisher_task_vectors[t][k]) for k in keys]

        bins = 30
        ax.hist(std_norms, bins=bins, alpha=0.6, label="Standard", color="steelblue", density=True)
        ax.hist(fsh_norms, bins=bins, alpha=0.6, label="Fisher", color="darkorange", density=True)
        ax.set_title(names[t])
        ax.set_xlabel("Frobenius norm")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    plt.suptitle("Per-layer Frobenius norm distribution", y=1.02)


# ---------------------------------------------------------------------------
# Function 3 — Per-layer strip plots
# ---------------------------------------------------------------------------

def _plot_layer_strip(values: np.ndarray, keys: List[str], title: str,
                      cmap: str, colorbar_label: str,
                      ax: Optional[plt.Axes] = None) -> None:
    """Render a 1×L strip of squares; color encodes `values`, text shows the value."""
    L = len(keys)
    if ax is None:
        _, ax = plt.subplots(figsize=(max(8, L // 2), 2))
    im = ax.imshow(values[np.newaxis, :], cmap=cmap, aspect="auto")
    labels = [next(iter(re.findall(r"\d+", k)), k) for k in keys]
    ax.set_xticks(range(L))
    ax.set_xticklabels(labels, rotation=90, fontsize=5)
    ax.set_yticks([])
    for j, v in enumerate(values):
        ax.text(j, 0, f"{v:.2f}", ha="center", va="center", fontsize=4, color="black")
    plt.colorbar(im, ax=ax, label=colorbar_label, fraction=0.046, pad=0.04)
    ax.set_title(title)


def plot_layer_norm_variance(
    task_vectors: List[Dict[str, Tensor]],
    title: str = "Per-layer norm variance across models",
    ax: Optional[plt.Axes] = None,
) -> None:
    """Strip where each cell = variance of Frobenius norms across tasks."""
    keys = sorted(task_vectors[0].keys(), key=lambda k: int(next(iter(re.findall(r"\d+", k)), 0)))
    values = np.array([
        float(np.var([_frobenius_norm(tv[k]) for tv in task_vectors]))
        for k in tqdm(keys, desc=title)
    ])
    _plot_layer_strip(values, keys, title, cmap="Greens",
                      colorbar_label="Norm variance across models", ax=ax)


def plot_layer_cosine_agreement(
    task_vectors: List[Dict[str, Tensor]],
    title: str = "Per-layer avg pairwise cosine similarity",
    ax: Optional[plt.Axes] = None,
) -> None:
    """Strip where each cell = average pairwise cosine similarity across all model pairs."""
    keys = sorted(task_vectors[0].keys(), key=lambda k: int(next(iter(re.findall(r"\d+", k)), 0)))
    T = len(task_vectors)
    values = []
    for k in tqdm(keys, desc=title):
        sims = []
        for i in range(T):
            for j in range(i + 1, T):
                vi, vj = task_vectors[i][k], task_vectors[j][k]
                sims.append(_layerwise_cosine(vi, vj))
        values.append(float(np.mean(sims)) if sims else 0.0)
    _plot_layer_strip(np.array(values), keys, title, cmap="Blues",
                      colorbar_label="Avg pairwise cosine similarity", ax=ax)


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from src.merging import OFTMerging
    from src.analysis.utils import fisher_task_vectors

    parser = argparse.ArgumentParser(description="Visualize OFT task vectors.")
    parser.add_argument("--adapter_paths", nargs="+", required=True,
                        help="Paths to OFT adapter dirs")
    parser.add_argument("--fisher_paths", nargs="+", required=True,
                        help="Paths to Fisher .safetensors files")
    parser.add_argument("--task_names", nargs="+", default=None,
                        help="Human-readable task labels")
    parser.add_argument("--lam",    type=float, default=1.0, help="Regularisation lambda")
    parser.add_argument("--output", type=str,   default="task_vector_analysis.png")
    args = parser.parse_args()

    T = len(args.adapter_paths)
    alphas = [1.0 / T] * T
    device = "cuda" if torch.cuda.is_available() else "cpu"
    merging = OFTMerging(lam=args.lam, alphas=alphas, device=device)

    std_tvs = merging.load_weights(args.adapter_paths)
    fishers  = merging.load_fishers(args.fisher_paths)
    fsh_tvs  = fisher_task_vectors(
        oft_params_per_task=std_tvs,
        fisher_per_task=fishers,
        alphas=alphas,
        merging=merging,
        lam=args.lam,
    )

    merged_std = merging.merge(args.adapter_paths, mode="plain")
    merged_fsh = merging.merge(args.adapter_paths, args.fisher_paths, mode="diagonal_fisher")

    all_std_tvs = std_tvs + [merged_std]
    all_fsh_tvs = fsh_tvs + [merged_fsh]
    base_names  = args.task_names or [f"Task {i}" for i in range(T)]
    all_names   = base_names + ["merged"]

    base, ext = args.output.rsplit(".", 1) if "." in args.output else (args.output, "png")

    # --- Figure 1: cosine similarity matrices ---
    fig1, (ax_sim_std, ax_sim_fsh) = plt.subplots(1, 2, figsize=(12, 6))
    plot_cosine_similarity_matrix(all_std_tvs, all_names, title="Standard — avg cosine similarity", ax=ax_sim_std)
    plot_cosine_similarity_matrix(all_fsh_tvs, all_names, title="Fisher — avg cosine similarity",   ax=ax_sim_fsh)
    fig1.tight_layout()
    p1 = f"{base}_cosine_similarity.{ext}"
    fig1.savefig(p1, bbox_inches="tight")
    print(f"Saved → {p1}")

    # --- Figure 2: norm distributions (all tasks + merged) ---
    N = T + 1
    fig2, axes_norm = plt.subplots(1, N, figsize=(5 * N, 4), sharey=False)
    if N == 1:
        axes_norm = [axes_norm]
    plot_norm_distributions(all_std_tvs, all_fsh_tvs, task_names=all_names, axes=list(axes_norm))
    fig2.tight_layout()
    p2 = f"{base}_norm_distributions.{ext}"
    fig2.savefig(p2, bbox_inches="tight")
    print(f"Saved → {p2}")

    # --- Figure 3: per-layer norm variance strip ---
    fig3, (ax3_std, ax3_fsh) = plt.subplots(2, 1, figsize=(20, 5))
    plot_layer_norm_variance(all_std_tvs, title="Standard — per-layer norm variance", ax=ax3_std)
    plot_layer_norm_variance(all_fsh_tvs, title="Fisher — per-layer norm variance",   ax=ax3_fsh)
    fig3.tight_layout()
    p3 = f"{base}_norm_variance.{ext}"
    fig3.savefig(p3, bbox_inches="tight")
    print(f"Saved → {p3}")

    # --- Figure 3b: per-layer avg pairwise cosine similarity strip ---
    fig3b, (ax3b_std, ax3b_fsh) = plt.subplots(2, 1, figsize=(20, 5))
    plot_layer_cosine_agreement(all_std_tvs, title="Standard — per-layer avg pairwise cosine similarity", ax=ax3b_std)
    plot_layer_cosine_agreement(all_fsh_tvs, title="Fisher — per-layer avg pairwise cosine similarity",   ax=ax3b_fsh)
    fig3b.tight_layout()
    p3b = f"{base}_cosine_agreement.{ext}"
    fig3b.savefig(p3b, bbox_inches="tight")
    print(f"Saved → {p3b}")
