from __future__ import annotations

import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np


# Interpolation

def plot_interpolation_curve(
    alphas: List[float],
    losses: List[float],
    title: str = "Loss interpolation",
    save_path: Optional[str] = None,
    xlabel: str = "alpha (0 = pretrained, 1 = fine-tuned)",
    ylabel: str = "Cross-entropy loss",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(alphas, losses, marker="o", linewidth=2, markersize=5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


# Visualization

def plot_cosine_similarity_matrix(
    sim_matrix: np.ndarray,
    task_names: List[str],
    title: str = "Avg. layerwise cosine similarity",
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> None:
    """Render a heatmap of a pre-computed (T, T) pairwise cosine-similarity matrix."""
    T = sim_matrix.shape[0]
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(4, T), max(3, T - 1)))

    im = ax.imshow(sim_matrix, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(T))
    ax.set_xticklabels(task_names, rotation=45, ha="right")
    ax.set_yticks(range(T))
    ax.set_yticklabels(task_names)
    for i in range(T):
        for j in range(T):
            ax.text(j, i, f"{sim_matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)

    if fig is not None:
        fig.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()


def plot_norm_distributions(
    norms_per_task: List[List[float]],
    task_names: List[str],
    title: str = "Per-layer Frobenius norm distribution",
    save_path: Optional[str] = None,
    axes: Optional[List[plt.Axes]] = None,
) -> None:
    """Histogram of per-layer Frobenius norms for each task.

    Args:
        norms_per_task: T lists, each containing one scalar norm per layer key.
        task_names:     Human-readable label per task.
    """
    T = len(norms_per_task)
    fig = None
    if axes is None:
        fig, axes_raw = plt.subplots(1, T, figsize=(5 * T, 4), sharey=False)
        axes = [axes_raw] if T == 1 else list(axes_raw)

    for t, ax in enumerate(axes):
        ax.hist(norms_per_task[t], bins=30, alpha=0.8, color="steelblue", density=True)
        ax.set_title(task_names[t])
        ax.set_xlabel("Frobenius norm")
        ax.set_ylabel("Density")

    if fig is not None:
        fig.suptitle(title, y=1.02)
        fig.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()


def _plot_layer_strip(
    values: np.ndarray,
    labels: List[str],
    title: str,
    cmap: str,
    colorbar_label: str,
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> None:
    """1xL colour strip where each cell encodes one scalar per layer."""
    L = len(labels)
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(min(max(8, L // 4), 40), 4))

    im = ax.imshow(values[np.newaxis, :], cmap=cmap, aspect="auto")
    ax.set_xticks(range(L))
    ax.set_xticklabels(labels, rotation=90, fontsize=5)
    ax.set_yticks([])
    for j, v in enumerate(values):
        ax.text(j, 0, f"{v:.2f}", ha="center", va="center", fontsize=4, color="black")
    plt.colorbar(im, ax=ax, label=colorbar_label, fraction=0.046, pad=0.04)
    ax.set_title(title)

    if fig is not None:
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()


def plot_layer_norm_variance(
    values: np.ndarray,
    labels: List[str],
    title: str = "Per-layer norm variance across models",
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> None:
    """Strip plot where each cell = variance of Frobenius norms across tasks."""
    _plot_layer_strip(
        values, labels, title,
        cmap="Greens", colorbar_label="Norm variance across models",
        save_path=save_path, ax=ax,
    )


def plot_layer_cosine_similarity(
    values: np.ndarray,
    labels: List[str],
    title: str = "Per-layer avg pairwise cosine similarity",
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> None:
    """Strip plot where each cell = avg pairwise cosine similarity across model pairs."""
    _plot_layer_strip(
        values, labels, title,
        cmap="Blues", colorbar_label="Avg pairwise cosine similarity",
        save_path=save_path, ax=ax,
    )


def plot_layer_geodesic_distance(
    values: np.ndarray,
    labels: List[str],
    title: str = "Per-layer avg pairwise geodesic distance (SO(n))",
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> None:
    """Strip plot where each cell = avg pairwise geodesic distance across model pairs."""
    _plot_layer_strip(
        values, labels, title,
        cmap="Oranges", colorbar_label="Avg pairwise geodesic distance",
        save_path=save_path, ax=ax,
    )


def plot_module_layer_distributions(
    module_data: dict,
    save_path: Optional[str] = None,
) -> None:
    """Three-panel plot: geodesic dist, cosine sim, norm variance — one line per module, x = layer.

    Args:
        module_data: {module_name: {"layers": [int], "geo": [float], "cos": [float], "mean": [float]}}
    """
    metrics = [
        ("geo", "Avg geodesic distance (SO(n))", "Oranges"),
        ("cos", "Avg cosine similarity",         "Blues"),
        ("mean", "Mean norm",                     "Greens"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False)
    for ax, (key, ylabel, cmap) in zip(axes, metrics):
        cmap_fn = plt.get_cmap(cmap)
        modules = list(module_data.keys())
        colors = [cmap_fn(0.4 + 0.5 * i / max(len(modules) - 1, 1)) for i in range(len(modules))]
        for mod, color in zip(modules, colors):
            d = module_data[mod]
            ax.plot(d["layers"], d[key], marker="o", markersize=3, linewidth=1.2, label=mod, color=color)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Layer")
        ax.legend(fontsize=7, ncol=4, loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.4)
    fig.suptitle("Per-module layer distributions (all models vs. all models)", y=1.01)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
