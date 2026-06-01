from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
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


def plot_fisher_vectors(
    norm_ratio: np.ndarray,
    angles: np.ndarray,
    task_labels: List[str],
    title: str = "Standard vs Fisher-full task vectors",
    save_path: Optional[str] = None,
) -> None:
    """2×2 grid: norm-ratio lines, angle lines, and their heatmaps (tasks × layers)."""
    T, L = norm_ratio.shape
    layer_idx = np.arange(L)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    ax_r, ax_a, ax_hr, ax_ha = axes.flat

    for t, lbl in enumerate(task_labels):
        c = colors[t % len(colors)]
        ax_r.plot(layer_idx, norm_ratio[t], color=c, lw=1.5, label=lbl)
        ax_a.plot(layer_idx, angles[t],     color=c, lw=1.5, label=lbl)

    ax_r.axhline(1.0, color="k", lw=0.8, ls=":", label="ratio = 1")
    ax_r.set_title("Norm ratio  ‖f_t‖ / ‖ξ_t‖")
    ax_r.set_xlabel("layer index")
    ax_r.grid(True, lw=0.3, alpha=0.5)
    ax_r.legend(fontsize=8)

    ax_a.set_title("Angle  ∠(ξ_t, f_t)  [°]")
    ax_a.set_xlabel("layer index")
    ax_a.grid(True, lw=0.3, alpha=0.5)
    ax_a.legend(fontsize=8)

    im_r = ax_hr.imshow(norm_ratio, aspect="auto", cmap="RdBu_r", vmin=0.5, vmax=1.5)
    ax_hr.set_yticks(range(T)); ax_hr.set_yticklabels(task_labels)
    ax_hr.set_xlabel("layer index"); ax_hr.set_title("Norm ratio  (heatmap)")
    fig.colorbar(im_r, ax=ax_hr, shrink=0.85)

    im_a = ax_ha.imshow(angles, aspect="auto", cmap="YlOrRd")
    ax_ha.set_yticks(range(T)); ax_ha.set_yticklabels(task_labels)
    ax_ha.set_xlabel("layer index"); ax_ha.set_title("Angle [°]  (heatmap)")
    fig.colorbar(im_a, ax=ax_ha, shrink=0.85)

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_fisher_stats(
    f_mean: np.ndarray,
    f_std: np.ndarray,
    all_vals: List[np.ndarray],
    task_labels: List[str],
    title: str = "Fisher matrix statistics",
    save_path: Optional[str] = None,
) -> None:
    """2×2 grid: mean/std per layer (log scale), log-mean heatmap, per-task box plot."""
    T, L = f_mean.shape
    layer_idx = np.arange(L)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    ax_mean, ax_std, ax_hm, ax_box = axes.flat

    for t, lbl in enumerate(task_labels):
        c = colors[t % len(colors)]
        ax_mean.plot(layer_idx, f_mean[t], color=c, lw=1.5, label=lbl)
        ax_std.plot(layer_idx,  f_std[t],  color=c, lw=1.5, label=lbl)

    ax_mean.set_title("Mean Fisher value per layer")
    ax_mean.set_xlabel("layer index"); ax_mean.set_yscale("log")
    ax_mean.grid(True, lw=0.3, alpha=0.5); ax_mean.legend(fontsize=8)

    ax_std.set_title("Std Fisher value per layer")
    ax_std.set_xlabel("layer index"); ax_std.set_yscale("log")
    ax_std.grid(True, lw=0.3, alpha=0.5); ax_std.legend(fontsize=8)

    im = ax_hm.imshow(np.log10(f_mean + 1e-30), aspect="auto", cmap="viridis")
    ax_hm.set_yticks(range(T)); ax_hm.set_yticklabels(task_labels)
    ax_hm.set_xlabel("layer index"); ax_hm.set_title("log₁₀(mean Fisher)  (heatmap)")
    fig.colorbar(im, ax=ax_hm, shrink=0.85)

    ax_box.boxplot(all_vals, labels=task_labels, showfliers=False)
    ax_box.set_yscale("log"); ax_box.set_title("Fisher value distribution per task")
    ax_box.set_ylabel("Fisher value"); ax_box.grid(True, lw=0.3, alpha=0.5, axis="y")

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_transport_effect(
    off_diag_frac: np.ndarray,
    diag_std_diff: np.ndarray,
    task_labels: List[str],
    title: str = "Parallel transport effect on Fisher",
    save_path: Optional[str] = None,
) -> None:
    """2×2 grid showing how transport fills off-diagonal mass and redistributes diagonal std.

    Args:
        off_diag_frac: (T, L) fraction of Frobenius norm in off-diagonal entries of F̃_t.
        diag_std_diff: (T, L) std(diag(F̃_t)) - std(f_t); negative = transport homogenises.
    """
    T, L = off_diag_frac.shape
    layer_idx = np.arange(L)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    ax_frac, ax_std, ax_hf, ax_hs = axes.flat

    for t, lbl in enumerate(task_labels):
        c = colors[t % len(colors)]
        ax_frac.plot(layer_idx, off_diag_frac[t], color=c, lw=1.5, label=lbl)
        ax_std.plot(layer_idx,  diag_std_diff[t],  color=c, lw=1.5, label=lbl)

    ax_frac.set_title("Off-diagonal fraction  ||off_diag(F̃)||_F / ||F̃||_F")
    ax_frac.set_xlabel("layer index")
    ax_frac.grid(True, lw=0.3, alpha=0.5)
    ax_frac.legend(fontsize=8)

    ax_std.axhline(0, color="k", lw=0.8, ls=":")
    ax_std.set_title("Δ std(diag)  =  std(diag(F̃_t)) − std(f_t)  [<0 → homogenisation]")
    ax_std.set_xlabel("layer index")
    ax_std.grid(True, lw=0.3, alpha=0.5)
    ax_std.legend(fontsize=8)

    im_f = ax_hf.imshow(off_diag_frac, aspect="auto", cmap="YlOrRd")
    ax_hf.set_yticks(range(T)); ax_hf.set_yticklabels(task_labels)
    ax_hf.set_xlabel("layer index"); ax_hf.set_title("Off-diagonal fraction  (heatmap)")
    fig.colorbar(im_f, ax=ax_hf, shrink=0.85)

    im_s = ax_hs.imshow(diag_std_diff, aspect="auto", cmap="RdBu_r")
    ax_hs.set_yticks(range(T)); ax_hs.set_yticklabels(task_labels)
    ax_hs.set_xlabel("layer index"); ax_hs.set_title("Δ std(diag)  (heatmap)")
    fig.colorbar(im_s, ax=ax_hs, shrink=0.85)

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_gradient_analysis(
    results: dict,
    family_name: str,
    save_path: Optional[str] = None,
) -> None:
    """2x3 grid comparing gradient vs task-vector alignment per task.

    Args:
        results: {task_name: {param_name: {cos_raw, cos_fisher, rel_diff_raw,
                  rel_diff_fisher, normed_diff_raw, normed_diff_fisher,
                  norm_tv, norm_grad, norm_precond, abs_diff_raw, abs_diff_fisher}}}
        family_name: used in the suptitle.
        save_path: if given, save figure there; otherwise show.
    """
    tasks = list(results.keys())
    x = np.arange(len(tasks))
    w = 0.35

    def _means(key):
        return [np.mean([v[key] for v in results[t].values()]) for t in tasks]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    # (a) Cosine similarity
    ax = axes[0, 0]
    ax.bar(x - w/2, _means("cos_raw"),    w, label="H0 = I")
    ax.bar(x + w/2, _means("cos_fisher"), w, label="H0 = avg(F)")
    ax.axhline(1.0, color="r", linestyle="--", alpha=0.4, label="ideal")
    ax.set_title("Cosine  cos(-H0_inv * grad_l, xi)")
    ax.set_ylabel("cosine"); ax.set_ylim(-1, 1)
    ax.set_xticks(x); ax.set_xticklabels(tasks, rotation=20, ha="right")
    ax.legend(fontsize=8)

    # (b) Relative difference
    ax = axes[0, 1]
    ax.bar(x - w/2, _means("rel_diff_raw"),    w, label="H0 = I")
    ax.bar(x + w/2, _means("rel_diff_fisher"), w, label="H0 = avg(F)")
    ax.axhline(0.0, color="r", linestyle="--", alpha=0.4, label="ideal")
    ax.set_title("Relative diff  ||xi - (-H0_inv*g)|| / ||xi||")
    ax.set_ylabel("relative mismatch")
    ax.set_xticks(x); ax.set_xticklabels(tasks, rotation=20, ha="right")
    ax.legend(fontsize=8)

    # (c) Normed diff (unit-sphere, fair scale)
    ax = axes[0, 2]
    ax.bar(x - w/2, _means("normed_diff_raw"),    w, label="H0 = I")
    ax.bar(x + w/2, _means("normed_diff_fisher"), w, label="H0 = avg(F)")
    ax.axhline(0.0, color="r", linestyle="--", alpha=0.4, label="ideal")
    ax.set_title("Normed diff  ||xi_hat - v_hat||  (fair scale)")
    ax.set_ylabel("direction mismatch (unit sphere)")
    ax.set_xticks(x); ax.set_xticklabels(tasks, rotation=20, ha="right")
    ax.legend(fontsize=8)

    # (d) Gradient norms
    ax = axes[1, 0]
    ax.bar(tasks, _means("norm_grad"), color="steelblue")
    ax.set_title("Mean gradient norm  ||grad_l_t||")
    ax.set_ylabel("||grad_l_t||")
    ax.set_xticklabels(tasks, rotation=20, ha="right")

    # (e) Task-vector norm vs preconditioned-gradient norm
    ax = axes[1, 1]
    ax.bar(x - w/2, _means("norm_tv"),      w, label="||xi_t||  (task vec)")
    ax.bar(x + w/2, _means("norm_precond"), w, label="||H0_inv*g_t||  (Fisher)")
    ax.set_title("Norm: task vector vs H0_inv * grad_l_t")
    ax.set_ylabel("||*||")
    ax.set_xticks(x); ax.set_xticklabels(tasks, rotation=20, ha="right")
    ax.legend(fontsize=8)

    # (f) Absolute difference norms
    ax = axes[1, 2]
    ax.bar(x - w/2, _means("abs_diff_raw"),    w, label="H0 = I")
    ax.bar(x + w/2, _means("abs_diff_fisher"), w, label="H0 = avg(F)")
    ax.axhline(0.0, color="r", linestyle="--", alpha=0.4, label="ideal")
    ax.set_title("Absolute diff  ||xi - (-H0_inv*g)||")
    ax.set_ylabel("||*||")
    ax.set_xticks(x); ax.set_xticklabels(tasks, rotation=20, ha="right")
    ax.legend(fontsize=8)

    fig.suptitle(f"Gradient vs Task Vector -- {family_name}", fontsize=13)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_oft_covariance_eigenvalues(
    eigenvalues: np.ndarray,
    labels: List[str],
    title: str = "Top eigenvalues of per-layer OFT covariance",
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> None:
    """Line plot: x = layer depth, one line per top-k eigenvalue rank.

    Args:
        eigenvalues: (L, K) array; eigenvalues[:, 0] is the largest eigenvalue.
    """
    L, K = eigenvalues.shape
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(min(max(8, L // 4), 40), 4))

    for i in range(K):
        ax.plot(range(L), eigenvalues[:, i], linewidth=1.5, label=f"$\\lambda_{{{i+1}}}$")

    ax.set_xticks(range(L))
    ax.set_xticklabels(labels, rotation=90, fontsize=5)
    ax.set_xlabel("Layer depth")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)

    if fig is not None:
        fig.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()


def plot_pca_task_vectors(
    vectors: Dict[str, np.ndarray],
    n_tasks: int,
    title: str = "PCA of task vectors",
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> None:
    """2-D PCA scatter of task vectors and merged solutions.

    Args:
        vectors: Ordered dict mapping label -> 1-D numpy vector.
            The first n_tasks entries are task vectors; the rest are merged solutions.
        n_tasks:    Number of task vector entries at the start of `vectors`.
        title:      Axes title.
        save_path:  If given, save the figure; otherwise show.
        ax:         If provided, draw into this Axes (no fig created).
    """
    labels = list(vectors.keys())
    X = np.stack(list(vectors.values())).astype(np.float64)  # (N, D)

    X_c = X - X.mean(axis=0, keepdims=True)
    _, s, Vt = np.linalg.svd(X_c, full_matrices=False)
    coords = X_c @ Vt[:2].T  # (N, 2)
    total_var = (s ** 2).sum()
    explained_var = (s[:2] ** 2) / (total_var + 1e-30)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    task_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    n_merged = len(labels) - n_tasks
    merged_markers = ["*", "D", "^", "P", "X"]
    merged_sizes   = [250, 120, 140, 140, 140]

    for i, label in enumerate(labels):
        x, y = coords[i]
        if i < n_tasks:
            ax.scatter(x, y, color=task_colors[i % len(task_colors)], s=80, zorder=4)
            ax.annotate(label, (x, y), textcoords="offset points", xytext=(5, 4), fontsize=8)
        else:
            j = i - n_tasks
            marker = merged_markers[j % len(merged_markers)]
            size   = merged_sizes[j % len(merged_sizes)]
            color  = task_colors[(n_tasks + j) % len(task_colors)]
            ax.scatter(x, y, color=color, marker=marker, s=size, zorder=5, label=label)
            ax.annotate(label, (x, y), textcoords="offset points", xytext=(5, 4), fontsize=8,
                        fontweight="bold")

    for i in range(n_tasks):
        ax.scatter([], [], color=task_colors[i % len(task_colors)], s=80, label=labels[i])

    ax.set_xlabel(f"PC1 ({explained_var[0]:.1%})")
    ax.set_ylabel(f"PC2 ({explained_var[1]:.1%})")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, linestyle="--", alpha=0.3)

    if fig is not None:
        fig.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()


def _pca_fit(frames: List[Dict[str, np.ndarray]]):
    """Fit PCA on all frames combined; return (center, pc, explained_var)."""
    all_X = np.vstack([np.stack(list(f.values())) for f in frames]).astype(np.float64)
    center = all_X.mean(axis=0, keepdims=True)
    _, s, Vt = np.linalg.svd(all_X - center, full_matrices=False)
    ev = (s[:2] ** 2) / ((s ** 2).sum() + 1e-30)
    return center, Vt[:2], ev


def _draw_pca_frame(ax, coords: np.ndarray, labels: List[str], n_tasks: int,
                    title: str, xlim, ylim, ev: np.ndarray) -> None:
    task_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    merged_markers = ["*", "D", "^", "P", "X"]
    merged_sizes   = [250, 120, 140, 140, 140]
    ax.clear()
    for i, lbl in enumerate(labels):
        x, y = coords[i]
        if i < n_tasks:
            ax.scatter(x, y, color=task_colors[i % len(task_colors)], s=80, zorder=4)
            ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(5, 4), fontsize=8)
        else:
            j = i - n_tasks
            ax.scatter(x, y, color=task_colors[(n_tasks + j) % len(task_colors)],
                       marker=merged_markers[j % len(merged_markers)],
                       s=merged_sizes[j % len(merged_sizes)], zorder=5, label=lbl)
            ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(5, 4),
                        fontsize=8, fontweight="bold")
    for i in range(n_tasks):
        ax.scatter([], [], color=task_colors[i % len(task_colors)], s=80, label=labels[i])
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xlabel(f"PC1 ({ev[0]:.1%})"); ax.set_ylabel(f"PC2 ({ev[1]:.1%})")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, linestyle="--", alpha=0.3)


def save_pca_gif(
    frames: List[Dict[str, np.ndarray]],
    frame_titles: List[str],
    n_tasks: int,
    title: str = "",
    save_path: str = "animation.gif",
    fps: int = 2,
) -> None:
    """Animate PCA frames (one per layer) into a GIF with consistent axes.

    Args:
        frames:       List of ordered dicts {label: 1-D vector}, one per layer.
        frame_titles: Title suffix for each frame (e.g. "Layer 0").
        n_tasks:      Number of task-vector entries at the start of each frame dict.
        title:        Common title prefix.
        save_path:    Output .gif path.
        fps:          Frames per second.
    """
    center, pc, ev = _pca_fit(frames)
    labels = list(frames[0].keys())

    projected = [
        np.stack([(v.astype(np.float64) - center[0]) @ pc.T for v in f.values()])
        for f in frames
    ]
    all_c = np.vstack(projected)
    pad = 0.15 * (all_c.max(0) - all_c.min(0)).clip(min=0.01)
    xlim = (all_c[:, 0].min() - pad[0], all_c[:, 0].max() + pad[0])
    ylim = (all_c[:, 1].min() - pad[1], all_c[:, 1].max() + pad[1])

    fig, ax = plt.subplots(figsize=(6, 5))

    def _update(i):
        _draw_pca_frame(ax, projected[i], labels, n_tasks,
                        f"{title} — {frame_titles[i]}", xlim, ylim, ev)

    anim = FuncAnimation(fig, _update, frames=len(frames), interval=500)
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    anim.save(save_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
