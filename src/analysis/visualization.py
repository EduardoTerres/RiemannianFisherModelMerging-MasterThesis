"""Visualize OFT task-vector geometry: pairwise cosine similarity and norm distributions."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import re

import numpy as np
import torch
from typing import Dict, List, Optional
from tqdm import tqdm

from src.analysis.plot_utils import (
    plot_cosine_similarity_matrix,
    plot_layer_cosine_similarity,
    plot_layer_geodesic_distance,
    plot_layer_norm_variance,
    plot_module_layer_distributions,
    plot_norm_distributions,
    plot_oft_covariance_eigenvalues,
)
from src.geometry import SOnManifold
from src.merging import OFTMerging
from src.paths import (
    ROOTDIR,
    MODEL_FAMILIES,
)
from src.utils import parse_device

_device = "cuda" if torch.cuda.is_available() else "cpu"
_manifold = SOnManifold()
_merging = OFTMerging(device=_device)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _oft_keys(task_vectors: List[Dict[str, torch.Tensor]]) -> List[str]:
    return sorted(
        task_vectors[0].keys(),
        key=lambda k: int(next(iter(re.findall(r"\d+", k)), 0)),
    )


def _layer_labels(keys: List[str]) -> List[str]:
    return [next(iter(re.findall(r"\d+", k)), k) for k in keys]


def _frobenius_norm(v: torch.Tensor) -> float:
    return v.norm(p="fro").item()


def _layerwise_cosine(vi: torch.Tensor, vj: torch.Tensor) -> float:
    fi, fj = vi.flatten(), vj.flatten()
    return (fi @ fj / (fi.norm().clamp(min=1e-8) * fj.norm().clamp(min=1e-8))).item()


def _geodesic_distance(Qi: torch.Tensor, Qj: torch.Tensor) -> float:
    """d(Q, R) = sqrt(dist_sq(Q, R)), averaged over blocks."""
    if Qi.dim() == 2:
        Qi, Qj = Qi.unsqueeze(0), Qj.unsqueeze(0)
    dist_sq = _manifold.dist_sq(Qi, Qj)  # (num_blocks,)
    return dist_sq.clamp(min=0).sqrt().mean().item()


# ---------------------------------------------------------------------------

def compute_cosine_similarity_matrix(
    task_vectors: List[Dict[str, torch.Tensor]],
) -> np.ndarray:
    """Build a (T, T) matrix of avg layerwise cosine similarities.

    Args:
        task_vectors: T dicts, each mapping layer key -> (num_blocks, d).

    Returns:
        (T, T) float64 array.
    """
    T = len(task_vectors)
    keys = _oft_keys(task_vectors)
    sim_matrix = np.zeros((T, T))
    for i in tqdm(range(T), desc="Cosine similarity"):
        for j in range(T):
            sims = [_layerwise_cosine(task_vectors[i][k], task_vectors[j][k]) for k in keys]
            sim_matrix[i, j] = float(np.mean(sims))
    return sim_matrix


def compute_geodesic_distance_matrix(
    task_vectors: List[Dict[str, torch.Tensor]],
) -> np.ndarray:
    """Build a (T, T) matrix of avg layerwise geodesic distances on SO(n).

    Args:
        task_vectors: T dicts, each mapping layer key -> (num_blocks, d, d).

    Returns:
        (T, T) float64 array.
    """
    T = len(task_vectors)
    keys = _oft_keys(task_vectors)
    dist_matrix = np.zeros((T, T))
    for i in tqdm(range(T), desc="Geodesic distance"):
        for j in range(T):
            dists = [_geodesic_distance(task_vectors[i][k], task_vectors[j][k]) for k in keys]
            dist_matrix[i, j] = float(np.mean(dists))
    return dist_matrix


def compute_layer_norms(
    task_vectors: List[Dict[str, torch.Tensor]],
) -> List[List[float]]:
    """Compute per-layer Frobenius norms for each task.

    Args:
        task_vectors: T dicts, each mapping layer key -> (num_blocks, d).

    Returns:
        T lists, each containing one norm scalar per layer key.
    """
    keys = _oft_keys(task_vectors)
    return [[_frobenius_norm(tv[k]) for k in keys] for tv in task_vectors]


def compute_layer_norm_variance(
    task_vectors: List[Dict[str, torch.Tensor]],
) -> tuple[np.ndarray, List[str]]:
    """Variance of Frobenius norms across tasks, per layer.

    Returns:
        values: (L,) array of variance scalars.
        labels: layer index strings for axis tick labels.
    """
    keys = _oft_keys(task_vectors)
    values = np.array([
        float(np.var([_frobenius_norm(tv[k]) for tv in task_vectors]))
        for k in tqdm(keys, desc="Norm variance")
    ])
    return values, _layer_labels(keys)


def compute_layer_cosine_similarity(
    task_vectors: List[Dict[str, torch.Tensor]],
) -> tuple[np.ndarray, List[str]]:
    """Avg pairwise cosine similarity across tasks, per layer.

    Returns:
        values: (L,) array of mean cosine scalars.
        labels: layer index strings for axis tick labels.
    """
    keys = _oft_keys(task_vectors)
    T = len(task_vectors)
    values = []
    for k in tqdm(keys, desc="Cosine similarity"):
        sims = [
            _layerwise_cosine(task_vectors[i][k], task_vectors[j][k])
            for i in range(T) for j in range(i + 1, T)
        ]
        values.append(float(np.mean(sims)) if sims else 0.0)
    return np.array(values), _layer_labels(keys)


def compute_module_layer_distributions(
    task_vectors: List[Dict[str, torch.Tensor]],
) -> dict:
    """Avg geodesic dist, cosine sim and norm variance per module per layer.

    Keys are parsed as layers.{N}.{module}.
    Returns a dict suitable for plot_module_layer_distributions.
    """
    keys = _oft_keys(task_vectors)
    T = len(task_vectors)
    module_buckets: dict[str, dict[int, dict]] = {}
    for k in keys:
        m = re.search(r"layers?\.(\d+)\.(.+?)(?:\.oft_\w+)?$", k)
        if m is None:
            continue
        layer, mod = int(m.group(1)), m.group(2)
        bucket = module_buckets.setdefault(mod, {})
        pairs_geo = [
            _geodesic_distance(task_vectors[i][k], task_vectors[j][k])
            for i in range(T) for j in range(i + 1, T)
        ]
        pairs_cos = [
            _layerwise_cosine(task_vectors[i][k], task_vectors[j][k])
            for i in range(T) for j in range(i + 1, T)
        ]
        norms = [_frobenius_norm(tv[k]) for tv in task_vectors]
        bucket[layer] = {
            "geo": float(np.mean(pairs_geo)) if pairs_geo else 0.0,
            "cos": float(np.mean(pairs_cos)) if pairs_cos else 0.0,
            "mean": float(np.mean(norms)),
        }
    result = {}
    for mod, layers_dict in sorted(module_buckets.items()):
        sorted_layers = sorted(layers_dict)
        result[mod] = {
            "layers": sorted_layers,
            "geo": [layers_dict[n]["geo"] for n in sorted_layers],
            "cos": [layers_dict[n]["cos"] for n in sorted_layers],
            "mean": [layers_dict[n]["mean"] for n in sorted_layers],
        }
    return result


def _skew_repr(params: torch.Tensor) -> torch.Tensor:
    """OFT (num_blocks, d) upper-triangular skew params -> flat vector xi."""
    return params.flatten()


def compute_oft_covariance_eigenvalues(
    task_vectors: List[Dict[str, torch.Tensor]],
    top_k: int = 5,
    alpha: Optional[List[float]] = None,
) -> tuple[np.ndarray, List[str]]:
    """Per-layer covariance Sigma^(l) = sum_t alpha_t * xi_t^(l) @ xi_t^(l).T; return top-k eigenvalues.

    Args:
        task_vectors: T dicts mapping layer key -> rotation tensor.
        top_k:        Number of leading eigenvalues to track.
        alpha:        Per-task weights (uniform 1/T if None).

    Returns:
        eigenvalues: (L, top_k) array, column 0 = largest eigenvalue per layer.
        labels:      Layer index strings for axis tick labels.
    """
    keys = _oft_keys(task_vectors)
    T = len(task_vectors)
    if alpha is None:
        alpha = [1.0 / T] * T

    eigenvalues = []
    for k in tqdm(keys, desc="OFT covariance eigenvalues"):
        xis = [_skew_repr(tv[k]).float().cpu() for tv in task_vectors]
        # X is (T, D); Sigma = X^T diag(alpha) X has the same non-zero eigenvalues
        # as the T×T gram G = X diag(alpha) X^T, avoiding the huge D×D matrix.
        X = torch.stack(xis)  # (T, D)
        a_sqrt = torch.tensor(alpha).sqrt().unsqueeze(1)  # (T, 1)
        X_scaled = a_sqrt * X  # (T, D)
        G = X_scaled @ X_scaled.T  # (T, T)
        vals = torch.linalg.eigvalsh(G).flip(0)[:top_k].numpy()
        if len(vals) < top_k:
            vals = np.pad(vals, (0, top_k - len(vals)))
        eigenvalues.append(vals)
    return np.array(eigenvalues), _layer_labels(keys)


def compute_layer_geodesic_distance(
    task_vectors: List[Dict[str, torch.Tensor]],
) -> tuple[np.ndarray, List[str]]:
    """Avg pairwise geodesic distance on SO(n) across tasks, per layer.

    Returns:
        values: (L,) array of mean geodesic distance scalars.
        labels: layer index strings for axis tick labels.
    """
    keys = _oft_keys(task_vectors)
    T = len(task_vectors)
    values = []
    for k in tqdm(keys, desc="Geodesic distance"):
        dists = [
            _geodesic_distance(task_vectors[i][k], task_vectors[j][k])
            for i in range(T) for j in range(i + 1, T)
        ]
        values.append(float(np.mean(dists)) if dists else 0.0)
    return np.array(values), _layer_labels(keys)


def main(args: argparse.Namespace):
    for family_name in args.model_family:
        _run(family_name, args)


def _run(family_name: str, args: argparse.Namespace):
    model_family = MODEL_FAMILIES[family_name]
    adapter_paths = model_family.adapter_paths

    task_names = args.task_names or [Path(p).name for p in adapter_paths]

    task_vectors = _merging.load_weights(adapter_paths)

    IMG_DIR = Path(args.save_path) / family_name
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    # --- Cosine similarity matrix ---
    sim_matrix = compute_cosine_similarity_matrix(task_vectors)
    plot_cosine_similarity_matrix(
        sim_matrix=sim_matrix,
        task_names=task_names,
        title="Avg. layerwise cosine similarity",
        save_path=str(IMG_DIR / "cosine_similarity.png"),
    )
    print(f"Saved -> {IMG_DIR / 'cosine_similarity.png'}")

    # --- Geodesic distance matrix ---
    dist_matrix = compute_geodesic_distance_matrix(task_vectors)
    plot_cosine_similarity_matrix(
        sim_matrix=dist_matrix,
        task_names=task_names,
        title="Avg. geodesic distance (SO(n))",
        save_path=str(IMG_DIR / "geodesic_distance.png"),
    )
    print(f"Saved -> {IMG_DIR / 'geodesic_distance.png'}")

    # --- Norm distributions ---
    norms_per_task = compute_layer_norms(task_vectors)
    plot_norm_distributions(
        norms_per_task=norms_per_task,
        task_names=task_names,
        save_path=str(IMG_DIR / "norm_distributions.png"),
    )
    print(f"Saved -> {IMG_DIR / 'norm_distributions.png'}")

    # --- Per-layer norm variance strip ---
    variance_values, labels = compute_layer_norm_variance(task_vectors)
    plot_layer_norm_variance(
        values=variance_values,
        labels=labels,
        save_path=str(IMG_DIR / "norm_variance.png"),
    )
    print(f"Saved -> {IMG_DIR / 'norm_variance.png'}")

    # --- Per-layer cosine similarity strip ---
    cosine_values, labels = compute_layer_cosine_similarity(task_vectors)
    plot_layer_cosine_similarity(
        values=cosine_values,
        labels=labels,
        save_path=str(IMG_DIR / "cosine_similarity_per_layer.png"),
    )
    print(f"Saved -> {IMG_DIR / 'cosine_similarity_per_layer.png'}")

    # --- Per-layer geodesic distance strip ---
    geodesic_values, labels = compute_layer_geodesic_distance(task_vectors)
    plot_layer_geodesic_distance(
        values=geodesic_values,
        labels=labels,
        save_path=str(IMG_DIR / "geodesic_distance_per_layer.png"),
    )
    print(f"Saved -> {IMG_DIR / 'geodesic_distance_per_layer.png'}")

    # --- Per-module layer distributions ---
    module_data = compute_module_layer_distributions(task_vectors)
    plot_module_layer_distributions(
        module_data=module_data,
        save_path=str(IMG_DIR / "module_layer_distributions.png"),
    )
    print(f"Saved -> {IMG_DIR / 'module_layer_distributions.png'}")

    # --- Per-layer OFT covariance eigenvalues ---
    cov_eigenvalues, labels = compute_oft_covariance_eigenvalues(task_vectors)
    plot_oft_covariance_eigenvalues(
        eigenvalues=cov_eigenvalues,
        labels=labels,
        save_path=str(IMG_DIR / "oft_covariance_eigenvalues.png"),
    )
    print(f"Saved -> {IMG_DIR / 'oft_covariance_eigenvalues.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize OFT task-vector geometry.")
    parser.add_argument(
        "--task-names", nargs="+", default=None, dest="task_names",
        help="Human-readable task labels.",
    )
    parser.add_argument(
        "--save-path", type=str, default=f"{ROOTDIR}/outputs/visualization", dest="save_path",
        help="Directory where output PNGs are saved.",
    )
    parser.add_argument(
        "--model-family", nargs="+", default=list(MODEL_FAMILIES), choices=list(MODEL_FAMILIES),
        dest="model_family", help="One or more model families to process.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use for interpolation and loss evaluation (e.g., 'gpu', 'cpu').",
    )
    args = parser.parse_args()
    args.device = parse_device(args.device)

    main(args)
