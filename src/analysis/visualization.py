"""Visualize OFT task-vector geometry: pairwise cosine similarity and norm distributions."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import re

import numpy as np
import torch
from typing import Dict, List
from tqdm import tqdm

from src.analysis.plot_utils import (
    plot_cosine_similarity_matrix,
    plot_layer_cosine_agreement,
    plot_layer_geodesic_agreement,
    plot_layer_norm_variance,
    plot_norm_distributions,
)
from src.geometry import SOnManifold
from src.merging import OFTMerging
from src.path import (
    ROOTDIR,
    LLAMA_ADAPTER_PATHS,
    QWEN_ADAPTER_PATHS,
)

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


def compute_layer_cosine_agreement(
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
    for k in tqdm(keys, desc="Cosine agreement"):
        sims = [
            _layerwise_cosine(task_vectors[i][k], task_vectors[j][k])
            for i in range(T) for j in range(i + 1, T)
        ]
        values.append(float(np.mean(sims)) if sims else 0.0)
    return np.array(values), _layer_labels(keys)


def compute_layer_geodesic_agreement(
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
    for k in tqdm(keys, desc="Geodesic agreement"):
        dists = [
            _geodesic_distance(task_vectors[i][k], task_vectors[j][k])
            for i in range(T) for j in range(i + 1, T)
        ]
        values.append(float(np.mean(dists)) if dists else 0.0)
    return np.array(values), _layer_labels(keys)


def main(args: argparse.Namespace):
    if args.model_family == "llama3.1":
        adapter_paths = LLAMA_ADAPTER_PATHS
    elif args.model_family == "qwen2.5":
        adapter_paths = QWEN_ADAPTER_PATHS
    else:
        raise ValueError(f"Unsupported model family: {args.model_family}")

    task_names = args.task_names or [Path(p).name for p in adapter_paths]

    task_vectors = _merging.load_weights(adapter_paths)

    IMG_DIR = Path(args.save_path) / args.model_family
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

    # --- Per-layer cosine agreement strip ---
    cosine_values, labels = compute_layer_cosine_agreement(task_vectors)
    plot_layer_cosine_agreement(
        values=cosine_values,
        labels=labels,
        save_path=str(IMG_DIR / "cosine_agreement.png"),
    )
    print(f"Saved -> {IMG_DIR / 'cosine_agreement.png'}")

    # --- Per-layer geodesic agreement strip ---
    geodesic_values, labels = compute_layer_geodesic_agreement(task_vectors)
    plot_layer_geodesic_agreement(
        values=geodesic_values,
        labels=labels,
        save_path=str(IMG_DIR / "geodesic_agreement.png"),
    )
    print(f"Saved -> {IMG_DIR / 'geodesic_agreement.png'}")


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
        "--model-family", type=str, default="llama3.1", choices=["llama3.1", "qwen2.5"],
        dest="model_family", help="Model family to use: 'llama3.1' or 'qwen2.5'.",
    )
    args = parser.parse_args()

    main(args)
