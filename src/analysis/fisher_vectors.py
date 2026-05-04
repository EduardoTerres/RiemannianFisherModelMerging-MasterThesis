"""
Visualize standard vs Fisher-full task vectors across layers.
Standard: xi_t, Fisher-full: v_t = (λI + Σ α_j F̃_j)^{-1}(λI + F̃_t) xi_t

Usage: python -m src.analysis.fisher_vectors [--family_name llama3.1|qwen2.5]
"""
from pathlib import Path
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import torch
from tqdm import tqdm

from src.merging import OFTMerging
from src.paths import MODEL_FAMILIES
from src.analysis.plot_utils import plot_fisher_vectors, plot_fisher_stats, plot_transport_effect

TASK_NAMES = ["socialiqa", "commonsense", "numinamath", "magicoder", "scienceqa"]


def task_of(p):
    for n in TASK_NAMES:
        if n in p: return n
    return os.path.basename(p)

def frob(t): return float(t.norm("fro"))

def angle_deg(u, v):
    u, v = u.float(), v.float()
    cos = (u * v).sum() / (u.norm("fro") * v.norm("fro") + 1e-12)
    return float(np.degrees(torch.acos(cos.clamp(-1, 1)).item()))


def compute_fisher_stats(fishers_by_key: dict, task_labels: list):
    """Compute per-layer Fisher mean/std and per-task value arrays for plotting."""
    oft_keys = sorted(fishers_by_key.keys())
    L, T = len(oft_keys), len(task_labels)
    f_mean, f_std = np.zeros((T, L)), np.zeros((T, L))
    for i, k in enumerate(oft_keys):
        for t, f in enumerate(fishers_by_key[k]):
            flat = f.float().flatten()
            f_mean[t, i] = flat.mean().item()
            f_std[t, i]  = flat.std().item()
    all_vals = [
        np.concatenate([fishers_by_key[k][t].float().flatten().cpu().numpy() for k in oft_keys])
        for t in range(T)
    ]
    return f_mean, f_std, all_vals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family_name", default="qwen2.5", choices=["llama3.1", "qwen2.5"])
    parser.add_argument("--save_path", default="outputs/fisher_analysis")
    args = parser.parse_args()

    IMG_DIR = Path(args.save_path) / args.family_name
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    family = MODEL_FAMILIES[args.family_name]
    adapter_paths = family.adapter_paths
    fisher_by_task = {task_of(p): p for p in family.fisher_paths}
    fisher_paths   = [fisher_by_task[task_of(p)] for p in adapter_paths]
    task_labels    = [task_of(p) for p in adapter_paths]

    T = len(adapter_paths)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    merger = OFTMerging(lam=0.0, alphas=[1.0] * T, device=device)

    print("Loading weights and fishers...")
    all_weights = merger.load_weights(adapter_paths)
    all_fishers = merger.load_fishers(fisher_paths)

    oft_keys = sorted(k for k in all_weights[0] if "oft_r" in k or "oft_" in k.lower())
    L = len(oft_keys)
    weights_by_key = {k: [w[k] for w in all_weights] for k in oft_keys}
    fishers_by_key = {k: [f[k] for f in all_fishers] for k in oft_keys}

    norm_ratio    = np.zeros((T, L))
    angles        = np.zeros((T, L))
    # off-diagonal Frobenius fraction: 0 = no transport, 1 = fully mixed
    off_diag_frac = np.zeros((T, L))
    # std of diag(F̃_t) minus std of f_t: negative = transport homogenises Fisher values
    diag_std_diff = np.zeros((T, L))

    for i, k in enumerate(tqdm(oft_keys, desc="Processing layers")):
        # Fisher-full task vectors via merging.py (uses parallel transport internally)
        full_vecs = merger.fisher_full_task_vectors(weights_by_key[k], fishers_by_key[k], [1] * T)
        for t in range(T):
            xi, vt = weights_by_key[k][t], full_vecs[t]
            norm_ratio[t, i] = frob(vt) / (frob(xi) + 1e-12)
            angles[t, i]     = angle_deg(xi, vt)
        del full_vecs

        # Parallel-transported Fishers: F̃_t = P_t diag(f_t) P_t^T  (see merging._fisher_merging)
        for t in range(T):
            oft_params_t = weights_by_key[k][t]
            num_blocks, son_dim = oft_params_t.shape
            block_size = int((1 + (1 + 8 * son_dim) ** 0.5) / 2)
            skew   = merger.oft_params_to_skew_matrix(oft_params_t, son_dim)
            Pt     = merger.manifold.compute_Pt(skew_matrix=skew, block_size=block_size)
            F_tilde = Pt @ torch.diag_embed(fishers_by_key[k][t]) @ Pt.transpose(-2, -1)

            # Off-diagonal fraction: ||off_diag(F̃)||_F / ||F̃||_F  (invariant to trace)
            diag_part   = torch.diag_embed(F_tilde.diagonal(dim1=-2, dim2=-1))
            off_diag    = F_tilde - diag_part
            frac = (off_diag.norm(dim=(-2, -1)) / (F_tilde.norm(dim=(-2, -1)) + 1e-12))
            off_diag_frac[t, i] = frac.mean().item()

            # How much transport redistributes Fisher across coordinates
            diag_std  = F_tilde.diagonal(dim1=-2, dim2=-1).float().std(dim=-1).mean().item()
            raw_std   = fishers_by_key[k][t].float().std(dim=-1).mean().item()
            diag_std_diff[t, i] = diag_std - raw_std

            del Pt, F_tilde, diag_part, off_diag

        torch.cuda.empty_cache()

    # ── Standard vs Fisher-full task vectors ──────────────────────────────────
    plot_fisher_vectors(
        norm_ratio, angles, task_labels,
        title=f"Standard vs Fisher-full task vectors — {args.family_name}",
        save_path=str(IMG_DIR / "fisher_vectors.png"),
    )

    # ── Raw Fisher statistics per layer / task ─────────────────────────────────
    f_mean, f_std, all_vals = compute_fisher_stats(fishers_by_key, task_labels)
    plot_fisher_stats(
        f_mean, f_std, all_vals, task_labels,
        title=f"Fisher matrix statistics — {args.family_name}",
        save_path=str(IMG_DIR / "fisher_stats.png"),
    )

    # ── Parallel-transport effect: off-diagonal mass and diagonal redistribution ──
    plot_transport_effect(
        off_diag_frac, diag_std_diff, task_labels,
        title=f"Parallel transport effect on Fisher — {args.family_name}",
        save_path=str(IMG_DIR / "fisher_transport_effect.png"),
    )


if __name__ == "__main__":
    main()
