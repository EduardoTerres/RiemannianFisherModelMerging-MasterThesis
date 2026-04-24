"""
Visualize standard vs Fisher-full task vectors across layers.
Standard: xi_t, Fisher-full: v_t = (λI + Σ α_j F̃_j)^{-1}(λI + F̃_t) xi_t

Usage: python -m src.analysis.fisher_vectors [--model llama3.1|qwen2.5]
"""
from pathlib import Path
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.merging import OFTMerging
from src.paths import MODEL_FAMILIES

TASK_NAMES = ["socialiqa", "commonsense", "numinamath", "magicoder", "scienceqa"]
COLORS     = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
SAVE_DIR = os.path.join(os.path.dirname(__file__), "fisher_analysis")

def task_of(p):
    for n in TASK_NAMES:
        if n in p: return n
    return os.path.basename(p)

def frob(t): return float(t.norm("fro"))

def angle_deg(u, v):
    u, v = u.float(), v.float()
    cos = (u * v).sum() / (u.norm("fro") * v.norm("fro") + 1e-12)
    return float(np.degrees(torch.acos(cos.clamp(-1, 1)).item()))


def plot_fisher_stats(fishers_by_key: dict, task_labels: list, model_name: str, out_dir: str):
    """Plot per-layer Fisher statistics: mean, std, min, max across elements."""
    oft_keys = sorted(fishers_by_key.keys())
    L = len(oft_keys)
    T = len(task_labels)

    # (T, L) arrays of per-layer scalar stats
    f_mean = np.zeros((T, L))
    f_std  = np.zeros((T, L))
    f_max  = np.zeros((T, L))

    for i, k in enumerate(oft_keys):
        for t, f in enumerate(fishers_by_key[k]):
            flat = f.float().flatten()
            f_mean[t, i] = flat.mean().item()
            f_std[t, i]  = flat.std().item()
            f_max[t, i]  = flat.max().item()

    layer_idx = np.arange(L)
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle(f"Fisher matrix statistics — {model_name}", fontsize=13, fontweight="bold")
    ax_mean, ax_std, ax_hm, ax_box = axes.flat

    for t, (lbl, c) in enumerate(zip(task_labels, COLORS)):
        ax_mean.plot(layer_idx, f_mean[t], color=c, lw=1.5, label=lbl)
        ax_std.plot(layer_idx,  f_std[t],  color=c, lw=1.5, label=lbl)

    ax_mean.set_title("Mean Fisher value per layer")
    ax_mean.set_xlabel("layer index"); ax_mean.set_yscale("log")
    ax_mean.grid(True, lw=0.3, alpha=0.5); ax_mean.legend(fontsize=8)

    ax_std.set_title("Std Fisher value per layer")
    ax_std.set_xlabel("layer index"); ax_std.set_yscale("log")
    ax_std.grid(True, lw=0.3, alpha=0.5); ax_std.legend(fontsize=8)

    # Heatmap of mean Fisher (tasks × layers)
    im = ax_hm.imshow(np.log10(f_mean + 1e-30), aspect="auto", cmap="viridis")
    ax_hm.set_yticks(range(T)); ax_hm.set_yticklabels(task_labels)
    ax_hm.set_xlabel("layer index"); ax_hm.set_title("log₁₀(mean Fisher)  (heatmap)")
    fig.colorbar(im, ax=ax_hm, shrink=0.85)

    # Box plot of overall Fisher distributions per task (all layers, all elements)
    all_vals = [
        np.concatenate([fishers_by_key[k][t].float().flatten().cpu().numpy() for k in oft_keys])
        for t in range(T)
    ]
    ax_box.boxplot(all_vals, labels=task_labels, showfliers=False)
    ax_box.set_yscale("log"); ax_box.set_title("Fisher value distribution per task")
    ax_box.set_ylabel("Fisher value"); ax_box.grid(True, lw=0.3, alpha=0.5, axis="y")

    fig.tight_layout()
    out = os.path.join(out_dir, "fisher_stats.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family_name", default="llama3.1", choices=["llama3.1", "qwen2.5"])
    parser.add_argument("--save_path", default="outputs/fisher_analysis", help="Directory to save the analysis results.")
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
    merger = OFTMerging(lam=0.0, alphas=[1 / T] * T, device=device)

    print("Loading weights and fishers...")
    all_weights = merger.load_weights(adapter_paths)
    all_fishers = merger.load_fishers(fisher_paths)

    oft_keys = sorted(k for k in all_weights[0] if "oft_r" in k or "oft_" in k.lower())
    L = len(oft_keys)
    weights_by_key = {k: [w[k] for w in all_weights] for k in oft_keys}
    fishers_by_key = {k: [f[k] for f in all_fishers] for k in oft_keys}

    norm_ratio = np.zeros((T, L))
    angles = np.zeros((T, L))

    for i, k in enumerate(tqdm(oft_keys, desc="Processing layers")):
        full_vecs = merger.fisher_full_task_vectors(weights_by_key[k], fishers_by_key[k], [0.5] * T)
        for t in range(T):
            xi, vt = weights_by_key[k][t], full_vecs[t]
            norm_ratio[t, i] = frob(vt) / (frob(xi) + 1e-12)
            angles[t, i]     = angle_deg(xi, vt)

    # ── Plot ──────────────────────────────────────────────────────────────────
    layer_idx = np.arange(L)
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle(f"Standard vs Fisher-full task vectors — {args.family_name}",
                 fontsize=13, fontweight="bold")

    ax_r, ax_a, ax_hr, ax_ha = axes.flat

    for t, (lbl, c) in enumerate(zip(task_labels, COLORS)):
        ax_r.plot(layer_idx, norm_ratio[t], color=c, lw=1.5, label=lbl)
        ax_a.plot(layer_idx, angles[t],     color=c, lw=1.5, label=lbl)

    ax_r.axhline(1.0, color="k", lw=0.8, ls=":", label="ratio = 1")
    ax_r.set_title("Norm ratio  ‖f_t‖ / ‖ξ_t‖")
    ax_r.set_xlabel("layer index"); ax_r.grid(True, lw=0.3, alpha=0.5); ax_r.legend(fontsize=8)

    ax_a.set_title("Angle  ∠(ξ_t, f_t)  [°]")
    ax_a.set_xlabel("layer index"); ax_a.grid(True, lw=0.3, alpha=0.5); ax_a.legend(fontsize=8)

    # Heatmaps: tasks × layers — reveal which tasks/layers deviate most
    im_r = ax_hr.imshow(norm_ratio, aspect="auto", cmap="RdBu_r", vmin=0.5, vmax=1.5)
    ax_hr.set_yticks(range(T)); ax_hr.set_yticklabels(task_labels)
    ax_hr.set_xlabel("layer index"); ax_hr.set_title("Norm ratio  (heatmap)")
    fig.colorbar(im_r, ax=ax_hr, shrink=0.85)

    im_a = ax_ha.imshow(angles, aspect="auto", cmap="YlOrRd")
    ax_ha.set_yticks(range(T)); ax_ha.set_yticklabels(task_labels)
    ax_ha.set_xlabel("layer index"); ax_ha.set_title("Angle [°]  (heatmap)")
    fig.colorbar(im_a, ax=ax_ha, shrink=0.85)

    fig.tight_layout()
    out = os.path.join(IMG_DIR, "fisher_vectors.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.show()

    # Second plot
    plot_fisher_stats(fishers_by_key, task_labels, args.family_name, IMG_DIR)

if __name__ == "__main__":
    main()
