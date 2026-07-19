import argparse
import math
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
from matplotlib.patches import Patch
import torch
from safetensors import safe_open
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
    parse_fisher_key,
    split_key,
    to_coords,
)
from src.diffusion.dataset_1 import CONCEPT_ADAPTERS, DIFFUSION_MERGE_PAIRS, STYLE_ADAPTERS

DEVICE = "cuda:0"
PLOT_DIR = REPO_ROOT / "outputs/diffusion/analysis/fim"
MAX_HIST_POINTS = 200_000
SIGNED_LOG_DIFF_EDGE = 40.0
STYLE_AVERAGE_NUM_BINS = 80
STYLE_AVERAGE_CACHE_DIR = PLOT_DIR / "sdxl_uncharted/style_histograms"
STYLE_AVERAGE_MODES = ("raw", "frobenius")


def kfac_path(entry):
    return entry["kfac_path"]


def pair_with_existing_fisher(kind):
    path_getter = kfac_path if kind == "KFAC" else lambda entry: entry["fim_path"]
    for pair in DIFFUSION_MERGE_PAIRS:
        if Path(path_getter(pair["concept"])).exists() and Path(path_getter(pair["style"])).exists():
            return pair
    return None


PAIR = pair_with_existing_fisher("KFAC") or pair_with_existing_fisher("FIM")
if PAIR is None:
    raise FileNotFoundError("No dataset pair has FIM or KFAC files for both concept and style.")
A = PAIR["concept"]
B = PAIR["style"]


def fim_paths(entry):
    return [
        ("FIM", entry["fim_path"]),
        ("KFAC", kfac_path(entry)),
    ]


def fim_path(entry, kind):
    return entry["fim_path"] if kind == "FIM" else kfac_path(entry)


def kfac_factors(fim, key, device):
    row = fim[f"{key}.row"].to(device).float()
    col = fim[f"{key}.col"].to(device).float()
    scale = fim[f"{key}.scale"].to(device).float()
    return zip(
        row.reshape(-1, row.shape[-2], row.shape[-1]),
        col.reshape(-1, col.shape[-2], col.shape[-1]),
        scale.reshape(-1),
    )


def kfac_quad(fim, key, delta):
    if f"{key}.row" not in fim:
        return None
    total = torch.tensor(0.0, device=delta.device)
    deltas = delta.reshape(-1, delta.shape[-2], delta.shape[-1])
    for (r, c, s), d_block in zip(kfac_factors(fim, key, delta.device), deltas):
        row, col = torch.triu_indices(
            d_block.shape[-1],
            d_block.shape[-1],
            offset=1,
            device=delta.device,
        )
        f_block = s * r[row[:, None], row[None, :]] * c[col[:, None], col[None, :]]
        vec = d_block[row, col]
        total += vec @ f_block @ vec
    return total


def kfac_diag_values(fim, key, device):
    return [
        s * torch.kron(torch.diagonal(c), torch.diagonal(r))
        for r, c, s in kfac_factors(fim, key, device)
    ]


def kfac_coord_diag_values(fim, key, device):
    values = []
    for r, c, s in kfac_factors(fim, key, device):
        diag = s * torch.diagonal(r).unsqueeze(1) * torch.diagonal(c).unsqueeze(0)
        row, col = torch.triu_indices(diag.shape[-2], diag.shape[-1], offset=1, device=device)
        values.append(diag[row, col])
    return torch.stack(values)


def fisher_trace(fim, device):
    total = torch.tensor(0.0, device=device)
    if any(k.endswith(".row") for k in fim):
        keys = sorted(k.removesuffix(".row") for k in fim if k.endswith(".row"))
        for key in keys:
            total += kfac_coord_diag_values(fim, key, device).sum()
        return total.item()

    for key, value in fim.items():
        if parse_fisher_key(key) is None:
            continue
        value = value.to(device=device).float()
        if value.dim() >= 2 and value.shape[-1] == value.shape[-2]:
            total += value.diagonal(dim1=-2, dim2=-1).sum()
        else:
            total += value.sum()
    return total.item()


def transported_delta(a_tensor, b_tensor):
    da = (b_tensor - a_tensor).float()
    da = 0.5 * (da - da.transpose(-1, -2))
    a_skew = 0.5 * (a_tensor.float() - a_tensor.float().transpose(-1, -2))
    rot = torch.matrix_exp(a_skew / 2)
    return rot @ da @ rot.transpose(-1, -2)


def build_kfac_map(adapter_state, fim, keys):
    adapter_sig, fim_sig = defaultdict(dict), defaultdict(dict)
    for key in keys:
        processor, _, _ = split_key(key)
        adapter_sig[processor][key[len(processor) + 1 :]] = tuple(adapter_state[key].shape)
    for key, tensor in fim.items():
        parsed = parse_fisher_key(key)
        if parsed is not None and key.endswith(".row"):
            layer_idx, tail = parsed
            fim_sig[layer_idx][tail.removesuffix(".row")] = tuple(tensor.shape)

    out, used = {}, set()
    for processor, signature in adapter_sig.items():
        matches = [
            idx
            for idx, candidate in fim_sig.items()
            if idx not in used and all(candidate.get(tail) == shape for tail, shape in signature.items())
        ]
        out[processor] = min(matches)
        used.add(out[processor])
    return out


def build_map(adapter_state, fim, keys):
    try:
        return build_fisher_map(adapter_state, fim, keys)
    except KeyError:
        return build_kfac_map(adapter_state, fim, keys)


def print_stats(values, label, kl=None):
    values = torch.cat([v.flatten().cpu() for v in values]).float()
    p50 = values.kthvalue(math.ceil(0.50 * values.numel())).values.item()
    p95 = values.kthvalue(math.ceil(0.95 * values.numel())).values.item()
    p99 = values.kthvalue(math.ceil(0.99 * values.numel())).values.item()
    mean = values.mean().item()
    print(f"{label} mean={mean:.8g}")
    print(f"{label} p50={p50:.8g}")
    print(f"{label} p95={p95:.8g}")
    print(f"{label} p99={p99:.8g}")
    if kl is not None:
        denom = kl if kl != 0.0 else float("nan")
        print(f"{label}/KL mean={mean / denom:.8g}")
        print(f"{label}/KL p50={p50 / denom:.8g}")
        print(f"{label}/KL p95={p95 / denom:.8g}")
        print(f"{label}/KL p99={p99 / denom:.8g}")


def print_block_normalized_stats(values, label):
    normalized = []
    for value in values:
        blocks = value.float().reshape(-1, value.shape[-1])
        norms = torch.linalg.vector_norm(blocks, dim=1, keepdim=True).clamp_min(1e-30)
        normalized.append((blocks / norms).flatten().cpu())

    values = torch.cat(normalized).float()
    p50 = values.kthvalue(math.ceil(0.50 * values.numel())).values.item()
    p95 = values.kthvalue(math.ceil(0.95 * values.numel())).values.item()
    p99 = values.kthvalue(math.ceil(0.99 * values.numel())).values.item()
    print(f"{label} block_norm1 mean={values.mean().item():.8g}")
    print(f"{label} block_norm1 p50={p50:.8g}")
    print(f"{label} block_norm1 p95={p95:.8g}")
    print(f"{label} block_norm1 p99={p99:.8g}")


def flat(values):
    return torch.cat([v.flatten().cpu() for v in values]).float()


def block_normalized_flat(values):
    normalized = []
    for value in values:
        blocks = value.float().reshape(-1, value.shape[-1])
        norms = torch.linalg.vector_norm(blocks, dim=1, keepdim=True).clamp_min(1e-30)
        normalized.append((blocks / norms).flatten().cpu())
    return torch.cat(normalized).float()


def trace_normalized_flat(values):
    normalized = []
    for value in values:
        blocks = value.float().reshape(-1, value.shape[-1])
        traces = blocks.sum(dim=1, keepdim=True).clamp_min(1e-30)
        normalized.append((blocks / traces).flatten().cpu())
    return torch.cat(normalized).float()


def print_tensor_stats(values, label):
    values = values.float().abs()
    p50 = values.kthvalue(math.ceil(0.50 * values.numel())).values.item()
    p95 = values.kthvalue(math.ceil(0.95 * values.numel())).values.item()
    p99 = values.kthvalue(math.ceil(0.99 * values.numel())).values.item()
    print(f"{label} mean={values.mean().item():.8g}")
    print(f"{label} p50={p50:.8g}")
    print(f"{label} p95={p95:.8g}")
    print(f"{label} p99={p99:.8g}")


def print_fa_fb_diffs(results_a, results_b):
    print("=" * 50)
    print("abs diff F_A-F_B")
    for kind in ("FIM", "KFAC"):
        if kind not in results_a or kind not in results_b:
            continue

        a, b = results_a[kind], results_b[kind]
        a_raw = flat(a["compare_values"])
        b_raw = flat(b["compare_values"])
        a_norm = block_normalized_flat(a["compare_values"])
        b_norm = block_normalized_flat(b["compare_values"])

        print("-" * 50)
        print(kind)
        print_tensor_stats(a_raw - b_raw, f"abs(F_A-F_B) {kind}")
        print_tensor_stats(
            a_raw / a["kl"] - b_raw / b["kl"],
            f"abs(F_A/KL_A_to_B-F_B/KL_B_to_A) {kind}",
        )
        print_tensor_stats(
            a_norm - b_norm,
            f"abs(norm(F_A)-norm(F_B)) {kind}",
        )


def log_abs(values):
    eps = torch.finfo(values.dtype).tiny
    return torch.log10(values.abs().clamp_min(eps))


def signed_log_abs(values):
    return values.sign() * log_abs(values)


def plot_fa_fb_histograms(results_a, results_b):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    for kind in ("FIM", "KFAC"):
        if kind not in results_a or kind not in results_b:
            continue

        a, b = results_a[kind], results_b[kind]
        rows = [
            ("raw", flat(a["compare_values"]), flat(b["compare_values"])),
            (
                "divided by KL",
                flat(a["compare_values"]) / a["kl"],
                flat(b["compare_values"]) / b["kl"],
            ),
            (
                "block Frobenius norm 1",
                block_normalized_flat(a["compare_values"]),
                block_normalized_flat(b["compare_values"]),
            ),
            (
                "block trace 1",
                trace_normalized_flat(a["compare_values"]),
                trace_normalized_flat(b["compare_values"]),
            ),
        ]

        fig, axes = plt.subplots(
            len(rows),
            2,
            figsize=(13, 3 * len(rows)),
            gridspec_kw={"width_ratios": [2.0, 1.0]},
            constrained_layout=True,
        )
        fig.suptitle(f"{kind}: {A['name']} vs {B['name']}")
        for row, (name, a_values, b_values) in enumerate(rows):
            ax = axes[row, 0]
            ax.hist(log_abs(a_values).numpy(), bins=100, alpha=0.55, label="F_A")
            ax.hist(log_abs(b_values).numpy(), bins=100, alpha=0.55, label="F_B")
            ax.set_title(f"{name}: log10(abs(.))")
            ax.set_ylabel("count")
            ax.legend()

            ax = axes[row, 1]
            ax.hist(signed_log_abs(a_values - b_values).numpy(), bins=100, color="tab:gray")
            ax.set_title(f"{name}: signed log10(abs(F_A - F_B))")

        save_path = PLOT_DIR / f"constant_timestep_kl_{A['name']}__{B['name']}_{kind.lower()}.png"
        fig.savefig(save_path, dpi=160)
        plt.close(fig)
        print(f"saved plot {save_path}")


def sample_for_hist(values):
    return sample_tensor(values, MAX_HIST_POINTS)


def sample_tensor(values, max_points):
    values = values.cpu().float().flatten()
    if values.numel() <= max_points:
        return values
    idx = torch.arange(max_points, dtype=torch.long) * values.numel() // max_points
    return values[idx]


def fisher_blocks(entry, kind):
    path = entry["fim_path"] if kind == "FIM" else kfac_path(entry)
    if not Path(path).exists():
        return None
    fim = load_file(path, device="cpu")
    if kind == "FIM":
        keys = sorted(k for k in fim if parse_fisher_key(k) is not None)
        return [fim[key].float().reshape(-1, fim[key].shape[-1]).cpu() for key in keys]

    blocks = []
    keys = sorted(k.removesuffix(".row") for k in fim if k.endswith(".row"))
    for key in keys:
        values = kfac_coord_diag_values(fim, key, torch.device("cpu"))
        blocks.append(values.float().reshape(-1, values.shape[-1]).cpu())
    return blocks


def normalized_fisher_vector(blocks, mode):
    values = []
    for block in blocks:
        block = block.float()
        if mode == "raw":
            values.append(block.flatten())
        elif mode == "frobenius":
            values.append((block / torch.linalg.vector_norm(block, dim=1, keepdim=True).clamp_min(1e-30)).flatten())
        elif mode == "trace":
            values.append((block / block.sum(dim=1, keepdim=True).clamp_min(1e-30)).flatten())
        else:
            raise ValueError(mode)
    return torch.cat(values).float()


def colored_diff_hist(ax, diff):
    diff = sample_for_hist(diff)
    neg = diff[diff < 0]
    pos = diff[diff >= 0]
    if neg.numel() > 0:
        ax.hist(signed_log_abs(neg).numpy(), bins=40, color="tab:red", alpha=0.75)
    if pos.numel() > 0:
        ax.hist(signed_log_abs(pos).numpy(), bins=40, color="tab:green", alpha=0.75)
    ax.axvline(0, color="black", linewidth=0.6)


def style_average_fim_diff_values(concept_tensor, style_tensor, mode):
    concept_tensor = concept_tensor.float()
    style_tensor = style_tensor.float()
    if mode == "raw":
        return concept_tensor.flatten() - style_tensor.flatten()
    if mode == "frobenius":
        concept_blocks = concept_tensor.reshape(-1, concept_tensor.shape[-1])
        style_blocks = style_tensor.reshape(-1, style_tensor.shape[-1])
        concept_blocks = concept_blocks / torch.linalg.vector_norm(concept_blocks, dim=1, keepdim=True).clamp_min(1e-30)
        style_blocks = style_blocks / torch.linalg.vector_norm(style_blocks, dim=1, keepdim=True).clamp_min(1e-30)
        return (concept_blocks - style_blocks).flatten()
    raise ValueError(mode)


def fim_pair_hist_counts(concept, style, bins, mode):
    counts = torch.zeros(len(bins) - 1)
    with safe_open(concept["fim_path"], framework="pt", device="cpu") as concept_fim, safe_open(
        style["fim_path"], framework="pt", device="cpu"
    ) as style_fim:
        keys = sorted(k for k in concept_fim.keys() if parse_fisher_key(k) is not None and k in style_fim.keys())
        max_points_per_key = max(1, MAX_HIST_POINTS // max(1, len(keys)))
        for key in keys:
            diff = style_average_fim_diff_values(concept_fim.get_tensor(key), style_fim.get_tensor(key), mode)
            values = signed_log_abs(sample_tensor(diff, max_points_per_key))
            counts += torch.histc(values, bins=len(bins) - 1, min=float(bins[0]), max=float(bins[-1]))

    total = counts.sum().clamp_min(1e-30)
    return counts / total


def plot_average_histogram(ax, histograms, bins):
    if not histograms:
        ax.text(0.5, 0.5, r"\textrm{missing}", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    heights = torch.stack(histograms).mean(dim=0).numpy()
    widths = bins[1:] - bins[:-1]
    centers = 0.5 * (bins[:-1] + bins[1:])
    colors = ["tab:red" if center < 0 else "tab:green" for center in centers]
    ax.bar(centers, heights, width=widths, color=colors, alpha=0.78, align="center")
    ax.axvline(0, color="black", linewidth=0.6)


def style_histogram_cache_path(style, mode):
    return STYLE_AVERAGE_CACHE_DIR / mode / f"{style['name']}.pt"


def style_average_bins():
    return torch.linspace(-SIGNED_LOG_DIFF_EDGE, SIGNED_LOG_DIFF_EDGE, STYLE_AVERAGE_NUM_BINS + 1)


def compute_style_average_histogram(style, bins, mode):
    if not Path(style["fim_path"]).exists():
        return None

    histograms = []
    for concept in CONCEPT_ADAPTERS:
        if Path(concept["fim_path"]).exists():
            histograms.append(fim_pair_hist_counts(concept, style, bins, mode))
    if not histograms:
        return None
    return torch.stack(histograms).mean(dim=0)


def compute_style_average_histogram_batch(style_batch_start, style_batch_size, bins, mode):
    (STYLE_AVERAGE_CACHE_DIR / mode).mkdir(parents=True, exist_ok=True)
    start = max(0, style_batch_start)
    end = min(len(STYLE_ADAPTERS), start + style_batch_size)
    for style in STYLE_ADAPTERS[start:end]:
        hist = compute_style_average_histogram(style, bins, mode)
        cache_path = style_histogram_cache_path(style, mode)
        torch.save({"style": style["name"], "mode": mode, "bins": bins.cpu(), "hist": hist}, cache_path)
        status = "missing" if hist is None else "ok"
        print(f"cached {mode} style histogram {style['name']} ({status}) at {cache_path}", flush=True)


def load_style_average_histograms(bins, mode):
    histograms_by_style = {}
    missing = []
    for style in STYLE_ADAPTERS:
        cache_path = style_histogram_cache_path(style, mode)
        if not cache_path.exists():
            histograms_by_style[style["name"]] = None
            missing.append(style["name"])
            continue
        payload = torch.load(cache_path, map_location="cpu")
        if payload.get("mode") != mode or not torch.equal(payload["bins"], bins.cpu()):
            histograms_by_style[style["name"]] = None
            missing.append(style["name"])
            continue
        histograms_by_style[style["name"]] = payload["hist"]
    return histograms_by_style, missing


def style_average_plot_paths(mode):
    paths = [PLOT_DIR / f"style_averaged_{mode}_diff_grid_fim.png"]
    if mode == "raw":
        paths.append(PLOT_DIR / "sdxl_uncharted" / "matrices.png")
    return paths


def save_style_average_plot(histograms_by_style, bins, mode):
    bins_np = bins.cpu().numpy()
    fig, axes = plt.subplots(
        2,
        6,
        figsize=(17.5, 6.2),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(w_pad=0.08, h_pad=0.1, hspace=0.08, wspace=0.04)

    for style_idx, (ax, style) in enumerate(zip(axes.flat, STYLE_ADAPTERS), start=1):
        hist = histograms_by_style[style["name"]]
        plot_average_histogram(ax, [] if hist is None else [hist], bins_np)
        ax.set_title(rf"\textrm{{Style {style_idx}}}", fontsize=22)
        ax.tick_params(labelsize=16)
        ax.grid(alpha=0.18)

    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\Delta F$", fontsize=20)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$\operatorname{sign}(\Delta)\log_{10}(|\Delta|)$", fontsize=20)

    fig.legend(
        handles=[
            Patch(facecolor="tab:red", alpha=0.78, label=r"$\Delta < 0$: style larger"),
            Patch(facecolor="tab:green", alpha=0.78, label=r"$\Delta \geq 0$: concept larger"),
        ],
        loc="outside upper center",
        ncols=2,
        frameon=False,
        fontsize=22,
    )

    for save_path in style_average_plot_paths(mode):
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        print(f"saved plot {save_path}", flush=True)
    plt.close(fig)


def paired_fim_hist(ax, concept_values, style_values):
    concept_values = sample_for_hist(concept_values)
    style_values = sample_for_hist(style_values)
    ax.hist(log_abs(concept_values).numpy(), bins=40, color="tab:blue", alpha=0.55, label="concept")
    ax.hist(log_abs(style_values).numpy(), bins=40, color="tab:orange", alpha=0.55, label="style")


def latex_text(text):
    return r"\textrm{" + str(text).replace("\\", r"\textbackslash{}").replace("_", r"\_") + "}"


def plot_all_pair_raw_diffs(device):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    for kind in ("FIM", "KFAC"):
        concept_blocks = {entry["name"]: fisher_blocks(entry, kind) for entry in CONCEPT_ADAPTERS}
        style_blocks = {entry["name"]: fisher_blocks(entry, kind) for entry in STYLE_ADAPTERS}

        modes = [
            ("raw", "raw"),
            ("frobenius", "per-block Frobenius norm 1"),
            ("trace", "per-block trace 1"),
        ]
        for mode, mode_title in modes:
            concept_vectors = {
                name: None if blocks is None else normalized_fisher_vector(blocks, mode)
                for name, blocks in concept_blocks.items()
            }
            style_vectors = {
                name: None if blocks is None else normalized_fisher_vector(blocks, mode)
                for name, blocks in style_blocks.items()
            }

            fig, axes = plt.subplots(
                len(CONCEPT_ADAPTERS),
                len(STYLE_ADAPTERS),
                figsize=(2.4 * len(STYLE_ADAPTERS), 1.9 * len(CONCEPT_ADAPTERS)),
                sharex=True,
                sharey=True,
                constrained_layout=True,
            )
            for row, concept in enumerate(CONCEPT_ADAPTERS):
                for col, style in enumerate(STYLE_ADAPTERS):
                    ax = axes[row, col]
                    c_vec = concept_vectors[concept["name"]]
                    s_vec = style_vectors[style["name"]]
                    if c_vec is None or s_vec is None:
                        ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
                        ax.set_axis_off()
                        continue
                    colored_diff_hist(ax, c_vec - s_vec)
                    if row == 0:
                        ax.set_title(latex_text(style["name"]), fontsize=8)
                    if col == 0:
                        ax.set_ylabel(latex_text(concept["name"]), fontsize=8)
                    ax.tick_params(labelsize=6)

            save_path = PLOT_DIR / f"all_concept_style_{mode}_diff_grid_{kind.lower()}.png"
            fig.savefig(save_path, dpi=160)
            plt.close(fig)
            print(f"saved plot {save_path}")


def plot_style_averaged_fim_diffs(device, style_batch_start=0, style_batch_size=None, render=True, mode="raw"):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    bins = style_average_bins()
    if style_batch_size is None:
        style_batch_size = len(STYLE_ADAPTERS)

    compute_style_average_histogram_batch(style_batch_start, style_batch_size, bins, mode)
    if not render:
        return

    histograms_by_style, missing = load_style_average_histograms(bins, mode)
    if missing:
        print(f"Skipping style-averaged {mode} FIM diff render; missing cached styles: {', '.join(missing)}")
        return

    save_style_average_plot(histograms_by_style, bins, mode)


def render_style_averaged_fim_diffs(mode="raw"):
    bins = style_average_bins()
    histograms_by_style, missing = load_style_average_histograms(bins, mode)
    if missing:
        raise FileNotFoundError(f"Missing cached {mode} style histograms: {', '.join(missing)}")
    save_style_average_plot(histograms_by_style, bins, mode)


def plot_all_pair_fim_and_diff_grids(device):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    for kind in ("FIM", "KFAC"):
        concept_blocks = {entry["name"]: fisher_blocks(entry, kind) for entry in CONCEPT_ADAPTERS}
        style_blocks = {entry["name"]: fisher_blocks(entry, kind) for entry in STYLE_ADAPTERS}

        modes = [
            ("raw", "raw"),
            ("frobenius", "per-block Frobenius norm 1"),
            ("trace", "per-block trace 1"),
        ]
        for mode, mode_title in modes:
            concept_vectors = {
                name: None if blocks is None else normalized_fisher_vector(blocks, mode)
                for name, blocks in concept_blocks.items()
            }
            style_vectors = {
                name: None if blocks is None else normalized_fisher_vector(blocks, mode)
                for name, blocks in style_blocks.items()
            }

            fig = plt.figure(
                figsize=(2.6 * len(STYLE_ADAPTERS), 3.0 * len(CONCEPT_ADAPTERS)),
            )
            outer = fig.add_gridspec(
                len(CONCEPT_ADAPTERS),
                len(STYLE_ADAPTERS),
                left=0.045,
                right=0.985,
                bottom=0.045,
                top=0.92,
                hspace=0.55,
                wspace=0.18,
            )
            fig.suptitle(
                f"{kind} {mode_title}: FIM distributions above signed differences; "
                "blue = concept, orange = style"
            )

            legend_handles = None
            legend_labels = None
            for row, concept in enumerate(CONCEPT_ADAPTERS):
                for col, style in enumerate(STYLE_ADAPTERS):
                    inner = outer[row, col].subgridspec(2, 1, height_ratios=[1.0, 1.35], hspace=0.0)
                    fim_ax = fig.add_subplot(inner[0])
                    diff_ax = fig.add_subplot(inner[1])

                    c_vec = concept_vectors[concept["name"]]
                    s_vec = style_vectors[style["name"]]
                    if c_vec is None or s_vec is None:
                        for ax in (fim_ax, diff_ax):
                            ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
                            ax.set_axis_off()
                        continue

                    paired_fim_hist(fim_ax, c_vec, s_vec)
                    colored_diff_hist(diff_ax, c_vec - s_vec)
                    if legend_handles is None:
                        legend_handles, legend_labels = fim_ax.get_legend_handles_labels()

                    if row == 0:
                        fim_ax.set_title(style["name"], fontsize=8)
                    if col == 0:
                        fim_ax.set_ylabel(concept["name"], fontsize=8)
                    else:
                        fim_ax.tick_params(labelleft=False)
                        diff_ax.tick_params(labelleft=False)

                    fim_ax.tick_params(labelsize=6, labelbottom=False)
                    diff_ax.tick_params(labelsize=6)
                    fim_ax.grid(alpha=0.18)
                    diff_ax.grid(alpha=0.18)

            if legend_handles is not None:
                fig.legend(
                    legend_handles,
                    legend_labels,
                    loc="upper right",
                    bbox_to_anchor=(0.995, 0.995),
                    fontsize=8,
                )

            save_path = PLOT_DIR / f"all_concept_style_{mode}_fim_and_diff_grid_{kind.lower()}.png"
            fig.savefig(save_path, dpi=160)
            plt.close(fig)
            print(f"saved plot {save_path}")


def directional_kl(source_state, target_state, fim, device):
    keys = sorted(k for k in source_state if k in target_state and is_oft_key(k))
    fmap = build_map(source_state, fim, keys)
    is_kfac = any(k.endswith(".row") for k in fim)

    quad = torch.tensor(0.0, device=device)
    for key in keys:
        fkey = fisher_key(key, fmap)
        delta = transported_delta(source_state[key], target_state[key])
        if is_kfac:
            quad += kfac_quad(fim, fkey, delta)
        else:
            quad += (fim[fkey].float() * to_coords(delta).square()).sum()
    return 0.5 * quad.item()


def print_all_pair_kl_ratios(device):
    print("=" * 50)
    print("paired KL concept-closeness t for all concept-style pairs")

    concept_states = {
        entry["name"]: load_file(entry["adapter_path"], device=str(device)) for entry in CONCEPT_ADAPTERS
    }
    style_states = {
        entry["name"]: load_file(entry["adapter_path"], device=str(device)) for entry in STYLE_ADAPTERS
    }

    for kind in ("FIM", "KFAC"):
        print("-" * 50)
        print(kind)
        concept_fishers = {}
        style_fishers = {}
        concept_traces = {}
        style_traces = {}
        for entry in CONCEPT_ADAPTERS:
            path = fim_path(entry, kind)
            if Path(path).exists():
                fim = load_file(path, device=str(device))
                concept_fishers[entry["name"]] = fim
                concept_traces[entry["name"]] = fisher_trace(fim, device)
        for entry in STYLE_ADAPTERS:
            path = fim_path(entry, kind)
            if Path(path).exists():
                fim = load_file(path, device=str(device))
                style_fishers[entry["name"]] = fim
                style_traces[entry["name"]] = fisher_trace(fim, device)

        for pair in DIFFUSION_MERGE_PAIRS:
            concept = pair["concept"]
            style = pair["style"]
            concept_fim = concept_fishers.get(concept["name"])
            style_fim = style_fishers.get(style["name"])
            if concept_fim is None or style_fim is None:
                print(f"{pair['name']} missing_{kind.lower()}=nan")
                continue

            kl_a_to_b = directional_kl(
                concept_states[concept["name"]],
                style_states[style["name"]],
                concept_fim,
                device,
            )
            kl_b_to_a = directional_kl(
                style_states[style["name"]],
                concept_states[concept["name"]],
                style_fim,
                device,
            )
            trace_b = style_traces[style["name"]]
            trace_a = concept_traces[concept["name"]]
            normalized_kl_a_to_b = kl_a_to_b / trace_b if trace_b != 0.0 else float("nan")
            normalized_kl_b_to_a = kl_b_to_a / trace_a if trace_a != 0.0 else float("nan")
            denom = normalized_kl_a_to_b + normalized_kl_b_to_a
            concept_closeness_t = normalized_kl_b_to_a / denom if denom != 0.0 else float("nan")
            print(f"{kind} {pair['name']} t={concept_closeness_t:.8g}")


def report_one(source, target, source_state, target_state, source_label, target_label, fim_kind, fim_path, device):
    if not Path(fim_path).exists():
        print(f"{fim_kind}: skip missing {fim_path}")
        return None

    fim = load_file(fim_path, device=str(device))

    keys = sorted(k for k in source_state if k in target_state and is_oft_key(k))
    fmap = build_map(source_state, fim, keys)
    is_kfac = any(k.endswith(".row") for k in fim)

    fim_values, compare_values, quad = [], [], torch.tensor(0.0, device=device)
    for key in keys:
        fkey = fisher_key(key, fmap)
        delta = transported_delta(source_state[key], target_state[key])
        if is_kfac:
            fim_values.extend(kfac_diag_values(fim, fkey, device))
            compare_values.append(kfac_coord_diag_values(fim, fkey, device))
            quad += kfac_quad(fim, fkey, delta)
        else:
            fim_values.append(fim[fkey])
            compare_values.append(fim[fkey])
            quad += (fim[fkey].float() * to_coords(delta).square()).sum()

    kl = 0.5 * quad.item()
    fisher_label = f"F_{source_label}"
    print(f"{fim_kind}: {fisher_label}={fim_path}")
    print_stats(fim_values, fisher_label, kl)
    print_block_normalized_stats(fim_values, fisher_label)
    print(f"xi_{target_label}^T {fisher_label} xi_{target_label}={quad.item():.8g}")
    print(f"KL_{source_label}_to_{target_label}={kl:.8g}")
    print(f"KL(p_{source_label} || p_{target_label})~={kl:.8g}")
    return {"compare_values": compare_values, "kl": kl}


def report(source, target, source_label, target_label, device):
    source_state = load_file(source["adapter_path"], device=str(device))
    target_state = load_file(target["adapter_path"], device=str(device))
    results = {}
    for fim_kind, path in fim_paths(source):
        result = report_one(source, target, source_state, target_state, source_label, target_label, fim_kind, path, device)
        if result is not None:
            results[fim_kind] = result
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Normalize and plot SDXL FIM statistics.")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument(
        "--only_style_average_raw_fim",
        action="store_true",
        help="Only compute cached style-averaged raw FIM diff histograms and optionally render the 2x6 plot.",
    )
    parser.add_argument(
        "--render_style_average_raw_fim",
        action="store_true",
        help="Only render the 2x6 style-averaged raw FIM diff plot from cached style histograms.",
    )
    parser.add_argument(
        "--style_average_mode",
        choices=STYLE_AVERAGE_MODES,
        default="raw",
        help="FIM normalization used for the style-averaged 2x6 diff plot.",
    )
    parser.add_argument("--style_batch_start", type=int, default=0)
    parser.add_argument("--style_batch_size", type=int, default=None)
    parser.add_argument(
        "--no_render",
        action="store_true",
        help="Compute style histogram cache files without assembling the final plot.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.render_style_average_raw_fim:
        render_style_averaged_fim_diffs(args.style_average_mode)
        return

    device = torch.device(args.device)
    if args.only_style_average_raw_fim:
        plot_style_averaged_fim_diffs(
            device,
            style_batch_start=args.style_batch_start,
            style_batch_size=args.style_batch_size,
            render=not args.no_render,
            mode=args.style_average_mode,
        )
        return

    print_all_pair_kl_ratios(device)
    print(f"A={A['name']} B={B['name']}")
    results_a = report(A, B, "A", "B", device)
    print("=" * 50)
    results_b = report(B, A, "B", "A", device)
    print_fa_fb_diffs(results_a, results_b)
    plot_fa_fb_histograms(results_a, results_b)
    plot_all_pair_raw_diffs(device)
    for mode in STYLE_AVERAGE_MODES:
        plot_style_averaged_fim_diffs(device, mode=mode)
    plot_all_pair_fim_and_diff_grids(device)


if __name__ == "__main__":
    main()
