import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.diffusion.dataset_1 import DIFFUSION_MERGE_PAIRS, get_pair


METHODS = (
    "orthofuse",
    "orthomerge",
    "fisher",
    "standard",
    "standard_rescaled",
    "fisher_rescaled",
)
PROJECTIONS = ("to_q_moft", "to_k_moft", "to_v_moft", "to_out_moft")


def parse_args():
    parser = argparse.ArgumentParser(description="Compute SDXL OFT merge stats and one combined plot.")
    parser.add_argument("--concept_name", default="cat")
    parser.add_argument("--style_name", default="01_08")
    parser.add_argument("--all_dataset", action="store_true")
    parser.add_argument("--output_dir", default="outputs/diffusion/analysis")
    parser.add_argument("--t", type=float, default=0.6)
    parser.add_argument("--alphas", type=float, nargs=2, default=(1.0, 1.0))
    parser.add_argument("--fisher_min", type=float, default=1e-8)
    parser.add_argument("--fisher_rescale", type=float, default=1e10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--progress_every", type=int, default=200)
    return parser.parse_args()


def is_oft_key(key):
    return key.endswith((".L", ".R")) and any(f".{projection}." in key for projection in PROJECTIONS)


def split_key(key):
    prefix, side = key.rsplit(".", 1)
    processor, projection_tail = prefix.split(".to_", 1)
    projection = "to_" + projection_tail.split(".", 1)[0]
    layer = processor.split(".attentions.", 1)[0] if ".attentions." in processor else processor
    return processor, layer, f"{projection}.{side}"


def compact_shape(tensor):
    if tensor.ndim == 2:
        return tuple(tensor.shape)
    if tensor.ndim == 3 and tensor.shape[-1] == tensor.shape[-2]:
        block_size = tensor.shape[-1]
        return (*tensor.shape[:-2], block_size * (block_size - 1) // 2)
    raise ValueError(f"Unexpected OFT tensor shape: {tuple(tensor.shape)}")


def to_coords(tensor):
    if tensor.ndim == 2:
        return tensor.float()
    skew = 0.5 * (tensor.float() - tensor.float().transpose(-1, -2))
    row, col = torch.triu_indices(tensor.shape[-1], tensor.shape[-1], offset=1, device=tensor.device)
    return skew[..., row, col]


def block_size_from_coords(coords):
    dim = coords.shape[-1]
    block_size = int((1 + math.sqrt(1 + 8 * dim)) / 2)
    if block_size * (block_size - 1) // 2 != dim:
        raise ValueError(f"Cannot infer SO(n) block size from dim={dim}")
    return block_size


def identity_stats(coords):
    coords = coords.float()
    norms = torch.linalg.vector_norm(coords, dim=-1)
    block_size = block_size_from_coords(coords)
    geodesic = math.sqrt(2.0) * norms.mean().item()
    cosine = (1.0 - 4.0 * norms.square() / block_size).clamp(-1.0, 1.0).mean().item()
    return {
        "geodesic": geodesic,
        "cosine_to_identity": cosine,
        "coord_norm": norms.mean().item(),
        "max_abs_coord": coords.abs().max().item(),
    }


def parse_fisher_key(key):
    parts = key.split(".", 2)
    if len(parts) == 3 and parts[0] == "layers" and parts[1].isdigit():
        return int(parts[1]), parts[2]
    return None


def build_fisher_map(adapter_state, fisher_state, keys):
    adapter_sig = defaultdict(dict)
    for key in keys:
        processor, _, _ = split_key(key)
        adapter_sig[processor][key[len(processor) + 1 :]] = compact_shape(adapter_state[key])

    fisher_sig = defaultdict(dict)
    for key, tensor in fisher_state.items():
        parsed = parse_fisher_key(key)
        if parsed is not None:
            layer_idx, tail = parsed
            fisher_sig[layer_idx][tail] = tuple(tensor.shape)

    out = {}
    used = set()
    for processor, signature in adapter_sig.items():
        matches = [
            idx
            for idx, candidate in fisher_sig.items()
            if idx not in used and all(candidate.get(tail) == shape for tail, shape in signature.items())
        ]
        if not matches:
            tail, shape = next(iter(signature.items()))
            raise KeyError(f"No Fisher layer matches {processor}: {tail} shape={shape}")
        out[processor] = min(matches)
        used.add(out[processor])
    return out


def fisher_key(adapter_key, processor_to_fisher):
    processor, _, _ = split_key(adapter_key)
    return f"layers.{processor_to_fisher[processor]}." + adapter_key[len(processor) + 1 :]


def standard(weights, alphas):
    stacked = torch.stack(weights).float()
    a = torch.tensor(alphas, dtype=stacked.dtype, device=stacked.device)
    return torch.einsum("t,t...->...", a, stacked)


def norm_rescale(weights, merged, alphas):
    stacked = torch.stack(weights).float().to(merged.device)
    a = torch.tensor(alphas, dtype=stacked.dtype, device=stacked.device)
    while a.dim() < stacked.dim():
        a = a.unsqueeze(-1)
    target = torch.linalg.vector_norm((a * stacked).flatten(1), dim=1).sum()
    source = torch.linalg.vector_norm(merged.float())
    return merged * target / source.clamp_min(1e-8)


def fisher(weights, fishers, alphas):
    ref = weights[0]
    a = torch.tensor(alphas, dtype=torch.float32, device=ref.device)
    numer = torch.zeros_like(ref, dtype=torch.float32)
    denom = torch.zeros_like(ref, dtype=torch.float32)
    for alpha, weight, fisher in zip(a, weights, fishers):
        fisher = fisher.float().to(ref.device)
        numer += alpha * fisher * weight.float().to(ref.device)
        denom += alpha * fisher
    return numer / denom.clamp_min(1e-8)


def standard_norm_rescale(weights, merged, alphas):
    stacked = torch.stack(weights).float().to(merged.device)
    a = torch.tensor(alphas, dtype=stacked.dtype, device=stacked.device).abs()
    target = (a / a.sum().clamp_min(1e-8) * torch.linalg.vector_norm(stacked.flatten(1), dim=1)).sum()
    source = torch.linalg.vector_norm(merged.float())
    return merged * target / source.clamp_min(1e-8)


def merge(method, concept, style, concept_fisher, style_fisher, args):
    weights = [concept, style]
    if method == "orthofuse":
        return (1.0 - args.t) * concept + args.t * style
    if method == "standard":
        return standard(weights, args.alphas)
    if method in {"orthomerge", "standard_rescaled"}:
        return norm_rescale(weights, standard(weights, args.alphas), args.alphas)

    fishers = [
        concept_fisher.clamp_min(args.fisher_min) * args.fisher_rescale,
        style_fisher.clamp_min(args.fisher_min) * args.fisher_rescale,
    ]
    merged = fisher(weights, fishers, args.alphas)
    return standard_norm_rescale(weights, merged, args.alphas) if method == "fisher_rescaled" else merged


diagonal_fisher = fisher


def selected_pairs(args):
    return DIFFUSION_MERGE_PAIRS if args.all_dataset else [get_pair(args.concept_name, args.style_name)]


def analyze_pair(pair, args):
    device = torch.device(args.device)
    concept_state = load_file(pair["concept"]["adapter_path"], device=str(device))
    style_state = load_file(pair["style"]["adapter_path"], device=str(device))
    concept_fisher = load_file(pair["concept"]["fim_path"], device=str(device))
    style_fisher = load_file(pair["style"]["fim_path"], device=str(device))

    keys = sorted(k for k in concept_state if is_oft_key(k))
    processor_to_fisher = build_fisher_map(concept_state, concept_fisher, keys)
    rows = []
    for idx, key in enumerate(keys, 1):
        if args.progress_every and (idx == 1 or idx % args.progress_every == 0):
            print(f"[analysis] {pair['name']} {idx}/{len(keys)} tensors", flush=True)
        processor, layer, module = split_key(key)
        concept = to_coords(concept_state[key])
        style = to_coords(style_state[key])
        cf = concept_fisher[fisher_key(key, processor_to_fisher)]
        sf = style_fisher[fisher_key(key, processor_to_fisher)]
        cache = {}
        for method in METHODS:
            cache_key = "standard_rescaled" if method == "orthomerge" else method
            if cache_key not in cache:
                cache[cache_key] = merge(method, concept, style, cf, sf, args)
            rows.append(
                {
                    "pair": pair["name"],
                    "method": method,
                    "key": key,
                    "processor": processor,
                    "layer": layer,
                    "module": module,
                    **identity_stats(cache[cache_key]),
                }
            )
    return rows


def aggregate(rows, group_key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row[group_key])].append(row)
    out = []
    for (method, group), values in grouped.items():
        out.append(
            {
                "method": method,
                "group": group,
                "n": len(values),
                "geodesic": sum(v["geodesic"] for v in values) / len(values),
                "cosine_to_identity": sum(v["cosine_to_identity"] for v in values) / len(values),
                "coord_norm": sum(v["coord_norm"] for v in values) / len(values),
                "max_abs_coord": max(v["max_abs_coord"] for v in values),
            }
        )
    return sorted(out, key=lambda row: (row["group"], row["method"]))


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(layer_rows, module_rows, output_path, title):
    layers = sorted({row["group"] for row in layer_rows})
    modules = sorted({row["group"] for row in module_rows})
    layer = {(row["method"], row["group"]): row for row in layer_rows}
    module = {(row["method"], row["group"]): row for row in module_rows}

    fig, axes = plt.subplots(2, 2, figsize=(18, 11), constrained_layout=True)
    ax_dist, ax_cos, ax_heat, ax_bar = axes.ravel()

    for method in METHODS:
        ax_dist.plot(layers, [layer[(method, x)]["geodesic"] for x in layers], marker="o", label=method)
        ax_cos.plot(layers, [layer[(method, x)]["cosine_to_identity"] for x in layers], marker="o", label=method)

    for ax in (ax_dist, ax_cos):
        ax.tick_params(axis="x", rotation=35)
        ax.grid(alpha=0.25)
    ax_dist.set_title("Layerwise distance to pretrained")
    ax_dist.set_ylabel("mean geodesic proxy")
    ax_cos.set_title("Layerwise cosine to pretrained")
    ax_cos.set_ylabel("mean trace/cosine proxy")

    heat = torch.tensor([[module[(method, x)]["geodesic"] for x in modules] for method in METHODS])
    im = ax_heat.imshow(heat.numpy(), aspect="auto", cmap="viridis")
    ax_heat.set_title("Module-wise distance to pretrained")
    ax_heat.set_yticks(range(len(METHODS)), METHODS)
    ax_heat.set_xticks(range(len(modules)), modules, rotation=45, ha="right")
    fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)

    avg = [sum(layer[(method, x)]["geodesic"] for x in layers) / len(layers) for method in METHODS]
    ax_bar.bar(METHODS, avg)
    ax_bar.set_title("Average distance by merge")
    ax_bar.tick_params(axis="x", rotation=35)

    handles, labels = ax_dist.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3)
    fig.suptitle(title, fontsize=16)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = selected_pairs(args)
    rows = []
    for pair in pairs:
        print(f"[analysis] pair={pair['name']}", flush=True)
        rows.extend(analyze_pair(pair, args))

    layer_rows = aggregate(rows, "layer")
    module_rows = aggregate(rows, "module")
    prefix = "all_pairs" if args.all_dataset else pairs[0]["name"]
    write_csv(output_dir / f"{prefix}_per_tensor.csv", rows)
    write_csv(output_dir / f"{prefix}_layer_stats.csv", layer_rows)
    write_csv(output_dir / f"{prefix}_module_stats.csv", module_rows)
    with open(output_dir / f"{prefix}_stats.json", "w", encoding="utf-8") as handle:
        json.dump({"pairs": [pair["name"] for pair in pairs], "layer": layer_rows, "module": module_rows}, handle, indent=2)
    plot(layer_rows, module_rows, output_dir / f"{prefix}_sdxl_merge_stats.png", f"SDXL merge stats ({prefix})")
    print(f"Saved stats to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
