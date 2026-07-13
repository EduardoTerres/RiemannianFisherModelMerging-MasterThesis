import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

USER = os.environ.get("USER", "user")
os.environ.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib-{USER}")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.diffusion.analysis.sdxl_merge_stats import (
    build_fisher_map,
    fisher_key,
    is_oft_key,
    split_key,
)
from src.diffusion.dataset_1 import DIFFUSION_MERGE_PAIRS, get_pair

CONCEPT_COLOR = "#4C78A8"
STYLE_COLOR = "#F58518"
TIE_COLOR = "#BAB0AC"


def parse_args():
    parser = argparse.ArgumentParser(description="Compare concept/style FIM term dominance.")
    parser.add_argument("--samples", nargs="+", default=["cat:pots"])
    parser.add_argument("--output_dir", default="outputs/diffusion/analysis")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eps", type=float, default=0.0)
    parser.add_argument("--force-compute", action="store_true")
    return parser.parse_args()


def selected_pairs(samples):
    pairs = []
    for sample in samples:
        if sample == "all_dataset_pairs":
            pairs.extend(DIFFUSION_MERGE_PAIRS)
            continue
        if ":" not in sample:
            raise ValueError("Each --samples value must be 'all_dataset_pairs' or '<concept>:<style>'.")
        concept_name, style_name = sample.split(":", 1)
        pairs.append(get_pair(concept_name, style_name))
    return pairs


def counts(concept_fim, style_fim, eps):
    cf = concept_fim.float().flatten()
    sf = style_fim.float().flatten()
    concept_gt = (cf > sf + eps).sum().item()
    style_gt = (sf > cf + eps).sum().item()
    ties = cf.numel() - concept_gt - style_gt
    diff = cf - sf
    return {
        "n": cf.numel(),
        "concept_gt": concept_gt,
        "style_gt": style_gt,
        "ties": ties,
        "concept_frac": concept_gt / cf.numel(),
        "style_frac": style_gt / cf.numel(),
        "tie_frac": ties / cf.numel(),
        "mean_log10_ratio": torch.log10((cf.clamp_min(1e-30)) / (sf.clamp_min(1e-30))).mean().item(),
        "mean_diff": diff.mean().item(),
        "max_abs_diff": diff.abs().max().item(),
    }


def analyze_pair(pair, args):
    device = torch.device(args.device)
    concept_state = load_file(pair["concept"]["adapter_path"], device=str(device))
    concept_fisher = load_file(pair["concept"]["fim_path"], device=str(device))
    style_fisher = load_file(pair["style"]["fim_path"], device=str(device))

    keys = sorted(k for k in concept_state if is_oft_key(k))
    processor_to_fisher = build_fisher_map(concept_state, concept_fisher, keys)
    rows = []

    for key in keys:
        processor, layer, module = split_key(key)
        fk = fisher_key(key, processor_to_fisher)
        rows.append(
            {
                "pair": pair["name"],
                "key": key,
                "processor": processor,
                "layer": layer,
                "module": module,
                **counts(concept_fisher[fk], style_fisher[fk], args.eps),
            }
        )
    return rows


def aggregate(rows, group_key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["pair"], row[group_key])].append(row)

    out = []
    for (pair, group), values in grouped.items():
        n = sum(row["n"] for row in values)
        concept_gt = sum(row["concept_gt"] for row in values)
        style_gt = sum(row["style_gt"] for row in values)
        ties = sum(row["ties"] for row in values)
        out.append(
            {
                "pair": pair,
                "group": group,
                "n": n,
                "concept_gt": concept_gt,
                "style_gt": style_gt,
                "ties": ties,
                "concept_frac": concept_gt / n,
                "style_frac": style_gt / n,
                "tie_frac": ties / n,
                "mean_log10_ratio": sum(row["mean_log10_ratio"] * row["n"] for row in values) / n,
                "mean_diff": sum(row["mean_diff"] * row["n"] for row in values) / n,
                "max_abs_diff": max(row["max_abs_diff"] for row in values),
            }
        )
    return sorted(out, key=lambda row: (row["pair"], row["group"]))


def average_groups(group_rows):
    grouped = defaultdict(list)
    for row in group_rows:
        grouped[row["group"]].append(row)

    out = []
    for group, values in grouped.items():
        n = sum(row["n"] for row in values)
        concept_gt = sum(row["concept_gt"] for row in values)
        style_gt = sum(row["style_gt"] for row in values)
        ties = sum(row["ties"] for row in values)
        out.append(
            {
                "pair": "average",
                "group": group,
                "n": n,
                "concept_gt": concept_gt,
                "style_gt": style_gt,
                "ties": ties,
                "concept_frac": concept_gt / n,
                "style_frac": style_gt / n,
                "tie_frac": ties / n,
                "mean_log10_ratio": sum(row["mean_log10_ratio"] * row["n"] for row in values) / n,
                "mean_diff": sum(row["mean_diff"] * row["n"] for row in values) / n,
                "max_abs_diff": max(row["max_abs_diff"] for row in values),
            }
        )
    return sorted(out, key=lambda row: row["group"])


def average_summary(summary_rows):
    n = sum(row["n"] for row in summary_rows)
    concept_gt = sum(row["concept_gt"] for row in summary_rows)
    style_gt = sum(row["style_gt"] for row in summary_rows)
    ties = sum(row["ties"] for row in summary_rows)
    return {
        "pair": "average",
        "group": "all_pairs",
        "n": n,
        "concept_gt": concept_gt,
        "style_gt": style_gt,
        "ties": ties,
        "concept_frac": concept_gt / n,
        "style_frac": style_gt / n,
        "tie_frac": ties / n,
        "mean_log10_ratio": sum(row["mean_log10_ratio"] * row["n"] for row in summary_rows) / n,
        "mean_diff": sum(row["mean_diff"] * row["n"] for row in summary_rows) / n,
        "max_abs_diff": max(row["max_abs_diff"] for row in summary_rows),
    }


def plot_percentage_square(ax, summary_rows):
    row = average_summary(summary_rows) if len(summary_rows) > 1 else summary_rows[0]
    parts = [
        ("concept > style", row["concept_frac"], CONCEPT_COLOR),
        ("style > concept", row["style_frac"], STYLE_COLOR),
        ("tie", row["tie_frac"], TIE_COLOR),
    ]
    y = 0.0
    for label, frac, color in parts:
        ax.add_patch(plt.Rectangle((0.0, y), 1.0, frac, facecolor=color, edgecolor="white", linewidth=1.5))
        if frac >= 0.04:
            ax.text(0.5, y + frac / 2, f"{label}\n{frac:.1%}", ha="center", va="center", fontsize=9)
        y += frac
    ax.set_title("Overall dominance")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")


def plot(summary_rows, layer_rows, violin_rows, output_path, title):
    pairs = [row["pair"] for row in summary_rows]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    if pairs == ["average"]:
        violin_data = [[-row["mean_log10_ratio"] for row in violin_rows]]
    else:
        violin_data = [[-row["mean_log10_ratio"] for row in violin_rows if row["pair"] == pair] for pair in pairs]

    axes[0].violinplot(violin_data, showmeans=True, showmedians=True)
    y_values = [value for values in violin_data for value in values]
    y_pad = 0.05 * (max(y_values) - min(y_values) or 1.0)
    y_min, y_max = min(y_values) - y_pad, max(y_values) + y_pad
    axes[0].axhspan(y_min, 0.0, color=CONCEPT_COLOR, alpha=0.12)
    axes[0].axhspan(0.0, y_max, color=STYLE_COLOR, alpha=0.12)
    axes[0].set_ylim(y_min, y_max)
    axes[0].axhline(0.0, color="black", linewidth=1, linestyle="--", alpha=0.8)
    axes[0].set_title("FIM log-ratio distribution")
    axes[0].set_ylabel("mean log10(style FIM / concept FIM)")
    axes[0].set_xticks(range(1, len(pairs) + 1), pairs, rotation=25, ha="right")

    plot_percentage_square(axes[1], summary_rows)

    for pair in pairs:
        series = [row for row in layer_rows if row["pair"] == pair]
        axes[2].plot([row["group"] for row in series], [row["concept_frac"] for row in series], marker="o", label=pair)
    axes[2].set_title("Concept-dominant terms by layer")
    axes[2].set_ylabel("fraction concept > style")
    axes[2].tick_params(axis="x", rotation=35)
    axes[2].legend(fontsize=8)

    for ax in (axes[0], axes[2]):
        ax.grid(alpha=0.25)
    fig.suptitle(title)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = selected_pairs(args.samples)
    average_plot = args.samples == ["all_dataset_pairs"]
    prefix = "all_pairs" if average_plot else pairs[0]["name"]
    json_path = output_dir / f"{prefix}_fim_contributions.json"
    if json_path.exists() and not args.force_compute:
        print(f"[fim_contributions] {json_path} exists; replotting without recompute.", flush=True)
        with open(json_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        summary_rows = payload["summary"]
        layer_rows = payload["layer"]
        plot_summary_rows = [average_summary(summary_rows)] if average_plot else summary_rows
        plot_layer_rows = average_groups(layer_rows) if average_plot else layer_rows
        plot_violin_rows = payload.get("per_tensor", summary_rows)
        plot(
            plot_summary_rows,
            plot_layer_rows,
            plot_violin_rows,
            output_dir / f"{prefix}_fim_contributions.png",
            f"FIM contributions ({prefix})",
        )
        print(f"Saved FIM contribution plot to {output_dir}", flush=True)
        return

    rows = []
    for pair in pairs:
        print(f"[fim_contributions] {pair['name']}", flush=True)
        rows.extend(analyze_pair(pair, args))

    summary_rows = aggregate(rows, "pair")
    layer_rows = aggregate(rows, "layer")
    module_rows = aggregate(rows, "module")
    plot_summary_rows = [average_summary(summary_rows)] if average_plot else summary_rows
    plot_layer_rows = average_groups(layer_rows) if average_plot else layer_rows

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump({"per_tensor": rows, "summary": summary_rows, "layer": layer_rows, "module": module_rows}, handle, indent=2)
    plot(
        plot_summary_rows,
        plot_layer_rows,
        rows,
        output_dir / f"{prefix}_fim_contributions.png",
        f"FIM contributions ({prefix})",
    )
    print(f"Saved FIM contribution stats to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
