import argparse
import ast
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
import torch
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.diffusion.analysis.rescaling_curve import (  # noqa: E402
    coords_to_skew,
    norm_ratio_rows,
    plot,
    plot_rescaled_norm,
)
from src.diffusion.analysis.sdxl_merge_stats import (  # noqa: E402
    build_fisher_map,
    fisher,
    fisher_key,
    is_oft_key,
    selected_pairs,
    to_coords,
)


def correction_mus():
    path = REPO_ROOT / "src/diffusion/correction_hyperparam_search.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "mus":
            for statement in node.body:
                if isinstance(statement, ast.Return):
                    return ast.literal_eval(statement.value)
    raise ValueError(f"Could not read mus() from {path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept_name", default="cat")
    parser.add_argument("--style_name", default="01_08")
    parser.add_argument("--all_dataset", action="store_true")
    parser.add_argument("--output_dir", default="outputs/diffusion/analysis")
    parser.add_argument("--num_points", type=int, default=21)
    parser.add_argument("--fisher_min", type=float, default=1e-14)
    parser.add_argument("--fisher_rescale", type=float, default=1e10)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def corrected(base_by_key, mu, alphas):
    scale = 1.0 + float(mu) * float(alphas[0]) * (1.0 - float(alphas[0]))
    return {key: scale * value for key, value in base_by_key.items()}


def analyze_pair(pair, args):
    device = torch.device(args.device)
    concept_state = load_file(pair["concept"]["adapter_path"], device=str(device))
    style_state = load_file(pair["style"]["adapter_path"], device=str(device))
    concept_fisher = load_file(pair["concept"]["fim_path"], device=str(device))
    style_fisher = load_file(pair["style"]["fim_path"], device=str(device))

    keys = sorted(k for k in concept_state if is_oft_key(k))
    processor_to_fisher = build_fisher_map(concept_state, concept_fisher, keys)
    grid = torch.linspace(0.0, 1.0, args.num_points).tolist()
    mu_values = correction_mus()
    rows = []

    for alpha_concept in grid:
        alphas = (1.0 - alpha_concept, alpha_concept)
        fisher_base = {}
        for key in keys:
            weights = [to_coords(concept_state[key]), to_coords(style_state[key])]
            fishers = [
                concept_fisher[fisher_key(key, processor_to_fisher)].clamp_min(args.fisher_min)
                * args.fisher_rescale,
                style_fisher[fisher_key(key, processor_to_fisher)].clamp_min(args.fisher_min)
                * args.fisher_rescale,
            ]
            fisher_base[key] = coords_to_skew(fisher(weights, fishers, alphas))

        for mu in mu_values:
            rows.append(
                {
                    "pair": pair["name"],
                    "method_pair": f"fisher_mu{mu:g}",
                    "alpha_concept": alphas[0],
                    "alpha_style": alphas[1],
                    **norm_ratio_rows(fisher_base, corrected(fisher_base, mu, alphas)),
                }
            )
    return rows


def average_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method_pair"], row["alpha_concept"])].append(row)
    out = []
    for (method_pair, alpha), values in sorted(grouped.items()):
        numeric = {
            key: sum(float(row[key]) for row in values) / len(values)
            for key in values[0]
            if key not in {"pair", "method_pair"}
        }
        out.append({"pair": "average", "method_pair": method_pair, **numeric})
    return out


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    pairs = selected_pairs(args)
    for pair in pairs:
        print(f"[correction_rescale_curve] {pair['name']}", flush=True)
        rows.extend(analyze_pair(pair, args))

    prefix = "all_pairs" if args.all_dataset else pairs[0]["name"]
    averaged = average_rows(rows) if args.all_dataset else rows
    plot(averaged, output_dir / f"{prefix}_correction_rescaling_curve.png")
    plot_rescaled_norm(averaged, output_dir / f"{prefix}_correction_rescaled_norm.png")
    print(f"Saved {prefix} correction rescaling curves to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
