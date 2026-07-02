import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.diffusion.dataset_1 import STYLE_ADAPTERS
from src.diffusion.interpolation_hyperparam_search import (
    PROMPTS,
    alpha_grid,
    folder,
    image_path,
    root,
    text_prompt,
)
from src.diffusion.results.pareto import (
    METRIC_KEYS,
    add_evaluator_args,
    add_pareto_flags,
    aggregate_rows,
    make_evaluator,
    output_prefix,
    plot_style_concept_curves,
    score_records,
    selected_pairs,
    SimilarityScorer,
    write_csv,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=REPO_ROOT / "outputs/diffusion")
    parser.add_argument("--results_folder", type=str, default="interpolation_hyperparam_search")
    parser.add_argument("--output_prefix", type=str, default=None)
    parser.add_argument("--samples", nargs="+", default=None)
    parser.add_argument("--num_points", type=int, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--mu", type=float, default=4.0)
    parser.add_argument("--version_start", type=int, default=0)
    parser.add_argument("--image_index", type=int, default=0)
    add_evaluator_args(parser)
    return parser.parse_args()


def all_style_samples():
    return [f"style:{style['name']}" for style in STYLE_ADAPTERS]


def sample_folder(args, pair):
    return root(args) / "samples" / folder(args, pair["name"])


def version_indices(path):
    indices = []
    for child in path.iterdir():
        if not child.is_dir():
            continue
        match = re.fullmatch(r"version_(\d+)", child.name)
        if match:
            indices.append(int(match.group(1)))
    return sorted(indices)


def assert_all_styles_present(args, pairs):
    present_styles = {
        pair["style"]["name"]
        for pair in pairs
        if sample_folder(args, pair).exists()
    }
    expected_styles = {style["name"] for style in STYLE_ADAPTERS}
    missing = sorted(expected_styles - present_styles)
    if missing:
        raise FileNotFoundError(
            "Missing interpolation outputs for dataset styles: "
            + ", ".join(missing)
        )


def infer_num_points(args, pairs):
    expected = None
    for pair in pairs:
        pair_folder = sample_folder(args, pair)
        if not pair_folder.exists():
            raise FileNotFoundError(f"Missing interpolation output folder: {pair_folder}")

        indices = version_indices(pair_folder)
        if not indices:
            raise FileNotFoundError(f"No version_* folders found in {pair_folder}")

        contiguous = list(range(args.version_start, max(indices) + 1))
        if indices != contiguous:
            raise ValueError(
                f"Expected contiguous version folders starting at version_{args.version_start} "
                f"in {pair_folder}; found {', '.join(f'version_{idx}' for idx in indices)}"
            )
        if expected is None:
            expected = indices
        elif indices != expected:
            raise ValueError(
                f"Inconsistent version folders in {pair_folder}; expected "
                f"{', '.join(f'version_{idx}' for idx in expected)}, found "
                f"{', '.join(f'version_{idx}' for idx in indices)}"
            )

    if expected is None:
        raise RuntimeError("Cannot infer --num_points without selected pairs.")
    return len(expected)


def records(args, pairs):
    alphas = alpha_grid(args.num_points)
    for pair in pairs:
        for point_idx, (t, alpha) in enumerate(alphas):
            for prompt_name, template in PROMPTS.items():
                yield {
                    "pair": pair,
                    "pair_name": pair["name"],
                    "prompt": prompt_name,
                    "mu": args.mu,
                    "step": point_idx,
                    "t": t,
                    "alpha_1": alpha[0],
                    "alpha_2": alpha[1],
                    "image_path": image_path(args, pair, point_idx, template),
                    "text_prompt": text_prompt(pair, template),
                }


def average_prompt_rows(rows, prompt_names, metric_keys):
    prompts = tuple(prompt_names)
    expected_prompts = set(prompts)
    missing_prompts = expected_prompts - {row["prompt"] for row in rows}
    if missing_prompts:
        raise ValueError(
            "Cannot make averaged prompt plot; missing prompt rows for: "
            + ", ".join(sorted(missing_prompts))
        )

    group_keys = ("mu", "step", "t", "alpha_1", "alpha_2")
    grouped = defaultdict(list)
    for row in rows:
        if row["prompt"] in expected_prompts:
            grouped[tuple(row[key] for key in group_keys)].append(row)

    averaged = []
    for key, values in grouped.items():
        group_prompts = {row["prompt"] for row in values}
        if group_prompts != expected_prompts:
            raise ValueError(
                "Cannot make averaged prompt plot; expected prompts "
                f"{', '.join(prompts)} for {dict(zip(group_keys, key))}, found "
                + ", ".join(sorted(group_prompts))
            )

        row = {group_key: value for group_key, value in zip(group_keys, key)}
        row["prompt"] = "average"
        row["n"] = sum(value["n"] for value in values)
        for metric in metric_keys:
            present = [value[metric] for value in values if metric in value]
            if len(present) != len(prompts):
                raise ValueError(
                    f"Cannot average metric {metric!r} for {dict(zip(group_keys, key))}; "
                    f"expected {len(prompts)} values, found {len(present)}"
                )
            row[metric] = sum(present) / len(present)
        averaged.append(row)

    averaged = sorted(averaged, key=lambda row: (row["mu"], row["step"]))
    add_pareto_flags(averaged)
    return averaged


def run(args):
    samples = args.samples or all_style_samples()
    pairs = selected_pairs(samples)
    if args.samples is None:
        assert_all_styles_present(args, pairs)
    if args.num_points is None:
        args.num_points = infer_num_points(args, pairs)
        print(f"[interpolation-results] inferred num_points={args.num_points}", flush=True)

    reference_root = args.reference_root or args.output_dir / "d1_images"
    scorer = SimilarityScorer(make_evaluator(args), reference_root)
    rows = score_records(records(args, pairs), scorer)
    if not rows:
        raise RuntimeError("No interpolation search images were found for the requested pairs.")

    group_keys = ("prompt", "mu", "step", "t", "alpha_1", "alpha_2")
    metric_keys = (*METRIC_KEYS, "clip_text")
    aggregate = aggregate_rows(rows, group_keys, metric_keys)
    aggregate = sorted(aggregate, key=lambda row: (row["prompt"], row["mu"], row["step"]))
    add_pareto_flags(aggregate, group_keys=("prompt",))

    prefix = args.output_prefix or output_prefix(pairs)
    csv_path = root(args) / f"{prefix}_pareto_frontier.csv"
    fieldnames = [
        *group_keys,
        "n",
        "clip_concept",
        "clip_style",
        "clip_text",
        "dino_concept",
        "dino_style",
        "clip_pareto",
        "dino_pareto",
    ]
    write_csv(csv_path, aggregate, fieldnames)
    for prompt_name in PROMPTS:
        prompt_rows = [row for row in aggregate if row["prompt"] == prompt_name]
        png_path = root(args) / f"{prefix}_pareto_frontier_{prompt_name}.png"
        plot_style_concept_curves(
            prompt_rows,
            png_path,
            f"Interpolation hyperparameter search ({prompt_name})",
            series_key="mu",
            label_key="alpha_1",
        )
        print(f"[interpolation-results] wrote {png_path}", flush=True)
    default_png_path = root(args) / f"{prefix}_pareto_frontier.png"
    averaged_rows = average_prompt_rows(aggregate, ("normal", "bicycle"), metric_keys)
    plot_style_concept_curves(
        averaged_rows,
        default_png_path,
        "Interpolation hyperparameter search (prompt average)",
        series_key="mu",
        label_key="alpha_1",
    )
    print(f"[interpolation-results] wrote {default_png_path}", flush=True)
    print(f"[interpolation-results] wrote {csv_path}", flush=True)


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
