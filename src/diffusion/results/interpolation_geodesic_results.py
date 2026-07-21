import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "OrthoFuse"))
sys.path.insert(0, str(REPO_ROOT))

from nb_utils.eval_sets import merge_test_set
from src.diffusion.geodesic_interpolation import (
    ORTHOFUSE_METHODS,
    alphas_from_t,
    gradient_args,
    method_output_name,
    orthofuse_args,
)
from src.diffusion.pipeline_outputs import (
    gradients_inference_folder_name,
    orthofuse_inference_folder_name,
)
from src.diffusion.results.pareto import (
    METRIC_KEYS,
    add_evaluator_args,
    add_pareto_flags,
    aggregate_rows,
    make_evaluator,
    plot_style_concept_curves,
    score_records,
    selected_pairs,
    SimilarityScorer,
    write_csv,
)


DEFAULT_T_VALUES = (0.0, 0.2, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 1.0)
DEFAULT_RESULTS_FOLDER = "samples_interpolation_geodesic"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default=str(REPO_ROOT / "src/diffusion/config/config.yaml"))
    parser.add_argument("--output_dir", type=Path, default=REPO_ROOT / "outputs/diffusion")
    parser.add_argument("--method_output_root", type=Path, default=None)
    parser.add_argument("--results_folder", type=str, default=DEFAULT_RESULTS_FOLDER)
    parser.add_argument("--plot_dir", type=Path, default=None)
    parser.add_argument("--output_prefix", type=str, default="interpolation_geodesic")
    parser.add_argument("--samples", nargs="+", default=["all_dataset_pairs"])
    parser.add_argument("--methods", nargs="+", default=["fisher_geodesic", "orthofuse"])
    parser.add_argument("--prompt_templates", nargs="+", default=None)
    parser.add_argument("--t_values", nargs="+", type=float, default=list(DEFAULT_T_VALUES))
    parser.add_argument("--num_images_per_medium_prompt", type=int, default=2)
    parser.add_argument("--num_images_per_base_prompt", type=int, default=0)
    parser.add_argument("--batch_size_medium", type=int, default=2)
    parser.add_argument("--batch_size_base", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--geodesic_backend", choices=["cayley"], default="cayley")
    parser.add_argument("--fisher_min", type=float, default=1e-14)
    parser.add_argument("--fisher_rescale", type=float, default=1e10)
    parser.add_argument("--fisher_backend", choices=["diagonal", "kfac"], default="diagonal")
    parser.add_argument("--fisher_correction_mu", type=float, default=None)
    parser.add_argument("--correction_mu", type=float, default=2.0)
    parser.add_argument(
        "--fim_normalization",
        choices=[
            "none",
            "trace",
            "frobenius",
            "layer-trace",
            "layer-frobenius",
            "kl",
        ],
        default="trace",
    )
    parser.add_argument("--orthofuse_postprocessing_method", type=str, default="curve_over_id")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replace_inference_output", action="store_true")
    parser.add_argument("--scores_cache_path", type=Path, default=None)
    parser.add_argument("--replace_scores_cache", action="store_true")
    add_evaluator_args(parser)
    return parser.parse_args()


def prompt(pair, template, placeholders=True):
    concept_key = "placeholder_token" if placeholders else "class_name"
    style_key = "placeholder_token" if placeholders else "name"
    return template.format(pair["concept"][concept_key], pair["style"][style_key])


def method_root(args, method):
    if args.method_output_root is None:
        return args.output_dir / args.results_folder
    return Path(args.method_output_root) / method_output_name(args, method)


def method_folder(args, method, pair, t, version):
    if method == "fisher_geodesic":
        run_args = gradient_args(
            args,
            alphas_from_t(t),
            version,
            merge_mode="geodesic",
            use_fishers=True,
            backend=args.geodesic_backend,
        )
        run_args.dataset_pair_name = pair["name"]
        return gradients_inference_folder_name(run_args)
    if method in ("fisher", "fisher_rescaled"):
        run_args = gradient_args(
            args,
            alphas_from_t(t),
            version,
            merge_mode=method,
            use_fishers=True,
            backend=args.geodesic_backend,
        )
        run_args.dataset_pair_name = pair["name"]
        return gradients_inference_folder_name(run_args)
    if method in ORTHOFUSE_METHODS:
        postprocessing_method = (
            args.orthofuse_postprocessing_method
            if method == "orthofuse"
            else ORTHOFUSE_METHODS[method]
        )
        run_args = orthofuse_args(args, t, version, postprocessing_method)
        run_args.dataset_pair_name = pair["name"]
        return orthofuse_inference_folder_name(run_args)
    raise ValueError(f"Unsupported method: {method}")


def records(args, pairs):
    for pair in pairs:
        for method in args.methods:
            for step, t in enumerate(args.t_values):
                folder = method_folder(args, method, pair, t, step)
                version_root = method_root(args, method) / "samples" / folder / f"version_{step}"
                for prompt_idx, template in enumerate(args.prompt_templates or merge_test_set):
                    prompt_dir = version_root / prompt(pair, template, placeholders=True)
                    for image_idx in range(args.num_images_per_medium_prompt):
                        yield {
                            "pair": pair,
                            "pair_name": pair["name"],
                            "method": method,
                            "step": step,
                            "t": t,
                            "prompt_idx": prompt_idx,
                            "image_idx": image_idx,
                            "image_path": prompt_dir / f"{image_idx}.png",
                            "text_prompt": prompt(pair, template, placeholders=False),
                        }


def run(args):
    plot_dir = args.plot_dir or args.output_dir / "tables"
    cache_path = args.scores_cache_path or plot_dir / f"{args.output_prefix}_raw.json"

    if cache_path.exists() and not args.replace_scores_cache:
        with cache_path.open("r", encoding="utf-8") as handle:
            rows = json.load(handle)
        print(f"[interpolation-geodesic-results] loaded cached scores from {cache_path}", flush=True)
    else:
        pairs = selected_pairs(args.samples)
        reference_root = args.reference_root or args.output_dir / "d1_images"
        scorer = SimilarityScorer(make_evaluator(args), reference_root)
        rows = score_records(records(args, pairs), scorer)
        if not rows:
            raise RuntimeError("No geodesic interpolation images were found.")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as handle:
            json.dump(rows, handle)
        print(f"[interpolation-geodesic-results] wrote {cache_path}", flush=True)

    if not rows:
        raise RuntimeError("No geodesic interpolation images were found.")

    metric_keys = (*METRIC_KEYS, "clip_text")
    raw_csv = plot_dir / f"{args.output_prefix}_raw.csv"
    write_csv(
        raw_csv,
        rows,
        [
            "pair_name",
            "method",
            "step",
            "t",
            "prompt_idx",
            "image_idx",
            "image_path",
            "clip_concept",
            "clip_style",
            "clip_text",
            "dino_concept",
            "dino_style",
        ],
    )

    group_keys = ("method", "step", "t")
    aggregate = aggregate_rows(rows, group_keys, metric_keys)
    aggregate = sorted(aggregate, key=lambda row: (row["method"], row["step"]))
    add_pareto_flags(aggregate)

    csv_path = plot_dir / f"{args.output_prefix}_pareto_frontier.csv"
    png_path = plot_dir / f"{args.output_prefix}_pareto_frontier.png"
    write_csv(
        csv_path,
        aggregate,
        [
            *group_keys,
            "n",
            "clip_concept",
            "clip_style",
            "clip_text",
            "dino_concept",
            "dino_style",
            "clip_pareto",
            "dino_pareto",
        ],
    )
    plot_style_concept_curves(
        aggregate,
        png_path,
        "Geodesic interpolation",
        series_key="method",
        label_key="t",
    )
    print(f"[interpolation-geodesic-results] wrote {csv_path}", flush=True)
    print(f"[interpolation-geodesic-results] wrote {raw_csv}", flush=True)
    print(f"[interpolation-geodesic-results] wrote {png_path}", flush=True)


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
