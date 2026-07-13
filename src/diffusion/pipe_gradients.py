import argparse
import sys
import warnings
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "OrthoFuse"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from moft.inferencer_sdxl import inferencers
from nb_utils.eval_sets import merge_base_set, merge_test_set
from src.diffusion.dataset_1 import DIFFUSION_MERGE_PAIRS, get_pair
from src.diffusion.pipeline_outputs import (
    existing_output_path,
    gradients_inference_folder_name,
)


warnings.filterwarnings("ignore")


def canonical_merge_mode(mode):
    if mode == "diagonal_fisher":
        return "fisher"
    if mode == "diagonal_fisher_rescaled":
        return "fisher_rescaled"
    return mode


def correction_mu(args):
    mu = getattr(args, "fisher_correction_mu", None)
    if mu is not None:
        return mu
    return getattr(args, "diagonal_fisher_correction_mu", None)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--checkpoint_idx", type=str, default=None)
    parser.add_argument("--moft_layers_concept_path", type=str, default=None)
    parser.add_argument("--moft_layers_style_path", type=str, default=None)
    parser.add_argument("--concept_fisher_path", type=str, default=None)
    parser.add_argument("--style_fisher_path", type=str, default=None)
    parser.add_argument("--fisher_min", type=float, default=None)
    parser.add_argument("--fisher_rescale", type=float, default=None)
    parser.add_argument("--fisher_backend", choices=["diagonal", "kfac"], default="diagonal")
    parser.add_argument("--alphas", type=float, nargs=2, default=None, metavar=("CONCEPT", "STYLE"))
    parser.add_argument("--merge_mode", type=str, default=None)
    parser.add_argument("--fisher_correction_mu", type=float, default=None)
    parser.add_argument("--diagonal_fisher_correction_mu", type=float, default=None)
    parser.add_argument("--geodesic_backend", choices=["cayley"], default="cayley")
    parser.add_argument("--geodesic_use_fishers", action="store_true")
    parser.add_argument(
        "--samples",
        type=str,
        default=None,
        help=(
            "Dataset pair selector: all_dataset_pairs, <concept>:<style>, "
            "concept:<concept>, or style:<style>."
        ),
    )
    parser.add_argument("--concept_name", type=str, default=None)
    parser.add_argument("--style_name", type=str, default=None)
    parser.add_argument("--dataset_pair_name", type=str, default=None)
    parser.add_argument("--rescale", action="store_true")
    parser.add_argument("--num_images_per_medium_prompt", type=int, default=1)
    parser.add_argument("--num_images_per_base_prompt", type=int, default=10)
    parser.add_argument("--batch_size_medium", type=int, default=1)
    parser.add_argument("--batch_size_base", type=int, default=10)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--replace_inference_output", action="store_true")
    parser.add_argument("--version", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def has_fisher_inputs(args):
    concept_fisher_path = getattr(args, "concept_fisher_path", None)
    style_fisher_path = getattr(args, "style_fisher_path", None)
    if concept_fisher_path is None and style_fisher_path is None:
        return False
    if concept_fisher_path is None or style_fisher_path is None:
        raise ValueError("Both concept_fisher_path and style_fisher_path are required.")
    return True


def prepare_merge_args(args):
    if getattr(args, "merge_mode", None) is not None:
        args.merge_mode = canonical_merge_mode(args.merge_mode)
        args.diagonal_fisher_correction_mu = correction_mu(args)
        return args
    if has_fisher_inputs(args):
        args.merge_mode = "fisher_rescaled" if getattr(args, "rescale", False) else "fisher"
        if getattr(args, "fisher_min", None) is None:
            args.fisher_min = 1e-8
        if getattr(args, "fisher_rescale", None) is None:
            args.fisher_rescale = 1e10
    else:
        args.merge_mode = "standard_rescaled" if getattr(args, "rescale", False) else "standard"
    args.diagonal_fisher_correction_mu = correction_mu(args)
    return args


def fisher_path_from_entry(entry, backend):
    key = "kfac_path" if backend == "kfac" else "fim_path"
    if key not in entry:
        raise KeyError(f"Missing {key!r} for {entry.get('type', 'entry')} {entry.get('name')!r}")
    return entry[key]


def apply_pair(args, pair):
    args.moft_layers_concept_path = pair["concept"]["adapter_path"]
    args.moft_layers_style_path = pair["style"]["adapter_path"]
    backend = getattr(args, "fisher_backend", "diagonal")
    args.concept_fisher_path = fisher_path_from_entry(pair["concept"], backend)
    args.style_fisher_path = fisher_path_from_entry(pair["style"], backend)
    args.dataset_pair_name = pair["name"]
    args.concept_class_name = pair["concept"]["class_name"]
    args.placeholder_token_concept = pair["concept"]["placeholder_token"]
    args.placeholder_token_style = pair["style"]["placeholder_token"]
    return args


def apply_pair_config(config, args):
    if getattr(args, "concept_class_name", None) is None:
        return config
    config["class_name"] = args.concept_class_name
    config["placeholder_token_concept"] = args.placeholder_token_concept
    config["placeholder_token_style"] = args.placeholder_token_style
    return config


def selected_pairs(args):
    if args.samples == "all_dataset_pairs":
        return DIFFUSION_MERGE_PAIRS
    if args.samples is not None:
        if ":" not in args.samples:
            raise ValueError(
                "samples must be 'all_dataset_pairs', '<concept_name>:<style_name>', "
                "'concept:<concept_name>', or 'style:<style_name>'."
            )
        left, right = args.samples.split(":", 1)
        if left == "concept":
            pairs = [pair for pair in DIFFUSION_MERGE_PAIRS if pair["concept"]["name"] == right]
            if not pairs:
                raise KeyError(f"Unknown concept: {right}")
            return pairs
        if left == "style":
            pairs = [pair for pair in DIFFUSION_MERGE_PAIRS if pair["style"]["name"] == right]
            if not pairs:
                raise KeyError(f"Unknown style: {right}")
            return pairs
        concept_name, style_name = left, right
        return [get_pair(concept_name, style_name)]
    if args.concept_name is not None or args.style_name is not None:
        if args.concept_name is None:
            pairs = [
                pair for pair in DIFFUSION_MERGE_PAIRS if pair["style"]["name"] == args.style_name
            ]
            if not pairs:
                raise KeyError(f"Unknown style: {args.style_name}")
            return pairs
        if args.style_name is None:
            pairs = [
                pair for pair in DIFFUSION_MERGE_PAIRS if pair["concept"]["name"] == args.concept_name
            ]
            if not pairs:
                raise KeyError(f"Unknown concept: {args.concept_name}")
            return pairs
        return [get_pair(args.concept_name, args.style_name)]
    return [None]


def print_selected_pairs(pairs):
    print("=" * 40, flush=True)
    print("Pairs to compute:", flush=True)
    if pairs == [None]:
        print("  custom adapter paths", flush=True)
    else:
        for pair in pairs:
            print(f"  {pair['name']}", flush=True)
    print("=" * 40, flush=True)


def run_pipe(args):
    args = prepare_merge_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("pipe_gradients.py requires a CUDA GPU.")
    if args.moft_layers_concept_path is None or args.moft_layers_style_path is None:
        raise ValueError("Adapter paths are required unless a dataset pair is selected.")

    with open(args.config_path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    config = apply_pair_config(config, args)
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir

    pipe = inferencers["gradients"](
        config,
        args,
        merge_test_set,
        merge_base_set,
        device="cuda",
    )
    pipe.setup()
    pipe.generate()


if __name__ == "__main__":
    args = parse_args()
    pairs = selected_pairs(args)
    print_selected_pairs(pairs)
    for pair in pairs:
        run_args = argparse.Namespace(**vars(args))
        if pair is not None:
            print(f"Running pair: {pair['name']}", flush=True)
            apply_pair(run_args, pair)
        prepare_merge_args(run_args)
        existing_path = existing_output_path(run_args, gradients_inference_folder_name)
        if existing_path is not None:
            print(f"Skipping existing Gradients output: {existing_path}", flush=True)
            continue
        run_pipe(run_args)
