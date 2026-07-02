import argparse
import sys
import warnings
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "OrthoFuse"))

from moft.inferencer_sdxl import inferencers
from nb_utils.eval_sets import merge_base_set, merge_test_set
from src.diffusion.dataset_1 import DIFFUSION_MERGE_PAIRS, get_pair


warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--checkpoint_idx", type=str, default=None)
    parser.add_argument("--moft_layers_concept_path", type=str, default=None)
    parser.add_argument("--moft_layers_style_path", type=str, default=None)
    parser.add_argument("--all_dataset", action="store_true")
    parser.add_argument("--debug", action="store_true")
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
    parser.add_argument("--t", type=float, default=0.6)
    parser.add_argument("--parameter", type=float, default=None)
    parser.add_argument("--postprocessing_method", type=str, default="curve_over_id")
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


def selected_pairs(args):
    if args.debug:
        return [get_pair(DIFFUSION_MERGE_PAIRS[0]["concept"]["name"], "01_08")]
    if args.all_dataset or args.samples == "all_dataset_pairs":
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
        return [get_pair(left, right)]
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


def apply_pair(args, pair):
    args.moft_layers_concept_path = pair["concept"]["adapter_path"]
    args.moft_layers_style_path = pair["style"]["adapter_path"]
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


def inference_folder_name(args):
    pair_name = getattr(args, "dataset_pair_name", None)
    pair_suffix = f"_{pair_name}" if pair_name else ""
    return (
        f"ns{args.num_inference_steps}_gs{args.guidance_scale}"
        f"_orthofuse_t{args.t}_method_{args.postprocessing_method}{pair_suffix}"
    )


def output_root(args):
    if args.output_dir is not None:
        return Path(args.output_dir)

    with open(args.config_path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if config.get("output_dir") is None:
        raise ValueError("output_dir is required either as an argument or in the config.")
    return Path(config["output_dir"])


def existing_output_path(args):
    root = output_root(args)
    if args.checkpoint_idx is not None:
        root = root / f"checkpoint-{args.checkpoint_idx}"

    folder = inference_folder_name(args)
    candidates = [
        root / folder,
        root / "samples" / folder,
    ]
    if args.version is not None:
        candidates.append(root / "samples" / folder / f"version_{args.version}")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def run_pipe(args):
    if not torch.cuda.is_available():
        raise RuntimeError("pipe_orthofuse.py requires a CUDA GPU.")
    if args.moft_layers_concept_path is None or args.moft_layers_style_path is None:
        raise ValueError("Adapter paths are required unless a dataset pair is selected.")

    with open(args.config_path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    config = apply_pair_config(config, args)
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir

    pipe = inferencers["moft_merge"](
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
    for pair in tqdm(pairs, desc="Running OrthoFuse pipeline", unit="pair"):
        run_args = argparse.Namespace(**vars(args))
        if pair is not None:
            tqdm.write(f"Running pair: {pair['name']}")
            apply_pair(run_args, pair)
        existing_path = existing_output_path(run_args)
        if existing_path is not None:
            tqdm.write(f"Skipping existing OrthoFuse output: {existing_path}")
            continue
        run_pipe(run_args)
