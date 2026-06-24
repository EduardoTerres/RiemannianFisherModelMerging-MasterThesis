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
    if args.all_dataset:
        return DIFFUSION_MERGE_PAIRS
    if args.concept_name is not None or args.style_name is not None:
        if args.concept_name is None or args.style_name is None:
            raise ValueError("Both concept_name and style_name are required.")
        return [get_pair(args.concept_name, args.style_name)]
    return [None]


def apply_pair(args, pair):
    args.moft_layers_concept_path = pair["concept"]["adapter_path"]
    args.moft_layers_style_path = pair["style"]["adapter_path"]
    args.dataset_pair_name = pair["name"]
    return args


def run_pipe(args):
    if not torch.cuda.is_available():
        raise RuntimeError("pipe_orthofuse.py requires a CUDA GPU.")
    if args.moft_layers_concept_path is None or args.moft_layers_style_path is None:
        raise ValueError("Adapter paths are required unless a dataset pair is selected.")

    with open(args.config_path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
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
    for pair in tqdm(pairs, desc="Running OrthoFuse pipeline", unit="pair"):
        run_args = argparse.Namespace(**vars(args))
        if pair is not None:
            tqdm.write(f"Running pair: {pair['name']}")
            apply_pair(run_args, pair)
        run_pipe(run_args)
