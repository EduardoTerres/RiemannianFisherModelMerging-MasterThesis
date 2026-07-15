import argparse
import sys
import warnings
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "OrthoFuse"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.diffusion import pipe_gradients
from src.diffusion.pipeline_outputs import existing_output_path, gradients_inference_folder_name


warnings.filterwarnings("ignore")


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
    parser.add_argument("--fisher_backend", choices=["diagonal", "kfac"], default="kfac")
    parser.add_argument("--fisher_correction_mu", type=float, default=None)
    parser.add_argument("--correction_mu", type=float, default=None)
    parser.add_argument(
        "--fim_normalization",
        choices=["none", "trace", "frobenius", "kl"],
        default="frobenius",
    )
    parser.add_argument("--t", type=float, default=0.6)
    parser.add_argument("--geodesic_backend", choices=["cayley"], default="cayley")
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


def prepare_fisher_geodesic_args(args):
    if args.t < 0.0 or args.t > 1.0:
        raise ValueError(f"Interpolation factor --t must be in [0, 1]; got {args.t:.6g}")
    args.merge_mode = "geodesic"
    args.geodesic_use_fishers = True
    args.alphas = [1.0 - args.t, args.t]
    args.rescale = False
    return args


if __name__ == "__main__":
    args = prepare_fisher_geodesic_args(parse_args())
    pairs = pipe_gradients.selected_pairs(args)
    pipe_gradients.print_selected_pairs(pairs)
    for pair in tqdm(pairs, desc="Running Fisher geodesic pipeline", unit="pair"):
        run_args = argparse.Namespace(**vars(args))
        if pair is not None:
            tqdm.write(f"Running pair: {pair['name']}")
            pipe_gradients.apply_pair(run_args, pair)
        existing_path = existing_output_path(run_args, gradients_inference_folder_name)
        if existing_path is not None and not run_args.replace_inference_output:
            tqdm.write(f"Skipping existing Fisher geodesic output: {existing_path}")
            continue
        pipe_gradients.run_pipe(run_args)
