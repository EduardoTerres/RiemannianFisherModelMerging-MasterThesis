import argparse
import os

from tqdm import tqdm

from pipe_gradients import apply_pair, parse_args, run_pipe, selected_pairs


def prepare_fim_args(args):
    if args.merge_mode is None:
        args.merge_mode = "diagonal_fisher"
    if args.fisher_min is None:
        args.fisher_min = 1e-8
    if args.fisher_rescale is None:
        args.fisher_rescale = 1e10
    if args.concept_fisher_path is None:
        args.concept_fisher_path = "/scratch-shared/eterres/fishers/cat_oft_lie_fim.safetensors"
    if args.style_fisher_path is None:
        args.style_fisher_path = args.concept_fisher_path
    for path in (args.concept_fisher_path, args.style_fisher_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing Fisher file: {path}")
    return args


if __name__ == "__main__":
    args = parse_args()
    pairs = selected_pairs(args)
    for pair in tqdm(pairs, desc="Running Fisher merge pipeline", unit="pair"):
        run_args = argparse.Namespace(**vars(args))
        if pair is not None:
            tqdm.write(f"Running pair: {pair['name']}")
            apply_pair(run_args, pair)
        run_pipe(prepare_fim_args(run_args))
