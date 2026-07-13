import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.diffusion.pipe_gradients import apply_pair, run_pipe, selected_pairs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--samples", default=None)
    parser.add_argument("--concept_name", default=None)
    parser.add_argument("--style_name", default=None)
    parser.add_argument("--factors", type=float, nargs="+", default=None)
    parser.add_argument("--num_points", type=int, default=7)
    parser.add_argument("--num_images_per_medium_prompt", type=int, default=1)
    parser.add_argument("--batch_size_medium", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--geodesic_backend", choices=["cayley"], default="cayley")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--version_start", type=int, default=0)
    parser.add_argument("--image_index", type=int, default=0)
    parser.add_argument("--montage_prompt", default="a {0} in {1} style")
    parser.add_argument("--replace_inference_output", action="store_true")
    return parser.parse_args()


def interpolation_factors(args):
    if args.factors is not None:
        return args.factors
    if args.num_points < 2:
        raise ValueError("--num_points must be at least 2.")
    return [i / (args.num_points - 1) for i in range(args.num_points)]


def root(args):
    return Path(args.output_dir) / "geodesic_interpolation"


def inference_folder(args, pair_name):
    return (
        f"ns{args.num_inference_steps}_gs{args.guidance_scale}"
        f"_gradients_geodesic_{args.geodesic_backend}_{pair_name}"
    )


def _image_path_for_folder(args, pair, version, folder):
    prompt = args.montage_prompt.format(
        pair["concept"]["placeholder_token"],
        pair["style"]["placeholder_token"],
    )
    return (
        root(args)
        / "samples"
        / folder
        / f"version_{version}"
        / prompt
        / f"{args.image_index}.png"
    )


def image_path(args, pair, version):
    return _image_path_for_folder(args, pair, version, inference_folder(args, pair["name"]))


def legacy_image_path(args, pair, version):
    folder = (
        f"ns{args.num_inference_steps}_gs{args.guidance_scale}"
        f"_gradients_geodesic_{pair['name']}"
    )
    return _image_path_for_folder(args, pair, version, folder)


def pipe_args(args, factor, version):
    return argparse.Namespace(
        config_path=args.config_path,
        output_dir=str(root(args)),
        checkpoint_idx=None,
        moft_layers_concept_path=None,
        moft_layers_style_path=None,
        concept_fisher_path=None,
        style_fisher_path=None,
        fisher_min=None,
        fisher_rescale=None,
        alphas=[1.0 - factor, factor],
        merge_mode="geodesic",
        geodesic_backend=args.geodesic_backend,
        samples=None,
        concept_name=None,
        style_name=None,
        dataset_pair_name=None,
        rescale=False,
        num_images_per_medium_prompt=args.num_images_per_medium_prompt,
        num_images_per_base_prompt=0,
        batch_size_medium=args.batch_size_medium,
        batch_size_base=1,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        replace_inference_output=args.replace_inference_output,
        version=version,
        seed=args.seed,
    )


def make_plot(paths, factors, save_path):
    tiles = [Image.open(path).convert("RGB") for path in paths]
    w, h = tiles[0].size
    label_h = 42
    font = ImageFont.load_default()
    canvas = Image.new("RGB", (w * len(tiles), h + label_h), "white")
    draw = ImageDraw.Draw(canvas)

    for idx, (tile, factor) in enumerate(zip(tiles, factors)):
        if tile.size != (w, h):
            tile = tile.resize((w, h), Image.Resampling.LANCZOS)
        x = idx * w
        canvas.paste(tile, (x, 0))
        label = f"t={factor:.2f}"
        bbox = draw.textbbox((0, 0), label, font=font)
        draw.text((x + (w - bbox[2]) / 2, h + 14), label, fill="black", font=font)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(save_path)


def main():
    args = parse_args()
    if args.samples is None and (args.concept_name is None or args.style_name is None):
        raise ValueError("Pass --samples, or both --concept_name and --style_name.")

    factors = interpolation_factors(args)
    for pair in selected_pairs(args):
        paths = []
        for idx, factor in enumerate(factors):
            version = args.version_start + idx
            run_args = pipe_args(args, factor, version)
            apply_pair(run_args, pair)
            print(f"[geodesic] {pair['name']} t={factor:.3f} version={version}", flush=True)
            run_pipe(run_args)
            path = image_path(args, pair, version)
            if not path.exists():
                legacy_path = legacy_image_path(args, pair, version)
                if legacy_path.exists():
                    path = legacy_path
                else:
                    raise FileNotFoundError(f"Expected generated image missing: {path}")
            paths.append(path)

        save_path = root(args) / f"{pair['name']}_geodesic_{args.geodesic_backend}.png"
        make_plot(paths, factors, save_path)
        print(f"[geodesic] plot saved to {save_path}", flush=True)


if __name__ == "__main__":
    main()
