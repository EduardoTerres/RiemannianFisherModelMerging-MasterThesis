import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.diffusion.pipe_gradients import apply_pair, run_pipe, selected_pairs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--samples", type=str, default=None)
    parser.add_argument("--concept_name", type=str, default=None)
    parser.add_argument("--style_name", type=str, default=None)
    parser.add_argument("--num_points", type=int, default=7)
    parser.add_argument("--num_images_per_medium_prompt", type=int, default=1)
    parser.add_argument("--batch_size_medium", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--version_start", type=int, default=0)
    parser.add_argument("--image_index", type=int, default=0)
    parser.add_argument("--montage_prompt", type=str, default="a {0} in {1} style")
    parser.add_argument("--fisher_min", type=float, default=1e-14)
    parser.add_argument("--fisher_rescale", type=float, default=1e10)
    parser.add_argument("--replace_inference_output", action="store_true")
    return parser.parse_args()


def alpha_grid(num_points):
    if num_points < 2:
        raise ValueError("--num_points must be at least 2.")
    return [(1.0 - i / (num_points - 1), i / (num_points - 1)) for i in range(num_points)]


def inference_folder(args, pair_name):
    return (
        f"ns{args.num_inference_steps}_gs{args.guidance_scale}"
        f"_gradients_diagonal_fisher_rescaled_{pair_name}"
    )


def interpolation_root(args):
    return Path(args.output_dir) / "interpolation"


def image_path(args, pair, version):
    prompt = args.montage_prompt.format(
        pair["concept"]["placeholder_token"],
        pair["style"]["placeholder_token"],
    )
    return (
        interpolation_root(args)
        / "samples"
        / inference_folder(args, pair["name"])
        / f"version_{version}"
        / prompt
        / f"{args.image_index}.png"
    )


def make_montage(image_paths, alphas, save_path):
    tiles = [Image.open(path).convert("RGB") for path in image_paths]
    w, h = tiles[0].size
    label_h = 42
    font = ImageFont.load_default()
    canvas = Image.new("RGB", (w * len(tiles), h + label_h), "white")
    draw = ImageDraw.Draw(canvas)

    for idx, (tile, (concept_alpha, style_alpha)) in enumerate(zip(tiles, alphas)):
        if tile.size != (w, h):
            tile = tile.resize((w, h), Image.Resampling.LANCZOS)
        x = idx * w
        canvas.paste(tile, (x, 0))
        label = f"alpha=({concept_alpha:.2f}, {style_alpha:.2f})"
        bbox = draw.textbbox((0, 0), label, font=font)
        draw.text((x + (w - bbox[2]) / 2, h + 14), label, fill="black", font=font)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(save_path)


def pipe_args(args, pair, alphas, version):
    return argparse.Namespace(
        config_path=args.config_path,
        output_dir=str(interpolation_root(args)),
        checkpoint_idx=None,
        moft_layers_concept_path=None,
        moft_layers_style_path=None,
        concept_fisher_path=None,
        style_fisher_path=None,
        fisher_min=args.fisher_min,
        fisher_rescale=args.fisher_rescale,
        alphas=list(alphas),
        merge_mode="diagonal_fisher_rescaled",
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


def main():
    args = parse_args()
    if args.samples is None and (args.concept_name is None or args.style_name is None):
        raise ValueError("Pass --samples, or both --concept_name and --style_name.")
    alphas = alpha_grid(args.num_points)

    for pair in selected_pairs(args):
        image_paths = []
        for idx, alpha in enumerate(alphas):
            version = args.version_start + idx
            run_args = pipe_args(args, pair, alpha, version)
            apply_pair(run_args, pair)
            print(f"[interpolate] {pair['name']} alpha={alpha} version={version}", flush=True)
            run_pipe(run_args)
            path = image_path(args, pair, version)
            if not path.exists():
                raise FileNotFoundError(f"Expected generated image missing: {path}")
            image_paths.append(path)

        save_path = (
            interpolation_root(args)
            / (
                f"{pair['name']}_diagonal_fisher_rescaled_"
                f"ns{args.num_inference_steps}_gs{args.guidance_scale}.png"
            )
        )
        make_montage(image_paths, alphas, save_path)
        print(f"[interpolate] montage saved to {save_path}", flush=True)


if __name__ == "__main__":
    main()
