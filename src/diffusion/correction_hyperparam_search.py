import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "OrthoFuse"))
sys.path.insert(0, str(REPO_ROOT))

from src.diffusion.dataset_1 import DIFFUSION_MERGE_PAIRS, get_pair
from src.diffusion.pipe_gradients import (
    apply_pair,
    run_pipe,
)

PROMPTS = {
    "bicycle": "a {0} riding a bicycle in {1} style",
    "normal": "a {0} in {1} style",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default=str(REPO_ROOT / "src/diffusion/config/config.yaml"))
    parser.add_argument("--output_dir", type=Path, default=REPO_ROOT / "outputs/diffusion")
    parser.add_argument(
        "--samples",
        nargs="+",
        default=["cat:01_01", "dog2:dolina", "dog6:gondoliers", "cat2:pots", "dog:03_04"],
    )
    parser.add_argument("--num_points", type=int, default=9)
    parser.add_argument("--num_images_per_medium_prompt", type=int, default=1)
    parser.add_argument("--batch_size_medium", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--fisher_min", type=float, default=1e-14)
    parser.add_argument("--fisher_rescale", type=float, default=1e10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--version_start", type=int, default=0)
    parser.add_argument("--image_index", type=int, default=0)
    parser.add_argument("--reference_root", type=Path, default=REPO_ROOT / "outputs/diffusion/d1_images")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--clip_model", type=str, default="ViT-B/32")
    parser.add_argument("--clip_pretrained", type=str, default="openai")
    parser.add_argument("--dino_model", type=str, default="dinov2_vits14")
    parser.add_argument("--dino_repo", type=str, default="facebookresearch/dinov2")
    parser.add_argument("--dino_source", type=str, default="github")
    parser.add_argument("--force-generate", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    return parser.parse_args()


def mus():
    return [-0.5, 0, 1, 1.5, 2, 3, 4, 6, 8]


def interpolation_grid(mu, num_points):
    if num_points < 2:
        raise ValueError("--num_points must be at least 2.")
    return [(t, (t, 1.0 - t)) for t in np.linspace(0.0, 1.0, num_points)]


def root(args):
    return args.output_dir / "correction_hyperparam_search"


def folder(args, pair_name, mu):
    return (
        f"ns{args.num_inference_steps}_gs{args.guidance_scale}"
        f"_gradients_fisher_mu{mu:g}_{pair_name}"
    )


def prompt(pair, template):
    return template.format(
        pair["concept"]["placeholder_token"],
        pair["style"]["placeholder_token"],
    )


def text_prompt(pair, template):
    return template.format(
        pair["concept"]["class_name"],
        pair["style"]["name"],
    )


def image_path(args, pair, mu, point_idx, template):
    return (
        root(args)
        / "samples"
        / folder(args, pair["name"], mu)
        / f"version_{args.version_start + point_idx}"
        / prompt(pair, template)
        / f"{args.image_index}.png"
    )


def expected_image_paths(args, pair, mu, alphas):
    return [
        image_path(args, pair, mu, point_idx, template)
        for point_idx in range(len(alphas))
        for template in PROMPTS.values()
    ]


def missing_image_paths(args, pair, mu, alphas):
    return [path for path in expected_image_paths(args, pair, mu, alphas) if not path.exists()]


def run_generation(args, pair, mu, alphas):
    for point_idx, (t, alpha) in enumerate(alphas):
        run_args = argparse.Namespace(
            config_path=args.config_path,
            output_dir=str(root(args)),
            checkpoint_idx=None,
            moft_layers_concept_path=None,
            moft_layers_style_path=None,
            concept_fisher_path=None,
            style_fisher_path=None,
            fisher_min=args.fisher_min,
            fisher_rescale=args.fisher_rescale,
            alphas=list(alpha),
            merge_mode="fisher",
            fisher_correction_mu=mu,
            parameter=None,
            postprocessing_method="no_modification",
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
            replace_inference_output=args.force_generate,
            version=args.version_start + point_idx,
            seed=args.seed,
        )
        apply_pair(run_args, pair)
        print(
            f"[correction] pair={pair['name']} mu={mu:g} "
            f"t={t:.3f} alpha_1={alpha[0]:.3f} alpha_2={alpha[1]:.3f}",
            flush=True,
        )
        run_pipe(run_args)
        for template in PROMPTS.values():
            expected = image_path(args, pair, mu, point_idx, template)
            if not expected.exists():
                raise FileNotFoundError(f"Expected generated image missing: {expected}")


def make_montage(args, pair, mu, alphas, prompt_name, template):
    paths = [image_path(args, pair, mu, idx, template) for idx in range(len(alphas))]
    tiles = [Image.open(path).convert("RGB") for path in paths]
    width, height = tiles[0].size
    label_h = 42
    font = ImageFont.load_default()
    canvas = Image.new("RGB", (width * len(tiles), height + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (tile, (t, alpha)) in enumerate(zip(tiles, alphas)):
        if tile.size != (width, height):
            tile = tile.resize((width, height), Image.Resampling.LANCZOS)
        x = idx * width
        canvas.paste(tile, (x, 0))
        label = f"t={t:.2f}"
        bbox = draw.textbbox((0, 0), label, font=font)
        draw.text((x + (width - bbox[2]) / 2, height + 14), label, fill="black", font=font)
    save_path = root(args) / "collage" / f"{pair['name']}_{prompt_name}_fisher_mu{mu:g}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(save_path)
    print(f"[correction] montage saved to {save_path}", flush=True)


def selected_pairs(args):
    pairs = []
    for sample in args.samples:
        if sample == "all_dataset_pairs":
            pairs.extend(DIFFUSION_MERGE_PAIRS)
            continue
        if ":" not in sample:
            raise ValueError(
                "Each --samples value must be 'all_dataset_pairs', '<concept>:<style>', "
                "'concept:<concept>', or 'style:<style>'."
            )
        left, right = sample.split(":", 1)
        if left == "concept":
            selected = [pair for pair in DIFFUSION_MERGE_PAIRS if pair["concept"]["name"] == right]
            if not selected:
                raise KeyError(f"Unknown concept: {right}")
            pairs.extend(selected)
            continue
        if left == "style":
            selected = [pair for pair in DIFFUSION_MERGE_PAIRS if pair["style"]["name"] == right]
            if not selected:
                raise KeyError(f"Unknown style: {right}")
            pairs.extend(selected)
            continue
        pairs.append(get_pair(left, right))
    return pairs


def main():
    args = parse_args()
    pairs = selected_pairs(args)

    if args.plot_only:
        from src.diffusion.results.correction_hyperparam_search_results import run

        run(args)
        return

    for pair in pairs:
        for mu in mus():
            alphas = interpolation_grid(mu, args.num_points)
            missing = missing_image_paths(args, pair, mu, alphas)
            if args.force_generate or missing:
                if missing and not args.force_generate:
                    print(
                        f"[correction] missing {len(missing)} images for pair={pair['name']} "
                        f"mu={mu:g}; generating missing outputs",
                        flush=True,
                    )
                run_generation(args, pair, mu, alphas)
            else:
                print(f"[correction] using existing images for pair={pair['name']} mu={mu:g}", flush=True)
            for prompt_name, template in PROMPTS.items():
                make_montage(args, pair, mu, alphas, prompt_name, template)


if __name__ == "__main__":
    main()
