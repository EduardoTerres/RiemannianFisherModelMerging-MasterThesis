import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.diffusion.pipe_gradients import (
    apply_pair as apply_gradient_pair,
    run_pipe as run_gradient_pipe,
    selected_pairs,
)
from src.diffusion.pipe_orthofuse import (
    apply_pair as apply_orthofuse_pair,
    run_pipe as run_orthofuse_pipe,
)


ORTHOFUSE_METHODS = {
    "orthofuse_geodesic": "no_modification",
    "orthofuse_geodesic_curve_over_id": "curve_over_id",
    "orthofuse_geodesic_rotation": "rotation",
}
METHODS = (
    "standard_geodesic",
    "fisher_geodesic",
    "standard_rescaled",
    "diagonal_fisher",
    "diagonal_fisher_rescaled",
    *ORTHOFUSE_METHODS.keys(),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--samples", type=str, default=None)
    parser.add_argument("--concept_name", type=str, default=None)
    parser.add_argument("--style_name", type=str, default=None)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--num_points", type=int, default=7)
    parser.add_argument("--num_images_per_medium_prompt", type=int, default=1)
    parser.add_argument("--batch_size_medium", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--geodesic_backend", choices=["cayley"], default="cayley")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--version_start", type=int, default=0)
    parser.add_argument("--image_index", type=int, default=0)
    parser.add_argument("--montage_prompt", type=str, default="a {0} in {1} style")
    parser.add_argument("--fisher_min", type=float, default=1e-14)
    parser.add_argument("--fisher_rescale", type=float, default=1e10)
    parser.add_argument("--orthofuse_postprocessing_method", type=str, default="no_modification")
    parser.add_argument("--replace_inference_output", action="store_true")
    return parser.parse_args()


def t_grid(num_points):
    if num_points < 2:
        raise ValueError("--num_points must be at least 2.")
    return [i / (num_points - 1) for i in range(num_points)]


def alphas_from_t(t):
    return (t, 1.0 - t)


def interpolation_root(args):
    return Path(args.output_dir) / "geodesic_interpolation"


def prompt_path(args, pair, folder, version):
    prompt = args.montage_prompt.format(
        pair["concept"]["placeholder_token"],
        pair["style"]["placeholder_token"],
    )
    return (
        interpolation_root(args)
        / "samples"
        / folder
        / f"version_{version}"
        / prompt
        / f"{args.image_index}.png"
    )


def gradient_folder(args, pair_name, merge_mode, use_fishers, backend):
    suffix = ""
    if merge_mode == "geodesic":
        suffix = f"_{backend}"
        if use_fishers:
            suffix += "_fisher"
    return (
        f"ns{args.num_inference_steps}_gs{args.guidance_scale}"
        f"_gradients_{merge_mode}{suffix}_{pair_name}"
    )


def orthofuse_folder(args, pair_name, t, postprocessing_method):
    return (
        f"ns{args.num_inference_steps}_gs{args.guidance_scale}"
        f"_orthofuse_t{t}_method_{postprocessing_method}_{pair_name}"
    )


def make_montage(image_paths, ts, save_path):
    tiles = [Image.open(path).convert("RGB") for path in image_paths]
    w, h = tiles[0].size
    label_h = 42
    font = ImageFont.load_default()
    canvas = Image.new("RGB", (w * len(tiles), h + label_h), "white")
    draw = ImageDraw.Draw(canvas)

    for idx, (tile, t) in enumerate(zip(tiles, ts)):
        if tile.size != (w, h):
            tile = tile.resize((w, h), Image.Resampling.LANCZOS)
        x = idx * w
        canvas.paste(tile, (x, 0))
        label = f"t={t:.2f}"
        bbox = draw.textbbox((0, 0), label, font=font)
        draw.text((x + (w - bbox[2]) / 2, h + 14), label, fill="black", font=font)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(save_path)


def gradient_args(args, alphas, version, merge_mode, use_fishers, backend):
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
        merge_mode=merge_mode,
        geodesic_backend=backend,
        geodesic_use_fishers=use_fishers,
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


def orthofuse_args(args, beta, version, postprocessing_method):
    return argparse.Namespace(
        config_path=args.config_path,
        output_dir=str(interpolation_root(args)),
        checkpoint_idx=None,
        moft_layers_concept_path=None,
        moft_layers_style_path=None,
        all_dataset=False,
        debug=False,
        concept_name=None,
        style_name=None,
        dataset_pair_name=None,
        t=beta,
        parameter=None,
        postprocessing_method=postprocessing_method,
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


def run_gradient_method(args, pair, ts, method, merge_mode, use_fishers, backend):
    paths = []
    for idx, t in enumerate(ts):
        version = args.version_start + idx
        alpha = alphas_from_t(t)
        run_args = gradient_args(args, alpha, version, merge_mode, use_fishers, backend)
        apply_gradient_pair(run_args, pair)
        print(
            f"[interpolate] {method} {pair['name']} t={t:.3f}",
            flush=True,
        )
        run_gradient_pipe(run_args)
        path = prompt_path(
            args,
            pair,
            gradient_folder(args, pair["name"], merge_mode, use_fishers, backend),
            version,
        )
        if not path.exists():
            raise FileNotFoundError(f"Expected generated image missing: {path}")
        paths.append(path)
    return paths


def run_orthofuse_method(args, pair, ts, method, postprocessing_method):
    paths = []
    for idx, t in enumerate(ts):
        version = args.version_start + idx
        run_args = orthofuse_args(args, t, version, postprocessing_method)
        apply_orthofuse_pair(run_args, pair)
        print(
            f"[interpolate] {method} {pair['name']} t={t:.3f} "
            f"post={postprocessing_method}",
            flush=True,
        )
        run_orthofuse_pipe(run_args)
        path = prompt_path(
            args,
            pair,
            orthofuse_folder(args, pair["name"], t, postprocessing_method),
            version,
        )
        if not path.exists():
            raise FileNotFoundError(f"Expected generated image missing: {path}")
        paths.append(path)
    return paths


def main():
    args = parse_args()
    if args.samples is None and (args.concept_name is None or args.style_name is None):
        raise ValueError("Pass --samples, or both --concept_name and --style_name.")

    ts = t_grid(args.num_points)

    for pair in selected_pairs(args):
        method_paths = {}
        if "standard_geodesic" in args.methods:
            method_paths["standard_geodesic"] = run_gradient_method(
                args,
                pair,
                ts,
                method="standard_geodesic",
                merge_mode="geodesic",
                use_fishers=False,
                backend=args.geodesic_backend,
            )
        if "fisher_geodesic" in args.methods:
            method_paths["fisher_geodesic"] = run_gradient_method(
                args,
                pair,
                ts,
                method="fisher_geodesic",
                merge_mode="geodesic",
                use_fishers=True,
                backend=args.geodesic_backend,
            )
        if "standard_rescaled" in args.methods:
            method_paths["standard_rescaled"] = run_gradient_method(
                args,
                pair,
                ts,
                method="standard_rescaled",
                merge_mode="standard_rescaled",
                use_fishers=False,
                backend=args.geodesic_backend,
            )
        for method in ("diagonal_fisher", "diagonal_fisher_rescaled"):
            if method in args.methods:
                method_paths[method] = run_gradient_method(
                    args,
                    pair,
                    ts,
                    method=method,
                    merge_mode=method,
                    use_fishers=True,
                    backend=args.geodesic_backend,
                )
        for method, postprocessing_method in ORTHOFUSE_METHODS.items():
            if method in args.methods:
                method_paths[method] = run_orthofuse_method(
                    args,
                    pair,
                    ts,
                    method=method,
                    postprocessing_method=postprocessing_method,
                )

        for method, paths in method_paths.items():
            save_path = (
                interpolation_root(args)
                / f"{pair['name']}_{method}_ns{args.num_inference_steps}_gs{args.guidance_scale}.png"
            )
            make_montage(paths, ts, save_path)
            print(f"[interpolate] montage saved to {save_path}", flush=True)


if __name__ == "__main__":
    main()
