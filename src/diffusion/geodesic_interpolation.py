import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "OrthoFuse"))

from nb_utils.eval_sets import merge_test_set
from src.diffusion.pipe_gradients import (
    apply_pair as apply_gradient_pair,
    run_pipe as run_gradient_pipe,
    selected_pairs,
)
from src.diffusion.pipe_orthofuse import (
    apply_pair as apply_orthofuse_pair,
    run_pipe as run_orthofuse_pipe,
)
from src.diffusion.pipeline_outputs import (
    gradients_inference_folder_name,
    orthofuse_inference_folder_name,
)


ORTHOFUSE_METHODS = {
    "orthofuse_geodesic": "no_modification",
    "orthofuse_geodesic_curve_over_id": "curve_over_id",
    "orthofuse_geodesic_rotation": "rotation",
    "orthofuse": None,
}
METHODS = (
    "standard_geodesic",
    "fisher_geodesic",
    "standard_rescaled",
    "fisher",
    "fisher_rescaled",
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
    parser.add_argument("--prompt_templates", nargs="+", default=None)
    parser.add_argument("--method_output_root", type=str, default=None)
    parser.add_argument("--num_points", type=int, default=7)
    parser.add_argument("--t_values", nargs="+", type=float, default=None)
    parser.add_argument("--num_images_per_medium_prompt", type=int, default=5)
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
    parser.add_argument("--orthofuse_postprocessing_method", type=str, default="no_modification")
    parser.add_argument(
        "--results_folder",
        type=str,
        default="samples_10_prompts/interpolation_geodesic",
    )
    parser.add_argument("--replace_inference_output", action="store_true")
    return parser.parse_args()


def t_grid(num_points):
    if num_points < 2:
        raise ValueError("--num_points must be at least 2.")
    return [i / (num_points - 1) for i in range(num_points)]


def interpolation_ts(args):
    ts = args.t_values if args.t_values is not None else t_grid(args.num_points)
    for t in ts:
        if t < 0.0 or t > 1.0:
            raise ValueError(f"Interpolation factor t must be in [0, 1]; got {t:.6g}")
    return ts


def alphas_from_t(t):
    return (1.0 - t, t)


def interpolation_root(args):
    return Path(args.output_dir) / args.results_folder


def method_output_name(args, method):
    if method == "fisher_geodesic":
        name = f"fisher_geodesic_{args.fisher_backend}"
        if args.correction_mu is not None:
            name += f"_corr_{args.correction_mu:g}_fim_{args.fim_normalization}"
        return name
    if method == "fisher":
        prefix = "diagonal_fisher" if args.fisher_backend == "diagonal" else f"fisher_{args.fisher_backend}"
        if args.fisher_correction_mu is not None:
            prefix += f"_mu_{args.fisher_correction_mu:g}"
        if args.fim_normalization != "none":
            prefix += f"_fim_{args.fim_normalization}"
        return prefix
    if method.startswith("orthofuse"):
        return "orthofuse"
    return method


def method_output_dir(args, method):
    if args.method_output_root is None:
        return interpolation_root(args)
    return Path(args.method_output_root) / method_output_name(args, method)


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


def version_root(args, method, folder, version):
    return method_output_dir(args, method) / "samples" / folder / f"version_{version}"


def generated_prompt_paths(args, pair):
    for template in args.prompt_templates or merge_test_set:
        prompt = template.format(
            pair["concept"]["placeholder_token"],
            pair["style"]["placeholder_token"],
        )
        for image_idx in range(args.num_images_per_medium_prompt):
            yield prompt, image_idx


def expected_image_paths(args, method, pair, folder, version):
    root = version_root(args, method, folder, version)
    return [
        root / prompt / f"{image_idx}.png"
        for prompt, image_idx in generated_prompt_paths(args, pair)
    ]


def missing_image_paths(args, method, pair, folder, version):
    return [
        path
        for path in expected_image_paths(args, method, pair, folder, version)
        if not path.exists()
    ]


def representative_image_path(args, method, pair, folder, version):
    path = version_root(args, method, folder, version) / (
        args.montage_prompt.format(
            pair["concept"]["placeholder_token"],
            pair["style"]["placeholder_token"],
        )
    ) / f"{args.image_index}.png"
    if path.exists():
        return path
    return expected_image_paths(args, method, pair, folder, version)[0]


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
        fisher_backend=args.fisher_backend,
        fisher_correction_mu=args.fisher_correction_mu if use_fishers and merge_mode == "fisher" else None,
        correction_mu=args.correction_mu if use_fishers and merge_mode == "geodesic" else None,
        fim_normalization=args.fim_normalization,
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
        prompt_templates=args.prompt_templates,
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
        prompt_templates=args.prompt_templates,
    )


def run_gradient_method(args, pair, ts, method, merge_mode, use_fishers, backend):
    paths = []
    for idx, t in enumerate(ts):
        version = args.version_start + idx
        alpha = alphas_from_t(t)
        run_args = gradient_args(args, alpha, version, merge_mode, use_fishers, backend)
        run_args.output_dir = str(method_output_dir(args, method))
        apply_gradient_pair(run_args, pair)
        folder = gradients_inference_folder_name(run_args)
        missing = missing_image_paths(args, method, pair, folder, version)
        print(
            f"[interpolate] {method} {pair['name']} t={t:.3f}",
            flush=True,
        )
        if args.replace_inference_output or missing:
            run_gradient_pipe(run_args)
        else:
            print(
                f"[interpolate] using existing output: {version_root(args, method, folder, version)}",
                flush=True,
            )
        missing = missing_image_paths(args, method, pair, folder, version)
        if missing:
            raise FileNotFoundError(f"Expected generated images missing, first: {missing[0]}")
        path = representative_image_path(args, method, pair, folder, version)
        paths.append(path)
    return paths


def run_orthofuse_method(args, pair, ts, method, postprocessing_method):
    paths = []
    for idx, t in enumerate(ts):
        version = args.version_start + idx
        run_args = orthofuse_args(args, t, version, postprocessing_method)
        run_args.output_dir = str(method_output_dir(args, method))
        apply_orthofuse_pair(run_args, pair)
        folder = orthofuse_inference_folder_name(run_args)
        missing = missing_image_paths(args, method, pair, folder, version)
        print(
            f"[interpolate] {method} {pair['name']} t={t:.3f} "
            f"post={postprocessing_method}",
            flush=True,
        )
        if args.replace_inference_output or missing:
            run_orthofuse_pipe(run_args)
        else:
            print(
                f"[interpolate] using existing output: {version_root(args, method, folder, version)}",
                flush=True,
            )
        missing = missing_image_paths(args, method, pair, folder, version)
        if missing:
            raise FileNotFoundError(f"Expected generated images missing, first: {missing[0]}")
        path = representative_image_path(args, method, pair, folder, version)
        paths.append(path)
    return paths


def main():
    args = parse_args()
    if args.samples is None and args.concept_name is None and args.style_name is None:
        raise ValueError("Pass --samples, --concept_name, or --style_name.")

    ts = interpolation_ts(args)

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
        for method in ("fisher", "fisher_rescaled"):
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
                if method == "orthofuse":
                    postprocessing_method = args.orthofuse_postprocessing_method
                method_paths[method] = run_orthofuse_method(
                    args,
                    pair,
                    ts,
                    method=method,
                    postprocessing_method=postprocessing_method
                    or args.orthofuse_postprocessing_method,
                )

        for method, paths in method_paths.items():
            save_path = (
                method_output_dir(args, method)
                / f"{pair['name']}_{method}_ns{args.num_inference_steps}_gs{args.guidance_scale}.png"
            )
            make_montage(paths, ts, save_path)
            print(f"[interpolate] montage saved to {save_path}", flush=True)


if __name__ == "__main__":
    main()
