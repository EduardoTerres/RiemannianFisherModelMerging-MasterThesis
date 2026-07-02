import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import PIL.Image
import torch
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "OrthoFuse"))
sys.path.insert(0, str(REPO_ROOT))

from nb_utils.clip_eval import DINOEvaluator
from src.diffusion.dataset_1 import DIFFUSION_MERGE_PAIRS, get_pair
from src.diffusion.pipe_gradients import (
    apply_pair,
    run_pipe,
)

# Any of these work
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
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


def output_prefix(pairs):
    return pairs[0]["name"] if len(pairs) == 1 else "combined_" + "__".join(pair["name"] for pair in pairs)


def folder(args, pair_name, mu):
    return (
        f"ns{args.num_inference_steps}_gs{args.guidance_scale}"
        f"_gradients_diagonal_fisher_mu{mu:g}_{pair_name}"
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
            merge_mode="diagonal_fisher",
            diagonal_fisher_correction_mu=mu,
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


def load_pil(path):
    return PIL.Image.open(path).convert("RGB")


def pil_images_to_clip_tensor(pil_images):
    images = [np.asarray(image) for image in pil_images]
    images = torch.from_numpy(np.transpose(np.stack(images), axes=(0, 3, 1, 2)))
    return torch.clamp(images / 127.5 - 1.0, min=-1.0, max=1.0)


@torch.no_grad()
def mean_similarity(left_features, right_features):
    return (left_features @ right_features.T).mean().item()


@torch.no_grad()
def image_features(evaluator, paths):
    pil_images = [load_pil(path) for path in paths]
    clip_features = torch.cat(
        [evaluator.get_image_features(pil_images_to_clip_tensor([image])) for image in pil_images],
        dim=0,
    )
    dino_features = evaluator.get_dino_image_features(pil_images)
    return clip_features, dino_features


@torch.no_grad()
def text_features(evaluator, text):
    return evaluator.get_text_features(text)


def concept_reference_paths(pair, reference_root):
    concept_dir = reference_root / pair["concept"]["name"]
    if not concept_dir.exists():
        concept_dir = Path(pair["concept"]["dataset_path"])
    paths = sorted(path for path in concept_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise FileNotFoundError(f"No concept reference images found in {concept_dir}")
    return paths


def style_reference_path(pair, reference_root):
    style_path = Path(pair["style"]["dataset_path"])
    local_path = reference_root / style_path.name
    if local_path.exists():
        return local_path
    if style_path.exists():
        return style_path
    raise FileNotFoundError(f"Missing style reference image for {pair['name']}: {local_path}")


def score(args, pairs):
    evaluator = DINOEvaluator(
        device=args.device,
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        dino_model=args.dino_model,
        dino_repo=args.dino_repo,
        dino_source=args.dino_source,
    )
    rows = []
    for pair in pairs:
        concept_clip, concept_dino = image_features(
            evaluator,
            concept_reference_paths(pair, args.reference_root),
        )
        style_clip, style_dino = image_features(
            evaluator,
            [style_reference_path(pair, args.reference_root)],
        )
        for mu in mus():
            alphas = interpolation_grid(mu, args.num_points)
            for point_idx, (t, alpha) in enumerate(alphas):
                for prompt_name, template in PROMPTS.items():
                    path = image_path(args, pair, mu, point_idx, template)
                    gen_clip, gen_dino = image_features(evaluator, [path])
                    prompt_clip = text_features(evaluator, text_prompt(pair, template))
                    rows.append(
                        {
                            "pair": pair["name"],
                            "prompt": prompt_name,
                            "mu": mu,
                            "step": point_idx,
                            "t": t,
                            "alpha_1": alpha[0],
                            "alpha_2": alpha[1],
                            "image_path": str(path),
                            "clip_concept": mean_similarity(concept_clip, gen_clip),
                            "clip_style": mean_similarity(style_clip, gen_clip),
                            "clip_text": mean_similarity(prompt_clip, gen_clip),
                            "dino_concept": mean_similarity(concept_dino, gen_dino),
                            "dino_style": mean_similarity(style_dino, gen_dino),
                        }
                    )
    return rows


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["prompt"], row["mu"], row["step"], row["t"], row["alpha_1"], row["alpha_2"])].append(row)
    result = []
    for (prompt_name, mu, step, t, alpha_1, alpha_2), values in grouped.items():
        row = {
            "prompt": prompt_name,
            "mu": mu,
            "step": step,
            "t": t,
            "alpha_1": alpha_1,
            "alpha_2": alpha_2,
            "n": len(values),
        }
        for key in ("clip_concept", "clip_style", "clip_text", "dino_concept", "dino_style"):
            row[key] = float(np.mean([value[key] for value in values]))
        result.append(row)
    return sorted(result, key=lambda row: (row["prompt"], row["mu"], row["step"]))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "prompt",
        "mu",
        "step",
        "t",
        "alpha_1",
        "alpha_2",
        "n",
        "clip_concept",
        "clip_style",
        "clip_text",
        "dino_concept",
        "dino_style",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows, save_path, title):
    import matplotlib.pyplot as plt

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    specs = (
        (axes[0], "clip_concept", "clip_style", "CLIP"),
        (axes[1], "dino_concept", "dino_style", "DINO"),
    )
    cmap = plt.get_cmap("viridis")
    mu_values = mus()
    denom = max(len(mu_values) - 1, 1)
    for ax, concept_key, style_key, title in specs:
        for idx, mu in enumerate(mu_values):
            curve = [row for row in rows if row["mu"] == mu]
            color = cmap(idx / denom)
            ax.plot(
                [row[concept_key] for row in curve],
                [row[style_key] for row in curve],
                marker="o",
                linewidth=1.5,
                markersize=3,
                color=color,
                label=f"mu={mu:g}",
            )
        ax.set_title(title)
        ax.set_xlabel("concept preservation")
        ax.set_ylabel("style preservation")
        ax.grid(True, alpha=0.25)
    axes[1].legend(loc="best", fontsize=7)
    fig.suptitle(title)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_text_similarity(rows, save_path, title):
    import matplotlib.pyplot as plt

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    mu_values = mus()
    denom = max(len(mu_values) - 1, 1)
    for idx, mu in enumerate(mu_values):
        curve = [row for row in rows if row["mu"] == mu]
        ax.plot(
            [row["t"] for row in curve],
            [row["clip_text"] for row in curve],
            marker="o",
            linewidth=1.5,
            markersize=3,
            color=cmap(idx / denom),
            label=f"mu={mu:g}",
        )
    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel("CLIP similarity to text prompt")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=7)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


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
    save_path = root(args) / "collage" / f"{pair['name']}_{prompt_name}_diagonal_fisher_mu{mu:g}.png"
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
            raise ValueError("Each --samples value must be 'all_dataset_pairs' or '<concept>:<style>'.")
        concept_name, style_name = sample.split(":", 1)
        pairs.append(get_pair(concept_name, style_name))
    return pairs


def main():
    args = parse_args()
    pairs = selected_pairs(args)

    if not args.plot_only:
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

    rows = aggregate(score(args, pairs))
    prefix = output_prefix(pairs)
    csv_path = root(args) / f"{prefix}_pareto_curves.csv"
    write_csv(csv_path, rows)
    for prompt_name in PROMPTS:
        prompt_rows = [row for row in rows if row["prompt"] == prompt_name]
        png_path = root(args) / f"{prefix}_pareto_curves_{prompt_name}.png"
        plot(prompt_rows, png_path, f"Diagonal Fisher correction search ({prompt_name})")
        print(f"[correction] wrote {png_path}", flush=True)
        text_png_path = root(args) / f"{prefix}_text_similarity_{prompt_name}.png"
        plot_text_similarity(prompt_rows, text_png_path, f"CLIP text similarity ({prompt_name})")
        print(f"[correction] wrote {text_png_path}", flush=True)
        if prompt_name == "normal":
            default_png_path = root(args) / f"{prefix}_pareto_curves.png"
            plot(prompt_rows, default_png_path, "Diagonal Fisher correction search")
            print(f"[correction] wrote {default_png_path}", flush=True)
    print(f"[correction] wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
