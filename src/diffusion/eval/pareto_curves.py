import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import PIL.Image
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "OrthoFuse"))
sys.path.insert(0, str(REPO_ROOT))

from src.diffusion.dataset_1 import DIFFUSION_MERGE_PAIRS, get_pair


METHODS = (
    "standard_geodesic",
    "fisher_geodesic",
    "standard_rescaled",
    "fisher",
    "fisher_rescaled",
    "orthofuse_geodesic",
    "orthofuse_geodesic_curve_over_id",
    "orthofuse_geodesic_rotation",
)

ORTHOFUSE_POSTPROCESSING = {
    "orthofuse_geodesic": "no_modification",
    "orthofuse_geodesic_curve_over_id": "curve_over_id",
    "orthofuse_geodesic_rotation": "rotation",
}

GRADIENT_FOLDERS = {
    "standard_geodesic": "gradients_geodesic_{backend}",
    "fisher_geodesic": "gradients_geodesic_{backend}_fisher",
    "standard_rescaled": "gradients_standard_rescaled",
    "fisher": "gradients_fisher",
    "fisher_rescaled": "gradients_fisher_rescaled",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/diffusion"))
    parser.add_argument("--samples", nargs="+", default=["cat:01_01"])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--num_points", type=int, default=10)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--geodesic_backend", choices=["cayley"], default="cayley")
    parser.add_argument("--version_start", type=int, default=0)
    parser.add_argument("--image_index", type=int, default=0)
    parser.add_argument("--montage_prompt", type=str, default="a {0} in {1} style")
    parser.add_argument("--reference_root", type=Path, default=None)
    parser.add_argument("--save_path", type=Path, default=None)
    parser.add_argument("--csv_path", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--clip_model", type=str, default=os.environ.get("ORTHOFUSE_CLIP_MODEL", "ViT-B/32"))
    parser.add_argument("--clip_pretrained", type=str, default=os.environ.get("ORTHOFUSE_CLIP_PRETRAINED", "openai"))
    parser.add_argument("--dino_model", type=str, default=os.environ.get("ORTHOFUSE_DINO_MODEL", "dinov2_vits14"))
    parser.add_argument("--dino_repo", type=str, default=os.environ.get("ORTHOFUSE_DINO_REPO", "facebookresearch/dinov2"))
    parser.add_argument("--dino_source", type=str, default=os.environ.get("ORTHOFUSE_DINO_SOURCE", "github"))
    return parser.parse_args()


def alpha_grid(num_points):
    if num_points < 2:
        raise ValueError("--num_points must be at least 2.")
    return [idx / (num_points - 1) for idx in range(num_points)]


def selected_pairs(samples):
    pairs = []
    for sample in samples:
        if sample == "all_dataset_pairs":
            pairs.extend(DIFFUSION_MERGE_PAIRS)
            continue
        if ":" not in sample:
            raise ValueError("Each --samples value must be 'all_dataset_pairs' or '<concept>:<style>'.")
        concept_name, style_name = sample.split(":", 1)
        pairs.append(get_pair(concept_name, style_name))
    return pairs


def interpolation_root(args):
    return args.output_dir / "geodesic_interpolation"


def style_reference_path(pair, reference_root):
    style_path = Path(pair["style"]["dataset_path"])
    local_path = reference_root / style_path.name
    if local_path.exists():
        return local_path
    if style_path.exists():
        return style_path
    raise FileNotFoundError(f"Missing style reference image for {pair['name']}: {local_path}")


def concept_reference_paths(pair, reference_root):
    concept_dir = reference_root / pair["concept"]["name"]
    if not concept_dir.exists():
        concept_dir = Path(pair["concept"]["dataset_path"])
    if not concept_dir.exists():
        raise FileNotFoundError(f"Missing concept reference directory for {pair['name']}: {concept_dir}")
    paths = sorted(path for path in concept_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise FileNotFoundError(f"No concept reference images found in {concept_dir}")
    return paths


def method_folder(args, pair_name, method, beta):
    prefix = f"ns{args.num_inference_steps}_gs{args.guidance_scale}"
    if method in GRADIENT_FOLDERS:
        body = GRADIENT_FOLDERS[method].format(backend=args.geodesic_backend)
        return f"{prefix}_{body}_{pair_name}"
    postprocessing = ORTHOFUSE_POSTPROCESSING[method]
    return f"{prefix}_orthofuse_t{beta}_method_{postprocessing}_{pair_name}"


def generated_image_path(args, pair, method, point_idx, beta):
    folder = method_folder(args, pair["name"], method, beta)
    prompt = args.montage_prompt.format(
        pair["concept"]["placeholder_token"],
        pair["style"]["placeholder_token"],
    )
    version = args.version_start + point_idx
    version_dir = interpolation_root(args) / "samples" / folder / f"version_{version}"
    exact_path = version_dir / prompt / f"{args.image_index}.png"
    if exact_path.exists():
        return exact_path
    candidates = sorted(version_dir.rglob(f"{args.image_index}.png")) if version_dir.exists() else []
    if candidates:
        return candidates[0]
    return exact_path


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


def score_pair(args, evaluator, pair, betas):
    reference_root = args.reference_root or args.output_dir / "d1_images"
    concept_refs = concept_reference_paths(pair, reference_root)
    style_refs = [style_reference_path(pair, reference_root)]
    concept_clip, concept_dino = image_features(evaluator, concept_refs)
    style_clip, style_dino = image_features(evaluator, style_refs)

    rows = []
    for method in args.methods:
        for point_idx, beta in enumerate(betas):
            image_path = generated_image_path(args, pair, method, point_idx, beta)
            if not image_path.exists():
                print(f"[skip] missing image: {image_path}", file=sys.stderr)
                continue
            gen_clip, gen_dino = image_features(evaluator, [image_path])
            rows.append(
                {
                    "pair": pair["name"],
                    "method": method,
                    "step": point_idx,
                    "t": beta,
                    "image_path": str(image_path),
                    "clip_concept": mean_similarity(concept_clip, gen_clip),
                    "clip_style": mean_similarity(style_clip, gen_clip),
                    "dino_concept": mean_similarity(concept_dino, gen_dino),
                    "dino_style": mean_similarity(style_dino, gen_dino),
                }
            )
    return rows


def aggregate_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["step"], row["t"])].append(row)

    aggregates = []
    for (method, step, beta), values in grouped.items():
        agg = {"method": method, "step": step, "t": beta, "n": len(values)}
        for metric in ("clip_concept", "clip_style", "dino_concept", "dino_style"):
            agg[metric] = float(np.mean([value[metric] for value in values]))
        aggregates.append(agg)
    return sorted(aggregates, key=lambda row: (row["method"], row["step"]))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "step",
        "t",
        "n",
        "clip_concept",
        "clip_style",
        "dino_concept",
        "dino_style",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(rows, save_path):
    import matplotlib.pyplot as plt

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    specs = [
        (axes[0], "clip_concept", "clip_style", "CLIP"),
        (axes[1], "dino_concept", "dino_style", "DINO"),
    ]

    methods = list(dict.fromkeys(row["method"] for row in rows))
    cmap = plt.get_cmap("tab10")
    for ax, concept_key, style_key, title in specs:
        for method_idx, method in enumerate(methods):
            method_rows = [row for row in rows if row["method"] == method]
            if not method_rows:
                continue
            color = cmap(method_idx % 10)
            xs = [row[concept_key] for row in method_rows]
            ys = [row[style_key] for row in method_rows]
            ax.plot(xs, ys, marker="o", linewidth=2, markersize=4, label=method, color=color)
            for row, x, y in zip(method_rows, xs, ys):
                ax.annotate(f"{row['t']:.2f}", (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_title(title)
        ax.set_xlabel("concept preservation")
        ax.set_ylabel("style preservation")
        ax.grid(True, alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)
    fig.suptitle("Geodesic interpolation Pareto curves")
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    betas = alpha_grid(args.num_points)
    pairs = selected_pairs(args.samples)
    save_path = args.save_path or interpolation_root(args) / "pareto_curves.png"
    csv_path = args.csv_path or save_path.with_suffix(".csv")

    from nb_utils.clip_eval import DINOEvaluator

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
        print(f"[pareto] scoring {pair['name']}", flush=True)
        rows.extend(score_pair(args, evaluator, pair, betas))

    if not rows:
        raise RuntimeError("No interpolation images were found for the requested pairs and methods.")

    aggregate = aggregate_rows(rows)
    write_csv(csv_path, aggregate)
    plot_curves(aggregate, save_path)
    print(f"[pareto] wrote {save_path}", flush=True)
    print(f"[pareto] wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
