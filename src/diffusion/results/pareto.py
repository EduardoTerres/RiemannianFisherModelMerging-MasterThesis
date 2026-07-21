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

from nb_utils.clip_eval import DINOEvaluator
from src.diffusion.dataset_1 import DIFFUSION_MERGE_PAIRS, get_pair


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
METRIC_KEYS = ("clip_concept", "clip_style", "dino_concept", "dino_style")


def add_evaluator_args(parser):
    parser.add_argument("--reference_root", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--clip_model", type=str, default=os.environ.get("ORTHOFUSE_CLIP_MODEL", "ViT-B/32"))
    parser.add_argument("--clip_pretrained", type=str, default=os.environ.get("ORTHOFUSE_CLIP_PRETRAINED", "openai"))
    parser.add_argument("--dino_model", type=str, default=os.environ.get("ORTHOFUSE_DINO_MODEL", "dinov2_vits14"))
    parser.add_argument("--dino_repo", type=str, default=os.environ.get("ORTHOFUSE_DINO_REPO", "facebookresearch/dinov2"))
    parser.add_argument("--dino_source", type=str, default=os.environ.get("ORTHOFUSE_DINO_SOURCE", "github"))
    return parser


def make_evaluator(args):
    return DINOEvaluator(
        device=args.device,
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        dino_model=args.dino_model,
        dino_repo=args.dino_repo,
        dino_source=args.dino_source,
    )


def selected_pairs(samples):
    pairs = []
    for sample in samples:
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


def output_prefix(pairs):
    return pairs[0]["name"] if len(pairs) == 1 else "combined_" + "__".join(pair["name"] for pair in pairs)


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


def style_reference_path(pair, reference_root):
    style_path = Path(pair["style"]["dataset_path"])
    local_path = reference_root / style_path.name
    if local_path.exists():
        return local_path
    if style_path.exists():
        return style_path
    raise FileNotFoundError(f"Missing style reference image for {pair['name']}: {local_path}")


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


class SimilarityScorer:
    def __init__(self, evaluator, reference_root):
        self.evaluator = evaluator
        self.reference_root = Path(reference_root)
        self._reference_cache = {}
        self._image_cache = {}
        self._text_cache = {}

    def reference_features(self, pair):
        pair_name = pair["name"]
        if pair_name not in self._reference_cache:
            concept_clip, concept_dino = image_features(
                self.evaluator,
                concept_reference_paths(pair, self.reference_root),
            )
            style_clip, style_dino = image_features(
                self.evaluator,
                [style_reference_path(pair, self.reference_root)],
            )
            self._reference_cache[pair_name] = {
                "concept_clip": concept_clip,
                "concept_dino": concept_dino,
                "style_clip": style_clip,
                "style_dino": style_dino,
            }
        return self._reference_cache[pair_name]

    def generated_features(self, image_path):
        image_path = Path(image_path)
        if image_path not in self._image_cache:
            self._image_cache[image_path] = image_features(self.evaluator, [image_path])
        return self._image_cache[image_path]

    def prompt_features(self, text):
        if text not in self._text_cache:
            self._text_cache[text] = text_features(self.evaluator, text)
        return self._text_cache[text]

    def score(self, pair, image_path, text_prompt=None):
        refs = self.reference_features(pair)
        gen_clip, gen_dino = self.generated_features(image_path)
        row = {
            "clip_concept": mean_similarity(refs["concept_clip"], gen_clip),
            "clip_style": mean_similarity(refs["style_clip"], gen_clip),
            "dino_concept": mean_similarity(refs["concept_dino"], gen_dino),
            "dino_style": mean_similarity(refs["style_dino"], gen_dino),
        }
        if text_prompt is not None:
            row["clip_text"] = mean_similarity(self.prompt_features(text_prompt), gen_clip)
        return row


def score_records(records, scorer, skip_missing=True):
    rows = []
    for record in records:
        image_path = Path(record["image_path"])
        if not image_path.exists():
            if skip_missing:
                print(f"[pareto] skip missing image: {image_path}", file=sys.stderr, flush=True)
                continue
            raise FileNotFoundError(f"Missing generated image: {image_path}")
        pair = record["pair"]
        row = {key: value for key, value in record.items() if key not in {"pair", "text_prompt"}}
        row["image_path"] = str(image_path)
        row.update(scorer.score(pair, image_path, record.get("text_prompt")))
        rows.append(row)
    return rows


def aggregate_rows(rows, group_keys, metric_keys):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)

    aggregates = []
    for key, values in grouped.items():
        row = {group_key: value for group_key, value in zip(group_keys, key)}
        row["n"] = len(values)
        for metric in metric_keys:
            present = [value[metric] for value in values if metric in value]
            if present:
                row[metric] = float(np.mean(present))
        aggregates.append(row)
    return aggregates


def pareto_frontier_indices(rows, concept_key, style_key):
    points = np.array([[row[concept_key], row[style_key]] for row in rows], dtype=float)
    frontier = []
    for idx, point in enumerate(points):
        dominated = np.any(
            np.all(points >= point, axis=1)
            & np.any(points > point, axis=1)
        )
        if not dominated:
            frontier.append(idx)
    return frontier


def add_pareto_flags(rows, group_keys=()):
    for row in rows:
        row["clip_pareto"] = False
        row["dino_pareto"] = False

    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)

    for group_rows in grouped.values():
        if all("clip_concept" in row and "clip_style" in row for row in group_rows):
            for idx in pareto_frontier_indices(group_rows, "clip_concept", "clip_style"):
                group_rows[idx]["clip_pareto"] = True
        if all("dino_concept" in row and "dino_style" in row for row in group_rows):
            for idx in pareto_frontier_indices(group_rows, "dino_concept", "dino_style"):
                group_rows[idx]["dino_pareto"] = True
    return rows


def sort_frontier(rows, concept_key):
    return sorted(rows, key=lambda row: row[concept_key])


def label_value(value):
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_style_concept_curves(rows, save_path, title, series_key, label_key=None):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "axes.titlesize": 20,
            "axes.labelsize": 18,
            "xtick.labelsize": 30,
            "ytick.labelsize": 30,
            "legend.fontsize": 13,
            "legend.title_fontsize": 14,
        }
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(19, 6.5))
    fig.subplots_adjust(left=0.1, right=0.855, bottom=0.28, top=0.9, wspace=0.12)
    specs = (
        (axes[0], "clip_concept", "clip_style", "CLIP"),
        (axes[1], "dino_concept", "dino_style", "DINO"),
    )
    series_values = list(dict.fromkeys(row[series_key] for row in rows))
    label_values = list(dict.fromkeys(row[label_key] for row in rows)) if label_key is not None else []
    if label_values:
        numeric_label_values = [float(value) for value in label_values]
        min_label = min(numeric_label_values)
        max_label = max(numeric_label_values)
        label_range = max(max_label - min_label, 1e-12)
        min_radius = 2.0
        max_radius = 11.0
        label_radii = {
            value: max_radius - (float(value) - min_label) / label_range * (max_radius - min_radius)
            for value in label_values
        }
        min_alpha = 0.35
        max_alpha = 1.0
        label_alphas = {
            value: max_alpha - (float(value) - min_label) / label_range * (max_alpha - min_alpha)
            for value in label_values
        }
    else:
        label_radii = {}
        label_alphas = {}
    cmap = plt.get_cmap("tab10" if len(series_values) <= 10 else "viridis")
    denom = max(len(series_values) - 1, 1)

    color_handles = []
    for ax, concept_key, style_key, axis_title in specs:
        for idx, series_value in enumerate(series_values):
            series_rows = [row for row in rows if row[series_key] == series_value]
            if not series_rows:
                continue
            color = cmap(idx % 10 if len(series_values) <= 10 else idx / denom)
            label = (
                rf"$\mu={label_value(series_value)}$"
                if series_key == "mu"
                else str(series_value)
            )
            if ax is axes[0]:
                color_handles.append(Line2D([0], [0], color=color, linewidth=2.0, label=label))
            ax.plot(
                [row[style_key] for row in series_rows],
                [row[concept_key] for row in series_rows],
                linewidth=2.0,
                color=color,
                label="_nolegend_",
            )
            if label_key is not None:
                for row in series_rows:
                    radius = label_radii[row[label_key]]
                    alpha = label_alphas[row[label_key]]
                    ax.scatter(
                        row[style_key],
                        row[concept_key],
                        marker="o",
                        s=radius**2,
                        color=color,
                        alpha=alpha,
                        zorder=3,
                    )
                    if float(row[label_key]) <= 0.5:
                        ax.annotate(
                            label_value(row[label_key]),
                            (row[style_key], row[concept_key]),
                            fontsize=13,
                            xytext=(4, 4),
                            textcoords="offset points",
                        )
            else:
                ax.scatter(
                    [row[style_key] for row in series_rows],
                    [row[concept_key] for row in series_rows],
                    marker="x",
                    s=42,
                    color=color,
                    linewidths=1.4,
                    zorder=3,
                )
        ax.set_title(axis_title)
        ax.grid(True, alpha=0.25)
    fig.supxlabel("Style similarity", y=0.08, fontsize=32)
    fig.supylabel("Concept similarity", x=0.035, fontsize=32)
    if color_handles:
        series_title = rf"${series_key}$" if series_key != "mu" else r"$\mu$"
        fig.legend(
            handles=color_handles,
            loc="center",
            bbox_to_anchor=(0.935, 0.73),
            frameon=False,
            title=series_title,
        )
    if label_key is not None:
        marker_handles = [
            Line2D(
                [0],
                [0],
                color="black",
                marker="o",
                linestyle="None",
                markersize=label_radii[value],
                alpha=label_alphas[value],
                label=f"t={label_value(value)}" if label_key == "t" else label_value(value),
            )
            for value in label_values
        ]
        fig.legend(
            handles=marker_handles,
            loc="center",
            bbox_to_anchor=(0.935, 0.29),
            frameon=False,
            title=rf"${label_key}$",
        )
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
