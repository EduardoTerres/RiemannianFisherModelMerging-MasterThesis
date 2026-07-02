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

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    specs = (
        (axes[0], "clip_concept", "clip_style", "clip_pareto", "CLIP"),
        (axes[1], "dino_concept", "dino_style", "dino_pareto", "DINO"),
    )
    series_values = list(dict.fromkeys(row[series_key] for row in rows))
    cmap = plt.get_cmap("tab10" if len(series_values) <= 10 else "viridis")
    denom = max(len(series_values) - 1, 1)

    for ax, concept_key, style_key, pareto_key, axis_title in specs:
        for idx, series_value in enumerate(series_values):
            series_rows = [row for row in rows if row[series_key] == series_value]
            if not series_rows:
                continue
            color = cmap(idx % 10 if len(series_values) <= 10 else idx / denom)
            ax.plot(
                [row[concept_key] for row in series_rows],
                [row[style_key] for row in series_rows],
                marker="o",
                linewidth=1.5,
                markersize=3,
                color=color,
                label=str(series_value),
            )
            if label_key is not None:
                for row in series_rows:
                    ax.annotate(
                        label_value(row[label_key]),
                        (row[concept_key], row[style_key]),
                        fontsize=7,
                        xytext=(3, 3),
                        textcoords="offset points",
                    )
        frontier = sort_frontier([row for row in rows if row.get(pareto_key)], concept_key)
        if frontier:
            ax.plot(
                [row[concept_key] for row in frontier],
                [row[style_key] for row in frontier],
                color="black",
                linewidth=2.0,
                linestyle="--",
                label="pareto frontier",
            )
        ax.set_title(axis_title)
        ax.set_xlabel("concept preservation")
        ax.set_ylabel("style preservation")
        ax.grid(True, alpha=0.25)
    axes[1].legend(loc="best", fontsize=7)
    fig.suptitle(title)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
