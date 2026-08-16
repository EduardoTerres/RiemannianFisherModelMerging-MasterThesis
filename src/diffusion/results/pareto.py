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
        if image_path.stat().st_size == 0:
            if skip_missing:
                print(f"[pareto] skip empty image: {image_path}", file=sys.stderr, flush=True)
                continue
            raise FileNotFoundError(f"Empty generated image: {image_path}")
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


def plot_style_concept_curves(
    rows,
    save_path,
    title,
    series_key,
    label_key=None,
    series_labels=None,
    legend_fontsize=13,
    legend_title_fontsize=14,
    series_legend_anchor=(0.935, 0.73),
    series_legend_ncol=1,
    label_legend_anchor=(0.935, 0.29),
    label_legend_ncol=1,
    label_legend_markersize=None,
    label_alpha=None,
    save_pdf=False,
    save_individual=False,
    individual_zoom_t=None,
    individual_zoom_group_count=2,
    individual_zoom_window=0.05,
):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import ConnectionPatch, Rectangle

    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "axes.titlesize": 20,
            "axes.labelsize": 18,
            "xtick.labelsize": 30,
            "ytick.labelsize": 30,
            "legend.fontsize": legend_fontsize,
            "legend.title_fontsize": legend_title_fontsize,
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
    if series_labels is not None and not isinstance(series_labels, dict):
        raise TypeError(
            "series_labels must be a dict mapping series value -> label, "
            "since series_values order (from the data) is not guaranteed to "
            "match the order labels were supplied in."
        )
    series_label_map = series_labels or {}
    label_values = list(dict.fromkeys(row[label_key] for row in rows)) if label_key is not None else []
    if label_values:
        numeric_label_values = [float(value) for value in label_values]
        min_label = min(numeric_label_values)
        max_label = max(numeric_label_values)
        label_range = max(max_label - min_label, 1e-12)
        min_radius = 4.0
        max_radius = 15.0
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

    def rows_at_zoom_t(concept_key, style_key):
        if individual_zoom_t is None or label_key is None:
            return []
        target = float(individual_zoom_t)
        return [
            row
            for row in rows
            if concept_key in row
            and style_key in row
            and abs(float(row[label_key]) - target) <= 1e-9
        ]

    def zoom_groups(concept_key, style_key):
        target_rows = sorted(rows_at_zoom_t(concept_key, style_key), key=lambda row: row[style_key])
        if len(target_rows) < 2:
            return []
        group_count = min(individual_zoom_group_count, len(target_rows))
        if group_count <= 1:
            return [target_rows]

        xs = np.array([row[style_key] for row in target_rows], dtype=float)
        split_after = sorted(np.argsort(np.diff(xs))[-(group_count - 1):] + 1)
        groups = []
        start = 0
        for stop in [*split_after, len(target_rows)]:
            groups.append(target_rows[start:stop])
            start = stop
        return [group for group in groups if group]

    def zoom_bounds(group, concept_key, style_key):
        group_series = {row[series_key] for row in group}
        target = float(individual_zoom_t)
        nearby_rows = [
            row
            for row in rows
            if row[series_key] in group_series
            and concept_key in row
            and style_key in row
            and abs(float(row[label_key]) - target) <= individual_zoom_window + 1e-9
        ]
        if not nearby_rows:
            nearby_rows = group

        x_values = np.array([row[style_key] for row in nearby_rows], dtype=float)
        y_values = np.array([row[concept_key] for row in nearby_rows], dtype=float)
        all_x = np.array([row[style_key] for row in rows if style_key in row], dtype=float)
        all_y = np.array([row[concept_key] for row in rows if concept_key in row], dtype=float)
        x_span = max(float(x_values.max() - x_values.min()), float(np.ptp(all_x)) * 0.012, 1e-4)
        y_span = max(float(y_values.max() - y_values.min()), float(np.ptp(all_y)) * 0.012, 1e-4)
        x_pad = x_span * 0.65
        y_pad = y_span * 1.6
        return (
            float(x_values.min() - x_pad),
            float(x_values.max() + x_pad),
            float(y_values.min() - y_pad),
            float(y_values.max() + y_pad),
        )

    def draw_panel(
        ax,
        concept_key,
        style_key,
        axis_title,
        collect_handles=False,
        fixed_label_radius=None,
        line_width=2.0,
    ):
        handles = []
        for idx, series_value in enumerate(series_values):
            series_rows = [row for row in rows if row[series_key] == series_value]
            if not series_rows:
                continue
            color = cmap(idx % 10 if len(series_values) <= 10 else idx / denom)
            label = (
                rf"$\mu={label_value(series_value)}$"
                if series_key == "mu"
                else series_label_map.get(series_value, str(series_value))
            )
            if collect_handles:
                handles.append(Line2D([0], [0], color=color, linewidth=5.0, label=label))
            ax.plot(
                [row[style_key] for row in series_rows],
                [row[concept_key] for row in series_rows],
                linewidth=line_width,
                color=color,
                label="_nolegend_",
            )
            if label_key is not None:
                for row in series_rows:
                    radius = fixed_label_radius or label_radii[row[label_key]]
                    alpha = label_alpha or label_alphas[row[label_key]]
                    ax.scatter(
                        row[style_key],
                        row[concept_key],
                        marker="o",
                        s=radius**2,
                        color=color,
                        alpha=alpha,
                        zorder=3,
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
        return handles

    def add_zoom_boxes(fig, ax, concept_key, style_key):
        groups = zoom_groups(concept_key, style_key)
        if not groups:
            return

        inset_left = 0.665
        inset_width = 0.205
        inset_height = 0.245
        inset_bottoms = [0.635, 0.355]
        for group, inset_bottom in zip(groups[:2], inset_bottoms, strict=False):
            x_min, x_max, y_min, y_max = zoom_bounds(group, concept_key, style_key)
            rect = Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                fill=False,
                edgecolor="black",
                linewidth=1.2,
                linestyle=(0, (2, 2)),
                zorder=4,
            )
            ax.add_patch(rect)

            inset_ax = fig.add_axes([inset_left, inset_bottom, inset_width, inset_height])
            draw_panel(
                inset_ax,
                concept_key,
                style_key,
                "",
                fixed_label_radius=13.5,
                line_width=6.0,
            )
            inset_ax.set_xlim(x_min, x_max)
            inset_ax.set_ylim(y_min, y_max)
            inset_ax.set_title("")
            inset_ax.tick_params(axis="both", labelsize=13, pad=1)
            inset_ax.grid(True, alpha=0.25)
            for spine in inset_ax.spines.values():
                spine.set_linewidth(1.0)
                spine.set_edgecolor("black")

            for source_xy, inset_xy in (
                ((x_min, y_min), (0, 0)),
                ((x_max, y_max), (1, 1)),
            ):
                fig.add_artist(
                    ConnectionPatch(
                        xyA=source_xy,
                        coordsA=ax.transData,
                        xyB=inset_xy,
                        coordsB=inset_ax.transAxes,
                        color="black",
                        linewidth=0.9,
                        linestyle=(0, (2, 2)),
                        zorder=4,
                    )
                )

            label_offsets = {0: (16, 12), 1: (-16, 12), 2: (-16, -12), 3: (16, 12), 4: (0, 18)}
            for row in group:
                idx = series_values.index(row[series_key])
                color = cmap(idx % 10 if len(series_values) <= 10 else idx / denom)
                inset_ax.annotate(
                    label_value(row[label_key]),
                    xy=(row[style_key], row[concept_key]),
                    xytext=label_offsets.get(idx, (0, 14)),
                    textcoords="offset points",
                    ha="center",
                    va="center",
                    fontsize=24,
                    fontweight="bold",
                    color=color,
                    clip_on=False,
                    zorder=5,
                )

    color_handles = []
    for ax, concept_key, style_key, axis_title in specs:
        color_handles.extend(draw_panel(ax, concept_key, style_key, axis_title, collect_handles=ax is axes[0]))
    fig.supxlabel("Style similarity", y=0.08, fontsize=32)
    fig.supylabel("Concept similarity", x=0.035, y=0.5, fontsize=32)
    if color_handles:
        series_title = None if series_key == "method" else (r"$\mu$" if series_key == "mu" else rf"${series_key}$")
        fig.legend(
            handles=color_handles,
            loc="center",
            bbox_to_anchor=series_legend_anchor,
            frameon=False,
            title=series_title,
            ncol=series_legend_ncol,
        )
    if label_key is not None:
        marker_handles = [
            Line2D(
                [0],
                [0],
                color="black",
                marker="o",
                linestyle="None",
                markersize=label_legend_markersize or label_radii[value],
                alpha=label_alpha or label_alphas[value],
                label=f"t={label_value(value)}" if label_key == "t" else label_value(value),
            )
            for value in label_values
        ]
        fig.legend(
            handles=marker_handles,
            loc="center",
            bbox_to_anchor=label_legend_anchor,
            frameon=False,
            ncol=label_legend_ncol,
            title=rf"${label_key}$",
        )
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if save_pdf:
        fig.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    if save_individual:
        for _unused_ax, concept_key, style_key, axis_title in specs:
            metric_path = save_path.with_name(f"{save_path.stem}_{axis_title.lower()}{save_path.suffix}")
            figsize = (13.0, 7.2) if individual_zoom_t is not None else (8.5, 7.2)
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            right = 0.6 if individual_zoom_t is not None else 0.78
            left = 0.0 if individual_zoom_t is not None else 0.16
            fig.subplots_adjust(left=left, right=right, bottom=0.31, top=0.9)
            color_handles = draw_panel(
                ax,
                concept_key,
                style_key,
                axis_title,
                collect_handles=True,
            )
            main_x_center = (left + right) / 2.0
            main_y_center = (0.31 + 0.9) / 2.0
            fig.supxlabel("Style similarity", x=main_x_center, y=0.155, fontsize=32)
            fig.supylabel("Concept similarity", x=-0.12, y=main_y_center, fontsize=32)
            if color_handles:
                fig.legend(
                    handles=color_handles,
                    loc="center",
                    bbox_to_anchor=((0.48, -0.015) if individual_zoom_t is not None else series_legend_anchor),
                    frameon=False,
                    title=None if series_key == "method" else series_title,
                    ncol=series_legend_ncol,
                )
            if label_key is not None:
                fig.legend(
                    handles=marker_handles,
                    loc="center",
                    bbox_to_anchor=(0.98, 0.53),
                    frameon=False,
                    ncol=1,
                    title=rf"${label_key}$",
                )
            add_zoom_boxes(fig, ax, concept_key, style_key)
            fig.savefig(metric_path, dpi=200, bbox_inches="tight")
            if save_pdf:
                fig.savefig(metric_path.with_suffix(".pdf"), bbox_inches="tight")
            plt.close(fig)
