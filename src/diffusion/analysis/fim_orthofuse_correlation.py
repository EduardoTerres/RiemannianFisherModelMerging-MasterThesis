import argparse
import csv
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

USER = os.environ.get("USER", "user")
os.environ.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib-{USER}")
os.environ.setdefault("XDG_CACHE_HOME", f"/tmp/cache-{USER}")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import PIL.Image
import torch
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "OrthoFuse"))
sys.path.insert(0, str(REPO_ROOT))

from src.diffusion.dataset_1 import DIFFUSION_MERGE_PAIRS, get_pair


PAIR_RE = re.compile(r"(?P<concept>cat2?|dog2?|dog3|dog6)__(?P<style>[^/]+)$")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


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
    from nb_utils.clip_eval import DINOEvaluator

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
            raise ValueError("Each --samples value must be 'all_dataset_pairs' or '<concept>:<style>'.")
        concept_name, style_name = sample.split(":", 1)
        pairs.append(get_pair(concept_name, style_name))
    return pairs


def concept_reference_paths(pair, reference_root):
    concept_dir = reference_root / pair["concept"]["name"]
    if not concept_dir.exists():
        concept_dir = Path(pair["concept"]["dataset_path"])
    paths = sorted(path for path in concept_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise FileNotFoundError(f"No concept references found for {pair['name']}: {concept_dir}")
    return paths


def style_reference_path(pair, reference_root):
    style_path = Path(pair["style"]["dataset_path"])
    local_path = reference_root / style_path.name
    if local_path.exists():
        return local_path
    if style_path.exists():
        return style_path
    raise FileNotFoundError(f"Missing style reference for {pair['name']}: {local_path}")


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


class SimilarityScorer:
    def __init__(self, evaluator, reference_root):
        self.evaluator = evaluator
        self.reference_root = Path(reference_root)
        self._reference_cache = {}
        self._image_cache = {}

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

    def score(self, pair, image_path):
        refs = self.reference_features(pair)
        gen_clip, gen_dino = self.generated_features(image_path)
        return {
            "clip_concept": mean_similarity(refs["concept_clip"], gen_clip),
            "clip_style": mean_similarity(refs["style_clip"], gen_clip),
            "dino_concept": mean_similarity(refs["concept_dino"], gen_dino),
            "dino_style": mean_similarity(refs["style_dino"], gen_dino),
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Correlate Orthofuse CLIP/DINO similarities with average adapter FIM."
    )
    parser.add_argument("--samples_root", type=Path, default=Path("outputs/diffusion/samples_10_prompts"))
    parser.add_argument("--method_dirs", nargs="+", default=["orthofuse"])
    parser.add_argument("--samples", nargs="+", default=["all_dataset_pairs"])
    parser.add_argument("--version", default=None, help="Use one version_N dir; default scores all versions.")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/diffusion/analysis/fim_orthofuse_correlation"))
    parser.add_argument("--force", action="store_true", help="Recompute scores even if pair CSV exists.")
    parser = add_evaluator_args(parser)
    return parser.parse_args()


def pair_lookup():
    return {pair["name"]: pair for pair in DIFFUSION_MERGE_PAIRS}


def parse_pair_from_run(run_dir):
    match = PAIR_RE.search(run_dir.name)
    if not match:
        return None
    name = f"{match.group('concept')}__{match.group('style')}"
    return pair_lookup().get(name)


def wanted_pair_names(samples):
    return {pair["name"] for pair in selected_pairs(samples)}


def iter_records(args):
    wanted = wanted_pair_names(args.samples)
    for method_dir in args.method_dirs:
        sample_base = args.samples_root / method_dir / "samples"
        if not sample_base.is_dir():
            print(f"[skip] missing sample dir: {sample_base}", file=sys.stderr)
            continue
        for run_dir in sorted(path for path in sample_base.iterdir() if path.is_dir()):
            pair = parse_pair_from_run(run_dir)
            if pair is None or pair["name"] not in wanted:
                continue
            version_dirs = [run_dir / args.version] if args.version else sorted(run_dir.glob("version_*"))
            for version_dir in version_dirs:
                if not version_dir.is_dir():
                    continue
                for image_path in sorted(path for path in version_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTS):
                    yield {
                        "method_dir": method_dir,
                        "run": run_dir.name,
                        "version": version_dir.name,
                        "pair": pair,
                        "prompt": image_path.parent.name,
                        "image_path": image_path,
                    }


def fim_mean(path, device):
    state = load_file(path, device=device)
    total, count = 0.0, 0
    for tensor in state.values():
        values = tensor.float().clamp_min(0.0)
        total += values.sum().item()
        count += values.numel()
    if count == 0:
        raise ValueError(f"Empty FIM file: {path}")
    mean = total / count
    return mean, math.log10(max(mean, 1e-30))


def fim_table(pairs, device):
    entries = {}
    for pair in pairs:
        for side in ("concept", "style"):
            adapter = pair[side]
            key = (side, adapter["name"])
            if key not in entries:
                mean, log10_mean = fim_mean(adapter["fim_path"], device)
                entries[key] = {"fim_mean": mean, "fim_log10_mean": log10_mean}
    return entries


def score_images(records, args):
    evaluator = make_evaluator(args)
    scorer = SimilarityScorer(evaluator, args.reference_root or Path("outputs/diffusion/analysis/d1_images"))
    rows = []
    for idx, record in enumerate(records, 1):
        if idx == 1 or idx % 100 == 0:
            print(f"[score] {idx} images", flush=True)
        score = scorer.score(record["pair"], record["image_path"])
        rows.append(
            {
                "method_dir": record["method_dir"],
                "run": record["run"],
                "version": record["version"],
                "pair": record["pair"]["name"],
                "concept": record["pair"]["concept"]["name"],
                "style": record["pair"]["style"]["name"],
                "prompt": record["prompt"],
                "image_path": str(record["image_path"]),
                **score,
            }
        )
    return rows


def aggregate_pair_scores(score_rows):
    grouped = defaultdict(list)
    for row in score_rows:
        grouped[(row["method_dir"], row["pair"])].append(row)
    out = []
    for (method_dir, pair_name), rows in sorted(grouped.items()):
        row = {
            "method_dir": method_dir,
            "pair": pair_name,
            "concept": rows[0]["concept"],
            "style": rows[0]["style"],
            "n_images": len(rows),
        }
        for metric in ("clip_concept", "clip_style", "dino_concept", "dino_style"):
            row[metric] = float(np.mean([item[metric] for item in rows]))
        row["clip_delta_concept_minus_style"] = row["clip_concept"] - row["clip_style"]
        row["dino_delta_concept_minus_style"] = row["dino_concept"] - row["dino_style"]
        out.append(row)
    return out


def add_fim_columns(pair_rows, pairs, device):
    pair_by_name = {pair["name"]: pair for pair in pairs}
    fims = fim_table(pairs, device)
    for row in pair_rows:
        pair = pair_by_name[row["pair"]]
        concept_fim = fims[("concept", pair["concept"]["name"])]
        style_fim = fims[("style", pair["style"]["name"])]
        row["concept_fim_mean"] = concept_fim["fim_mean"]
        row["style_fim_mean"] = style_fim["fim_mean"]
        row["concept_fim_log10_mean"] = concept_fim["fim_log10_mean"]
        row["style_fim_log10_mean"] = style_fim["fim_log10_mean"]
        row["fim_log10_delta_concept_minus_style"] = (
            row["concept_fim_log10_mean"] - row["style_fim_log10_mean"]
        )
    return pair_rows


def ranks(values):
    order = np.argsort(values)
    ranks_out = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        ranks_out[order[i : j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks_out


def corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan"), int(len(x))
    pearson = float(np.corrcoef(x, y)[0, 1])
    spearman = float(np.corrcoef(ranks(x), ranks(y))[0, 1])
    return pearson, spearman, int(len(x))


def correlation_rows(pair_rows):
    specs = [
        ("concept_fim_log10_mean", "clip_concept", "higher concept FIM vs CLIP concept sim"),
        ("concept_fim_log10_mean", "dino_concept", "higher concept FIM vs DINO concept sim"),
        ("style_fim_log10_mean", "clip_style", "higher style FIM vs CLIP style sim"),
        ("style_fim_log10_mean", "dino_style", "higher style FIM vs DINO style sim"),
        ("fim_log10_delta_concept_minus_style", "clip_delta_concept_minus_style", "FIM delta vs CLIP sim delta"),
        ("fim_log10_delta_concept_minus_style", "dino_delta_concept_minus_style", "FIM delta vs DINO sim delta"),
    ]
    out = []
    for method_dir in sorted({row["method_dir"] for row in pair_rows}):
        rows = [row for row in pair_rows if row["method_dir"] == method_dir]
        for x_key, y_key, label in specs:
            pearson, spearman, n = corr([row[x_key] for row in rows], [row[y_key] for row in rows])
            out.append(
                {
                    "method_dir": method_dir,
                    "x": x_key,
                    "y": y_key,
                    "label": label,
                    "pearson": pearson,
                    "spearman": spearman,
                    "n": n,
                }
            )
    return out


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in list(row.items()):
            if key in {"method_dir", "run", "version", "pair", "concept", "style", "prompt", "image_path", "x", "y", "label"}:
                continue
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                pass
    return rows


def short_label(key):
    return {
        "concept_fim_log10_mean": "log10 concept FIM",
        "style_fim_log10_mean": "log10 style FIM",
        "fim_log10_delta_concept_minus_style": "log10 FIM concept - style",
        "clip_concept": "CLIP concept",
        "clip_style": "CLIP style",
        "dino_concept": "DINO concept",
        "dino_style": "DINO style",
        "clip_delta_concept_minus_style": "CLIP concept - style",
        "dino_delta_concept_minus_style": "DINO concept - style",
    }.get(key, key)


def fit_line(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2 or np.std(x) == 0:
        return None
    xs = np.linspace(x.min(), x.max(), 100)
    slope, intercept = np.polyfit(x, y, 1)
    return xs, slope * xs + intercept


def concept_color(name):
    return "#4C78A8" if name.startswith("cat") else "#F58518"


def summarize_by(rows, key, fim_key, metrics):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    out = []
    for name, values in sorted(grouped.items()):
        summary = {
            key: name,
            "fim": float(np.mean([row[fim_key] for row in values])),
            "n_pairs": len(values),
        }
        for metric in metrics:
            vals = np.asarray([row[metric] for row in values], dtype=float)
            summary[f"{metric}_mean"] = float(vals.mean())
            summary[f"{metric}_std"] = float(vals.std(ddof=0))
        out.append(summary)
    return out


def annotate_points(ax, rows, x_key, y_key, label_key, x_offset=3, y_offset=3):
    for row in rows:
        ax.annotate(
            str(row[label_key]),
            (row[x_key], row[y_key]),
            fontsize=8,
            xytext=(x_offset, y_offset),
            textcoords="offset points",
        )


def plot_adapter_summary(pair_rows, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    method_dir = sorted({row["method_dir"] for row in pair_rows})[0]
    rows = [row for row in pair_rows if row["method_dir"] == method_dir]
    concepts = summarize_by(rows, "concept", "concept_fim_log10_mean", ("clip_concept", "dino_concept"))
    styles = summarize_by(rows, "style", "style_fim_log10_mean", ("clip_style", "dino_style"))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    concept_specs = (("clip_concept", "CLIP", "#4C78A8"), ("dino_concept", "DINO", "#72B7B2"))
    style_specs = (("clip_style", "CLIP", "#54A24B"), ("dino_style", "DINO", "#B279A2"))

    ax = axes[0]
    for metric, label, color in concept_specs:
        y = [row[f"{metric}_mean"] for row in concepts]
        yerr = [row[f"{metric}_std"] for row in concepts]
        ax.errorbar(
            [row["fim"] for row in concepts],
            y,
            yerr=yerr,
            fmt="o",
            markersize=7,
            capsize=3,
            color=color,
            label=label,
        )
    annotate_points(ax, concepts, "fim", "clip_concept_mean", "concept")
    ax.set_title("Concept adapters")
    ax.set_xlabel("average concept FIM, log10")
    ax.set_ylabel("mean similarity to concept references")
    ax.legend(title="metric")
    ax.grid(alpha=0.25)

    ax = axes[1]
    for metric, label, color in style_specs:
        y = [row[f"{metric}_mean"] for row in styles]
        yerr = [row[f"{metric}_std"] for row in styles]
        ax.errorbar(
            [row["fim"] for row in styles],
            y,
            yerr=yerr,
            fmt="o",
            markersize=7,
            capsize=3,
            color=color,
            label=label,
        )
    annotate_points(ax, styles, "fim", "clip_style_mean", "style")
    ax.set_title("Style adapters")
    ax.set_xlabel("average style FIM, log10")
    ax.set_ylabel("mean similarity to style reference")
    ax.legend(title="metric")
    ax.grid(alpha=0.25)

    fig.suptitle(f"Adapter-level FIM vs preservation ({method_dir})")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_delta_ranked(pair_rows, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    method_dir = sorted({row["method_dir"] for row in pair_rows})[0]
    rows = sorted(
        [row for row in pair_rows if row["method_dir"] == method_dir],
        key=lambda row: row["fim_log10_delta_concept_minus_style"],
    )
    x = np.arange(len(rows))
    colors = [
        "#4C78A8" if row["fim_log10_delta_concept_minus_style"] >= 0.0 else "#B279A2"
        for row in rows
    ]

    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True, constrained_layout=True)
    ax = axes[0]
    ax.bar(x, [row["fim_log10_delta_concept_minus_style"] for row in rows], color=colors, alpha=0.85)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_ylabel("log10 concept FIM - style FIM")
    ax.set_title("Pairs sorted by FIM balance")
    ax.text(0.01, 0.08, "style FIM higher", transform=ax.transAxes, color="#B279A2", fontsize=9)
    ax.text(0.86, 0.88, "concept FIM higher", transform=ax.transAxes, color="#4C78A8", fontsize=9)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    ax.plot(x, [row["clip_delta_concept_minus_style"] for row in rows], marker="o", markersize=3, label="CLIP")
    ax.plot(x, [row["dino_delta_concept_minus_style"] for row in rows], marker="o", markersize=3, label="DINO")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_ylabel("concept similarity - style similarity")
    ax.set_xlabel("merge pair")
    ax.set_xticks(x[:: max(1, len(x) // 18)], [rows[i]["pair"] for i in x[:: max(1, len(x) // 18)]], rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Does the FIM-dominant side also dominate similarity?")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_correlation_heatmap(corr_rows, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [row["label"].replace("higher ", "").replace(" vs ", "\nvs ") for row in corr_rows]
    values = np.asarray([[float(row["pearson"]), float(row["spearman"])] for row in corr_rows], dtype=float)

    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    image = ax.imshow(values, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_xticks([0, 1], ["Pearson", "Spearman"])
    ax.set_yticks(np.arange(len(labels)), labels)
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            val = values[row_idx, col_idx]
            ax.text(col_idx, row_idx, f"{val:.2f}", ha="center", va="center", color="black")
    ax.set_title("Correlation summary")
    fig.colorbar(image, ax=ax, label="correlation")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_scatter_grid(pair_rows, corr_rows, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    method_dir = corr_rows[0]["method_dir"] if corr_rows else "orthofuse"
    rows = [row for row in pair_rows if row["method_dir"] == method_dir]
    specs = [
        ("concept_fim_log10_mean", "clip_concept", "Concept FIM vs concept preservation"),
        ("style_fim_log10_mean", "clip_style", "Style FIM vs style preservation"),
        ("fim_log10_delta_concept_minus_style", "clip_delta_concept_minus_style", "FIM balance vs CLIP balance"),
        ("fim_log10_delta_concept_minus_style", "dino_delta_concept_minus_style", "FIM balance vs DINO balance"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, (x_key, y_key, label) in zip(axes.ravel(), specs):
        x = [row[x_key] for row in rows]
        y = [row[y_key] for row in rows]
        colors = [concept_color(row["concept"]) for row in rows]
        ax.scatter(x, y, c=colors, s=34, alpha=0.78, edgecolor="white", linewidth=0.4)
        line = fit_line(x, y)
        if line is not None:
            ax.plot(*line, color="black", linewidth=1.3, alpha=0.75)
        corr_row = next((row for row in corr_rows if row["x"] == x_key and row["y"] == y_key), None)
        suffix = f" r={float(corr_row['pearson']):.2f}" if corr_row else ""
        ax.set_title(f"{label}{suffix}")
        ax.set_xlabel(short_label(x_key))
        ax.set_ylabel(short_label(y_key))
        ax.grid(alpha=0.25)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#4C78A8", label="cat concepts", markersize=8),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#F58518", label="dog concepts", markersize=8),
    ]
    axes[0, 0].legend(handles=handles, loc="best")
    fig.suptitle(f"Pair-level FIM vs preservation ({method_dir})")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_correlation_summary(corr_rows, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [row["label"] for row in corr_rows]
    y = np.arange(len(labels))
    pearson = [float(row["pearson"]) for row in corr_rows]
    spearman = [float(row["spearman"]) for row in corr_rows]

    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    ax.barh(y - 0.18, pearson, height=0.34, label="Pearson", color="#4C78A8")
    ax.barh(y + 0.18, spearman, height=0.34, label="Spearman", color="#F58518")
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_xlim(-1.0, 1.0)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("correlation")
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    fig.suptitle("FIM-similarity correlations")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_pngs(pair_rows, corr_rows, output_dir):
    if not pair_rows or not corr_rows:
        return []
    first_method = sorted({row["method_dir"] for row in corr_rows})[0]
    method_corrs = [row for row in corr_rows if row["method_dir"] == first_method]
    paths = [
        output_dir / "fim_adapter_preservation.png",
        output_dir / "fim_delta_ranked.png",
        output_dir / "fim_correlation_heatmap.png",
        output_dir / "fim_similarity_scatter.png",
        output_dir / "fim_similarity_correlations.png",
    ]
    plot_adapter_summary(pair_rows, paths[0])
    plot_delta_ranked(pair_rows, paths[1])
    plot_correlation_heatmap(method_corrs, paths[2])
    plot_scatter_grid(pair_rows, method_corrs, paths[3])
    plot_correlation_summary(method_corrs, paths[4])
    return paths


def main():
    args = parse_args()
    pair_csv = args.output_dir / "pair_scores_with_fim.csv"
    score_csv = args.output_dir / "image_scores.csv"
    corr_csv = args.output_dir / "correlations.csv"

    if pair_csv.exists() and corr_csv.exists() and not args.force:
        pair_rows = read_csv(pair_csv)
        corr_rows = read_csv(corr_csv)
        png_paths = write_pngs(pair_rows, corr_rows, args.output_dir)
        print(f"[reuse] {pair_csv}")
        print(f"[reuse] {corr_csv}")
        for png_path in png_paths:
            print(f"[done] wrote {png_path}")
        return

    if pair_csv.exists() and not args.force:
        pair_rows = read_csv(pair_csv)
        corr_rows = correlation_rows(pair_rows)
        write_csv(corr_csv, corr_rows)
        png_paths = write_pngs(pair_rows, corr_rows, args.output_dir)
        print(f"[reuse] {pair_csv}")
        print(f"[done] wrote {corr_csv}")
        for png_path in png_paths:
            print(f"[done] wrote {png_path}")
        return

    records = list(iter_records(args))
    if not records:
        raise RuntimeError(f"No images found under {args.samples_root} for {args.method_dirs}")
    score_rows = score_images(records, args)
    pair_rows = aggregate_pair_scores(score_rows)
    selected = [get_pair(row["concept"], row["style"]) for row in pair_rows]
    pair_rows = add_fim_columns(pair_rows, selected, "cpu")
    corr_rows = correlation_rows(pair_rows)

    write_csv(score_csv, score_rows)
    write_csv(pair_csv, pair_rows)
    write_csv(corr_csv, corr_rows)
    png_paths = write_pngs(pair_rows, corr_rows, args.output_dir)
    print(f"[done] wrote {score_csv}")
    print(f"[done] wrote {pair_csv}")
    print(f"[done] wrote {corr_csv}")
    for png_path in png_paths:
        print(f"[done] wrote {png_path}")
    for row in corr_rows:
        print(
            f"{row['method_dir']}: {row['label']} pearson={row['pearson']:.3f} "
            f"spearman={row['spearman']:.3f} n={row['n']}"
        )


if __name__ == "__main__":
    main()
