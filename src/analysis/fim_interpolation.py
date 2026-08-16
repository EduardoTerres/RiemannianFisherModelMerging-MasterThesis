from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from safetensors.torch import load_file
from tqdm import tqdm

ROOTDIR = Path(__file__).resolve().parents[2]
if str(ROOTDIR) not in sys.path:
    sys.path.insert(0, str(ROOTDIR))

MPLCONFIGDIR = ROOTDIR / "outputs" / ".matplotlib"
XDG_CACHE_HOME = ROOTDIR / "outputs" / ".cache"
for path in (MPLCONFIGDIR, XDG_CACHE_HOME):
    path.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_HOME))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from src.analysis.plot_utils import dataset_plot_label
from src.geometry import SOnManifold
from src.merging import OFTMerging
from src.dataset.dataset_3 import DATASET_3_TEST, DATASET_3_TRAIN
from src.utils import parse_device

FIM_SUBDIR = "fim"
IMG_SUBDIR = "imgs"
ORANGE = "#f97316"
YELLOW = "#facc15"
WHITE = "#ffffff"
MODELS_DIR = Path("/path/to/models")
FISHERS_DIR = Path("/path/to/fishers")


@dataclass(frozen=True)
class ModelFamily:
    adapter_paths: list[str]
    fisher_finetuned_paths: list[str]
    fisher_pretrained_paths: list[str]


ADAPTER_TASK_NAMES = {
    "social_iqa": "socialiqa",
    "commonsense_qa": "commonsense",
    "science_qa": "scienceqa",
}


def adapter_task_name(task: str) -> str:
    return ADAPTER_TASK_NAMES.get(task, task)


def fisher_path(model_family: str, task: str, adapter_tag: str, model_state: str) -> str:
    return str(FISHERS_DIR / model_family / task / f"{adapter_tag}_{model_state}.safetensors")


def family_paths(prefix: str, adapter_dir: str, family_name: str) -> ModelFamily:
    tasks = [tag for tag, *_ in DATASET_3_TRAIN]
    adapter_paths = [
        str(MODELS_DIR / adapter_dir / f"{prefix}_{adapter_task_name(task)}")
        for task in tasks
    ]
    finetuned_paths = [
        fisher_path(family_name, task, f"{prefix}_{adapter_task_name(task)}", "finetuned")
        for task in tasks
    ]
    pretrained_paths = [
        fisher_path(family_name, task, f"{prefix}_{adapter_task_name(task)}", "pretrained")
        for task in tasks
    ]
    return ModelFamily(adapter_paths, finetuned_paths, pretrained_paths)


MODEL_FAMILIES = {
    "llama3.1": family_paths(
        "llama3-1_8b_finetune",
        "Llama-3.1-8B_OFT_dataset3_adapters",
        "llama3.1",
    ),
    "qwen2.5": family_paths(
        "qwen2.5_3b_finetune",
        "Qwen-2.5-3B_OFT_dataset3_adapters",
        "qwen2.5",
    ),
}

_device = "cuda" if torch.cuda.is_available() else "cpu"
_manifold = SOnManifold()
_merging = OFTMerging(device=_device)


def interpolate(
    start_model: Optional[Dict[str, torch.Tensor]],
    end_model: Dict[str, torch.Tensor],
    alpha: float,
) -> Dict[str, torch.Tensor]:
    interpolated: Dict[str, torch.Tensor] = {}

    for key, end_params in end_model.items():
        if not is_oft_key(key):
            interpolated[key] = end_params
            continue

        num_blocks, son_dimension = end_params.shape
        block_size = int((1 + (1 + 8 * son_dimension) ** 0.5) / 2)
        end_skew = _merging.oft_params_to_skew_matrix(end_params, son_dimension)
        r_end = torch.matrix_exp(end_skew)

        if start_model is None:
            r_start = torch.eye(block_size, dtype=end_params.dtype, device=end_params.device)
            r_start = r_start.unsqueeze(0).expand(num_blocks, -1, -1)
        else:
            start_params = start_model[key]
            start_skew = _merging.oft_params_to_skew_matrix(start_params, son_dimension)
            r_start = torch.matrix_exp(start_skew)

        tangent = _manifold.exact_log(r_start, r_end)
        r_interp = _manifold.exact_exp(r_start, alpha * tangent)
        omega_interp = _manifold.exact_log(r_start, r_interp)
        interpolated[key] = _merging.skew_matrix_to_oft_params(omega_interp)

    return interpolated


def resolve_device(device: str) -> str:
    if device.lower().strip() == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return parse_device(device)


def is_oft_key(key: str) -> bool:
    key_lower = key.lower()
    return "oft_r" in key_lower or "oft_" in key_lower


def load_summed_fisher(paths: Sequence[str], device: str) -> Dict[str, torch.Tensor]:
    summed: Dict[str, torch.Tensor] = {}
    for path in paths:
        fisher_path = Path(path)
        if not fisher_path.exists():
            raise FileNotFoundError(fisher_path)
        fisher = load_file(str(fisher_path), device=device)
        for key, value in fisher.items():
            value = value.float()
            if key not in summed:
                summed[key] = value.clone()
            else:
                if summed[key].shape != value.shape:
                    raise ValueError(f"Fisher shape mismatch for {key}: {summed[key].shape} vs {value.shape}")
                summed[key].add_(value)
    return summed


def fisher_key_for_weight(weight_key: str, summed_fisher: Dict[str, torch.Tensor]) -> str | None:
    candidates = [
        weight_key,
        weight_key.replace(".weight", ".default.weight"),
        weight_key.replace(".default.weight", ".weight"),
    ]
    for candidate in candidates:
        if candidate in summed_fisher:
            return candidate

    normalized = weight_key.replace(".weight", ".default.weight")
    matches = [key for key in summed_fisher if key.endswith(normalized) or normalized.endswith(key)]
    if len(matches) == 1:
        return matches[0]
    return None


def build_fisher_key_map(
    weights: Dict[str, torch.Tensor],
    summed_fisher: Dict[str, torch.Tensor],
) -> dict[str, str]:
    key_map = {}
    missing = []
    for key, value in weights.items():
        if not is_oft_key(key):
            continue
        fisher_key = fisher_key_for_weight(key, summed_fisher)
        if fisher_key is None:
            missing.append(key)
            continue
        if value.shape != summed_fisher[fisher_key].shape:
            raise ValueError(
                f"Shape mismatch for {key} and {fisher_key}: "
                f"{tuple(value.shape)} vs {tuple(summed_fisher[fisher_key].shape)}"
            )
        key_map[key] = fisher_key

    if not key_map:
        raise RuntimeError(
            "No OFT adapter weights matched the summed Fisher. "
            f"Adapter sample={list(weights)[:3]}, Fisher sample={list(summed_fisher)[:3]}"
        )
    if missing:
        print(f"  Warning: {len(missing)} OFT weights had no matching Fisher key.")
    return key_map


def fim_bilinear(
    weights: Dict[str, torch.Tensor],
    summed_fisher: Dict[str, torch.Tensor],
    key_map: dict[str, str],
    device: str,
) -> float:
    total = torch.zeros((), dtype=torch.float64, device=device)
    for weight_key, fisher_key in key_map.items():
        theta = weights[weight_key].to(device=device, dtype=torch.float64)
        fisher = summed_fisher[fisher_key].to(device=device, dtype=torch.float64)
        total = total + (theta * fisher * theta).sum()
    return float(total.cpu())


def interpolate_fim_values(
    end_model: Dict[str, torch.Tensor],
    summed_fisher: Dict[str, torch.Tensor],
    interpolation_grid: list[float],
    device: str,
) -> list[float]:
    key_map = build_fisher_key_map(end_model, summed_fisher)
    values = []
    for alpha in tqdm(interpolation_grid, desc="FIM bilinear"):
        interpolated_weights = interpolate(None, end_model, alpha=alpha)
        value = fim_bilinear(interpolated_weights, summed_fisher, key_map, device)
        values.append(value)
    return values


def save_values_json(
    path: Path,
    task: str,
    alphas: list[float],
    values: list[float],
    args: argparse.Namespace,
) -> None:
    payload = {
        "task": task,
        "fisher_state": args.fisher_state,
        "alphas": alphas,
        "fim_bilinear": values,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_values_json(path: Path) -> tuple[list[float], list[float]]:
    payload = json.loads(path.read_text())
    return payload["alphas"], payload["fim_bilinear"]


def plot_fim_curve(
    alphas: Sequence[float],
    values: Sequence[float],
    title: str,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=WHITE)
    ax.set_facecolor(WHITE)
    ax.plot(
        alphas,
        values,
        color=ORANGE,
        marker="o",
        markerfacecolor=YELLOW,
        markeredgecolor=ORANGE,
        linewidth=2,
        markersize=5,
    )
    ax.set_xlabel("alpha (0 = pretrained, 1 = fine-tuned)")
    ax.set_ylabel(r"$\theta^\top F_{\mathrm{sum}}\theta$")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_joint_fim_curves(
    series: Sequence[tuple[str, Sequence[float], Sequence[float]]],
    alpha_end: float,
    title: str,
    save_stem: Path,
) -> None:
    cmap = LinearSegmentedColormap.from_list("orange_yellow_white", [ORANGE, YELLOW, WHITE])
    usable = [(label, np.asarray(a), np.asarray(v)) for label, a, v in series if label != "math500"]
    if not usable:
        return

    fig, ax = plt.subplots(figsize=(18, 5.4), facecolor=WHITE)
    ax.set_facecolor(WHITE)
    colors = [cmap(x) for x in np.linspace(0.0, 0.72, len(usable))]
    for color, (label, alphas, values) in zip(colors, usable, strict=True):
        mask = (alphas >= -1e-9) & (alphas <= alpha_end + 1e-9)
        if not np.any(mask):
            continue
        ax.plot(
            alphas[mask],
            values[mask],
            marker="^",
            linewidth=2.4,
            markersize=6.4,
            color=color,
            label=dataset_plot_label(label),
        )

    ax.set_xlim(0, alpha_end)
    ax.set_xlabel(r"Geodesic parameter $\alpha$", fontsize=16)
    ax.set_ylabel(r"$\theta^\top F_{\mathrm{sum}}\theta$", fontsize=16)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(True, linestyle="--", alpha=0.45)
    label_y = ax.get_ylim()[1] + 0.04 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    for x_value, label, ha in (
        (0.0, r"\textbf{Pretrained} $\alpha=0$", "left"),
        (1.0, r"\textbf{Finetuned} $\alpha=1$", "center"),
    ):
        if x_value <= alpha_end:
            ax.axvline(x_value, color=ORANGE, linestyle="--", linewidth=1.4, alpha=0.85)
            ax.text(x_value, label_y, label, color=ORANGE, ha=ha, va="bottom", fontsize=16, clip_on=False)
    ax.legend(fontsize=13, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax.set_title(title)

    fig.tight_layout()
    save_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(save_stem.with_suffix(suffix), dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_outputs(
    family_name: str,
    task_tag: str,
    interpolation_grid: list[float],
    values: list[float],
    args: argparse.Namespace,
) -> None:
    fim_dir = Path(args.save_path) / family_name / FIM_SUBDIR
    img_dir = Path(args.save_path) / family_name / IMG_SUBDIR
    fim_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    np.save(fim_dir / f"{task_tag}.npy", np.array(values))
    save_values_json(fim_dir / f"{task_tag}.json", task_tag, interpolation_grid, values, args)
    plot_fim_curve(
        interpolation_grid,
        values,
        title=f"FIM interpolation: pretrained -> {task_tag}",
        save_path=img_dir / f"{task_tag}.png",
    )


def save_joint_plots(family_name: str, task_tags: list[str], args: argparse.Namespace) -> None:
    fim_dir = Path(args.save_path) / family_name / FIM_SUBDIR
    img_dir = Path(args.save_path) / family_name / IMG_SUBDIR
    series = []
    for task_tag in task_tags:
        json_path = fim_dir / f"{task_tag}.json"
        if json_path.exists():
            alphas, values = load_values_json(json_path)
            series.append((task_tag, alphas, values))
    if not series:
        return

    family_tag = "llama" if family_name.startswith("llama") else "qwen" if family_name.startswith("qwen") else family_name
    for alpha_end, suffix in ((1.0, "0_1"), (2.0, "0_2")):
        save_stem = img_dir / f"fim_interpolation_{family_tag}_{suffix}"
        plot_joint_fim_curves(
            series,
            alpha_end,
            title=f"{family_name} FIM interpolation ({alpha_end:g})",
            save_stem=save_stem,
        )
        print(f"  Saved joint plots to {save_stem}.png and {save_stem}.pdf")


def run_family(family_name: str, args: argparse.Namespace) -> None:
    model_family = MODEL_FAMILIES[family_name]
    dataset_specs = DATASET_3_TEST if args.dataset_split == "test" else DATASET_3_TRAIN
    task_tags = [spec[0] for spec in dataset_specs]
    adapter_by_task = {
        task_tag: adapter_path
        for (task_tag, *_), adapter_path in zip(dataset_specs, model_family.adapter_paths, strict=True)
    }
    fisher_paths = (
        model_family.fisher_pretrained_paths
        if args.fisher_state == "pretrained"
        else model_family.fisher_finetuned_paths
    )
    interpolation_grid = np.linspace(
        args.interpolation_start,
        args.interpolation_end,
        args.num_points,
    ).tolist()
    fim_dir = Path(args.save_path) / family_name / FIM_SUBDIR

    specs_to_compute = []
    for spec in dataset_specs:
        task_tag = spec[0]
        json_path = fim_dir / f"{task_tag}.json"
        if json_path.exists() and not args.force_compute:
            print(f"\n---[{task_tag}]--- Found cached FIM interpolation: {json_path}")
            if args.plots:
                alphas, values = load_values_json(json_path)
                plot_fim_curve(alphas, values, f"FIM interpolation: pretrained -> {task_tag}", Path(args.save_path) / family_name / IMG_SUBDIR / f"{task_tag}.png")
            continue
        specs_to_compute.append(spec)

    if not specs_to_compute:
        save_joint_plots(family_name, task_tags, args)
        return

    print(f"\n[{family_name}] Loading summed {args.fisher_state} FIM...")
    summed_fisher = load_summed_fisher(fisher_paths, args.device)

    for task_tag, *_ in specs_to_compute:
        adapter_path = adapter_by_task[task_tag]
        print(f"\n---[{task_tag}]--- Loading adapter: {adapter_path}")
        end_model = load_file(f"{adapter_path}/adapter_model.safetensors", device=args.device)
        values = interpolate_fim_values(end_model, summed_fisher, interpolation_grid, args.device)
        save_outputs(family_name, task_tag, interpolation_grid, values, args)

    save_joint_plots(family_name, task_tags, args)


def main(args: argparse.Namespace) -> None:
    for family_name in args.model_family:
        run_family(family_name, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-points", type=int, default=5)
    parser.add_argument("--interpolation-start", type=float, default=0.0)
    parser.add_argument("--interpolation-end", type=float, default=1.0)
    parser.add_argument("--save-path", type=str, default=f"{ROOTDIR}/outputs/fim_interpolation")
    parser.add_argument("--dataset-split", choices=["train", "test"], default="test")
    parser.add_argument("--fisher-state", choices=["finetuned", "pretrained"], default="finetuned")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--force-compute", action="store_true")
    parser.add_argument("--model-family", nargs="+", default=list(MODEL_FAMILIES), choices=list(MODEL_FAMILIES))
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    args.device = resolve_device(args.device)
    main(args)
