from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from safetensors.torch import load_file
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "outputs" / "mergeschema_fim"
sys.path.insert(0, str(REPO_ROOT))

MPLCONFIGDIR = REPO_ROOT / "outputs" / ".matplotlib"
XDG_CACHE_HOME = REPO_ROOT / "outputs" / ".cache"
for path in (MPLCONFIGDIR, XDG_CACHE_HOME):
    path.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_HOME))

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.dataset.dataset_3 import DATASET_3_TRAIN
from src.merging import OFTMerging
from src.paths import MODEL_FAMILIES_D3
from src.utils import parse_device

WATER_CMAP = LinearSegmentedColormap.from_list(
    "water_dark_to_white",
    ["#123a66", "#1f6f9e", "#28a9c7", "#9de3eb", "#ffffff"],
)

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 16,
    "axes.labelsize": 18,
    "axes.titlesize": 20,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 15,
    "legend.title_fontsize": 15,
    "figure.titlesize": 20,
})


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def resolve_device(device: str) -> str:
    if device.lower().strip() == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return parse_device(device)


def adapter_paths_for_tasks(model_name: str, names: Sequence[str]) -> list[str]:
    family = MODEL_FAMILIES_D3[model_name]
    adapter_by_task = {
        task: family.adapter_paths[i]
        for i, (task, *_) in enumerate(DATASET_3_TRAIN)
    }
    return [adapter_by_task[name] for name in names]


def fisher_paths_for_tasks(model_name: str, names: Sequence[str]) -> list[str]:
    family = MODEL_FAMILIES_D3[model_name]
    fisher_by_task = {
        task: family.fisher_finetuned_paths[i]
        for i, (task, *_) in enumerate(DATASET_3_TRAIN)
    }
    return [fisher_by_task[name] for name in names]


def load_summed_fisher(paths: Sequence[str], device: str) -> dict[str, torch.Tensor]:
    summed: dict[str, torch.Tensor] = {}
    for path in paths:
        fisher_path = Path(path)
        if not fisher_path.exists():
            raise FileNotFoundError(fisher_path)
        fisher = load_file(str(fisher_path), device=device)
        print(f"Loaded Fisher: {fisher_path}", flush=True)
        for key, value in fisher.items():
            value = value.float()
            if key not in summed:
                summed[key] = value.clone()
            else:
                if summed[key].shape != value.shape:
                    raise ValueError(
                        f"Fisher shape mismatch for {key}: "
                        f"{tuple(summed[key].shape)} vs {tuple(value.shape)}"
                    )
                summed[key].add_(value)
    return summed


def is_oft_key(key: str) -> bool:
    key_lower = key.lower()
    return "oft_r" in key_lower or "oft_" in key_lower


def fisher_key_for_weight(
    weight_key: str,
    summed_fisher: dict[str, torch.Tensor],
) -> str | None:
    candidates = [
        weight_key,
        weight_key.replace(".weight", ".default.weight"),
        weight_key.replace(".default.weight", ".weight"),
    ]
    for candidate in candidates:
        if candidate in summed_fisher:
            return candidate

    normalized = weight_key.replace(".weight", ".default.weight")
    matches = [
        key for key in summed_fisher
        if key.endswith(normalized) or normalized.endswith(key)
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def build_fisher_key_map(
    weights: dict[str, torch.Tensor],
    summed_fisher: dict[str, torch.Tensor],
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
        print(f"Warning: {len(missing)} OFT weights had no matching Fisher key.", flush=True)
    return key_map


def pair_interpolated_weights(
    weights: list[dict[str, torch.Tensor]],
    x: float,
    y: float,
) -> dict[str, torch.Tensor]:
    if len(weights) != 2:
        raise ValueError(f"Expected exactly two task vectors, got {len(weights)}.")
    return {
        key: x * weights[0][key].float() + y * weights[1][key].float()
        for key in weights[0]
    }


def fim_bilinear(
    weights: dict[str, torch.Tensor],
    summed_fisher: dict[str, torch.Tensor],
    key_map: dict[str, str],
    device: str,
) -> float:
    total = torch.zeros((), dtype=torch.float64, device=device)
    for weight_key, fisher_key in key_map.items():
        theta = weights[weight_key].to(device=device, dtype=torch.float64)
        fisher = summed_fisher[fisher_key].to(device=device, dtype=torch.float64)
        total = total + (theta * fisher * theta).sum()
    return float(total.cpu())


def flatten_weights(
    weights: dict[str, torch.Tensor],
    keys: Sequence[str],
) -> torch.Tensor:
    return torch.cat([weights[key].float().cpu().flatten() for key in keys])


def pair_orthomerge_coefficients(weights: list[dict[str, torch.Tensor]]) -> tuple[float, float]:
    keys = list(weights[0])
    vecs = [flatten_weights(weight, keys) for weight in weights]
    correction = (
        sum(torch.linalg.vector_norm(vec) for vec in vecs)
        / torch.linalg.vector_norm(sum(vecs)).clamp(min=1e-8)
    )
    alpha = float(correction)
    return alpha, alpha


def fit_two_vector_lstsq(
    first: torch.Tensor,
    second: torch.Tensor,
    target: torch.Tensor,
) -> tuple[float, float]:
    gram = torch.tensor(
        [
            [torch.dot(first, first), torch.dot(first, second)],
            [torch.dot(first, second), torch.dot(second, second)],
        ],
        dtype=torch.float64,
    )
    rhs = torch.tensor(
        [torch.dot(first, target), torch.dot(second, target)],
        dtype=torch.float64,
    )
    try:
        solution = torch.linalg.solve(gram, rhs)
    except RuntimeError:
        solution = torch.linalg.lstsq(gram, rhs).solution
    return float(solution[0]), float(solution[1])


def project_weights_to_pair_plane(
    weights: list[dict[str, torch.Tensor]],
    target_weights: dict[str, torch.Tensor],
) -> tuple[float, float, float]:
    keys = list(weights[0])
    vecs = [flatten_weights(weight, keys).float() for weight in weights]
    target = flatten_weights(target_weights, keys).float()
    x, y = fit_two_vector_lstsq(vecs[0], vecs[1], target)
    residual = torch.linalg.vector_norm(x * vecs[0] + y * vecs[1] - target).item()
    return x, y, residual


def fisher_merged_weights(
    merger: OFTMerging,
    weights: list[dict[str, torch.Tensor]],
    fishers: list[dict[str, torch.Tensor]],
    mode: str,
) -> dict[str, torch.Tensor]:
    if len(fishers) != len(weights):
        raise ValueError(f"Expected {len(weights)} Fisher files, got {len(fishers)}.")

    merged = {}
    for key in weights[0]:
        fisher_layer = [merger._layer_fisher(fisher, key) for fisher in fishers]
        merged[key] = merger.merge_formula(
            [weight[key] for weight in weights],
            fisher_list=fisher_layer,
            mode=mode,
        )
    return merged


def fisher_projection(
    args: argparse.Namespace,
    names: Sequence[str],
    mode: str,
    weights: list[dict[str, torch.Tensor]] | None = None,
) -> tuple[float, float, float]:
    device = args.device if weights is not None else "cpu"
    merger = OFTMerging(lam=args.lam, device=device, fisher_backend="diagonal")
    if weights is None:
        weights = merger.load_weights(adapter_paths_for_tasks(args.model_name, names))
    fishers = merger.load_fishers(fisher_paths_for_tasks(args.model_name, names))
    merged = fisher_merged_weights(merger, weights, fishers, mode=mode)
    return project_weights_to_pair_plane(weights, merged)


def evaluate_grid(
    args: argparse.Namespace,
    cache: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    tuple[float, float],
    tuple[float, float] | None,
    tuple[float, float] | None,
]:
    names = args.tasks
    all_task_names = [task for task, *_ in DATASET_3_TRAIN]
    adapter_paths = adapter_paths_for_tasks(args.model_name, names)
    fisher_paths = fisher_paths_for_tasks(args.model_name, all_task_names)

    print(f"Using device: {args.device}", flush=True)
    print(f"Merge axes: {names[0]}, {names[1]}", flush=True)
    print("Interpolated model: x * task_a + y * task_b.", flush=True)
    print(f"Summed finetuned FIM tasks ({len(all_task_names)}): {', '.join(all_task_names)}", flush=True)
    print(f"Grid: {args.num_points}x{args.num_points} from {args.min_coeff} to {args.max_coeff}", flush=True)

    merger = OFTMerging(device=args.device)
    weights = merger.load_weights(adapter_paths)
    if len(weights) != 2:
        raise RuntimeError(f"Expected two adapter weight dicts, loaded {len(weights)}.")
    merge_xy = pair_orthomerge_coefficients(weights)

    diagonal_fisher_xy = None
    if not args.skip_diagonal_fisher:
        fisher_x, fisher_y, fisher_residual = fisher_projection(
            args,
            names,
            mode="diagonal_fisher",
            weights=weights,
        )
        diagonal_fisher_xy = (fisher_x, fisher_y)
        print(
            "Diagonal Fisher projection: "
            f"x={fisher_x:.4f}, y={fisher_y:.4f}, residual={fisher_residual:.4g}",
            flush=True,
        )

    fisher_xy = None
    if not args.skip_fisher:
        fisher_xy = diagonal_fisher_xy
        if fisher_xy is not None:
            print("Fisher marker reuses Diagonal Fisher projection for diagonal FIM.", flush=True)

    summed_fisher = load_summed_fisher(fisher_paths, args.device)
    key_map = build_fisher_key_map(weights[0], summed_fisher)

    xs = np.linspace(args.min_coeff, args.max_coeff, args.num_points)
    ys = np.linspace(args.min_coeff, args.max_coeff, args.num_points)
    z = np.zeros((len(ys), len(xs)))
    grid = [(row, col, x, y) for row, y in enumerate(ys) for col, x in enumerate(xs)]
    for row, col, x, y in tqdm(grid, desc="FIM grid points"):
        merged = pair_interpolated_weights(weights, float(x), float(y))
        z[row, col] = fim_bilinear(merged, summed_fisher, key_map, args.device)
        tqdm.write(f"x={x:.3f}, y={y:.3f}, fim={z[row, col]:.6g}")

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache,
        xs=xs,
        ys=ys,
        z=z,
        tasks=np.array(names),
        fim_tasks=np.array(all_task_names),
        interpolation_rule=np.array("x * task_a + y * task_b"),
        merge_x=merge_xy[0],
        merge_y=merge_xy[1],
        diagonal_fisher_x=np.nan if diagonal_fisher_xy is None else diagonal_fisher_xy[0],
        diagonal_fisher_y=np.nan if diagonal_fisher_xy is None else diagonal_fisher_xy[1],
        fisher_x=np.nan if fisher_xy is None else fisher_xy[0],
        fisher_y=np.nan if fisher_xy is None else fisher_xy[1],
        fim_state=np.array("finetuned"),
    )
    print(f"Saved FIM grid to {cache}", flush=True)
    return xs, ys, z, names, merge_xy, diagonal_fisher_xy, fisher_xy


def marker_handles(tasks: Sequence[str]) -> list[Line2D]:
    return [
        Line2D([0], [0], color="black", marker="o", linestyle="None", markersize=9, label="Pretrained"),
        Line2D([0], [0], color="black", marker="^", linestyle="None", markersize=10, label=latex_escape(tasks[0].capitalize())),
        Line2D([0], [0], color="black", marker="s", linestyle="None", markersize=9, label=latex_escape(tasks[1].capitalize())),
        Line2D([0], [0], color="black", marker="D", linestyle="None", markersize=9, label=r"\textsc{Lie sum}"),
        Line2D([0], [0], color="black", marker="P", linestyle="None", markersize=10, label=r"\textsc{Diagonal Fisher}"),
        Line2D([0], [0], color="black", marker="X", linestyle="None", markersize=10, label=r"\textsc{Fisher}"),
        Line2D([0], [0], color="black", marker="*", linestyle="None", markersize=14, label=r"\textsc{OrthoMerge}"),
    ]


def save_legend(tasks: Sequence[str], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.0, 1.8))
    ax.axis("off")
    ax.legend(handles=marker_handles(tasks), loc="center", frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", transparent=True)
    pdf_output = output.with_suffix(".pdf")
    if pdf_output != output:
        fig.savefig(pdf_output, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"Saved {output}")
    if pdf_output != output:
        print(f"Saved {pdf_output}")


def add_tangent_background(ax, plane_extent: tuple[float, float]) -> None:
    lo, hi = plane_extent
    ax.set_facecolor("#eeeeee")
    xline = np.array([lo, hi])
    for offset in np.arange(2 * lo, 2 * hi + 0.001, 0.25):
        ax.plot(xline, xline + offset, color="black", lw=0.45, alpha=0.38, zorder=0)


def plot(
    xs: np.ndarray,
    ys: np.ndarray,
    z: np.ndarray,
    tasks: Sequence[str],
    merge_xy: tuple[float, float],
    diagonal_fisher_xy: tuple[float, float] | None,
    fisher_xy: tuple[float, float] | None,
    output: Path,
    plane_extent: tuple[float, float],
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6.6))
    add_tangent_background(ax, plane_extent)
    heatmap = ax.contourf(xs, ys, z, levels=40, cmap=WATER_CMAP, zorder=1)
    fig.colorbar(
        heatmap,
        ax=ax,
        pad=0.02,
        label=r"$\theta^\top F_{\mathrm{sum}}\theta$",
    )
    points = {
        "Pretrained": ((0, 0), "o", 110),
        latex_escape(tasks[0].capitalize()): ((1, 0), "^", 125),
        latex_escape(tasks[1].capitalize()): ((0, 1), "s", 110),
        r"\textsc{Gradients} standard": ((1, 1), "D", 115),
        r"\textsc{OrthoMerge}": (merge_xy, "*", 230),
    }
    if diagonal_fisher_xy is not None:
        points[r"\textsc{Diagonal Fisher}"] = (diagonal_fisher_xy, "P", 150)
    if fisher_xy is not None:
        points[r"\textsc{Fisher}"] = (fisher_xy, "X", 150)
    for label, (xy, marker, size) in points.items():
        ax.scatter(*xy, s=size, marker=marker, color="black", linewidth=1.2, label=label, zorder=3)
        dx, dy = (0.035, -0.10) if label == "Pretrained" else (0.035, 0.035)
        if label == r"\textsc{Fisher}" and diagonal_fisher_xy is not None and np.allclose(xy, diagonal_fisher_xy):
            dx, dy = (0.035, -0.15)
        ax.text(xy[0] + dx, xy[1] + dy, label, fontsize=13, weight="bold", zorder=4, color="white")
    arrow_ends = [(1, 0), (0, 1), (1, 1), merge_xy]
    if diagonal_fisher_xy is not None:
        arrow_ends.append(diagonal_fisher_xy)
    if fisher_xy is not None:
        arrow_ends.append(fisher_xy)
    for end in arrow_ends:
        ax.annotate("", xy=end, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="black", lw=2.3))
    ax.set(
        xlabel=rf"\texttt{{{latex_escape(tasks[0])}}} tangent coefficient",
        ylabel=rf"\texttt{{{latex_escape(tasks[1])}}} tangent coefficient",
    )
    ax.set_aspect("equal")
    ax.set_xlim(*plane_extent)
    ax.set_ylim(*plane_extent)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    pdf_output = output.with_suffix(".pdf")
    if pdf_output != output:
        fig.savefig(pdf_output, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")
    if pdf_output != output:
        print(f"Saved {pdf_output}")


def load_marker(
    data: np.lib.npyio.NpzFile,
    x_key: str,
    y_key: str,
) -> tuple[float, float] | None:
    if {x_key, y_key}.issubset(data.files) and not np.isnan(data[x_key]):
        return (float(data[x_key]), float(data[y_key]))
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--model-name", default="llama3.1", choices=list(MODEL_FAMILIES_D3))
    parser.add_argument("--tasks", nargs=2, default=["coqa", "triviaqa"])
    parser.add_argument("--num-points", type=int, default=20)
    parser.add_argument("--min-coeff", type=float, default=-1.0)
    parser.add_argument("--max-coeff", type=float, default=1.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache", type=Path, default=OUTPUT_DIR / "merge_schema_fim.npz")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "merge_schema_fim.png")
    parser.add_argument("--plane-min", type=float, default=-1.5)
    parser.add_argument("--plane-max", type=float, default=1.5)
    parser.add_argument("--lam", type=float, default=0.0, help="Regularisation coefficient for Fisher merge markers.")
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--from-saved", action="store_true", help="Only plot from --cache; error if it is missing.")
    parser.add_argument("--skip-diagonal-fisher", action="store_true", help="Do not compute or plot the diagonal Fisher merge marker.")
    parser.add_argument("--skip-fisher", action="store_true", help="Do not plot the Fisher marker.")
    args = parser.parse_args()
    args.device = resolve_device(args.device)

    if args.from_saved and not args.cache.exists():
        raise FileNotFoundError(f"--from-saved requested, but saved FIM grid does not exist: {args.cache}")

    if args.from_saved or (args.cache.exists() and not args.recompute):
        print(f"Loading saved FIM grid from {args.cache}", flush=True)
        data = np.load(args.cache, allow_pickle=True)
        xs, ys, z, tasks = data["xs"], data["ys"], data["z"], data["tasks"].tolist()
        if {"merge_x", "merge_y"}.issubset(data.files):
            merge_xy = (float(data["merge_x"]), float(data["merge_y"]))
        else:
            merger = OFTMerging(device="cpu")
            weights = merger.load_weights(adapter_paths_for_tasks(args.model_name, tasks))
            merge_xy = pair_orthomerge_coefficients(weights)
        diagonal_fisher_xy = None if args.skip_diagonal_fisher else load_marker(data, "diagonal_fisher_x", "diagonal_fisher_y")
        fisher_xy = None if args.skip_fisher else load_marker(data, "fisher_x", "fisher_y")
    else:
        print("Evaluating FIM bilinear grid...", flush=True)
        xs, ys, z, tasks, merge_xy, diagonal_fisher_xy, fisher_xy = evaluate_grid(args, args.cache)

    plot(
        xs,
        ys,
        z,
        tasks,
        merge_xy,
        diagonal_fisher_xy,
        fisher_xy,
        args.output,
        (args.plane_min, args.plane_max),
    )


if __name__ == "__main__":
    main()
