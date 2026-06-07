from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "outputs" / "mergeschema"
sys.path.insert(0, str(REPO_ROOT))

from src.dataset.dataset_2 import DATASET_2_TEST, DATASET_2_TRAIN, build_loader
from src.merging import OFTMerging
from src.paths import MODEL_FAMILIES_D2

WATER_CMAP = LinearSegmentedColormap.from_list(
    "water_dark_to_white",
    ["#123a66", "#1f6f9e", "#28a9c7", "#9de3eb", "#ffffff"],
)


def task_specs(names: list[str]):
    specs = {task: spec for task, *spec in DATASET_2_TEST}
    return [(task, *specs[task]) for task in names]


def adapter_paths_for_tasks(model_name: str, names: list[str]) -> list[str]:
    family = MODEL_FAMILIES_D2[model_name]
    adapter_by_task = {task: family.adapter_paths[i] for i, (task, *_) in enumerate(DATASET_2_TRAIN)}
    return [adapter_by_task[name] for name in names]


def apply_weights(model: torch.nn.Module, weights: dict[str, torch.Tensor], device: str) -> None:
    params = dict(model.named_parameters())
    for key, val in weights.items():
        target = params.get(key.replace(".weight", ".default.weight"), params.get(key))
        if target is not None:
            target.data.copy_(val.to(device))


@torch.inference_mode()
def loss(model: torch.nn.Module, loader, device: str, desc: str | None = None) -> float:
    total, tokens = 0.0, 0
    for batch in tqdm(loader, desc=desc, leave=False):
        batch = {key: val.to(device) for key, val in batch.items()}
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        logits = out.logits[..., :-1, :].contiguous()
        labels = batch["labels"][..., 1:].contiguous()
        mask = labels.ne(-100)
        safe_labels = labels.masked_fill(~mask, 0)
        vals = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), safe_labels.view(-1), reduction="none")
        total += (vals.view_as(labels) * mask).sum().item()
        tokens += mask.sum().item()
    return total / max(tokens, 1)


def merged_at(merger: OFTMerging, weights: list[dict[str, torch.Tensor]], x: float, y: float) -> dict[str, torch.Tensor]:
    return {
        key: merger.merge_formula([w[key] for w in weights], mode="standard", alphas=torch.tensor([x, y], device=merger.device))
        for key in weights[0]
    }


def corrected_average(merger: OFTMerging, weights: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    alphas = torch.tensor([0.5, 0.5], device=merger.device)
    return {key: merger.merge_formula([w[key] for w in weights], mode="standard_rescaled", alphas=alphas) for key in weights[0]}


def flatten_weights(weights: dict[str, torch.Tensor], keys) -> torch.Tensor:
    return torch.cat([weights[key].float().cpu().flatten() for key in keys])


def project_to_coefficients(weights: list[dict[str, torch.Tensor]], merged: dict[str, torch.Tensor]) -> tuple[float, float]:
    keys = list(weights[0])
    v1, v2, vm = flatten_weights(weights[0], keys), flatten_weights(weights[1], keys), flatten_weights(merged, keys)
    gram = torch.tensor([[torch.dot(v1, v1), torch.dot(v1, v2)], [torch.dot(v2, v1), torch.dot(v2, v2)]])
    rhs = torch.tensor([torch.dot(v1, vm), torch.dot(v2, vm)])
    coeffs = torch.linalg.solve(gram + 1e-8 * torch.eye(2), rhs)
    return float(coeffs[0]), float(coeffs[1])


def task_vector_geometry(weights: list[dict[str, torch.Tensor]]) -> tuple[float, float, float]:
    norms_1, norms_2, angles = [], [], []
    for key in weights[0]:
        v1 = weights[0][key].float().flatten()
        v2 = weights[1][key].float().flatten()
        n1 = torch.linalg.vector_norm(v1)
        n2 = torch.linalg.vector_norm(v2)
        if n1 == 0 or n2 == 0:
            continue
        cos = torch.clamp(torch.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        norms_1.append(n1.item())
        norms_2.append(n2.item())
        angles.append(torch.arccos(cos).item())
    if not angles:
        raise ValueError("Could not compute task-vector geometry: no nonzero OFT vectors found.")
    return float(np.mean(norms_1)), float(np.mean(norms_2)), float(np.mean(angles))


def geometry_from_adapters(args: argparse.Namespace, names: list[str]) -> tuple[float, float, float]:
    merger = OFTMerging(device="cpu")
    return task_vector_geometry(merger.load_weights(adapter_paths_for_tasks(args.model_name, names)))


def corrected_projection_from_adapters(args: argparse.Namespace, names: list[str]) -> tuple[float, float]:
    merger = OFTMerging(device="cpu")
    weights = merger.load_weights(adapter_paths_for_tasks(args.model_name, names))
    return project_to_coefficients(weights, corrected_average(merger, weights))


def require_cuda(device: str) -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for loss computation; rerun on a GPU node or use --from_saved.")
    if not device.startswith("cuda"):
        raise RuntimeError(f"Loss computation must run on CUDA, got --device {device!r}.")
    return device


def evaluate_grid(args: argparse.Namespace, cache: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    args.device = require_cuda(args.device)
    family = MODEL_FAMILIES_D2[args.model_name]
    names = args.tasks
    selected = task_specs(names)
    adapter_paths = adapter_paths_for_tasks(args.model_name, names)

    print(f"Using device: {args.device}", flush=True)
    print(f"Tasks: {names[0]}, {names[1]} | samples per task: {args.num_samples}", flush=True)
    print(f"Grid: {args.num_points}x{args.num_points} from {args.min_coeff} to {args.max_coeff}", flush=True)
    print("Loading tokenizer and fixed eval loaders...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(family.base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    loaders = [
        build_loader(path, dset, split, formatter, tokenizer, args.num_samples, args.batch_size, args.max_length, task, str(args.dataset_cache_dir))
        for task, path, dset, split, formatter in selected
    ]

    print("Loading selected adapters and base model...", flush=True)
    merger = OFTMerging(alphas=[1.0, 1.0], device=args.device)
    weights = merger.load_weights(adapter_paths)
    r1, r2, theta = task_vector_geometry(weights)
    merge_x, merge_y = project_to_coefficients(weights, corrected_average(merger, weights))
    base = AutoModelForCausalLM.from_pretrained(
        family.base_model_path,
        torch_dtype=torch.bfloat16 if "cuda" in args.device else torch.float32,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, adapter_paths[0], is_trainable=False)
    model.enable_adapter_layers()
    model.to(args.device).eval()

    xs = np.linspace(args.min_coeff, args.max_coeff, args.num_points)
    ys = np.linspace(args.min_coeff, args.max_coeff, args.num_points)
    task_losses = np.zeros((len(names), len(ys), len(xs)))
    grid = [(row, col, x, y) for row, y in enumerate(ys) for col, x in enumerate(xs)]
    for row, col, x, y in tqdm(grid, desc="grid points"):
        apply_weights(model, merged_at(merger, weights, float(x), float(y)), args.device)
        for idx, (name, loader) in enumerate(zip(names, loaders, strict=True)):
            task_losses[idx, row, col] = loss(model, loader, args.device, desc=f"{name} loss")
        tqdm.write(f"x={x:.3f}, y={y:.3f}, sum={task_losses[:, row, col].sum():.4f}")
    z = task_losses.sum(axis=0)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache,
        xs=xs,
        ys=ys,
        z=z,
        task_losses=task_losses,
        tasks=np.array(names),
        r1=r1,
        r2=r2,
        theta=theta,
        merge_x=merge_x,
        merge_y=merge_y,
    )
    print(f"Saved losses to {cache}", flush=True)
    return xs, ys, z, task_losses, names


def plot(xs: np.ndarray, ys: np.ndarray, z: np.ndarray, tasks: list[str], merge_xy: tuple[float, float], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6.6))
    heatmap = ax.contourf(xs, ys, z, levels=40, cmap=WATER_CMAP)
    fig.colorbar(heatmap, ax=ax, pad=0.02, label=f"{tasks[0]} loss + {tasks[1]} loss")
    points = {
        "Pretrained": ((0, 0), "o", 80),
        f"{tasks[0].capitalize()}": ((1, 0), "^", 90),
        f"{tasks[1].capitalize()}": ((0, 1), "s", 80),
        "OrthoMerge + correction": (merge_xy, "*", 180),
    }
    for label, (xy, marker, size) in points.items():
        ax.scatter(*xy, s=size, marker=marker, color="black", linewidth=1.2, label=label, zorder=3)
        dx, dy = (0.035, -0.10) if label == "Pretrained" else (0.035, 0.035)
        ax.text(xy[0] + dx, xy[1] + dy, label, fontsize=10, weight="bold", zorder=4, color="black")
    for end in [(1, 0), (0, 1), merge_xy]:
        ax.annotate("", xy=end, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="black", lw=2.3))
    ax.set(
        xlabel=f"{tasks[0]} tangent coefficient",
        ylabel=f"{tasks[1]} tangent coefficient",
        title="Loss landscape with finetunes and OrthoMerge + correction",
    )
    ax.set_aspect("equal")
    ax.legend(loc="center left", bbox_to_anchor=(1.18, 0.5), frameon=False, title="Points")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")


def plot_geometry(
    xs: np.ndarray,
    ys: np.ndarray,
    z: np.ndarray,
    tasks: list[str],
    geometry: tuple[float, float, float],
    merge_xy: tuple[float, float],
    output: Path,
) -> None:
    r1, r2, theta = geometry
    v1 = np.array([r1, 0.0])
    v2 = np.array([r2 * np.cos(theta), r2 * np.sin(theta)])
    merge = merge_xy[0] * v1 + merge_xy[1] * v2
    grid_x, grid_y = np.meshgrid(xs, ys)
    geom_x = grid_x * v1[0] + grid_y * v2[0]
    geom_y = grid_x * v1[1] + grid_y * v2[1]

    fig, ax = plt.subplots(figsize=(8.5, 6.6))
    heatmap = ax.tricontourf(geom_x.ravel(), geom_y.ravel(), z.ravel(), levels=40, cmap=WATER_CMAP)
    fig.colorbar(heatmap, ax=ax, pad=0.02, label=f"{tasks[0]} loss + {tasks[1]} loss")
    points = {
        "pretrained": (np.array([0.0, 0.0]), "o", 80),
        f"{tasks[0]} finetune": (v1, "^", 90),
        f"{tasks[1]} finetune": (v2, "s", 80),
        "OrthoMerge + correction": (merge, "*", 180),
    }
    for label, (xy, marker, size) in points.items():
        ax.scatter(*xy, s=size, marker=marker, color="black", linewidth=1.2, label=label, zorder=3)
        ax.text(xy[0] + 0.035 * max(r1, r2), xy[1] + 0.035 * max(r1, r2), label, fontsize=10, weight="bold", zorder=4, color="black")
    for end in [v1, v2, merge]:
        ax.annotate("", xy=end, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="black", lw=2.3))
    ax.set(
        xlabel="OFT geometry axis 1",
        ylabel="OFT geometry axis 2",
        title="Loss landscape in average OFT task-vector geometry",
    )
    ax.set_aspect("equal")
    ax.legend(loc="center left", bbox_to_anchor=(1.18, 0.5), frameon=False, title="Points")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")


def plot_manifold_3d(
    xs: np.ndarray,
    ys: np.ndarray,
    image_path: Path,
    output: Path,
    curvature: float,
) -> None:
    radius = 1.0 / max(curvature, 1e-6)
    xmin, xmax, ymin, ymax = xs.min(), xs.max(), ys.min(), ys.max()
    mx, my = 0.12 * (xmax - xmin), 0.12 * (ymax - ymin)
    px = np.linspace(xmin - mx, xmax + mx, 90)
    py = np.linspace(ymin - my, ymax + my, 90)
    plane_x, plane_y = np.meshgrid(px, py)
    plane_lift = 0.05 * max(xmax - xmin, ymax - ymin)
    plane_z = np.full_like(plane_x, radius + plane_lift)

    span = min(max(abs(xmin), abs(xmax), abs(ymin), abs(ymax)) * 0.95, radius * 0.45)
    sx, sy = np.meshgrid(np.linspace(-span, span, 70), np.linspace(-span, span, 70))
    inside = sx**2 + sy**2 <= radius**2
    sz = np.where(inside, np.sqrt(radius**2 - sx**2 - sy**2), np.nan)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(sx, sy, sz, color="white", edgecolor="black", linewidth=0.25, alpha=0.95, shade=False)

    ax.plot_surface(plane_x, plane_y, plane_z, color="#d8d8d8", alpha=0.28, shade=False)
    for t in np.linspace(px.min(), px.max(), 12):
        ax.plot([t, t], [py.min(), ymin], [radius + plane_lift, radius + plane_lift], color="#7a7a7a", lw=0.6)
        ax.plot([t, t], [ymax, py.max()], [radius + plane_lift, radius + plane_lift], color="#7a7a7a", lw=0.6)
    for t in np.linspace(py.min(), py.max(), 12):
        ax.plot([px.min(), xmin], [t, t], [radius + plane_lift, radius + plane_lift], color="#7a7a7a", lw=0.6)
        ax.plot([xmax, px.max()], [t, t], [radius + plane_lift, radius + plane_lift], color="#7a7a7a", lw=0.6)

    img = plt.imread(image_path)
    ix = np.linspace(xmin, xmax, img.shape[1])
    iy = np.linspace(ymin, ymax, img.shape[0])
    img_x, img_y = np.meshgrid(ix, iy)
    ax.plot_surface(img_x, img_y, np.full_like(img_x, radius + plane_lift + 0.01), facecolors=img[::-1], shade=False)

    ax.set_axis_off()
    ax.view_init(elev=24, azim=-56)
    ax.set_box_aspect((1, 1, 0.32))
    ax.legend(
        handles=[
            Patch(facecolor="white", edgecolor="black", label="SO(n) manifold sketch"),
            Patch(facecolor="#d8d8d8", edgecolor="#7a7a7a", label="tangent plane outside sampled grid"),
            Line2D([0], [0], color="black", marker="*", linestyle="None", label="points/arrows in tangent plot"),
        ],
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--model-name", default="llama3.1", choices=list(MODEL_FAMILIES_D2))
    parser.add_argument("--tasks", nargs=2, default=["drop", "triviaqa"])
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--num-points", type=int, default=11)
    parser.add_argument("--min-coeff", type=float, default=-0.25)
    parser.add_argument("--max-coeff", type=float, default=1.25)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dataset-cache-dir", type=Path, default=REPO_ROOT / "data" / "hf_cache")
    parser.add_argument("--cache", type=Path, default=OUTPUT_DIR / "merge_schema_losses.npz")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "merge_schema.png")
    parser.add_argument("--geometry-output", type=Path, default=OUTPUT_DIR / "merge_schema_geometry.png")
    parser.add_argument("--manifold-output", type=Path, default=OUTPUT_DIR / "merge_schema_3d.png")
    parser.add_argument("--curvature", type=float, default=0.06)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--from-saved", action="store_true", help="Only plot from --cache; error if it is missing.")
    args = parser.parse_args()

    if args.from_saved and not args.cache.exists():
        raise FileNotFoundError(f"--from-saved requested, but saved losses do not exist: {args.cache}")

    if args.cache.exists() and (args.from_saved or not args.recompute):
        print(f"Loading saved losses from {args.cache}", flush=True)
        data = np.load(args.cache, allow_pickle=True)
        xs, ys, z, tasks = data["xs"], data["ys"], data["z"], data["tasks"].tolist()
        if {"r1", "r2", "theta"}.issubset(data.files):
            geometry = (float(data["r1"]), float(data["r2"]), float(data["theta"]))
        else:
            geometry = geometry_from_adapters(args, tasks)
        if {"merge_x", "merge_y"}.issubset(data.files):
            merge_xy = (float(data["merge_x"]), float(data["merge_y"]))
        else:
            merge_xy = corrected_projection_from_adapters(args, tasks)
    else:
        print("Evaluating loss grid...", flush=True)
        xs, ys, z, _, tasks = evaluate_grid(args, args.cache)
        data = np.load(args.cache, allow_pickle=True)
        geometry = (float(data["r1"]), float(data["r2"]), float(data["theta"]))
        merge_xy = (float(data["merge_x"]), float(data["merge_y"]))
    plot(xs, ys, z, tasks, merge_xy, args.output)
    plot_geometry(xs, ys, z, tasks, geometry, merge_xy, args.geometry_output)
    plot_manifold_3d(xs, ys, args.output, args.manifold_output, args.curvature)


if __name__ == "__main__":
    main()
