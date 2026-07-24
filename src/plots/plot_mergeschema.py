from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from peft import PeftConfig, PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "outputs" / "mergeschema"
sys.path.insert(0, str(REPO_ROOT))

from src.dataset.dataset_3 import DATASET_3_TEST, DATASET_3_TRAIN, build_loader
from src.merging import OFTMerging
from src.paths import MODEL_FAMILIES_D3

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


def task_specs(names: list[str]):
    specs = {task: spec for task, *spec in DATASET_3_TEST}
    return [(task, *specs[task]) for task in names]


def adapter_paths_for_tasks(model_name: str, names: list[str]) -> list[str]:
    family = MODEL_FAMILIES_D3[model_name]
    adapter_by_task = {task: family.adapter_paths[i] for i, (task, *_) in enumerate(DATASET_3_TRAIN)}
    return [adapter_by_task[name] for name in names]


def peft_config_for_adapter(adapter_path: str, candidate_paths: list[str]) -> PeftConfig | None:
    adapter_dir = Path(adapter_path)
    if (adapter_dir / "adapter_config.json").exists():
        return None
    for candidate in candidate_paths:
        candidate_dir = Path(candidate)
        if (candidate_dir / "adapter_config.json").exists():
            print(f"Reusing adapter config from {candidate_dir}", flush=True)
            return PeftConfig.from_pretrained(candidate_dir)
    raise FileNotFoundError(
        f"No adapter_config.json found for {adapter_path} or any candidate adapter path."
    )


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


def merged_at(
    merger: OFTMerging,
    weights: list[dict[str, torch.Tensor]],
    merge_indices: tuple[int, int],
    x: float,
    y: float,
) -> dict[str, torch.Tensor]:
    alphas = torch.full((len(weights),), min(x, y), device=merger.device)
    alphas[merge_indices[0]] = x
    alphas[merge_indices[1]] = y
    return {
        key: merger.merge_formula([w[key] for w in weights], mode="standard", alphas=alphas)
        for key in weights[0]
    }


def flatten_weights(weights: dict[str, torch.Tensor], keys) -> torch.Tensor:
    return torch.cat([weights[key].float().cpu().flatten() for key in keys])


def orthomerge_coefficients(weights: list[dict[str, torch.Tensor]]) -> tuple[float, float]:
    keys = list(weights[0])
    vecs = [flatten_weights(weight, keys) for weight in weights]
    c = sum(torch.linalg.vector_norm(vec) for vec in vecs) / torch.linalg.vector_norm(sum(vecs)).clamp(min=1e-8)
    alpha = float(c / len(vecs))
    return alpha, alpha


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


def orthomerge_coefficients_from_adapters(args: argparse.Namespace, names: list[str]) -> tuple[float, float]:
    merger = OFTMerging(device="cpu")
    weights = merger.load_weights(adapter_paths_for_tasks(args.model_name, names))
    return orthomerge_coefficients(weights)


def require_cuda(device: str) -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for loss computation; rerun on a GPU node or use --from_saved.")
    if not device.startswith("cuda"):
        raise RuntimeError(f"Loss computation must run on CUDA, got --device {device!r}.")
    return device


def evaluate_grid(args: argparse.Namespace, cache: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    args.device = require_cuda(args.device)
    family = MODEL_FAMILIES_D3[args.model_name]
    names = args.tasks
    merge_names = [task for task, *_ in DATASET_3_TRAIN]
    merge_indices = (merge_names.index(names[0]), merge_names.index(names[1]))
    eval_specs = DATASET_3_TEST
    eval_names = [task for task, *_ in eval_specs]
    adapter_paths = adapter_paths_for_tasks(args.model_name, merge_names)

    print(f"Using device: {args.device}", flush=True)
    print(f"Merge axes: {names[0]}, {names[1]} | samples per eval task: {args.num_samples}", flush=True)
    print(f"Merged tasks ({len(merge_names)}): {', '.join(merge_names)}", flush=True)
    print("Non-axis merge coefficients use min(x, y).", flush=True)
    print(f"Loss tasks ({len(eval_names)}): {', '.join(eval_names)}", flush=True)
    print(f"Grid: {args.num_points}x{args.num_points} from {args.min_coeff} to {args.max_coeff}", flush=True)
    print("Loading tokenizer and fixed eval loaders...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(family.base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    loaders = [
        build_loader(path, dset, split, formatter, tokenizer, args.num_samples, args.batch_size, args.max_length, task, str(args.dataset_cache_dir))
        for task, path, dset, split, formatter in eval_specs
    ]

    print("Loading selected adapters and base model...", flush=True)
    merger = OFTMerging(alphas=[1.0] * len(merge_names), device=args.device)
    weights = merger.load_weights(adapter_paths)
    merge_x, merge_y = orthomerge_coefficients(weights)
    base = AutoModelForCausalLM.from_pretrained(
        family.base_model_path,
        torch_dtype=torch.bfloat16 if "cuda" in args.device else torch.float32,
        trust_remote_code=True,
    )
    peft_config = peft_config_for_adapter(adapter_paths[0], [*adapter_paths, *family.adapter_paths])
    model = PeftModel.from_pretrained(base, adapter_paths[0], is_trainable=False, config=peft_config)
    model.enable_adapter_layers()
    model.to(args.device).eval()

    xs = np.linspace(args.min_coeff, args.max_coeff, args.num_points)
    ys = np.linspace(args.min_coeff, args.max_coeff, args.num_points)
    task_losses = np.zeros((len(eval_names), len(ys), len(xs)))
    grid = [(row, col, x, y) for row, y in enumerate(ys) for col, x in enumerate(xs)]
    for row, col, x, y in tqdm(grid, desc="grid points"):
        apply_weights(model, merged_at(merger, weights, merge_indices, float(x), float(y)), args.device)
        for idx, (name, loader) in enumerate(zip(eval_names, loaders, strict=True)):
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
        merge_tasks=np.array(merge_names),
        merge_indices=np.array(merge_indices),
        non_axis_alpha_rule=np.array("min(x, y)"),
        eval_tasks=np.array(eval_names),
        merge_x=merge_x,
        merge_y=merge_y,
    )
    print(f"Saved losses to {cache}", flush=True)
    return xs, ys, z, task_losses, names, eval_names


def marker_handles(tasks: list[str]) -> list[Line2D]:
    return [
        Line2D([0], [0], color="black", marker="o", linestyle="None", markersize=9, label="Pretrained"),
        Line2D([0], [0], color="black", marker="^", linestyle="None", markersize=10, label=latex_escape(tasks[0].capitalize())),
        Line2D([0], [0], color="black", marker="s", linestyle="None", markersize=9, label=latex_escape(tasks[1].capitalize())),
        Line2D([0], [0], color="black", marker="D", linestyle="None", markersize=9, label=r"\textsc{Lie sum}"),
        Line2D([0], [0], color="black", marker="*", linestyle="None", markersize=14, label=r"\textsc{OrthoMerge}"),
    ]


def save_legend(tasks: list[str], output: Path) -> None:
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
    tasks: list[str],
    merge_xy: tuple[float, float],
    output: Path,
    plane_extent: tuple[float, float],
    num_loss_tasks: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6.6))
    add_tangent_background(ax, plane_extent)
    heatmap = ax.contourf(xs, ys, z, levels=40, cmap=WATER_CMAP, zorder=1)
    fig.colorbar(
        heatmap,
        ax=ax,
        pad=0.02,
        label=rf"sum of {num_loss_tasks} task losses",
    )
    points = {
        "Pretrained": ((0, 0), "o", 110),
        latex_escape(tasks[0].capitalize()): ((1, 0), "^", 125),
        latex_escape(tasks[1].capitalize()): ((0, 1), "s", 110),
        r"\textsc{Gradients} standard": ((1, 1), "D", 115),
        r"\textsc{OrthoMerge}": (merge_xy, "*", 230),
    }
    for label, (xy, marker, size) in points.items():
        ax.scatter(*xy, s=size, marker=marker, color="black", linewidth=1.2, label=label, zorder=3)
        dx, dy = (0.035, -0.10) if label == "Pretrained" else (0.035, 0.035)
        ax.text(xy[0] + dx, xy[1] + dy, label, fontsize=13, weight="bold", zorder=4, color="white")
    for end in [(1, 0), (0, 1), (1, 1), merge_xy]:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--model-name", default="llama3.1", choices=list(MODEL_FAMILIES_D3))
    parser.add_argument("--tasks", nargs=2, default=["coqa", "triviaqa"])
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--num-points", type=int, default=20)
    parser.add_argument("--min-coeff", type=float, default=-1.0)
    parser.add_argument("--max-coeff", type=float, default=1.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dataset-cache-dir", type=Path, default=REPO_ROOT / "data" / "hf_cache")
    parser.add_argument("--cache", type=Path, default=OUTPUT_DIR / "merge_schema_losses.npz")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "merge_schema.png")
    parser.add_argument("--plane-min", type=float, default=-1.5)
    parser.add_argument("--plane-max", type=float, default=1.5)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--from-saved", action="store_true", help="Only plot from --cache; error if it is missing.")
    args = parser.parse_args()

    if args.from_saved and not args.cache.exists():
        raise FileNotFoundError(f"--from-saved requested, but saved losses do not exist: {args.cache}")

    if args.from_saved:
        print(f"Loading saved losses from {args.cache}", flush=True)
        data = np.load(args.cache, allow_pickle=True)
        xs, ys, z, tasks = data["xs"], data["ys"], data["z"], data["tasks"].tolist()
        eval_tasks = data["eval_tasks"].tolist() if "eval_tasks" in data.files else tasks
        if {"merge_x", "merge_y"}.issubset(data.files):
            merge_xy = (float(data["merge_x"]), float(data["merge_y"]))
        else:
            merge_xy = orthomerge_coefficients_from_adapters(args, tasks)
    else:
        print("Evaluating loss grid...", flush=True)
        xs, ys, z, _, tasks, eval_tasks = evaluate_grid(args, args.cache)
        data = np.load(args.cache, allow_pickle=True)
        merge_xy = (float(data["merge_x"]), float(data["merge_y"]))
    plane_extent = (args.plane_min, args.plane_max)
    plot(xs, ys, z, tasks, merge_xy, args.output, plane_extent, len(eval_tasks))


if __name__ == "__main__":
    main()
