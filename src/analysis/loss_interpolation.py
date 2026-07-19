from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json

import numpy as np
import torch
from typing import Dict, List, Optional
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel
from tqdm import tqdm

from src.analysis.plot_utils import (
    plot_interpolation_curve,
    plot_joint_normalized_interpolation_curves,
)
from src.geometry import SOnManifold
from src.merging import OFTMerging
from src.paths import (
    ROOTDIR,
    MODEL_FAMILIES,
)
from src.dataset.dataset_3 import DATASET_3_TEST, DATASET_3_TRAIN, build_loader
from src.utils import parse_device

LOSS_SUBDIR = "loss"
IMG_SUBDIR = "imgs"

_device = "cuda" if torch.cuda.is_available() else "cpu"
_manifold = SOnManifold()
_merging = OFTMerging(device=_device)


def load_adapter_config(adapter_path: str, adapter_paths: list[str]) -> PeftConfig | None:
    """Use a sibling PEFT config when an adapter directory only contains weights."""
    adapter_dir = Path(adapter_path)
    if (adapter_dir / "adapter_config.json").exists():
        return None

    for candidate in adapter_paths:
        candidate_dir = Path(candidate)
        if (candidate_dir / "adapter_config.json").exists():
            print(f"  Reusing adapter config from {candidate_dir}")
            return PeftConfig.from_pretrained(candidate_dir)

    return None


def interpolate(
    start_model: Optional[Dict[str, torch.Tensor]],
    end_model: Dict[str, torch.Tensor],
    alpha: float,
) -> Dict[str, torch.Tensor]:
    """Geodesic interpolation on SO(n) between two OFT adapter weight dicts.

    Args:
        start_model: OFT weights {key: (num_blocks, d)}. None = pretrained (Identity on all blocks).
        end_model:   OFT weights {key: (num_blocks, d)} for the fine-tuned target.
        alpha:       Interpolation position in [0, 1]. 0 = start, 1 = end.

    Returns:
        Dict {key: (num_blocks, d)} with geodesically interpolated OFT parameters.
    """
    interpolated: Dict[str, torch.Tensor] = {}

    for key, end_params in end_model.items():
        is_oft = "oft_r" in key or ("oft_" in key.lower() and "classifier" not in key.lower())
        if not is_oft:
            interpolated[key] = end_params
            continue

        num_blocks, son_dimension = end_params.shape
        block_size = int((1 + (1 + 8 * son_dimension) ** 0.5) / 2)

        end_skew = _merging.oft_params_to_skew_matrix(end_params, son_dimension)
        R_end = torch.matrix_exp(end_skew)

        if start_model is None:
            R_start = torch.eye(block_size, dtype=end_params.dtype, device=end_params.device)
            R_start = R_start.unsqueeze(0).expand(num_blocks, -1, -1)
        else:
            start_params = start_model[key]
            start_skew = _merging.oft_params_to_skew_matrix(start_params, son_dimension)
            R_start = torch.matrix_exp(start_skew)

        tangent = _manifold.exact_log(R_start, R_end)
        R_interp = _manifold.exact_exp(R_start, alpha * tangent)
        omega_interp = _manifold.exact_log(R_start, R_interp)
        interpolated[key] = _merging.skew_matrix_to_oft_params(omega_interp)

    return interpolated


def naive_interpolate(
    end_model: Dict[str, torch.Tensor],
    alpha: float,
) -> Dict[str, torch.Tensor]:
    def _is_oft(k: str) -> bool:
        return "oft_r" in k or "oft_" in k.lower()

    return {k: alpha * v if _is_oft(k) else v for k, v in end_model.items()}


def logl_loss(
    model: torch.nn.Module,
    weights: Dict[str, torch.Tensor],
    loader,
    device: str,
) -> float:
    """Apply interpolated OFT weights to a PeftModel and return average cross-entropy loss.

    Args:
        model:   PeftModel with OFT adapter layers already loaded.
        weights: OFT weight dict {param_name: (num_blocks, d)}.
        loader:  DataLoader yielding batches with 'input_ids' and 'attention_mask'.
        device:  Torch device string.

    Returns:
        Scalar average cross-entropy loss over the loader.
    """
    param_dict = dict(model.named_parameters())
    n_matched = 0
    for key, val in weights.items():
        # named_parameters inserts the adapter name: oft_R.weight -> oft_R.default.weight
        peft_key = key.replace(".weight", ".default.weight")
        target = param_dict.get(peft_key)
        if target is None:
            target = param_dict.get(key)
        if target is not None:
            target.data.copy_(val.to(device))
            n_matched += 1

    if n_matched == 0:
        raise RuntimeError(
            f"No weight keys matched. "
            f"\n  weights sample:     {list(weights.keys())[:3]}"
            f"\n  param_dict sample:  {list(param_dict.keys())[:3]}"
        )

    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


def interpolate_model(
    start_model: Optional[Dict[str, torch.Tensor]],
    end_model: Dict[str, torch.Tensor],
    interpolation_grid: List[float],
    model: torch.nn.Module,
    loader,
    device: str,
) -> List[float]:
    """Sweep an interpolation grid between two models and collect losses at each point.

    Args:
        start_model:        OFT weights {key: (num_blocks, d)}, or None for pretrained.
        end_model:          OFT weights {key: (num_blocks, d)} for the fine-tuned target.
        interpolation_grid: Sequence of alpha values in [0, 1].
        model:              PeftModel used for loss evaluation.
        loader:             DataLoader for the evaluation dataset.
        device:             Torch device string.

    Returns:
        List of scalar losses, one per alpha in interpolation_grid.
    """
    interpolation_losses = []
    with tqdm(interpolation_grid, desc="Interpolating...") as pbar:
        for alpha in pbar:
            interpolated_weights = interpolate(start_model, end_model, alpha=alpha)
            loss = logl_loss(model, interpolated_weights, loader, device)
            pbar.set_postfix(alpha=f"{alpha:.2f}", loss=f"{loss:.4f}")
            interpolation_losses.append(loss)
    return interpolation_losses


def main(args: argparse.Namespace):
    for family_name in args.model_family:
        _run(family_name, args)


def resolve_device(device: str) -> str:
    if device.lower().strip() == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return parse_device(device)


def save_losses_json(
    path: Path,
    task: str,
    alphas: list[float],
    losses: list[float],
    args: argparse.Namespace,
) -> None:
    payload = {
        "task": task,
        "dataset_split": args.dataset_split,
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "alphas": alphas,
        "losses": losses,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_losses_json(path: Path) -> tuple[list[float], list[float]]:
    payload = json.loads(path.read_text())
    return payload["alphas"], payload["losses"]


def save_interpolation_outputs(
    family_name: str,
    task_tag: str,
    interpolation_grid: list[float],
    interpolation_losses: list[float],
    args: argparse.Namespace,
) -> None:
    loss_dir = Path(args.save_path) / family_name / LOSS_SUBDIR
    img_dir = Path(args.save_path) / family_name / IMG_SUBDIR
    loss_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    loss_path = loss_dir / f"{task_tag}.npy"
    json_path = loss_dir / f"{task_tag}.json"
    np.save(loss_path, np.array(interpolation_losses))
    save_losses_json(json_path, task_tag, interpolation_grid, interpolation_losses, args)
    print(f"  Saved losses to {loss_path}")
    print(f"  Saved losses JSON to {json_path}")

    save_interpolation_plot(family_name, task_tag, interpolation_grid, interpolation_losses, args)


def save_interpolation_plot(
    family_name: str,
    task_tag: str,
    interpolation_grid: list[float],
    interpolation_losses: list[float],
    args: argparse.Namespace,
) -> None:
    img_dir = Path(args.save_path) / family_name / IMG_SUBDIR
    img_dir.mkdir(parents=True, exist_ok=True)
    img_path = img_dir / f"{task_tag}.png"
    plot_interpolation_curve(
        alphas=interpolation_grid,
        losses=interpolation_losses,
        title=f"Loss interpolation: pretrained -> {task_tag}",
        save_path=str(img_path),
    )
    print(f"  Saved plot to {img_path}")


def save_joint_interpolation_plots(
    family_name: str,
    task_tags: list[str],
    args: argparse.Namespace,
) -> None:
    loss_dir = Path(args.save_path) / family_name / LOSS_SUBDIR
    img_dir = Path(args.save_path) / family_name / IMG_SUBDIR
    series = []
    missing = []

    for task_tag in task_tags:
        json_path = loss_dir / f"{task_tag}.json"
        if not json_path.exists():
            missing.append(task_tag)
            continue
        alphas, losses = load_losses_json(json_path)
        series.append((task_tag, alphas, losses))

    if missing:
        print(f"  Skipping missing tasks in joint plot: {', '.join(missing)}")
    if not series:
        return

    family_tag = "llama" if family_name.startswith("llama") else "qwen" if family_name.startswith("qwen") else family_name
    for alpha_end, suffix in ((1.0, "0_1"), (2.0, "0_2")):
        save_stem = img_dir / f"loss_interpolation_{family_tag}_{suffix}"
        plot_joint_normalized_interpolation_curves(
            series=series,
            alpha_end=alpha_end,
            title=f"{family_name} normalized loss interpolation ({alpha_end:g})",
            save_stem=str(save_stem),
        )
        print(f"  Saved joint plots to {save_stem}.png and {save_stem}.pdf")


def _run(family_name: str, args: argparse.Namespace):
    model_family = MODEL_FAMILIES[family_name]
    base_model_path = model_family.base_model_path
    adapter_paths = model_family.adapter_paths
    dataset_specs = DATASET_3_TEST if args.dataset_split == "test" else DATASET_3_TRAIN

    device = args.device
    interpolation_grid = np.linspace(
        args.interpolation_start,
        args.interpolation_end,
        args.num_points,
    ).tolist()
    loss_dir = Path(args.save_path) / family_name / LOSS_SUBDIR
    task_tags = [spec[0] for spec in dataset_specs]
    adapter_by_task = {
        task_tag: adapter_path
        for (task_tag, *_), adapter_path in zip(dataset_specs, adapter_paths, strict=True)
    }

    specs_to_compute = []
    for spec in dataset_specs:
        task_tag = spec[0]
        json_path = loss_dir / f"{task_tag}.json"
        if json_path.exists() and not args.force_compute:
            print(f"\n---[{task_tag}]--- Found cached losses: {json_path}")
            if args.plots:
                alphas, losses = load_losses_json(json_path)
                save_interpolation_plot(family_name, task_tag, alphas, losses, args)
            continue

        if json_path.exists() and args.force_compute:
            print(
                f"\n---[{task_tag}]--- Recomputing cached losses because "
                f"--force-compute is set: {json_path}"
            )
        specs_to_compute.append(spec)

    if not specs_to_compute:
        print(f"\n[{family_name}] All interpolation JSON files already exist. Nothing to compute.")
        save_joint_interpolation_plots(family_name, task_tags, args)
        return

    dataset_specs = specs_to_compute
    tasks_to_compute = ", ".join(spec[0] for spec in dataset_specs)
    print(f"\n[{family_name}] Computing interpolation losses for: {tasks_to_compute}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float32, device_map=None
    )

    start_model = None  # pretrained = Identity on all SO(n) blocks

    for task_tag, dataset_path, dataset_name, split, doc_to_text_fn in tqdm(
        dataset_specs, desc="Interpolating..."
    ):
        adapter_path = adapter_by_task[task_tag]
        print(f"\n---[{task_tag}]--- Loading adapter: {adapter_path}")

        end_model = load_file(f"{adapter_path}/adapter_model.safetensors", device="cpu")
        peft_config = load_adapter_config(adapter_path, adapter_paths)

        model = PeftModel.from_pretrained(
            base,
            adapter_path,
            is_trainable=False,
            config=peft_config,
        )
        model.enable_adapter_layers()
        model.to(device)

        loader = build_loader(
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            split=split,
            doc_to_text_fn=doc_to_text_fn,
            tokenizer=tokenizer,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            max_length=args.max_length,
            task=task_tag,
            cache_dir=str(args.dataset_cache_dir) if args.dataset_cache_dir else None,
        )

        interpolation_losses = interpolate_model(
            start_model=start_model,
            end_model=end_model,
            interpolation_grid=interpolation_grid,
            model=model,
            loader=loader,
            device=device,
        )

        save_interpolation_outputs(
            family_name,
            task_tag,
            interpolation_grid,
            interpolation_losses,
            args,
        )

    save_joint_interpolation_plots(family_name, task_tags, args)


if __name__ == "__main__":
    # Input args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-points", type=int, default=5,
        help="Number of interpolation points between pretrained and adapter.",
    )
    parser.add_argument(
        "--interpolation-start", type=float, default=0.0,
        help="First alpha value in the interpolation grid.",
    )
    parser.add_argument(
        "--interpolation-end", type=float, default=1.0,
        help="Last alpha value in the interpolation grid.",
    )
    parser.add_argument(
        "--num-samples", type=int, default=512,
        help="Number of dataset samples used to evaluate loss at each interpolation point.")
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="DataLoader batch size for loss evaluation.")
    parser.add_argument(
        "--max-length", type=int, default=1024,
        help="Maximum token length for input sequences.")
    parser.add_argument(
        "--save-path", type=str, default=f"{ROOTDIR}/outputs/interpolation",
        help="Directory where interpolation plot PNGs are saved.",
    )
    parser.add_argument(
        "--dataset-split", choices=["train", "test"], default="test",
        help="Dataset_3 split specs to use for loss evaluation.",
    )
    parser.add_argument(
        "--dataset-cache-dir",
        type=Path,
        default=ROOTDIR / "data" / "hf_cache",
        help="Hugging Face datasets cache directory.",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Regenerate plots from saved JSON losses when available.",
    )
    parser.add_argument(
        "--force-compute",
        action="store_true",
        help="Recompute interpolation losses even when the task JSON already exists.",
    )
    parser.add_argument(
        "--model-family", nargs="+", default=list(MODEL_FAMILIES), choices=list(MODEL_FAMILIES),
        help="One or more model families to process.",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device to use for interpolation and loss evaluation (auto, gpu/cuda, cpu, or mps).",
    )
    args = parser.parse_args()
    args.device = resolve_device(args.device)

    main(args)
