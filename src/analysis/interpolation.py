from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse

import numpy as np
import torch
from typing import Dict, List, Optional
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

from src.analysis.plot_utils import plot_interpolation_curve
from src.geometry import SOnManifold
from src.merging import OFTMerging
from src.path import (
    ROOTDIR,
    LLAMA_ADAPTER_PATHS,
    LLAMA_BASE_MODEL_PATH,
    QWEN_ADAPTER_PATHS,
    QWEN_BASE_MODEL_PATH,
)
from src.dataset.dataset_1 import DATASET_1
from src.scripts.compute_fisher import build_loader

LOSS_SUBDIR = "loss"
IMG_SUBDIR = "imgs"

_device = "cuda" if torch.cuda.is_available() else "cpu"
_manifold = SOnManifold()
_merging = OFTMerging(device=_device)


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

        num_blocks, d = end_params.shape
        block_size = int((1 + (1 + 8 * d) ** 0.5) / 2)

        end_skew = _merging.oft_params_to_skew_matrix(end_params, block_size)
        R_end = torch.matrix_exp(end_skew)

        if start_model is None:
            R_start = torch.eye(block_size, dtype=end_params.dtype, device=end_params.device)
            R_start = R_start.unsqueeze(0).expand(num_blocks, -1, -1)
        else:
            start_params = start_model[key]
            start_skew = _merging.oft_params_to_skew_matrix(start_params, block_size)
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
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
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
    if args.model_family == "llama3.1":
        base_model_path = LLAMA_BASE_MODEL_PATH
        adapter_paths = LLAMA_ADAPTER_PATHS
    else:
        base_model_path = QWEN_BASE_MODEL_PATH
        adapter_paths = QWEN_ADAPTER_PATHS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    interpolation_grid = np.linspace(0, 1, args.num_points).tolist()

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float32, device_map=None
    )

    start_model = None  # pretrained = Identity on all SO(n) blocks

    for (task_tag, dataset_path, dataset_name, split, doc_to_text), adapter_path in tqdm(
        zip(DATASET_1, adapter_paths), desc="Interpolating..."
    ):
        print(f"\n[{task_tag}] Loading adapter: {adapter_path}")

        end_model = load_file(f"{adapter_path}/adapter_model.safetensors", device="cpu")

        model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
        model.enable_adapter_layers()
        model.to(device)

        loader = build_loader(
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            split=split,
            doc_to_text=doc_to_text,
            tokenizer=tokenizer,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )

        interpolation_losses = interpolate_model(
            start_model=start_model,
            end_model=end_model,
            interpolation_grid=interpolation_grid,
            model=model,
            loader=loader,
            device=device,
        )

        # Save losss interpolation to plot
        LOSS_DIR = Path(args.save_path) / args.model_family / LOSS_SUBDIR
        IMG_DIR = Path(args.save_path) / args.model_family / IMG_SUBDIR
        LOSS_DIR.mkdir(parents=True, exist_ok=True)
        IMG_DIR.mkdir(parents=True, exist_ok=True)

        loss_path = LOSS_DIR / f"{task_tag}.npy"
        np.save(loss_path, np.array(interpolation_losses))
        print(f"  Saved losses to {loss_path}")

        img_path = IMG_DIR / f"{task_tag}.png"
        plot_interpolation_curve(
            alphas=interpolation_grid,
            losses=interpolation_losses,
            title=f"Loss interpolation: pretrained -> {task_tag}",
            save_path=str(img_path),
        )
        print(f"  Saved plot to {img_path}")


if __name__ == "__main__":
    # Input args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-points", type=int, default=5,
        help="Number of interpolation points between pretrained and adapter.",
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
        "--model-family", type=str, default="llama3.1", choices=["llama3.1", "qwen2.5"],
        help="Model family to use: 'llama3.1' or 'qwen2.5'.",
    )
    args = parser.parse_args()

    main(args)
