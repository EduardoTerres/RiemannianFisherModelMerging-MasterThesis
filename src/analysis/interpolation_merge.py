from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.dataset.dataset_1 import DATASET_1_TRAIN as DATASET_1, build_loader
from src.merging import OFTMerging
from src.paths import MODEL_FAMILIES, ROOTDIR
from src.utils import parse_device

LOSS_SUBDIR = "loss"
IMG_SUBDIR = "imgs"

def apply_weights(
    model: torch.nn.Module,
    weights: Dict[str, torch.Tensor],
    coeff: float,
    device: str,
) -> None:
    param_dict = dict(model.named_parameters())
    n_matched = 0
    for key, val in weights.items():
        peft_key = key.replace(".weight", ".default.weight")
        target = param_dict.get(peft_key, param_dict.get(key))
        if target is not None:
            target.data.copy_((coeff * val).to(device))
            n_matched += 1

    if n_matched == 0:
        raise RuntimeError(f"No weight keys matched. Sample: {list(weights.keys())[:3]}")


def logl_loss(
    model: torch.nn.Module,
    loader,
    device: str,
) -> float:
    total_loss = 0.0
    n_batches = 0
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Evaluating"):
            loss = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            ).loss
            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


def plot_losses(
    coeffs: List[float],
    task_losses: Dict[str, List[float]],
    save_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for task, losses in task_losses.items():
        ax.plot(coeffs, losses, marker="o", linewidth=2, markersize=4, label=task)
    ax.set_xlabel("coefficient on merged task vector")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main(args: argparse.Namespace) -> None:
    if args.device != "cuda":
        raise ValueError("interpolation_merge.py must run on CUDA. Use --device cuda.")
    if not torch.cuda.is_available():
        raise ValueError("CUDA requested but torch.cuda.is_available() is False.")
    if args.merge_method != "gradients":
        raise ValueError("interpolation_merge.py uses OFTMerging; use --merge-method gradients.")

    for family_name in args.model_family:
        _run(family_name, args)


def _run(family_name: str, args: argparse.Namespace) -> None:
    model_family = MODEL_FAMILIES[family_name]
    merger = OFTMerging(lam=args.lam, alphas=args.alphas, device=args.device)

    merged_weights = merger.merge(
        adapter_paths=model_family.adapter_paths,
        fisher_paths=model_family.fisher_paths if "fisher" in args.merge_mode else None,
        mode=args.merge_mode,
    )
    coeffs = np.linspace(0, args.n, args.num_points).tolist()

    tokenizer = AutoTokenizer.from_pretrained(model_family.base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        model_family.base_model_path, torch_dtype=torch.float32, device_map=None
    )
    model = PeftModel.from_pretrained(base, model_family.adapter_paths[0], is_trainable=False)
    model.enable_adapter_layers()
    model.to(args.device)
    model.eval()

    task_loaders = {}
    for task_tag, dataset_path, dataset_name, split, doc_to_text in DATASET_1:
        task_loaders[task_tag] = build_loader(
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            split=split,
            doc_to_text=doc_to_text,
            tokenizer=tokenizer,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )

    task_losses = {task: [] for task in task_loaders}
    sum_losses = []
    with tqdm(coeffs, desc="Interpolating merged direction") as pbar:
        for coeff in pbar:
            apply_weights(model, merged_weights, coeff, args.device)
            losses = {}
            for task, loader in task_loaders.items():
                losses[task] = logl_loss(model, loader, args.device)
                task_losses[task].append(losses[task])
            sum_losses.append(sum(losses.values()))
            pbar.set_postfix(coeff=f"{coeff:.3f}", sum_loss=f"{sum_losses[-1]:.4f}")

    loss_dir = Path(args.save_path) / family_name / args.merge_mode / LOSS_SUBDIR
    img_dir = Path(args.save_path) / family_name / args.merge_mode / IMG_SUBDIR
    loss_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        loss_dir / "merged_direction_losses.npz",
        coeffs=np.array(coeffs),
        sum_losses=np.array(sum_losses),
        **{task: np.array(losses) for task, losses in task_losses.items()},
    )
    plot_losses(
        coeffs,
        task_losses,
        img_dir / "all_losses.png",
        f"{family_name}: base -> merged ({args.merge_mode})",
    )
    plot_losses(
        coeffs,
        {"sum": sum_losses},
        img_dir / "sum_loss.png",
        f"{family_name}: summed loss ({args.merge_mode})",
    )
    print(f"Saved losses to {loss_dir / 'merged_direction_losses.npz'}")
    print(f"Saved plots to {img_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-points", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--save-path", type=str, default=f"{ROOTDIR}/outputs/interpolation_merge")
    parser.add_argument(
        "--model-family",
        nargs="+",
        default=list(MODEL_FAMILIES),
        choices=list(MODEL_FAMILIES),
    )
    parser.add_argument("--merge-method", type=str, default="gradients", choices=["gradients"])
    parser.add_argument(
        "--merge-mode",
        type=str,
        default="standard",
        choices=["standard", "diagonal_fisher", "fisher"],
    )
    parser.add_argument("--lam", type=float, default=0.0)
    parser.add_argument("--alphas", type=float, nargs="+", default=None)
    parser.add_argument("--n", type=float, default=3.0, help="Max coefficient; 1 is the merge.")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    args.device = parse_device(args.device)
    main(args)
