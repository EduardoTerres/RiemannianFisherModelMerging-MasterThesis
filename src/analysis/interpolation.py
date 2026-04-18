"""
Loss-curve interpolation between pretrained weights (R=I) and finetuned OFT models.

Two task-vector flavours:
  - standard_task_vectors : xi_t = log(theta_t)  (plain Lie-algebra vectors)
  - fisher_task_vectors   : transported Fisher-weighted vectors F_tilde_t * xi_t
"""

from __future__ import annotations

import os

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Callable, Dict, List, Optional

import torch
import matplotlib.pyplot as plt
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

from src.utils.path import ROOTDIR, OFT_LLAMA_MODELS_DIR
from src.scripts.compute_fisher import (
    build_loader,
    _siqa_text, _csqa_text, _minerva_text, _humaneval_text, _scienceqa_text,
)
from src.merging import OFTMerging
from src.geometry import SOnManifold

from src.analysis.utils import fisher_task_vectors, _is_oft_param_name


def interpolate_oft_params(
    oft_params_t: Tensor,
    s: float,
) -> Tensor:
    """
    Geodesic interpolation theta(s) = exp(s * log(theta_t)) for s in [0, 1].

    Since OFT params store the upper-triangle of log(theta_t), scaling by s gives
    the interpolated skew-symmetric matrix, returned as upper-triangle params.

    Args:
        oft_params_t: (num_blocks, d) OFT parameters of the finetuned model.
        s: interpolation coefficient (0 = pretrained I, 1 = finetuned theta_t).

    Returns:
        (num_blocks, d) OFT parameters of theta(s).
    """
    return s * oft_params_t


def interpolate_oft_params_between(
    oft_params_base: Tensor,
    oft_params_target: Tensor,
    s: float,
    merging: OFTMerging,
    manifold: SOnManifold,
) -> Tensor:
    """
    Left-trivialized geodesic interpolation between two non-identity OFT points.

    theta(s) = theta_base * Exp(s * Log_{theta_base}(theta_target)),  s in [0, 1].
    """
    d = oft_params_base.shape[-1]
    block_size = int((1 + (1 + 8 * d) ** 0.5) / 2)

    S_base = merging.oft_params_to_skew_matrix(oft_params_base, block_size)
    S_target = merging.oft_params_to_skew_matrix(oft_params_target, block_size)

    theta_base = manifold.cayley_exp(S_base)
    theta_target = manifold.cayley_exp(S_target)

    tangent_at_base = manifold.log(theta_base, theta_target)
    theta_s = manifold.exp(theta_base, s * tangent_at_base)
    S_s = manifold.cayley_inverse_log(theta_s)

    return merging.skew_matrix_to_oft_params(S_s)

def plot_loss_curves(
    loss_fn: Callable[[float], float],
    label: str,
    num_points: int = 11,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot loss curve along the geodesic from I (pretrained, s=0) to finetuned theta_t (s=1).

    Args:
        loss_fn: callable (s: float) -> scalar loss. Must inject s-scaled task vectors
                 into the model internally.
        label: curve label shown in the plot.
        num_points: number of uniformly-spaced interpolation steps.
        save_path: if provided, saves the figure; otherwise shows it.
    """
    alphas = torch.linspace(0.0, 1.0, num_points).tolist()
    losses = [
        loss_fn(s)
        for s in tqdm(alphas, desc=f"Interpolating {label}", leave=False)
    ]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(alphas, losses, marker="o", markersize=3, label=label)
    ax.axvline(0.0, color="gray", linewidth=0.8, linestyle=":")
    ax.axvline(1.0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("s  (0 = pretrained I,  1 = finetuned theta_t)")
    ax.set_ylabel("Loss")
    ax.set_title(f"Loss curve -- {label}")
    ax.legend(fontsize=8)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved -> {save_path}")
    else:
        plt.show()
    plt.close(fig)


def make_llm_loss_fn(
    model,
    loader,
    task_vectors: Dict[str, Tensor],
    device: str,
    base_vectors: Optional[Dict[str, Tensor]] = None,
) -> Callable[[float], float]:
    """
    Return `loss_fn(s: float) -> float`.

    For each `s`:
      - if `base_vectors is None`: inject `s * task_vectors` (pretrained -> task),
      - else: inject left-trivialized geodesic interpolation (base -> task).
    """
    # TODO: remove the defaults
    model_params = {name.replace(".default", ""): p for name, p in model.named_parameters() if name.replace(".default", "") in task_vectors}

    merging_local: Optional[OFTMerging] = None
    manifold_local: Optional[SOnManifold] = None
    if base_vectors is not None:
        merging_local = OFTMerging(device=device)
        manifold_local = SOnManifold()

    def loss_fn(s: float) -> float:
        with torch.no_grad():
            for name, vec in task_vectors.items():
                if base_vectors is None:
                    curr = s * vec
                else:
                    assert merging_local is not None
                    assert manifold_local is not None
                    curr = interpolate_oft_params_between(
                        oft_params_base=base_vectors[name],
                        oft_params_target=vec,
                        s=s,
                        merging=merging_local,
                        manifold=manifold_local,
                    )
                model_params[name].data.copy_(curr.to(device))

            total, n = 0.0, 0
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                    labels = input_ids.masked_fill(attention_mask == 0, -100)
                else:
                    labels = input_ids

                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                total += out.loss.item()
                n += 1
        return total / max(n, 1)

    return loss_fn

# MAIN
if __name__ == "__main__":
    BASE_MODEL    = f"{OFT_LLAMA_MODELS_DIR}/Llama-3.1-8B"
    ADAPTERS_DIR  = f"{OFT_LLAMA_MODELS_DIR}/Llama-3.1-8B_OFT_adapters"
    FISHER_DIR    = f"{ROOTDIR}/data/empirical_diagonal_fishers"
    OUTPUT_DIR    = f"{ROOTDIR}/outputs/loss_curves"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Each entry: (tag, dataset_path, dataset_name, split, doc_to_text)
    # Datasets mirror those used in compute_fisher.py / eval harness.
    TASKS = [
        ("socialiqa",   "allenai/social_i_qa",       None,      "validation", _siqa_text),
        ("commonsense", "tau/commonsense_qa",          None,      "validation", _csqa_text),
        ("numinamath",  "HuggingFaceH4/MATH-500",     "default", "test",  _minerva_text),
        ("magicoder",   "evalplus/humanevalplus",      None,      "test",  _humaneval_text),
        ("scienceqa",   "derek-thomas/ScienceQA",      None,      "test", _scienceqa_text),
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    MODE = "fisher"  # "standard" or "fisher"
    LAM  = 1.0
    BASE: Optional[str] = None  # None -> pretrained (identity). Otherwise adapter tag/path.
    # BASE = "outputs/OrthoMerge_Llama-3.1-8B-fisher/merged_adapter/"

    def resolve_adapter_path(base: str) -> str:
        if "/" in base:
            return base
        return f"{ADAPTERS_DIR}/llama3-1_8b_finetune_{base}"

    # Optional non-identity base adapter (for base -> task interpolation)
    base_oft_named: Optional[Dict[str, Tensor]] = None
    if BASE is not None:
        if MODE != "standard":
            raise ValueError("Non-identity BASE currently supports MODE='standard' only.")

        base_adapter_path = resolve_adapter_path(BASE)
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float32,
            device_map=None,
        )
        base_peft = PeftModel.from_pretrained(
            base_model,
            base_adapter_path,
            is_trainable=False,
        )
        if hasattr(base_peft, "enable_adapter_layers"):
            base_peft.enable_adapter_layers()
        base_peft.to(device).eval()

        base_oft_named = {
            name: p.data.clone()
            for name, p in base_peft.named_parameters()
            if _is_oft_param_name(name)
        }
        if not base_oft_named:
            raise RuntimeError(f"No OFT parameters found for BASE adapter: {base_adapter_path}")

        del base_peft, base_model
        torch.cuda.empty_cache()

    # Pass 1 (fisher mode): collect OFT params and Fishers for all tasks jointly
    all_oft_named: List[Dict[str, Tensor]] = []
    all_fisher_dicts: List[Dict[str, Tensor]] = []
    block_size: Optional[int] = None

    if MODE == "fisher":
        merging = OFTMerging(lam=LAM, device=device)
        print("Pre-loading OFT params and Fishers for all tasks...")
        for tag, *_ in TASKS:
            adapter_path = f"{ADAPTERS_DIR}/llama3-1_8b_finetune_{tag}"
            fisher_path  = f"{FISHER_DIR}/llama3-1_8b_finetune_{tag}.safetensors"

            _base = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL, torch_dtype=torch.float32, device_map=None
            )
            _model = PeftModel.from_pretrained(_base, adapter_path, is_trainable=False)
            if hasattr(_model, "enable_adapter_layers"):
                _model.enable_adapter_layers()

            oft_named = {name.replace(".default", ""): p.data.clone().to(device)
                         for name, p in _model.named_parameters()
                         if _is_oft_param_name(name)}
            del _model, _base
            torch.cuda.empty_cache()

            if not oft_named:
                raise RuntimeError(f"No OFT params found for adapter: {tag}")
            if block_size is None:
                d = next(iter(oft_named.values())).shape[-1]
                block_size = int((1 + (1 + 8 * d) ** 0.5) / 2)

            fishers = merging.load_fishers([fisher_path])[0]
            all_oft_named.append(oft_named)
            all_fisher_dicts.append(fishers)

        T = len(TASKS)
        alphas = [0.5] * T
        all_task_vectors = fisher_task_vectors(
            all_oft_named, all_fisher_dicts, alphas, merging, LAM,
        )
        tag_to_task_vectors = {tag: tv for (tag, *_), tv in zip(TASKS, all_task_vectors)}

    # Pass 2: load each model and evaluate interpolation curve
    for tag, dataset_path, dataset_name, split, doc_to_text in TASKS:
        adapter_path = f"{ADAPTERS_DIR}/llama3-1_8b_finetune_{tag}"

        print(f"\n=== {tag} ===")
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.float32, device_map=None
        )
        model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
        if hasattr(model, "enable_adapter_layers"):
            model.enable_adapter_layers()
        model.to(device).eval()

        oft_named = {name.replace(".default", ""): p.data.clone()
                     for name, p in model.named_parameters()
                     if _is_oft_param_name(name)}

        if not oft_named:
            raise RuntimeError(
                "No OFT parameters found in model.named_parameters(). "
                "Check adapter loading and OFT key filtering."
            )

        if base_oft_named is not None:
            missing = set(oft_named) - set(base_oft_named)
            if missing:
                raise RuntimeError(
                    f"BASE adapter is missing {len(missing)} OFT keys required by task '{tag}'."
                )

        task_vectors = tag_to_task_vectors[tag] if MODE == "fisher" else oft_named

        loader = build_loader(
            dataset_path, dataset_name, split, doc_to_text,
            tokenizer, num_samples=128, batch_size=64, max_length=32,
        )

        plot_loss_curves(
            loss_fn=make_llm_loss_fn(
                model,
                loader,
                task_vectors,
                device,
                base_vectors=base_oft_named,
            ),
            label=f"{tag} ({MODE})",
            num_points=7,
            save_path=f"{OUTPUT_DIR}/{tag}_{MODE}.png",
        )

        del model, base
        torch.cuda.empty_cache()
