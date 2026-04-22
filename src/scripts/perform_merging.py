import argparse
import os
import sys
import shutil

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.merging import OFTMerging
from src.utils import parse_device
from src.paths import MODEL_FAMILIES


def parse_args():
    parser = argparse.ArgumentParser(description="Merge OFT adapters with Riemannian merging.")
    parser.add_argument(
        "--model_family", type=str, choices=list(MODEL_FAMILIES), required=True,
    )
    parser.add_argument(
        "--merge_mode", type=str, choices=["standard", "diagonal_fisher", "fisher"],
        help="Merging strategy: 'standard' (weighted average), 'diagonal_fisher' (element-wise Fisher weighting), or 'linear_system' (full transported-Fisher solve).",  # noqa: E501
    )
    parser.add_argument(
        "--lam", type=float, default=1.0,
        help="Regularisation coefficient lambda used in fisher merge modes.",
    )
    parser.add_argument(
        "--alphas", type=float, nargs="+", default=None,
        help="Per-task weights (must match number of adapters). Defaults to uniform.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/merged",
        help="Directory where the merged adapter and optionally the full model are saved.",
    )
    parser.add_argument(
        "--save_merged_model", action="store_true",
        help="If set, bake the merged adapter into the base model and save the full merged model.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use (e.g., 'gpu', 'cpu').",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.device = parse_device(args.device)

    model_family = MODEL_FAMILIES[args.model_family]
    base_model_path = model_family.base_model_path
    adapter_paths = model_family.adapter_paths
    fisher_paths = model_family.fisher_paths

    merging = OFTMerging(lam=args.lam, alphas=args.alphas, device=args.device)
    merged_weights = merging.merge(
        adapter_paths=adapter_paths,
        fisher_paths=fisher_paths,
        mode=args.merge_mode,
    )

    merged_adapter_dir = os.path.join(args.output_dir, "merged_adapter")
    os.makedirs(merged_adapter_dir, exist_ok=True)
    save_file(merged_weights, os.path.join(merged_adapter_dir, "adapter_model.safetensors"))

    config_src = os.path.join(adapter_paths[0], "adapter_config.json")
    if os.path.exists(config_src):
        shutil.copy(config_src, os.path.join(merged_adapter_dir, "adapter_config.json"))

    print(f"Saved merged adapter to: {merged_adapter_dir}")

    if not args.save_merged_model:
        sys.exit(0)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        device_map=args.device,
    )

    merged_model = PeftModel.from_pretrained(base_model, merged_adapter_dir).merge_and_unload()

    model_save_dir = os.path.join(args.output_dir, "merged_model")
    os.makedirs(model_save_dir, exist_ok=True)
    merged_model.save_pretrained(model_save_dir)
    AutoTokenizer.from_pretrained(base_model_path).save_pretrained(model_save_dir)
    print(f"Saved merged model to: {model_save_dir}")


if __name__ == "__main__":
    main()
