import argparse
import os
import sys
import shutil

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.merging import OFTMerging


def parse_args():
    parser = argparse.ArgumentParser(description="Merge OFT adapters with Riemannian merging.")
    parser.add_argument("--language_model_name", type=str, required=True,
                        help="Base LLM name or local path (e.g. 'models/Llama-3.1-8B/').")
    parser.add_argument("--adapter_paths", type=str, nargs="+", required=True,
                        help="Paths to OFT adapter directories to merge.")
    parser.add_argument("--fisher_paths", type=str, nargs="+", default=None,
                        help="Paths to diagonal Fisher dicts, one per adapter. Required for fisher merge modes.")
    parser.add_argument(
        "--merge_mode",
        type=str,
        choices=["plain", "diagonal_fisher", "linear_system"],
        default="plain",
        help="Merging strategy: 'plain' (weighted average), 'diagonal_fisher' (element-wise Fisher weighting), or 'linear_system' (full transported-Fisher solve).",
    )
    parser.add_argument("--lam", type=float, default=1.0,
                        help="Regularisation coefficient lambda used in fisher merge modes.")
    parser.add_argument("--alpha", type=float, nargs="+", default=None,
                        help="Per-task weights (must match number of adapters). Defaults to uniform.")
    parser.add_argument("--output_dir", type=str, default="outputs/merged",
                        help="Directory where the merged adapter and optionally the full model are saved.")
    parser.add_argument("--save_merged_model", action="store_true",
                        help="If set, bake the merged adapter into the base model and save the full merged model.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU id to use (-1 for CPU).")
    return parser.parse_args()


def main():
    args = parse_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"

    merging = OFTMerging(lam=args.lam, alpha=args.alpha, device=device)
    merged_weights = merging.merge(
        adapter_paths=args.adapter_paths,
        fisher_paths=args.fisher_paths,
        mode=args.merge_mode,
    )

    merged_adapter_dir = os.path.join(args.output_dir, "merged_adapter")
    os.makedirs(merged_adapter_dir, exist_ok=True)
    save_file(merged_weights, os.path.join(merged_adapter_dir, "adapter_model.safetensors"))

    config_src = os.path.join(args.adapter_paths[0], "adapter_config.json")
    if os.path.exists(config_src):
        shutil.copy(config_src, os.path.join(merged_adapter_dir, "adapter_config.json"))

    print(f"Saved merged adapter to: {merged_adapter_dir}")

    if not args.save_merged_model:
        sys.exit(0)

    base_model = AutoModelForCausalLM.from_pretrained(
        args.language_model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        device_map=None,
    ).to(device)

    merged_model = PeftModel.from_pretrained(base_model, merged_adapter_dir).merge_and_unload()

    model_save_dir = os.path.join(args.output_dir, "merged_model")
    os.makedirs(model_save_dir, exist_ok=True)
    merged_model.save_pretrained(model_save_dir)
    AutoTokenizer.from_pretrained(args.language_model_name).save_pretrained(model_save_dir)
    print(f"Saved merged model to: {model_save_dir}")


if __name__ == "__main__":
    main()
