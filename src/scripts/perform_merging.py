import argparse
import os
import sys
import shutil

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


import torch
import wandb
from safetensors.torch import save_file
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.merging import OFTMerging, WudiOFTMerging, OFTKarcherMerging, AdaMergingPP
from src.utils import parse_device
from src.paths import MODEL_FAMILIES_D2 as MODEL_FAMILIES, ModelFamily, WANDB_PROJECT
# from src.dataset.dataset_1 import DATASET_1_TEST, build_loader
from src.dataset.dataset_2 import DATASET_2_TEST, build_loader


def _build_alpha_optimizer_inputs(
    model_family: ModelFamily,
    device: str,
    num_samples: int = 128,
    batch_size: int = 4,
    max_length: int = 128,
) -> tuple:
    tokenizer = AutoTokenizer.from_pretrained(model_family.base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    task_names = [tag for tag, *_ in DATASET_TEST]
    task_loaders = [
        build_loader(
            dataset_path=ds_path,
            dataset_name=ds_name,
            split=split,
            doc_to_text=doc_to_text,
            tokenizer=tokenizer,
            num_samples=num_samples,
            batch_size=batch_size,
            max_length=max_length,
        )
        for _, ds_path, ds_name, split, doc_to_text in DATASET_TEST
    ]
    peft_model = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(
            model_family.base_model_path, torch_dtype=torch.float32, device_map=None
        ),
        model_family.adapter_paths[0],
        is_trainable=False,
    )
    peft_model.enable_adapter_layers()
    peft_model.to(device)
    return peft_model, task_loaders, task_names


def _build_adamerging(
    model_family: ModelFamily,
    device: str,
    num_samples: int = 256,
    batch_size: int = 16,
    max_length: int = 32,
) -> dict[str, Tensor]:
    tokenizer = AutoTokenizer.from_pretrained(model_family.base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    task_names = [tag for tag, *_ in DATASET_TEST]
    task_loaders = [
        build_loader(
            dataset_path=ds_path,
            dataset_name=ds_name,
            split=split,
            doc_to_text=doc_to_text,
            tokenizer=tokenizer,
            num_samples=num_samples,
            batch_size=batch_size,
            max_length=max_length,
        )
        for _, ds_path, ds_name, split, doc_to_text in DATASET_TEST
    ]

    base_model = AutoModelForCausalLM.from_pretrained(
        model_family.base_model_path, torch_dtype=torch.float32, device_map=None
    )
    peft_model = PeftModel.from_pretrained(
        base_model, model_family.adapter_paths[0], is_trainable=False
    )
    peft_model.enable_adapter_layers()
    peft_model.to(device)
    peft_model.eval()

    wandb.init(project=WANDB_PROJECT, name=f"adamerging-{model_family.name}")

    merging = AdaMergingPP(device=device)
    merged_weights = merging.merge(
        adapter_paths=model_family.adapter_paths,
        model=peft_model,
        task_loaders=task_loaders,
        task_names=task_names,
    )
    return merged_weights


def parse_args():
    parser = argparse.ArgumentParser(description="Merge OFT adapters with Riemannian merging.")
    parser.add_argument(
        "--model_family", type=str, choices=list(MODEL_FAMILIES), required=True,
    )
    parser.add_argument(
        "--merge_method", type=str, choices=["gradients", "wudi", "karcher", "adamerging"],
        help="Merging method: 'gradients', 'wudi', 'karcher' (Karcher mean on SO(n)), 'adamerging'.",  # noqa: E501
    )
    parser.add_argument(
        "--merge_mode",
        type=str,
        choices=[
            "standard",
            "standard_rescaled",
            "diagonal_fisher",
            "diagonal_fisher_rescaled",
            "diagonal_fisher_kl_rescaled",
            "fisher",
        ],
        help="Merging strategy: standard, rescaled standard, diagonal Fisher, rescaled diagonal Fisher, KL-rescaled diagonal Fisher, or full Fisher.",  # noqa: E501
    )
    parser.add_argument(
        "--lam", type=float, default=0.0,
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
    parser.add_argument(
        "--optimize_alphas",
        type=str,
        choices=["adamerging", "adamergingpp", "adamerging_equal", "adamergingpp_equal"],
        default=None,
        help="If set, optimize alphas via entropy minimization; *_equal ties all alphas.",  # noqa: E501
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.device = parse_device(args.device)

    model_family = MODEL_FAMILIES[args.model_family]
    base_model_path = model_family.base_model_path
    adapter_paths = model_family.adapter_paths

    use_wandb = not (
        args.merge_method == "gradients"
        and args.merge_mode in {"standard", "standard_rescaled", "diagonal_fisher"}
    )
    if use_wandb:
        wandb.init(
            project=WANDB_PROJECT,
            name=f"{args.merge_method}-{args.merge_mode}-{model_family.name}",
            config=vars(args),
        )

    merged_weights = None
    if args.merge_method == "adamerging":
        merged_weights = _build_adamerging(model_family=model_family, device=args.device)
    else:
        if args.merge_method == "gradients":
            merging = OFTMerging(lam=args.lam, alphas=args.alphas, device=args.device)
        elif args.merge_method == "wudi":
            merging = WudiOFTMerging(device=args.device)
        elif args.merge_method == "karcher":
            merging = OFTKarcherMerging(lam=args.lam, alphas=args.alphas, device=args.device)
        else:
            raise ValueError(f"Unsupported merge method: {args.merge_method}")

        merge_kwargs: dict = dict(
            adapter_paths=adapter_paths,
            fisher_paths=model_family.fisher_paths if "fisher" in args.merge_mode else None,
            mode=args.merge_mode,
            optimize_alphas=args.optimize_alphas,
        )
        if args.optimize_alphas is not None:
            model, task_loaders, task_names = _build_alpha_optimizer_inputs(
                model_family=model_family, device=args.device,
            )
            merge_kwargs["model"] = model
            merge_kwargs["task_loaders"] = task_loaders
            merge_kwargs["task_names"] = task_names

        merged_weights = merging.merge(**merge_kwargs)

    if not merged_weights:
        print("Merging failed. No weights returned.")
        sys.exit(1)

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
