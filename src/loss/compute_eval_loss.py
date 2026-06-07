from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.dataset.dataset_2 import DATASET_2_TEST as DATASET_TEST, build_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute target-token eval loss on dataset_2 tests.")
    parser.add_argument("--eval-type", choices=["pretrained", "finetunes"], default="pretrained")
    parser.add_argument("--model", choices=["llama", "qwen"], default="llama")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs" / "eval_loss")
    parser.add_argument("--dataset-cache-dir", type=Path, default=REPO_ROOT / "data" / "hf_cache")
    parser.add_argument("--run-name")
    parser.add_argument("--model-name")
    parser.add_argument("--peft-model", type=Path)
    parser.add_argument("--task-id", type=int, choices=range(len(DATASET_TEST)))
    parser.add_argument("--task", choices=[task for task, *_ in DATASET_TEST])
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def model_family_key(model: str) -> str:
    return {
        "llama": "llama3.1",
        "qwen": "qwen2.5",
    }[model]


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    from src.paths import MODEL_FAMILIES_D2

    family_key = args.model_name or model_family_key(args.model)
    family = MODEL_FAMILIES_D2[family_key]
    args.model_name = family.name
    args.model_path = args.model_path or Path(family.base_model_path)
    args.run_name = args.run_name or args.eval_type
    return args


def selected_tasks(args: argparse.Namespace):
    if args.task_id is not None:
        return [DATASET_TEST[args.task_id]]
    if args.task is not None:
        return [spec for spec in DATASET_TEST if spec[0] == args.task]
    return DATASET_TEST


def load_model(args: argparse.Namespace):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    if args.peft_model:
        model = PeftModel.from_pretrained(model, args.peft_model, is_trainable=False)
        model.enable_adapter_layers()
    model.to(args.device)
    model.eval()
    return model, tokenizer


def adapter_path_for_task(args: argparse.Namespace, task: str) -> Path:
    from src.paths import MODEL_FAMILIES_D2

    family = MODEL_FAMILIES_D2[args.model_name]
    task_names = [task_name for task_name, *_ in DATASET_TEST]
    task_id = task_names.index(task)
    return Path(family.adapter_paths[task_id])


@torch.inference_mode()
def eval_loss(model, loader, device: str) -> tuple[float, int]:
    total_nll = 0.0
    total_tokens = 0
    for batch in tqdm(loader, desc="loss", leave=False):
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = batch["labels"][..., 1:].contiguous()
        mask = shift_labels.ne(-100)
        safe_labels = shift_labels.masked_fill(~mask, 0)
        token_losses = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            safe_labels.view(-1),
            reduction="none",
        ).view_as(shift_labels)
        total_nll += (token_losses * mask).sum().item()
        total_tokens += mask.sum().item()
    return total_nll / max(total_tokens, 1), total_tokens


def save_metrics(output_root: Path, task: str, loss: float, tokens: int, args: argparse.Namespace) -> None:
    output_dir = output_root / args.run_name / args.model_name / task
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "results": {
            task: {
                "eval_loss": loss,
                "perplexity": math.exp(loss) if loss < 100 else float("inf"),
                "num_tokens": tokens,
            }
        },
        "config": {
            "model_path": str(args.model_path),
            "peft_model": str(args.peft_model) if args.peft_model else None,
            "num_samples": args.num_samples,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")


def compute_eval_task(args: argparse.Namespace) -> None:
    model, tokenizer = load_model(args)

    for task, dataset_path, dataset_name, split, formatter in selected_tasks(args):
        print(f"Computing eval loss for {task}")
        loader = build_loader(
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            split=split,
            doc_to_text_fn=formatter,
            tokenizer=tokenizer,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            max_length=args.max_length,
            task=task,
            cache_dir=str(args.dataset_cache_dir) if args.dataset_cache_dir else None,
        )
        loss, tokens = eval_loss(model, loader, args.device)
        save_metrics(args.output_root, task, loss, tokens, args)
        print(f"{task}: eval_loss={loss:.6f}, tokens={tokens}")


def compute_eval_pretrained(args: argparse.Namespace) -> None:
    args.peft_model = None
    args.run_name = args.run_name or "pretrained"
    compute_eval_task(args)


def compute_eval_finetunes(args: argparse.Namespace) -> None:
    args.run_name = args.run_name or "finetunes"
    for task, *_ in selected_tasks(args):
        task_args = argparse.Namespace(**vars(args))
        task_args.task = task
        task_args.task_id = None
        task_args.peft_model = args.peft_model or adapter_path_for_task(args, task)
        compute_eval_task(task_args)


def main() -> None:
    args = resolve_args(parse_args())
    if args.eval_type == "pretrained":
        compute_eval_pretrained(args)
    else:
        compute_eval_finetunes(args)


if __name__ == "__main__":
    main()
