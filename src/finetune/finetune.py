"""Fine-tune fresh identity-initialized OFT adapters."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from datasets import Dataset, load_dataset
from peft import OFTConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from src.dataset.dataset_3 import DATASET_3_TRAIN, doc_to_text, expand_doc
from src.paths import (
    LLAMA_D3_ADAPTERS_FOLDER as LLAMA_ADAPTERS_FOLDER,
    LLAMA_BASE_MODEL_PATH,
    QWEN_D3_ADAPTERS_FOLDER as QWEN_ADAPTERS_FOLDER,
    QWEN_BASE_MODEL_PATH,
    ROOTDIR,
)

NEW_ADAPTER_TASKS = DATASET_3_TRAIN

WANDB_PROJECT = "thesis-finetunes"
WANDB_ENTITY = None
WANDB_GROUP = "oft-new-adapters"
WANDB_TAGS = ["oft", "finetune"]
WANDB_LOG_MODEL = "false"
CACHE_DIR = ROOTDIR / "data" / "hf_cache"
SCRATCH_CHECKPOINT_ROOT = Path(
    os.environ.get(
        "FINETUNE_CHECKPOINT_ROOT",
        "/scratch-shared/eterres/MasterThesis/finetune_checkpoints",
    )
)
FINAL_OUTPUT_ROOT = ROOTDIR / "outputs" / "models"
SEED = 42
MAX_LENGTH = 1024
EVAL_FRACTION = 0.10
TRAIN_SIZE_MAX = 2**16
TRAIN_EVAL_SIZE_MAX = 2**12
PER_DEVICE_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.0
WARMUP_RATIO = 0.0
LR_SCHEDULER_TYPE = "linear"
NUM_TRAIN_EPOCHS = 1
SAVE_TOTAL_LIMIT = None
LOGGING_STEPS = 1
EVALS_PER_EPOCH = 10
TOTAL_SAVED_CHECKPOINTS_PER_EPOCH = 10
SAVE_STRATEGY = "steps"
OPTIM = "adamw_torch"
MAX_GRAD_NORM = 1.0
DEBUG_TRAIN_SAMPLES = 32
DEBUG_EPOCHS = 2
WANDB_CONFIG = {
    "adapter_type": "OFT",
    "lr_scheduler_type": LR_SCHEDULER_TYPE,
    "warmup_ratio": WARMUP_RATIO,
    "weight_decay": WEIGHT_DECAY,
    "optim": OPTIM,
    "evals_per_epoch": EVALS_PER_EPOCH,
    "total_saved_checkpoints_per_epoch": TOTAL_SAVED_CHECKPOINTS_PER_EPOCH,
    "effective_batch_size": PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
}


@dataclass(frozen=True)
class FinetuneModel:
    name: str
    base_model_path: str
    adapters_folder: str
    reference_adapter_name: str
    output_dir_name: str
    output_prefix: str

    @property
    def reference_adapter_dir(self) -> Path:
        return Path(self.adapters_folder) / self.reference_adapter_name


MODELS = {
    "llama3.1": FinetuneModel(
        name="llama3.1",
        base_model_path=LLAMA_BASE_MODEL_PATH,
        adapters_folder=LLAMA_ADAPTERS_FOLDER,
        reference_adapter_name="llama3-1_8b_finetune_numinamath",  # copies OFT adapters format
        output_dir_name="Llama-3.1-8B_OFT_dataset3_adapters",
        output_prefix="llama3-1_8b",
    ),
    "qwen2.5": FinetuneModel(
        name="qwen2.5",
        base_model_path=QWEN_BASE_MODEL_PATH,
        adapters_folder=QWEN_ADAPTERS_FOLDER,
        reference_adapter_name="qwen2.5_3b_finetune_numinamath",  # copies OFT adapters format
        output_dir_name="Qwen-2.5-3B_OFT_dataset3_adapters",
        output_prefix="qwen2.5_3b",
    ),
}


def oft_weight_l2_norm(model) -> float:
    total = None
    with torch.no_grad():
        for name, param in model.named_parameters():
            if ".oft_" not in name:
                continue
            value = param.detach().float().pow(2).sum()
            total = value if total is None else total + value
    return 0.0 if total is None else total.sqrt().item()


class WandbStepCallback(TrainerCallback):
    """Log Trainer metrics with the actual Trainer global step as the W&B x-axis."""

    def __init__(self):
        self.warned_no_run = False

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError("wandb is not installed in this environment.") from exc
        if wandb.run is None:
            if not self.warned_no_run:
                print("[wandb] no active run; skipping W&B metric logging", flush=True)
                self.warned_no_run = True
            return

        payload = {"train/global_step": state.global_step}
        for key, value in logs.items():
            if key == "epoch":
                payload["train/epoch"] = value
            elif key.startswith("eval_"):
                payload[f"eval/{key.removeprefix('eval_')}"] = value
            elif key not in {"total_flos"}:
                payload[f"train/{key}"] = value
        if state.epoch is not None:
            payload.setdefault("train/epoch", state.epoch)
        model = kwargs.get("model")
        if model is not None:
            payload["train/oft_weight_l2_norm"] = oft_weight_l2_norm(model)
        wandb.log(payload, step=state.global_step)


class CheckpointPrintCallback(TrainerCallback):
    """Print scratch checkpoint locations as soon as Trainer saves them."""

    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        print(
            f"[checkpoint] step={state.global_step} epoch={state.epoch} "
            f"saved to {checkpoint_path}",
            flush=True,
        )


def eval_steps_from_evals_per_epoch(train_size: int) -> int | None:
    if EVALS_PER_EPOCH < 1:
        raise ValueError("EVALS_PER_EPOCH must be at least 1")
    if EVALS_PER_EPOCH == 1:
        return None

    train_batches_per_epoch = math.ceil(train_size / PER_DEVICE_BATCH_SIZE)
    update_steps_per_epoch = math.ceil(train_batches_per_epoch / GRADIENT_ACCUMULATION_STEPS)
    return max(1, update_steps_per_epoch // (EVALS_PER_EPOCH - 1))


def save_steps_from_checkpoints_per_epoch(train_size: int) -> int:
    if TOTAL_SAVED_CHECKPOINTS_PER_EPOCH < 1:
        raise ValueError("TOTAL_SAVED_CHECKPOINTS_PER_EPOCH must be at least 1")

    train_batches_per_epoch = math.ceil(train_size / PER_DEVICE_BATCH_SIZE)
    update_steps_per_epoch = math.ceil(train_batches_per_epoch / GRADIENT_ACCUMULATION_STEPS)
    return max(1, update_steps_per_epoch // TOTAL_SAVED_CHECKPOINTS_PER_EPOCH)


def task_info(task_name: str) -> tuple[str, str, str | None, str]:
    tasks = {
        name: (name, path, subset, split)
        for name, path, subset, split, _ in NEW_ADAPTER_TASKS
    }
    if task_name not in tasks:
        raise ValueError(f"Unknown task {task_name!r}. Choose from {sorted(tasks)}")
    return tasks[task_name]


def read_reference(model_info: FinetuneModel) -> tuple[dict, dict[str, tuple[int, ...]]]:
    with open(model_info.reference_adapter_dir / "adapter_config.json") as f:
        config = json.load(f)
    with open(model_info.reference_adapter_dir / "adapter_model.safetensors", "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    specs = {k: tuple(v["shape"]) for k, v in header.items() if k != "__metadata__"}
    return config, specs


def target_regex_from_specs(specs: dict[str, tuple[int, ...]]) -> str:
    modules: list[str] = []
    for key in specs:
        stem = key.split(".oft_")[0]
        modules.append(re.escape(stem.removeprefix("base_model.model.")))
    return r"^(?:" + "|".join(sorted(set(modules))) + r")$"


def build_model(
    model_info: FinetuneModel,
    reference_config: dict,
    specs: dict[str, tuple[int, ...]],
):
    model = AutoModelForCausalLM.from_pretrained(
        model_info.base_model_path,
        torch_dtype=torch.bfloat16,
        device_map=None,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    oft_config = OFTConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_regex_from_specs(specs),
        r=reference_config.get("r", 0),
        oft_block_size=reference_config["oft_block_size"],
        module_dropout=reference_config.get("module_dropout", 0.0),
        coft=reference_config.get("coft", False),
        eps=reference_config.get("eps", 6e-5),
        block_share=reference_config.get("block_share", False),
        use_cayley_neumann=reference_config.get("use_cayley_neumann", True),
        num_cayley_neumann_terms=reference_config.get("num_cayley_neumann_terms", 5),
        init_weights=True,
        bias="none",
    )
    model = get_peft_model(model, oft_config)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    verify_oft_shapes(model, specs)
    model.print_trainable_parameters()
    return model


def normalize_param_name(name: str) -> str:
    return name.replace(".default", "").removeprefix("base_model.model.")


def verify_oft_shapes(model, specs: dict[str, tuple[int, ...]]) -> None:
    expected = {k.removeprefix("base_model.model."): shape for k, shape in specs.items()}
    found = {
        normalize_param_name(name): tuple(param.shape)
        for name, param in model.named_parameters()
        if ".oft_" in name
    }
    missing = sorted(set(expected) - set(found))
    extra = sorted(set(found) - set(expected))
    wrong = sorted(k for k in expected.keys() & found.keys() if expected[k] != found[k])
    if missing or extra or wrong:
        raise RuntimeError(
            f"OFT adapter mismatch: missing={missing[:3]}, extra={extra[:3]}, wrong={wrong[:3]}"
        )


def output_dir_name(model_info: FinetuneModel, debug: bool) -> str:
    return f"{model_info.output_dir_name}_debug" if debug else model_info.output_dir_name


def load_split_dataset(path: str, subset: str | None, split: str) -> Dataset:
    return load_dataset(path, subset, split=split, trust_remote_code=True, cache_dir=str(CACHE_DIR))


def maybe_sample_dataset(dataset: Dataset, max_size: int | None, split_name: str) -> Dataset:
    if max_size is None:
        return dataset
    original_size = len(dataset)
    capped_size = min(max_size, original_size)
    print(
        f"[data] {split_name}_size before_cap={original_size} cap={max_size} after_cap={capped_size}",
        flush=True,
    )
    return dataset.shuffle(seed=SEED).select(range(capped_size))


def build_datasets(
    task: str,
    path: str,
    subset: str | None,
    split_name: str,
    tokenizer,
    max_samples: int | None,
) -> tuple[Dataset, Dataset]:
    raw = load_split_dataset(path, subset, split_name).shuffle(seed=SEED)
    if max_samples is not None:
        raw = raw.select(range(min(max_samples, len(raw))))
    if len(raw) < 2:
        raise ValueError(f"Need at least 2 samples to split {task!r}; got {len(raw)}")
    split = raw.train_test_split(test_size=EVAL_FRACTION, seed=SEED)

    def join_prompt_and_target(prompt: str, target: str) -> tuple[list[int], list[int], list[int]]:
        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
        if len(prompt_ids) + len(target_ids) > MAX_LENGTH and target_ids:
            target_budget = min(len(target_ids), MAX_LENGTH - 1)
            prompt_budget = MAX_LENGTH - target_budget
            prompt_ids = prompt_ids[-prompt_budget:]
            target_ids = target_ids[:target_budget]
        input_ids = (prompt_ids + target_ids)[:MAX_LENGTH]
        labels = ([-100] * len(prompt_ids) + target_ids)[:MAX_LENGTH]
        attention_mask = [1] * len(input_ids)
        pad_len = MAX_LENGTH - len(input_ids)
        if pad_len > 0:
            input_ids += [tokenizer.pad_token_id] * pad_len
            labels += [-100] * pad_len
            attention_mask += [0] * pad_len
        return input_ids, attention_mask, labels

    def tokenize(batch: dict[str, list[Any]]) -> dict[str, Any]:
        source_docs = [dict(zip(batch.keys(), values)) for values in zip(*batch.values())]
        docs = [
            expanded
            for doc in source_docs
            for expanded in expand_doc(task, doc)
        ]
        if not docs:
            return {"input_ids": [], "attention_mask": [], "labels": []}
        prompts, texts = zip(*(doc_to_text(task, doc) for doc in docs))
        rows = {"input_ids": [], "attention_mask": [], "labels": []}
        for prompt, text in zip(prompts, texts):
            target = text[len(prompt):]
            input_ids, attention_mask, labels = join_prompt_and_target(prompt, target)
            rows["input_ids"].append(input_ids)
            rows["attention_mask"].append(attention_mask)
            rows["labels"].append(labels)
        return rows

    columns = raw.column_names
    train_raw = maybe_sample_dataset(split["train"], TRAIN_SIZE_MAX, "train")
    eval_raw = maybe_sample_dataset(split["test"], TRAIN_EVAL_SIZE_MAX, "eval")
    train = train_raw.map(tokenize, batched=True, remove_columns=columns)
    eval_ds = eval_raw.map(tokenize, batched=True, remove_columns=columns)
    return train, eval_ds


def _clean_for_log(text: str) -> str:
    return text.replace("\0", "")


def print_debug_batch_preview(tokenizer, train_ds: Dataset, eval_ds: Dataset, num_examples: int = 2) -> None:
    print("[debug] tokenized batch preview", flush=True)
    for split_name, dataset in (("train", train_ds), ("eval", eval_ds)):
        print(f"[debug] {split_name}_size={len(dataset)}", flush=True)
        for i in range(min(num_examples, len(dataset))):
            item = dataset[i]
            input_ids = item["input_ids"]
            attention_mask = item["attention_mask"]
            labels = item["labels"]
            active_ids = [token for token, mask in zip(input_ids, attention_mask) if mask]
            prompt_ids = [
                token
                for token, label, mask in zip(input_ids, labels, attention_mask)
                if mask and label == -100
            ]
            label_ids = [label for label in labels if label != -100]
            print(f"[debug] {split_name}[{i}] active_tokens={len(active_ids)} label_tokens={len(label_ids)}", flush=True)
            print(f"[debug] {split_name}[{i}] FULL:\n{_clean_for_log(tokenizer.decode(active_ids))}", flush=True)
            print(f"[debug] {split_name}[{i}] PROMPT_MASKED:\n{_clean_for_log(tokenizer.decode(prompt_ids))}", flush=True)
            print(f"[debug] {split_name}[{i}] LABELS:\n{_clean_for_log(tokenizer.decode(label_ids))}", flush=True)


def print_run_specs(
    model_info: FinetuneModel,
    task: str,
    path: str,
    subset: str | None,
    split_name: str,
    train_ds: Dataset,
    eval_ds: Dataset,
    args: TrainingArguments,
    debug: bool,
) -> None:
    print("[run] fine-tune specs", flush=True)
    print(f"[run] model_family={model_info.name}", flush=True)
    print(f"[run] task={task}", flush=True)
    print(f"[run] dataset={path} subset={subset or 'None'} split={split_name}", flush=True)
    print(f"[run] train_size={len(train_ds)} eval_size={len(eval_ds)}", flush=True)
    print(f"[run] debug={debug} seed={args.seed} max_length={MAX_LENGTH}", flush=True)
    print(
        f"[run] epochs={args.num_train_epochs} learning_rate={args.learning_rate} "
        f"scheduler={args.lr_scheduler_type}",
        flush=True,
    )
    print(
        f"[run] per_device_batch_size={args.per_device_train_batch_size} "
        f"gradient_accumulation_steps={args.gradient_accumulation_steps} "
        f"effective_batch_size={args.per_device_train_batch_size * args.gradient_accumulation_steps}",
        flush=True,
    )
    print(
        f"[run] evals_per_epoch={EVALS_PER_EPOCH} eval_strategy={args.eval_strategy} "
        f"eval_steps={args.eval_steps}",
        flush=True,
    )
    print(
        f"[run] saved_checkpoints_per_epoch={TOTAL_SAVED_CHECKPOINTS_PER_EPOCH} "
        f"save_strategy={args.save_strategy} save_steps={args.save_steps} "
        f"save_total_limit={args.save_total_limit}",
        flush=True,
    )


def maybe_init_wandb(
    model_info: FinetuneModel,
    task: str,
    debug: bool,
    run_name: str,
    train_size: int,
    eval_size: int,
    specs: dict[str, tuple[int, ...]],
) -> None:
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb is not installed in this environment.") from exc
    if wandb.run is not None:
        print(f"[wandb] using existing run: {wandb.run.name}", flush=True)
        return
    config = {
        **WANDB_CONFIG,
        "model_family": model_info.name,
        "base_model": model_info.base_model_path,
        "reference_adapter": model_info.reference_adapter_name,
        "task": task,
        "debug": debug,
        "train_size": train_size,
        "eval_size": eval_size,
        "train_size_max": TRAIN_SIZE_MAX,
        "train_eval_size_max": TRAIN_EVAL_SIZE_MAX,
        "max_length": MAX_LENGTH,
        "learning_rate": LEARNING_RATE,
        "num_oft_tensors": len(specs),
        "oft_shapes": sorted({shape for shape in specs.values()}),
    }
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        group=f"{model_info.name}-{WANDB_GROUP}",
        tags=WANDB_TAGS + [model_info.name] + (["debug"] if debug else []),
        name=run_name,
        config=config,
    )
    wandb.define_metric("train/global_step")
    wandb.define_metric("*", step_metric="train/global_step")
    print(f"[wandb] initialized run: {wandb.run.name} ({wandb.run.url})", flush=True)


def log_initial_eval(metrics: dict[str, float], model) -> None:
    metrics = {**metrics, "epoch": 0.0}
    print(metrics, flush=True)
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb is not installed in this environment.") from exc
    if wandb.run is None:
        print("[wandb] no active run; skipping initial eval metric logging", flush=True)
        return

    payload = {"train/global_step": 0, "train/epoch": 0.0}
    for key, value in metrics.items():
        if key == "epoch":
            continue
        if key.startswith("eval_"):
            payload[f"eval/{key.removeprefix('eval_')}"] = value
    payload["train/oft_weight_l2_norm"] = oft_weight_l2_norm(model)
    wandb.log(payload, step=0)


def train(model_family: str, task_name: str, debug: bool) -> None:
    model_info = MODELS[model_family]
    task, path, subset, split_name = task_info(task_name)
    os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)
    os.environ.setdefault("WANDB_LOG_MODEL", WANDB_LOG_MODEL)
    if WANDB_ENTITY:
        os.environ.setdefault("WANDB_ENTITY", WANDB_ENTITY)

    reference_config, specs = read_reference(model_info)
    tokenizer = AutoTokenizer.from_pretrained(model_info.base_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = build_model(model_info, reference_config, specs)
    train_ds, eval_ds = build_datasets(
        task,
        path,
        subset,
        split_name,
        tokenizer,
        DEBUG_TRAIN_SAMPLES if debug else None,
    )
    if debug:
        print_debug_batch_preview(tokenizer, train_ds, eval_ds)
    run_dir = f"{model_info.output_prefix}_finetune_{task}"
    adapter_output_dir_name = output_dir_name(model_info, debug)
    checkpoint_dir = SCRATCH_CHECKPOINT_ROOT / adapter_output_dir_name / run_dir
    final_output_dir = FINAL_OUTPUT_ROOT / adapter_output_dir_name / run_dir
    run_name = f"{model_info.name}-oft-{task}{'-debug' if debug else ''}"
    print(f"[checkpoint] intermediate checkpoints will be saved to {checkpoint_dir}", flush=True)
    print(f"[final] final adapter will be saved to {final_output_dir}", flush=True)
    eval_steps = eval_steps_from_evals_per_epoch(len(train_ds))
    save_steps = save_steps_from_checkpoints_per_epoch(len(train_ds))
    args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=DEBUG_EPOCHS if debug else NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        bf16=True,
        gradient_checkpointing=True,
        optim=OPTIM,
        max_grad_norm=MAX_GRAD_NORM,
        eval_strategy="no" if eval_steps is None else "steps",
        eval_steps=eval_steps,
        save_strategy=SAVE_STRATEGY,
        save_steps=save_steps,
        logging_strategy="steps",
        logging_steps=LOGGING_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        report_to=[],
        run_name=run_name,
        remove_unused_columns=False,
        seed=SEED,
    )
    args.run_name = run_name
    print_run_specs(model_info, task, path, subset, split_name, train_ds, eval_ds, args, debug)
    maybe_init_wandb(model_info, task, debug, args.run_name, len(train_ds), len(eval_ds), specs)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        callbacks=[WandbStepCallback(), CheckpointPrintCallback()],
    )
    print("[eval] running pre-training evaluation", flush=True)
    initial_eval_metrics = trainer.evaluate(metric_key_prefix="eval")
    log_initial_eval(initial_eval_metrics, model)
    trainer.train()
    print(f"[final] saving final adapter to {final_output_dir}", flush=True)
    trainer.save_model(str(final_output_dir))
    tokenizer.save_pretrained(str(final_output_dir))
    print(f"[final] saved final adapter and tokenizer to {final_output_dir}", flush=True)
    latest_checkpoint = max(
        checkpoint_dir.glob("checkpoint-*"),
        key=lambda path: int(path.name.rsplit("-", 1)[-1]),
        default=None,
    )
    if latest_checkpoint is not None:
        pointer_path = final_output_dir / "latest_scratch_checkpoint.txt"
        pointer_path.write_text(f"{latest_checkpoint}\n")
        print(
            f"[checkpoint] latest scratch checkpoint is {latest_checkpoint}; "
            f"wrote pointer to {pointer_path}",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-family", choices=MODELS.keys(), default="llama3.1")
    parser.add_argument("--task", choices=[task[0] for task in NEW_ADAPTER_TASKS], default=None)
    parser.add_argument("--task-index", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--list-tasks", action="store_true")
    args = parser.parse_args()

    if args.list_tasks:
        for i, (task, _, _, _, _) in enumerate(NEW_ADAPTER_TASKS):
            print(f"{i}: {task}")
        return args

    slurm_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    model_family = os.environ.get("MODEL_FAMILY")
    if model_family is not None:
        if model_family not in MODELS:
            parser.error(f"MODEL_FAMILY must be one of {sorted(MODELS)}")
        args.model_family = model_family

    if args.task is None and args.task_index is None and slurm_task_id is not None:
        args.task_index = int(slurm_task_id)
    elif args.debug and args.task is None and args.task_index is None:
        args.task = NEW_ADAPTER_TASKS[0][0]

    if (args.task is None) == (args.task_index is None):
        parser.error("Pass exactly one of --task or --task-index, unless --debug selects task 0.")
    if args.task_index is not None:
        if not 0 <= args.task_index < len(NEW_ADAPTER_TASKS):
            parser.error(f"--task-index must be in [0, {len(NEW_ADAPTER_TASKS) - 1}]")
        args.task = NEW_ADAPTER_TASKS[args.task_index][0]
    return args


if __name__ == "__main__":
    cli_args = parse_args()
    if not cli_args.list_tasks:
        train(cli_args.model_family, cli_args.task, cli_args.debug)
