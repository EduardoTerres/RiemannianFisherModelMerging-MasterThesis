"""
Entrypoint to compute Fisher matrices for all eval-harness tasks.
Prompt formats match lm-evaluation-harness / bigcode-evaluation-harness exactly
so the Fisher is computed on the same distribution as the evaluation.
"""
import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from torch.utils.data import DataLoader
from safetensors.torch import save_file
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.path import ROOTDIR, OFT_LLAMA_MODELS_DIR
from src.fisher import (
    compute_diagonal_fim,
    compute_empirical_fisher,
    compute_empirical_diagonal_fisher,
    compute_kfac,
)

# Task definitions: (task_tag, dataset_path, dataset_name, split, doc_to_text)
# doc_to_text(doc) -> str   — same format as the eval harness uses

def _siqa_text(doc):
    choices = [doc["answerA"], doc["answerB"], doc["answerC"]]
    correct = choices[int(doc["label"]) - 1]
    return f"Q: {doc['context']} {doc['question']}\nA: {correct}"

def _csqa_text(doc):
    letters = ["A", "B", "C", "D", "E"]
    choices_text = "\n".join(
        f"{l}. {t}" for l, t in zip(letters, doc["choices"]["text"])
    )
    return f"Question: {doc['question'].strip()}\n{choices_text}\nAnswer: {doc['answerKey']}"

def _minerva_text(doc):
    return f"Problem:\n{doc['problem']}\n\nSolution:\n{doc['solution']}"

def _humaneval_text(doc):
    return doc["prompt"] + doc["canonical_solution"]

def _scienceqa_text(doc):
    _INDEX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}

    question = doc["question"]
    options = doc["choices"]
    answer_index = int(doc["answer"])
    answer_letter = _INDEX_TO_LETTER.get(answer_index, "A")

    choice_lines = []
    for i, opt in enumerate(options):
        label = _INDEX_TO_LETTER.get(i, chr(ord("A") + i))
        choice_lines.append(f"{label}. {opt}")
    choices_str = "\n".join(choice_lines)

    return (
        f"Question: {question}\n"
        f"Choices:\n{choices_str}\n\n"
        f"Answer: {answer_letter}"
    )

# Execution parameters

BASE_MODEL_PATH = f"{OFT_LLAMA_MODELS_DIR}/Llama-3.1-8B"
ADAPTERS_PATH   = f"{OFT_LLAMA_MODELS_DIR}/Llama-3.1-8B_OFT_adapters"

EVAL_TASKS = [
    # (tag, dataset_path, dataset_name, split, doc_to_text, adapter_path)
    ("social_iqa",      "allenai/social_i_qa",    None,      "train", _siqa_text,     f"{ADAPTERS_PATH}/llama3-1_8b_finetune_socialiqa"),  # noqa: E501
    ("commonsense_qa",  "tau/commonsense_qa",      None,      "train", _csqa_text,     f"{ADAPTERS_PATH}/llama3-1_8b_finetune_commonsense"),  # noqa: E501
    ("numinamath", "HuggingFaceH4/MATH-500",  "default", "test",  _minerva_text,  f"{ADAPTERS_PATH}/llama3-1_8b_finetune_numinamath"),  # noqa: E501
    ("humanevalplus",   "evalplus/humanevalplus",  None,      "test",  _humaneval_text, f"{ADAPTERS_PATH}/llama3-1_8b_finetune_magicoder"),  # noqa: E501
    ("science_qa",      "derek-thomas/ScienceQA",     None,    "train", _scienceqa_text, f"{ADAPTERS_PATH}/llama3-1_8b_finetune_scienceqa"),  # noqa: E501
]


def build_loader(
    dataset_path: str,
    dataset_name: str | None,
    split: str,
    doc_to_text,
    tokenizer,
    num_samples: int,
    batch_size: int,
    max_length: int,
) -> DataLoader:
    dataset = load_dataset(dataset_path, dataset_name, split=split, trust_remote_code=True)
    dataset = dataset.select(range(min(num_samples, len(dataset))))

    def tokenize(batch):
        texts = [
            doc_to_text({k: batch[k][i] for k in batch})
            for i in range(len(batch[next(iter(batch))]))
        ]
        return tokenizer(texts, truncation=True, max_length=max_length, padding="max_length")

    dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return DataLoader(dataset, batch_size=batch_size)

def compute_and_save_fim(
    base_model_path: str,
    adapter_path: str,
    task_tag: str,
    dataset_path: str,
    dataset_name: str | None,
    split: str,
    doc_to_text,
    save_path: str,
    num_samples: int = 512,
    batch_size: int = 4,
    max_length: int = 512,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float32, device_map=None
    )
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=True)
    model.enable_adapter_layers()
    model.to(device)
    model.eval()

    loader = build_loader(
        dataset_path, dataset_name, split, doc_to_text,
        tokenizer, num_samples, batch_size, max_length,
    )
    # diag_fisher = compute_diagonal_fim(model, loader, device)
    # diag_fisher = compute_empirical_diagonal_fisher(model, loader, device)
    # fisher = compute_kfac(model, loader, device)
    fisher = FIM(model=model, loader=loader, representation=PMatDiag)
    print(fisher)
    exit(0)

    save_file(fisher, save_path)
    print(f"[{task_tag}] Fisher saved to {save_path}")

def compute_all_fishers(
    output_dir: str,
    num_samples: int,
    batch_size: int,
    max_length: int,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    total = len(EVAL_TASKS)
    for i, (task_tag, dataset_path, dataset_name, split, doc_to_text, adapter_path) in tqdm(
        enumerate(EVAL_TASKS, 1), total=total, desc="Computing FIMs"
    ):
        model_tag = os.path.basename(adapter_path.rstrip("/"))
        save_path = os.path.join(output_dir, f"{model_tag}.safetensors")
        print(f"\n[{i}/{total}] Task: {task_tag}  adapter={model_tag}  split={split}")
        print(f"         Save : {save_path}")

        if os.path.exists(save_path):
            print("[WARNING] Already exists, skipping.")
            continue

        compute_and_save_fim(
            base_model_path=BASE_MODEL_PATH,
            adapter_path=adapter_path,
            task_tag=task_tag,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            split=split,
            doc_to_text=doc_to_text,
            save_path=save_path,
            num_samples=num_samples,
            batch_size=batch_size,
            max_length=max_length,
        )
        print("[OK] Done.")

    print(f"All fishers saved to {output_dir}/")


if __name__ == "__main__":
    compute_all_fishers(
        output_dir=f"{ROOTDIR}/data/fishers",
        num_samples=512,
        batch_size=8,
        max_length=256,
    )
