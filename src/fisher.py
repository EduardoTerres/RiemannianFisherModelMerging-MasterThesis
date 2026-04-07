"""
This file contains the functionality for computing Fisher Information Matrix
or its approximations.
"""
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from safetensors.torch import save_file


def compute_fim_rank_1() -> None:
    """
    Compute the rank-1 approximation of the Fisher Information Matrix.
    """
    pass


def compute_diagional_fim(
    model_name_or_path: str,
    dataset_path: str,
    save_path: str,
    dataset_name: str | None = None,
    num_samples: int = 512,
    batch_size: int = 4,
    max_length: int = 512,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    """
    Compute the diagonal approximation of the Fisher Information Matrix.

    Loads the model, tokenizes the dataset, and accumulates squared gradients
    of the log-likelihood w.r.t. each parameter (empirical Fisher diagonal).
    Saves the result as a safetensors file at save_path.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float32,
        device_map=None,
    ).to(device)
    model.eval()

    dataset = get_dataset(dataset_path, dataset_name)
    dataset = dataset.select(range(min(num_samples, len(dataset))))

    # Find the text column
    text_col = next(
        (c for c in ("text", "sentence", "question", "ctx") if c in dataset.column_names),
        dataset.column_names[0]
    )

    def tokenize(batch):
        return tokenizer(
            batch[text_col],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    loader = DataLoader(dataset, batch_size=batch_size)

    # Accumulate diagonal Fisher: E[grad log p * grad log p]
    diag_fisher = {
        n: torch.zeros_like(p, device="cpu")
        for n, p in model.named_parameters() if p.requires_grad
    }

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        model.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        loss.backward()

        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                diag_fisher[n] += (p.grad.detach().cpu() ** 2)

    n_batches = len(loader)
    diag_fisher = {n: v / n_batches for n, v in diag_fisher.items()}

    save_file(diag_fisher, save_path)
    print(f"Diagonal FIM saved to {save_path}")

    return diag_fisher


def get_dataset(dataset_path: str, dataset_name: str | None = None, **kwargs):
    """Load the train split, similar to lm-evaluation-harness."""
    return load_dataset(dataset_path, dataset_name, split="train", **kwargs)

def test_get_dataset():
    social_iqa = get_dataset("social_i_qa", trust_remote_code=True)
    commonsense_qa = get_dataset("commonsense_qa", trust_remote_code=True)

    print(social_iqa)

    assert len(social_iqa) > 0
    assert len(commonsense_qa) > 0

def test_compute_diagonal_fim(
    model_name_or_path: str = "../OrthoMerge/models/Llama-3.1-8B",
    dataset_path: str = "tau/commonsense_qa",
    dataset_name: str = "default",
    save_path: str = "/tmp/diag_fisher_test.safetensors",
):
    diag_fisher = compute_diagional_fim(
        model_name_or_path=model_name_or_path,
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        save_path=save_path,
        num_samples=8,
        batch_size=2,
        max_length=64,
    )
    assert len(diag_fisher) > 0, "Fisher dict is empty"
    for name, val in diag_fisher.items():
        assert (val >= 0).all(), f"Negative Fisher value for {name}"
    print(f"OK: {len(diag_fisher)} parameter tensors, saved to {save_path}")


# (dataset_path, dataset_name, text_field)
EVAL_DATASETS = [
    ("social_i_qa",      None,           "context"),
    ("commonsense_qa",   None,           "question"),
    ("EleutherAI/hendrycks_math", "all", "problem"),
    ("openai_humaneval",  None,          "prompt"),
]


def compute_all_fishers(
    model_name_or_path: str = "../OrthoMerge/models/Llama-3.1-8B",
    output_dir: str = "data/fishers",
    num_samples: int = 512,
    batch_size: int = 4,
    max_length: int = 512,
) -> None:
    """Compute and save the diagonal FIM for each dataset in EVAL_DATASETS."""
    import os

    os.makedirs(output_dir, exist_ok=True)
    model_tag = os.path.basename(model_name_or_path.rstrip("/"))
    total = len(EVAL_DATASETS)

    for i, (dataset_path, dataset_name, _) in enumerate(EVAL_DATASETS, 1):
        dataset_tag = (dataset_name or dataset_path).replace("/", "_")
        save_path = os.path.join(output_dir, f"{model_tag}__{dataset_tag}.safetensors")

        print(f"\n[{i}/{total}] Dataset: {dataset_path} ({dataset_name or 'default'})")
        print(f"         Save  : {save_path}")

        if os.path.exists(save_path):
            print("         -> already exists, skipping.")
            continue

        compute_diagional_fim(
            model_name_or_path=model_name_or_path,
            dataset_path=dataset_path,
            save_path=save_path,
            dataset_name=dataset_name,
            num_samples=num_samples,
            batch_size=batch_size,
            max_length=max_length,
        )
        print(f"         -> done.")

    print(f"\nAll fishers saved to {output_dir}/")


def inspect_fisher(path: str, n: int = 10) -> None:
    """Load a saved diagonal FIM and print the first n parameter entries."""
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        for key in keys:
            tensor = f.get_tensor(key)
            print(f"{key}: shape={list(tensor.shape)}, min={tensor.min():.4e}, max={tensor.max():.4e}, mean={tensor.mean():.4e}")


if __name__ == "__main__":
    # test_compute_diagonal_fim()
    # inspect_fisher("/tmp/diag_fisher_test.safetensors", n=5)
    compute_all_fishers()
