"""Fisher Information Matrix computations and entrypoint to compute and save FIMs."""
import argparse
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.constants import (
    ROOTDIR,
    LLAMA_ADAPTER_PATHS,
    LLAMA_BASE_MODEL_PATH,
    QWEN_BASE_MODEL_PATH,
    QWEN_ADAPTER_PATHS,
)
from src.dataset.dataset_1 import DATASET_1_TRAIN as DATASET_1, build_loader
from src.utils import parse_device

def compute_diagonal_fim(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
) -> dict:
    """
    Compute the diagonal empirical Fisher: E[grad log p]^2.
    Args:
        model:  an already-loaded, eval-mode CausalLM
        loader: DataLoader yielding dicts with 'input_ids' and 'attention_mask'
        device: torch device string
    Returns:
        dict mapping parameter name -> diagonal Fisher tensor (on CPU)
    """
    fisher = {}
    for n, p in model.named_parameters():
        if p.requires_grad:
            fisher[n] = torch.zeros_like(p, device=device)

    model.eval()
    count = 0

    for batch in tqdm(loader, desc="Computing Diagonal FIM"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Standard causal LM: predict next token
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        shift_logits = outputs.logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        shift_mask = attention_mask[:, 1:]

        log_probs = F.log_softmax(shift_logits, dim=-1)
        # Per-token NLL, masked and averaged
        token_nll = F.nll_loss(
            log_probs.view(-1, log_probs.size(-1)),
            shift_labels.reshape(-1),
            reduction="none",
        ).view(shift_labels.shape)

        loss = (token_nll * shift_mask).sum() / shift_mask.sum()

        model.zero_grad()
        loss.backward()

        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] += p.grad.data ** 2

        count += 1

    for n in fisher:
        fisher[n] /= count
        fisher[n] = fisher[n].cpu()

    return fisher


def compute_empirical_diagonal_fisher(model, loader, device):
    """
    Compute the empirical diagonal Fisher for the trainable parameters
    of a causal LM / PEFT model.

    Returns
    -------
    fisher : dict[str, torch.Tensor]
        Diagonal Fisher tensors on CPU, same shapes as the parameters.
    """
    model.eval()

    named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not named_params:
        raise ValueError("No trainable parameters with requires_grad=True were found.")

    param_names, params = zip(*named_params)
    fisher = [torch.zeros_like(p, dtype=torch.float32, device="cpu") for p in params]
    n_sequences = 0

    for batch in tqdm(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        log_probs = F.log_softmax(outputs.logits[:, :-1, :], dim=-1)
        token_log_probs = log_probs.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

        if attention_mask is not None:
            token_log_probs = token_log_probs * attention_mask[:, 1:].to(token_log_probs.dtype)

        seq_log_probs = token_log_probs.sum(dim=-1)   # (B,)
        batch_size = seq_log_probs.shape[0]

        for i in range(batch_size):
            model.zero_grad(set_to_none=True)
            grads = torch.autograd.grad(
                seq_log_probs[i],
                params,
                retain_graph=(i < batch_size - 1),
                create_graph=False,
                allow_unused=True,
            )
            for j, g in enumerate(grads):
                if g is not None:
                    fisher[j] += g.detach().float().cpu() ** 2
            n_sequences += 1

    if n_sequences == 0:
        raise ValueError("No sequences were processed.")

    return {n: f / n_sequences for n, f in zip(param_names, fisher)}

def compute_true_fisher(model, loader, device):
    """
    True diagonal Fisher: sample labels from the model's own distribution
    instead of using observed labels (avoids empirical Fisher bias).
    """
    model.eval()
    named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not named_params:
        raise ValueError("No trainable parameters found.")

    param_names, params = zip(*named_params)
    accum = [torch.zeros_like(p, dtype=torch.float32, device="cpu") for p in params]
    n_sequences = 0

    for batch in tqdm(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            sampled = torch.multinomial(
                F.softmax(logits[:, :-1, :].reshape(-1, logits.shape[-1]), dim=-1), 1
            ).reshape(input_ids.shape[0], -1)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        log_probs = F.log_softmax(outputs.logits[:, :-1, :], dim=-1)
        token_log_probs = log_probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        if attention_mask is not None:
            token_log_probs = token_log_probs * attention_mask[:, 1:].to(token_log_probs.dtype)
        seq_log_probs = token_log_probs.sum(dim=-1)

        for i in range(seq_log_probs.shape[0]):
            model.zero_grad(set_to_none=True)
            grads = torch.autograd.grad(
                seq_log_probs[i], params,
                retain_graph=(i < seq_log_probs.shape[0] - 1),
                create_graph=False, allow_unused=True,
            )
            for j, g in enumerate(grads):
                if g is not None:
                    accum[j] += g.detach().float().cpu() ** 2
            n_sequences += 1

    if n_sequences == 0:
        raise ValueError("No sequences processed.")
    return {n: f / n_sequences for n, f in zip(param_names, accum)}


def compute_empirical_fisher(model, loader, device):
    """
    Compute the diagonal empirical Fisher Information Matrix, layerwise.

    The empirical Fisher is approximated as:
        F_theta = E_{(x,y) ~ data} [ (grad log p(y|x; theta))^2 ]

    We use the true labels from the loader (empirical Fisher), computing
    per-sample gradients of the log-likelihood and averaging their squares.

    Args:
        model:  nn.Module. Output assumed to be logits for classification.
        loader: DataLoader yielding (inputs, targets).
        device: torch device.

    Returns:
        dict[str, Tensor]: {param_name: fisher_tensor} with the same shape
        as each parameter. Only parameters with requires_grad=True are included.
    """
    model.eval()
    model.to(device)

    # Initialize Fisher accumulators (same shape as each trainable parameter)
    fisher = {
        name: torch.zeros_like(p)
        for name, p in model.named_parameters()
        if p.requires_grad
    }

    n_samples = 0

    for batch in tqdm(loader, desc="Computing Empirical FIM"):
        batch = {k: v.to(device) for k, v in batch.items()}
        batch_size = batch["input_ids"].size(0)

        # Process one sample at a time to get true per-sample gradients.
        # (Batch gradients would give grad of the *sum*, whose square is not
        # the same as the sum of squared per-sample gradients.)
        for i in range(batch_size):
            model.zero_grad(set_to_none=True)

            sample = {k: v[i:i + 1] for k, v in batch.items()}
            loss = model(**sample, labels=sample["input_ids"]).loss

            loss.backward()

            for name, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[name] += p.grad.detach() ** 2

            n_samples += 1

    # Average over the dataset
    for name in fisher:
        fisher[name] /= max(n_samples, 1)

    model.zero_grad(set_to_none=True)
    return fisher


def compute_kfac(model, loader, device):
    """
    K-FAC: approximate the Fisher block for each nn.Linear as A ⊗ G,
    where A = E[a aᵀ] (input covariance) and G = E[δ δᵀ] (output-gradient covariance).

    Returns
    -------
    dict[str, tuple[Tensor, Tensor]]
        {module_name: (A, G)} both on CPU.
    """
    model.eval()
    model.to(device)

    A, G = {}, {}
    handles = []

    # Capture forward inputs and backward output-gradients for every Linear layer.
    acts, grads = {}, {}
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue

        def _fwd(n):
            def hook(_, inp, __):
                acts[n] = inp[0].detach().float()
            return hook

        def _bwd(n):
            def hook(_, __, g_out):
                grads[n] = g_out[0].detach().float()
            return hook

        handles += [
            mod.register_forward_hook(_fwd(name)),
            mod.register_full_backward_hook(_bwd(name)),
        ]

    n_samples = 0
    for batch in tqdm(loader, desc="Computing K-FAC"):
        batch = {k: v.to(device) for k, v in batch.items()}
        for s in range(batch["input_ids"].size(0)):
            model.zero_grad(set_to_none=True)
            sample = {k: v[s:s+1] for k, v in batch.items()}

            # Mask padding in labels so loss (and its gradients) ignores pad tokens.
            labels = sample["input_ids"].clone()
            if "attention_mask" in sample:
                labels[sample["attention_mask"] == 0] = -100
            model(**sample, labels=labels).loss.backward()

            # attention_mask: (1, T) → boolean mask over token positions
            mask = sample.get("attention_mask")  # (1, T) or None
            if mask is not None:
                mask = mask[0].bool().cpu()  # (T,)

            for name in list(acts.keys() & grads.keys()):
                a = acts[name].reshape(-1, acts[name].shape[-1]).cpu()   # (T, d_in)
                g = grads[name].reshape(-1, grads[name].shape[-1]).cpu() # (T, d_out)

                if mask is not None and mask.shape[0] == a.shape[0]:
                    a = a[mask]
                    g = g[mask]

                T = a.shape[0]
                if T == 0:
                    continue

                if name not in A:
                    A[name] = torch.zeros(a.shape[1], a.shape[1])
                    G[name] = torch.zeros(g.shape[1], g.shape[1])

                A[name].addmm_(a.t(), a, alpha=1.0 / T)
                G[name].addmm_(g.t(), g, alpha=1.0 / T)

            n_samples += 1

    for h in handles:
        h.remove()

    if n_samples == 0:
        raise ValueError("No sequences processed.")

    return {n: (A[n] / n_samples, G[n] / n_samples) for n in A}


def compute_and_save_fim(
    base_model_path: str,
    adapter_path: str,
    task_tag: str,
    dataset_path: str,
    dataset_name: str | None,
    split: str,
    doc_to_text,
    save_path: str,
    device: str,
    num_samples: int = 512,
    batch_size: int = 4,
    max_length: int = 512,
) -> None:
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
    fisher = dict()

    fisher = compute_diagonal_fim(model, loader, device)
    # fisher = compute_empirical_diagonal_fisher(model, loader, device)
    # fisher = compute_kfac(model, loader, device)

    save_file(fisher, save_path)
    print(f"[{task_tag}] Fisher saved to {save_path}")


def compute_all_fishers(
    base_model_path: str,
    adapter_paths: list,
    output_dir: str,
    device: str,
    num_samples: int,
    batch_size: int,
    max_length: int,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    total = len(DATASET_1)
    for i, ((task_tag, dataset_path, dataset_name, split, doc_to_text), adapter_path) in tqdm(
        enumerate(zip(DATASET_1, adapter_paths), 1), total=total, desc="Computing FIMs"
    ):
        model_tag = os.path.basename(adapter_path.rstrip("/"))
        save_path = os.path.join(output_dir, f"{model_tag}.safetensors")
        print(f"\n[{i}/{total}] Task: {task_tag}  adapter={model_tag}  split={split}")
        print(f"         Save : {save_path}")

        if os.path.exists(save_path):
            print("[WARNING] Already exists, skipping.")
            continue

        compute_and_save_fim(
            base_model_path=base_model_path,
            adapter_path=adapter_path,
            task_tag=task_tag,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            split=split,
            doc_to_text=doc_to_text,
            save_path=save_path,
            device=device,
            num_samples=num_samples,
            batch_size=batch_size,
            max_length=max_length,
        )

    print(f"All fishers saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-family", type=str, default="llama3.1", choices=["llama3.1", "qwen2.5"],
        dest="model_family", help="Model family to use: 'llama3.1' or 'qwen2.5'.",
    )
    parser.add_argument("--output-dir", type=str, default=f"{ROOTDIR}/data/fishers")
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="Device to use for FIM computation (e.g., 'gpu', 'cpu').",
    )
    args = parser.parse_args()
    args.device = parse_device(args.device)

    if args.model_family == "llama3.1":
        base_model_path = LLAMA_BASE_MODEL_PATH
        adapter_paths = LLAMA_ADAPTER_PATHS
    elif args.model_family == "qwen2.5":
        base_model_path = QWEN_BASE_MODEL_PATH
        adapter_paths = QWEN_ADAPTER_PATHS
    else:
        raise ValueError(f"Unsupported model family: {args.model_family}")

    compute_all_fishers(
        base_model_path=base_model_path,
        adapter_paths=adapter_paths,
        output_dir=args.output_dir,
        device=args.device,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
