"""
Fisher Information Matrix computations (math only).
All I/O (model/dataset loading, file saving) lives in src/scripts/compute_fisher.py.
"""
import torch
from torch.utils.data import DataLoader


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
    diag_fisher = {
        n: torch.zeros_like(p, device="cpu")
        for n, p in model.named_parameters() if p.requires_grad
    }

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        model.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        outputs.loss.backward()

        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                diag_fisher[n] += p.grad.detach().cpu() ** 2

    n_batches = len(loader)
    return {n: v / n_batches for n, v in diag_fisher.items()}
