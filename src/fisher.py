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


def compute_diagonal_fim_true(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    max_batches: int | None = None,
    topk: int = 50,
    renormalize_topk: bool = True,
) -> tuple[dict, int]:
    """
    Compute the true diagonal Fisher using model probabilities (not labels).

    diag(F) = E_x sum_t sum_c p(c | x_{<t}) [grad log p(c | x_{<t})]^2

    Approximates the inner expectation over the vocabulary using only the top-k
    tokens under the model's distribution at each position.

    Args:
        model:            PEFT-wrapped HuggingFace CausalLM (adapter params require_grad=True)
        dataloader:       yields dicts with 'input_ids' and optionally 'attention_mask'
        device:           torch device string
        max_batches:      stop after this many batches (None = full dataloader)
        topk:             number of top-probability tokens used to approximate E_vocab
        renormalize_topk: if True, renormalize top-k probs to sum to 1

    Returns:
        (diag_fisher, n_positions)
        diag_fisher:  dict {param_name: tensor} same shape as trainable params, accumulated on CPU
        n_positions:  total number of token positions processed
    """
    trainable_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    diag_fisher = {n: torch.zeros_like(p, device="cpu") for n, p in trainable_params}

    model.eval()
    n_positions = 0

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)                          # (B, T)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits                                         # (B, T, V)

        # Valid positions: all but the last token (each predicts the next)
        logits = logits[:, :-1, :].contiguous()                            # (B, T-1, V)
        B, T, _ = logits.shape

        log_probs_all = torch.log_softmax(logits.float(), dim=-1)          # (B, T-1, V)
        probs_all = log_probs_all.exp()

        # Select top-k tokens per position
        topk_probs, topk_ids = probs_all.topk(topk, dim=-1)               # (B, T-1, k)

        if renormalize_topk:
            topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        # Count valid (non-padding) positions
        if attention_mask is not None:
            valid_mask = attention_mask[:, :-1].bool()                     # (B, T-1)
        else:
            valid_mask = torch.ones(B, T, dtype=torch.bool, device=device)

        n_positions += valid_mask.sum().item()

        # Accumulate Fisher contribution position by position
        for b in range(B):
            for t in range(T):
                if not valid_mask[b, t]:
                    continue

                # grad sum_c p(c) [grad log p(c)]^2  at this position
                fisher_accum = {n: torch.zeros_like(p) for n, p in trainable_params}

                for ki in range(topk):
                    c = topk_ids[b, t, ki].item()
                    p_c = topk_probs[b, t, ki]                             # scalar

                    log_p_c = log_probs_all[b, t, c]                      # scalar (with grad)

                    grads = torch.autograd.grad(
                        log_p_c, [p for _, p in trainable_params],
                        retain_graph=True,
                        allow_unused=True,
                    )

                    for (n, _), g in zip(trainable_params, grads):
                        if g is not None:
                            fisher_accum[n] += p_c.item() * (g.detach() ** 2)

                for n, v in fisher_accum.items():
                    diag_fisher[n] += v.cpu()

    return diag_fisher, n_positions
