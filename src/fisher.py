"""
Fisher Information Matrix computations (math only).
All I/O (model/dataset loading, file saving) lives in src/scripts/compute_fisher.py.
"""
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm


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

    for batch in tqdm(loader):
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
) -> dict:
    """
    Compute the true diagonal Fisher using model probabilities (not labels).

    diag(F) = E_x sum_t sum_c p(c | x_{<t}) [grad log p(c | x_{<t})]^2

    Approximates the inner expectation over the vocabulary using only the top-k
    tokens under the model's distribution at each position.

    Instead of doing B*T*topk backward passes, this vectorizes over all (b, t)
    positions and does one backward pass per top-k token, reducing to `topk`
    backward passes per batch. Cross-terms between positions are included in the
    squared gradient, which is the standard batched-gradient approximation.

    Args:
        model:            PEFT-wrapped HuggingFace CausalLM (adapter params require_grad=True)
        dataloader:       yields dicts with 'input_ids' and optionally 'attention_mask'
        device:           torch device string
        max_batches:      stop after this many batches (None = full dataloader)
        topk:             number of top-probability tokens used to approximate E_vocab
        renormalize_topk: if True, renormalize top-k probs to sum to 1

    Returns:
        diag_fisher:  dict {param_name: tensor} same shape as trainable params, on CPU
    """
    trainable_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    param_names = [n for n, _ in trainable_params]
    params = [p for _, p in trainable_params]

    diag_fisher = [torch.zeros_like(p, device="cpu") for p in params]
    n_batches = 0

    model.eval()

    for batch_idx, batch in enumerate(tqdm(dataloader)):
        if max_batches is not None and batch_idx >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)                   # (B, T)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :].float()                  # (B, T-1, V)
        B, T, _ = logits.shape

        log_probs = torch.log_softmax(logits, dim=-1)               # (B, T-1, V)

        # Weights are detached — no grad needed through p_c
        with torch.no_grad():
            topk_probs, topk_ids = log_probs.exp().topk(topk, dim=-1)  # (B, T-1, k)
            if renormalize_topk:
                topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        # Float mask over valid (non-padding) positions
        if attention_mask is not None:
            valid_mask = attention_mask[:, :-1].float()             # (B, T-1)
        else:
            valid_mask = torch.ones(B, T, device=device)

        # One backward pass per top-k rank; vectorized over all (b, t) positions.
        for ki in range(topk):
            ids_ki = topk_ids[:, :, ki]                             # (B, T-1)
            weights_ki = topk_probs[:, :, ki]                       # (B, T-1), detached

            # log p for the ki-th top token at every position
            log_p_ki = log_probs.gather(-1, ids_ki.unsqueeze(-1)).squeeze(-1)  # (B, T-1)

            scalar = (weights_ki * valid_mask * log_p_ki).sum()

            grads = torch.autograd.grad(
                scalar, params,
                retain_graph=(ki < topk - 1),
                create_graph=False,
                allow_unused=True,
            )

            for j, g in enumerate(grads):
                if g is not None:
                    diag_fisher[j] += g.detach().float().cpu() ** 2

        n_batches += 1

    return {n: f / max(n_batches, 1) for n, f in zip(param_names, diag_fisher)}


def compute_empirical_fisher(model, loader, device):
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
