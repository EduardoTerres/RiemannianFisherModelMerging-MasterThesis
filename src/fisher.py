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
