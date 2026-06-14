"""Fisher Information Matrix computations and entrypoint to compute and save FIMs."""
import argparse
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.merging import OFTMerging
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.paths import FISHERS_DIR as FIM_OUTPUT_ROOT, MODEL_FAMILIES_D3 as MODEL_FAMILIES
from src.dataset.dataset_3 import DATASET_3_TRAIN as TASKS, build_loader
from src.utils import parse_device

WANDB_PROJECT = "fim"
WANDB_ENTITY = None
WANDB_MODE = os.environ.get("WANDB_MODE", "online")
WANDB_LOG_EVERY_BATCH = 1
WANDB_METRIC_MAX_ENTRIES = 200_000

_merging = OFTMerging()
_manifold = _merging.manifold


def log_fisher_convergence_metrics(
    fisher: dict[str, torch.Tensor],
    normalizer: int,
    metric_prefix: str | None,
    previous_fishers: dict[str, torch.Tensor] | None,
    metric_step: int,
) -> dict[str, torch.Tensor] | None:
    if metric_prefix is None:
        return previous_fishers
    try:
        import wandb
    except ImportError:
        return previous_fishers
    if wandb.run is None:
        return previous_fishers

    running_fishers = {
        name: (value.detach().float() / max(normalizer, 1)).cpu()
        for name, value in fisher.items()
    }
    flat = torch.cat([value.flatten() for value in running_fishers.values()])
    metric_flat = flat
    if metric_flat.numel() > WANDB_METRIC_MAX_ENTRIES:
        step = metric_flat.numel() / WANDB_METRIC_MAX_ENTRIES
        sample_idx = (torch.arange(WANDB_METRIC_MAX_ENTRIES) * step).long()
        metric_flat = metric_flat[sample_idx]
    payload = {
        f"{metric_prefix}/step": metric_step,
        f"{metric_prefix}/num_samples": normalizer,
        f"{metric_prefix}/mean": flat.mean().item(),
        f"{metric_prefix}/p50": torch.quantile(metric_flat, 0.50).item(),
        f"{metric_prefix}/p90": torch.quantile(metric_flat, 0.90).item(),
        f"{metric_prefix}/p99": torch.quantile(metric_flat, 0.99).item(),
        f"{metric_prefix}/max": flat.max().item(),
        f"{metric_prefix}/top_1pct_mass": (
            metric_flat.topk(max(1, metric_flat.numel() // 100)).values.sum()
            / metric_flat.sum().clamp_min(1e-12)
        ).item(),
    }

    if previous_fishers is not None:
        prev_flat = torch.cat([previous_fishers[name].flatten() for name in running_fishers])
        diff = flat - prev_flat
        payload[f"{metric_prefix}/rel_fro_change"] = (
            diff.norm() / prev_flat.norm().clamp_min(1e-12)
        ).item()
        payload[f"{metric_prefix}/cosine_to_prev"] = F.cosine_similarity(
            flat,
            prev_flat,
            dim=0,
            eps=1e-12,
        ).item()

        layer_changes = torch.tensor([
            ((running - previous_fishers[name]).norm()
             / previous_fishers[name].norm().clamp_min(1e-12)).item()
            for name, running in running_fishers.items()
        ])
        payload.update({
            f"{metric_prefix}/layer_rel_fro_change_mean": layer_changes.mean().item(),
            f"{metric_prefix}/layer_rel_fro_change_p90": torch.quantile(layer_changes, 0.90).item(),
            f"{metric_prefix}/layer_rel_fro_change_max": layer_changes.max().item(),
        })

    wandb.log(payload)
    return {
        name: value.clone()
        for name, value in running_fishers.items()
    }


def compute_empirical_diagonal_transported_fisher(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    metric_prefix: str | None = None,
    wandb_step_offset: int = 0,
) -> dict:
    """
    Compute the diagonal empirical Fisher: E[grad log p]^2.

    For OFT parameters stored in upper-triangle coordinates, this computes the
    transported diagonal Fisher

        diag(P_t F_t P_t^T)

    by transporting each per-sample gradient before squaring:

        g_tilde = P_t g
        fisher += g_tilde^2

    Assumes the following objects already exist globally or in scope:

        _merging.oft_params_to_skew_matrix(...)
        _manifold.compute_Pt(...)

    Args:
        model:  an already-loaded, eval-mode CausalLM
        loader: DataLoader yielding dicts with 'input_ids' and 'attention_mask'
        device: torch device string

    Returns:
        dict mapping parameter name -> diagonal Fisher tensor on CPU
    """

    model.eval()
    model.to(device)

    named_params = {
        name: p
        for name, p in model.named_parameters()
        if p.requires_grad
    }

    fisher = {
        name: torch.zeros_like(p, device="cpu")
        for name, p in named_params.items()
    }

    def infer_block_size_from_son_dimension(son_dimension: int) -> int:
        """
        Solve d = n(n-1)/2 for n.
        """
        block_size = int((1 + (1 + 8 * son_dimension) ** 0.5) / 2)

        if block_size * (block_size - 1) // 2 != son_dimension:
            raise ValueError(
                f"Invalid so(n) dimension: {son_dimension}. "
                "Expected d = n(n-1)/2."
            )

        return block_size

    def is_oft_coordinate_parameter(p: torch.Tensor) -> bool:
        """
        OFT coordinates are expected to have shape:

            (num_blocks, son_dimension)

        where son_dimension = block_size * (block_size - 1) // 2.

        This check is intentionally conservative.
        """
        if p.ndim != 2:
            return False

        son_dimension = p.shape[-1]

        try:
            infer_block_size_from_son_dimension(son_dimension)
            return True
        except ValueError:
            return False

    @torch.no_grad()
    def precompute_transport_matrices() -> dict:
        """
        Precompute P_t for each OFT parameter tensor.

        For each OFT parameter tensor A_t with shape

            (num_blocks, son_dimension),

        we construct the skew matrix Omega_t and then compute

            P_t = P_{theta_t -> theta_LLM}.
        """
        transport_matrices = {}

        for name, p in named_params.items():
            if not is_oft_coordinate_parameter(p):
                continue

            son_dimension = p.shape[-1]
            block_size = infer_block_size_from_son_dimension(son_dimension)

            skew_matrix = _merging.oft_params_to_skew_matrix(
                p.detach(),
                son_dimension,
            )

            Pt = _manifold.compute_Pt(
                skew_matrix=skew_matrix,
                block_size=block_size,
            )

            transport_matrices[name] = Pt.detach()

        return transport_matrices

    transport_matrices = precompute_transport_matrices()

    def sequence_nll(logits, input_ids, attention_mask):
        """
        CausalLM negative log-likelihood for one sample.

        Since Fisher uses grad log p squared, we can use the gradient of
        negative log-likelihood because the sign disappears after squaring.
        """
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous()

        vocab_size = shift_logits.shape[-1]

        token_losses = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            reduction="none",
        )

        token_losses = token_losses.view_as(shift_labels)

        nll = (token_losses * shift_mask).sum()

        return nll

    previous_fishers = None

    def log_convergence_metrics(batch_idx: int, num_samples: int) -> None:
        nonlocal previous_fishers
        if metric_prefix is None or batch_idx % WANDB_LOG_EVERY_BATCH:
            return
        previous_fishers = log_fisher_convergence_metrics(
            fisher=fisher,
            normalizer=num_samples,
            metric_prefix=metric_prefix,
            previous_fishers=previous_fishers,
            metric_step=wandb_step_offset + num_samples,
        )

    num_samples = 0

    for batch_idx, batch in enumerate(tqdm(loader, desc="Computing transported diagonal FIM"), 1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        batch_size = input_ids.shape[0]

        for b in range(batch_size):
            model.zero_grad(set_to_none=True)

            sample_input_ids = input_ids[b:b + 1]
            sample_attention_mask = attention_mask[b:b + 1]

            outputs = model(
                input_ids=sample_input_ids,
                attention_mask=sample_attention_mask,
            )

            loss = sequence_nll(
                logits=outputs.logits,
                input_ids=sample_input_ids,
                attention_mask=sample_attention_mask,
            )

            loss.backward()

            with torch.no_grad():
                for name, p in named_params.items():
                    if p.grad is None:
                        continue

                    grad = p.grad.detach()

                    if name in transport_matrices:
                        Pt = transport_matrices[name]

                        # grad has shape:
                        #   (num_blocks, son_dimension)
                        #
                        # Pt has shape:
                        #   (num_blocks, son_dimension, son_dimension)
                        #
                        # transported_grad[b] = Pt[b] @ grad[b]
                        transported_grad = torch.einsum(
                            "bij,bj->bi",
                            Pt,
                            grad,
                        )

                        fisher[name] += transported_grad.pow(2).cpu()

                    else:
                        # Fallback: ordinary diagonal empirical Fisher.
                        fisher[name] += grad.pow(2).cpu()

            num_samples += 1
        log_convergence_metrics(batch_idx, num_samples)

    if num_samples == 0:
        raise ValueError("The DataLoader produced zero samples.")

    for name in fisher:
        fisher[name] /= num_samples

    return fisher

def compute_empirical_diagonal_fisher(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    metric_prefix: str | None = None,
    wandb_step_offset: int = 0,
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
    previous_fishers = None

    def log_convergence_metrics(num_batches: int) -> None:
        nonlocal previous_fishers
        if metric_prefix is None or num_batches % WANDB_LOG_EVERY_BATCH:
            return
        previous_fishers = log_fisher_convergence_metrics(
            fisher=fisher,
            normalizer=num_batches,
            metric_prefix=metric_prefix,
            previous_fishers=previous_fishers,
            metric_step=wandb_step_offset + num_batches,
        )

    for batch in tqdm(loader, desc="Computing Diagonal FIM"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        shift_logits = outputs.logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        label_mask = (shift_labels != -100).float()

        # Clamp -100 to 0 so nll_loss doesn't index out of bounds; masked out below
        shift_labels_safe = shift_labels.clone()
        shift_labels_safe[shift_labels == -100] = 0

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_nll = F.nll_loss(
            log_probs.view(-1, log_probs.size(-1)),
            shift_labels_safe.reshape(-1),
            reduction="none",
        ).view(shift_labels.shape)

        loss = (token_nll * label_mask).sum() / label_mask.sum().clamp(min=1)

        # if count == 0 and tokenizer is not None:
        #     for b in range(input_ids.shape[0]):
        #         prompt = tokenizer.decode(input_ids[b], skip_special_tokens=False)
        #         fisher_ids = shift_labels[b][label_mask[b].bool()].tolist()
        #         print(f"\n--- Sample {b} ---")
        #         print(f"Prompt: {prompt!r}")
        #         decoded = tokenizer.decode(fisher_ids, skip_special_tokens=False)
        #         print(f"Loss tokens ({len(fisher_ids)}): ids={fisher_ids}  text={decoded!r}")

        model.zero_grad()
        loss.backward()

        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] += p.grad.data ** 2

        count += 1
        log_convergence_metrics(count)

    if count == 0:
        raise ValueError("The DataLoader produced zero batches.")

    for n in fisher:
        fisher[n] /= count
        fisher[n] = fisher[n].cpu()

    return fisher

def compute_diagonal_fisher(model, loader, device, tokenizer=None):
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
        labels = batch["labels"].to(device)

        answer_mask = (labels[:, 1:] != -100).float()  # (B, T-1)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            sampled = torch.multinomial(
                F.softmax(logits[:, :-1, :].reshape(-1, logits.shape[-1]), dim=-1), 1
            ).reshape(input_ids.shape[0], -1)  # (B, T-1)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        log_probs = F.log_softmax(outputs.logits[:, :-1, :], dim=-1)
        token_log_probs = log_probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        token_log_probs = token_log_probs * answer_mask

        seq_log_probs = token_log_probs.sum(dim=-1)   # (B,)
        batch_size = seq_log_probs.shape[0]

        if n_sequences == 0 and tokenizer is not None:
            for b in range(batch_size):
                prompt = tokenizer.decode(input_ids[b], skip_special_tokens=False)
                ans_mask_b = answer_mask[b].bool()
                backprop_ids = sampled[b][ans_mask_b].tolist()
                backprop_text = tokenizer.decode(backprop_ids, skip_special_tokens=False)
                print(f"\n--- Sample {b} ---")
                print(f"Prompt:  {prompt!r}")
                print(f"Backprop: {backprop_text!r}")

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


def maybe_init_wandb(
    model_family: str,
    task_tag: str,
    adapter_tag: str,
    debug: bool,
    enabled: bool,
) -> None:
    if not enabled:
        return
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb is not installed in this environment.") from exc
    if wandb.run is not None:
        return
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        mode=WANDB_MODE,
        name=f"fim-{model_family}-{task_tag}",
        group="compute-fim",
        config={
            "model_family": model_family,
            "task": task_tag,
            "adapter": adapter_tag,
            "debug": debug,
        },
    )
    for model_state in ("pretrained", "finetuned"):
        step_metric = f"fim/{model_state}/step"
        wandb.define_metric(step_metric)
        wandb.define_metric(f"fim/{model_state}/*", step_metric=step_metric)


def zero_trainable_adapter_parameters(model: torch.nn.Module) -> None:
    with torch.no_grad():
        for _, param in model.named_parameters():
            if param.requires_grad:
                param.zero_()


def load_adapter_model(
    base_model_path: str,
    adapter_path: str,
    device: str,
    zero_adapter: bool,
) -> torch.nn.Module:
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float32, device_map=None
    )
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=True)
    # If the model is pretrained we still want the OFT for backpropagating for FIM computation,
    # but we want to zero out the adapter parameters so they don't contribute to the FIM.
    if zero_adapter:
        zero_trainable_adapter_parameters(model)
    model.enable_adapter_layers()
    model.to(device)
    model.eval()
    return model


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
    model_state: str = "finetuned",
    log_to_wandb: bool = False,
    dataset_cache_dir: str | None = None,
    wandb_step_offset: int = 0,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_adapter_model(
        base_model_path=base_model_path,
        adapter_path=adapter_path,
        device=device,
        zero_adapter=(model_state == "pretrained"),
    )

    loader = build_loader(
        dataset_path, dataset_name, split, doc_to_text,
        tokenizer, num_samples, batch_size, max_length, task=task_tag,
        cache_dir=dataset_cache_dir,
    )

    metric_prefix = f"fim/{model_state}" if log_to_wandb else None
    if model_state == "pretrained":
        fisher = compute_empirical_diagonal_fisher(
            model,
            loader,
            device,
            metric_prefix=metric_prefix,
            wandb_step_offset=wandb_step_offset,
        )
    elif model_state == "finetuned":
        fisher = compute_empirical_diagonal_transported_fisher(
            model,
            loader,
            device,
            metric_prefix=metric_prefix,
            wandb_step_offset=wandb_step_offset,
        )
    else:
        raise ValueError(f"Unsupported model_state: {model_state}")

    fisher = {name: value.clamp_min(1e-6) for name, value in fisher.items()}
    save_file(fisher, str(save_path))
    print(f"[{task_tag}] {model_state} Fisher saved to {save_path}", flush=True)


def compute_task_fishers(
    base_model_path: str,
    adapter_path: str,
    fisher_paths_by_state: dict[str, str],
    model_family: str,
    task_index: int,
    device: str,
    num_samples: int,
    batch_size: int,
    max_length: int,
    dataset_cache_dir: str | None,
    debug: bool,
    log_to_wandb: bool,
    force_compute: bool,
) -> None:
    task_tag, dataset_path, dataset_name, split, doc_to_text = TASKS[task_index]
    adapter_tag = os.path.basename(adapter_path.rstrip("/"))
    output_dir = FIM_OUTPUT_ROOT / model_family / task_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {task_tag}", flush=True)
    print(f"Adapter model: {adapter_tag}", flush=True)
    print(f"Task index: {task_index}", flush=True)
    print(f"Debug: {debug}", flush=True)

    maybe_init_wandb(model_family, task_tag, adapter_tag, debug, log_to_wandb)

    for model_state in ("pretrained", "finetuned"):
        save_path = Path(fisher_paths_by_state[model_state])
        save_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"\nComputing {model_state} FIM", flush=True)
        print(f"Save: {save_path}", flush=True)
        if save_path.exists() and not force_compute:
            print("[WARNING] Already exists, skipping.", flush=True)
            continue
        if save_path.exists() and force_compute:
            print("[WARNING] Already exists, recomputing due to --force-compute.", flush=True)
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
            model_state=model_state,
            log_to_wandb=log_to_wandb,
            dataset_cache_dir=dataset_cache_dir,
            wandb_step_offset=0,
        )

    print(f"FIMs saved to {output_dir}/", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-family", type=str, default="llama3.1", choices=list(MODEL_FAMILIES),
        dest="model_family", help="Model family to use.",
    )
    parser.add_argument("--task-index", type=int, required=True, choices=range(len(TASKS)))
    parser.add_argument("--num-samples", type=int, default=2048)  # 1024
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=1024)  # 1024
    parser.add_argument("--dataset-cache-dir", type=str, default=str(Path(__file__).resolve().parents[1] / "data" / "hf_cache"))
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument(
        "--force-compute",
        action="store_true",
        help="Recompute and overwrite existing FIM files instead of skipping them.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use for FIM computation (e.g., 'gpu', 'cpu').",
    )
    args = parser.parse_args()
    args.device = parse_device(args.device)
    if args.debug:
        args.num_samples = 2
        args.batch_size = 1
        args.max_length = min(args.max_length, 128)

    model_family = MODEL_FAMILIES[args.model_family]
    base_model_path = model_family.base_model_path
    adapter_path = model_family.adapter_paths[args.task_index]
    fisher_paths_by_state = {
        "pretrained": model_family.fisher_pretrained_paths[args.task_index],
        "finetuned": model_family.fisher_finetuned_paths[args.task_index],
    }

    compute_task_fishers(
        base_model_path=base_model_path,
        adapter_path=adapter_path,
        fisher_paths_by_state=fisher_paths_by_state,
        model_family=args.model_family,
        task_index=args.task_index,
        device=args.device,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        max_length=args.max_length,
        dataset_cache_dir=args.dataset_cache_dir,
        debug=args.debug,
        log_to_wandb=not args.no_wandb,
        force_compute=args.force_compute,
    )
