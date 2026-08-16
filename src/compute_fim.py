"""Fisher Information Matrix computations and entrypoint to compute and save FIMs."""
import argparse
import copy
import os
import sys
from pathlib import Path
from types import MethodType
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel, get_peft_model
from peft.tuners.oft.layer import OFTLayer

from src.paths import FISHERS_DIR as FIM_OUTPUT_ROOT, MODEL_FAMILIES_D3 as MODEL_FAMILIES
from src.dataset.dataset_3 import DATASET_3_TRAIN as TASKS, build_loader
from src.utils import parse_device

WANDB_PROJECT = "fim"
WANDB_ENTITY = None
WANDB_MODE = os.environ.get("WANDB_MODE", "online")
WANDB_LOG_EVERY_BATCH = 1
WANDB_METRIC_MAX_ENTRIES = 200_000
DEFAULT_MATERIALIZED_ADAPTERS_DIR = Path("/path/to/materialized_adapters")


def infer_oft_block_size(so_dimension: int) -> int:
    """Solve d = n(n-1)/2 for the OFT block size n."""
    block_size = int((1 + (1 + 8 * so_dimension) ** 0.5) / 2)
    if block_size * (block_size - 1) // 2 != so_dimension:
        raise ValueError(
            f"Invalid so(n) dimension {so_dimension}; expected n(n-1)/2."
        )
    return block_size


def oft_coordinates_to_skew(coordinates: torch.Tensor) -> torch.Tensor:
    """Expand upper-triangle OFT coordinates into skew-symmetric blocks."""
    if coordinates.ndim != 2:
        raise ValueError(
            "OFT coordinates must have shape (num_blocks, n(n-1)/2), "
            f"got {tuple(coordinates.shape)}."
        )
    block_size = infer_oft_block_size(coordinates.shape[-1])
    row, col = torch.triu_indices(
        block_size,
        block_size,
        offset=1,
        device=coordinates.device,
    )
    skew = coordinates.new_zeros(coordinates.shape[0], block_size, block_size)
    skew[:, row, col] = coordinates
    return skew - skew.transpose(-1, -2)


def skew_to_oft_coordinates(skew: torch.Tensor) -> torch.Tensor:
    """Extract strict upper-triangle OFT coordinates from skew blocks."""
    row, col = torch.triu_indices(
        skew.shape[-1],
        skew.shape[-1],
        offset=1,
        device=skew.device,
    )
    return skew[:, row, col]


def unnormalized_cayley(skew: torch.Tensor) -> torch.Tensor:
    """Cayley(Omega) = (I - Omega)^-1 (I + Omega)."""
    eye = torch.eye(
        skew.shape[-1],
        dtype=skew.dtype,
        device=skew.device,
    ).expand_as(skew)
    return torch.linalg.solve(eye - skew, eye + skew)


@torch.no_grad()
def principal_rotation_sqrt(rotation: torch.Tensor) -> torch.Tensor:
    """Compute the principal square root of each SO(n) block once via a polar factor."""
    eye = torch.eye(
        rotation.shape[-1],
        dtype=rotation.dtype,
        device=rotation.device,
    ).expand_as(rotation)
    left, singular_values, right_h = torch.linalg.svd(eye + rotation)
    if torch.any(singular_values[..., -1] < 1e-6):
        raise ValueError(
            "Cannot stably compute theta_t^(1/2): a trained OFT rotation "
            "has an eigenvalue too close to -1."
        )
    rotation_half = left @ right_h
    residual = (rotation_half @ rotation_half - rotation).norm(dim=(-2, -1))
    if torch.any(residual > 2e-4 * rotation.shape[-1]):
        raise ValueError(
            "Principal rotation square-root validation failed; "
            f"maximum residual is {residual.max().item():.3e}."
        )
    return rotation_half


def local_oft_forward(self, inputs: torch.Tensor) -> torch.Tensor:
    """
    Apply a fresh Cayley chart on the right of the materialized rotation.

    PEFT normally applies a fresh input rotation before the rotation already
    absorbed in the base weight, which would compose as Cayley(Omega) theta_t.
    Conjugating the fresh input rotation makes the effective model rotation
    theta_t Cayley(Omega), while retaining Omega=0 as the materialized model.
    """
    required_dtype = inputs.dtype
    coordinates = self.weight
    rotation_delta = unnormalized_cayley(
        oft_coordinates_to_skew(coordinates)
    )
    theta = self._fim_materialized_rotation
    input_rotation = theta @ rotation_delta @ theta.transpose(-1, -2)

    rank = self.in_features // self.block_size
    if self.block_share:
        input_rotation = input_rotation.expand(rank, -1, -1)
    input_blocks = inputs.to(coordinates.dtype).reshape(
        *inputs.shape[:-1],
        rank,
        self.block_size,
    )
    rotated = torch.einsum(
        "...rk,rkc->...rc",
        input_blocks,
        input_rotation,
    )
    return rotated.reshape_as(inputs).to(required_dtype)


@torch.no_grad()
def materialize_trained_oft(
    model: PeftModel,
    rotation_path: Path,
) -> tuple[torch.nn.Module, dict[str, torch.Tensor]]:
    """
    Materialize the trained OFT rotations blockwise and unload the adapter.

    Only the (num_blocks, n, n) rotations are saved. No full model-width
    rotation, Fisher, or coordinate transport matrix is formed.
    """
    rotations_by_parameter = {}
    rotations_to_save = {}

    for module_name, module in model.named_modules():
        if not isinstance(module, OFTLayer):
            continue
        active_adapters = [
            name for name in module.active_adapters if name in module.oft_R
        ]
        if len(active_adapters) != 1:
            raise ValueError(
                f"Expected one active OFT adapter in {module_name}, "
                f"found {active_adapters}."
            )

        adapter_name = active_adapters[0]
        rotation_module = module.oft_R[adapter_name]
        coordinates = rotation_module.weight.detach().float()
        rotation = unnormalized_cayley(oft_coordinates_to_skew(coordinates))

        base_weight = module.get_base_layer().weight
        block_size = rotation.shape[-1]
        rank = module.in_features // block_size
        block_rotation = (
            rotation.expand(rank, -1, -1)
            if rotation_module.block_share
            else rotation
        )
        if block_rotation.shape[0] != rank:
            raise ValueError(
                f"OFT block count mismatch in {module_name}: "
                f"rotation has {block_rotation.shape[0]} blocks, expected {rank}."
            )

        weight_blocks = base_weight.detach().float().reshape(
            base_weight.shape[0],
            rank,
            block_size,
        )
        # PEFT applies x @ theta before the linear layer, hence W_t = W @ theta^T.
        materialized = torch.einsum(
            "orb,rab->ora",
            weight_blocks,
            block_rotation,
        ).reshape_as(base_weight)
        base_weight.copy_(materialized.to(base_weight.dtype))

        parameter_name = f"{module_name}.oft_R.{adapter_name}.weight"
        rotations_by_parameter[parameter_name] = rotation.cpu()
        rotations_to_save[f"{parameter_name}.rotation"] = rotation.cpu().contiguous()

    if not rotations_by_parameter:
        raise ValueError("The trained model contains no active OFT layers.")

    rotation_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        rotations_to_save,
        str(rotation_path),
        metadata={
            "format": "blockwise_oft_rotations",
            "cayley": "(I-Omega)^-1(I+Omega)",
        },
    )
    print(f"Materialized OFT rotations saved to {rotation_path}", flush=True)

    # The base weights above already contain the rotations, so unload without merging.
    base_model = model.unload()
    # PEFT leaves this marker on some base nn.Module implementations. Remove it
    # before attaching the genuinely fresh adapter.
    if hasattr(base_model, "peft_config"):
        delattr(base_model, "peft_config")
    return base_model, rotations_by_parameter


def attach_fresh_zero_oft(
    base_model: torch.nn.Module,
    adapter_config: PeftConfig,
    device: str,
    rotations_by_parameter: dict[str, torch.Tensor] | None,
) -> tuple[PeftModel, dict[str, torch.Tensor]]:
    """Attach a trainable zero OFT chart and precompute theta_t^(1/2) per block."""
    fresh_config = copy.deepcopy(adapter_config)
    fresh_config.inference_mode = False
    fresh_config.init_weights = True
    fresh_config.module_dropout = 0.0
    # The PEFT Neumann implementation has d Cayley_0 = 2 id, as required here.
    fresh_config.use_cayley_neumann = True

    model = get_peft_model(base_model, fresh_config)
    model.enable_adapter_layers()
    model.to(device)
    model.eval()

    named_params = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not named_params:
        raise ValueError("Fresh OFT attachment produced no trainable parameters.")

    rotation_modules = {}
    for module_name, module in model.named_modules():
        if not isinstance(module, OFTLayer):
            continue
        for adapter_name in module.active_adapters:
            if adapter_name in module.oft_R:
                parameter_name = f"{module_name}.oft_R.{adapter_name}.weight"
                rotation_modules[parameter_name] = module.oft_R[adapter_name]

    theta_halves = {}
    rotations_by_parameter = rotations_by_parameter or {}
    for name, parameter in named_params.items():
        if ".oft_R." not in name or not name.endswith(".weight"):
            raise ValueError(f"Unexpected non-OFT trainable parameter: {name}.")
        if torch.count_nonzero(parameter.detach()).item():
            raise ValueError(f"Fresh OFT coordinates are not zero: {name}.")

        block_size = infer_oft_block_size(parameter.shape[-1])
        rotation = rotations_by_parameter.get(name)
        if rotation is None:
            if rotations_by_parameter:
                raise KeyError(f"No materialized trained rotation found for {name}.")
            rotation = torch.eye(
                block_size,
                dtype=torch.float32,
            ).expand(parameter.shape[0], block_size, block_size).clone()
        expected_shape = (parameter.shape[0], block_size, block_size)
        if tuple(rotation.shape) != expected_shape:
            raise ValueError(
                f"Rotation shape mismatch for {name}: got {tuple(rotation.shape)}, "
                f"expected {expected_shape}."
            )
        rotation = rotation.to(device=parameter.device, dtype=torch.float32)
        theta_halves[name] = principal_rotation_sqrt(rotation)

        rotation_module = rotation_modules.get(name)
        if rotation_module is None:
            raise KeyError(f"No fresh OFT rotation module found for {name}.")
        rotation_module.register_buffer(
            "_fim_materialized_rotation",
            rotation,
            persistent=False,
        )
        rotation_module.forward = MethodType(local_oft_forward, rotation_module)

    unexpected = set(rotations_by_parameter) - set(named_params)
    if unexpected:
        raise KeyError(
            "Materialized rotations did not match fresh OFT parameters: "
            + ", ".join(sorted(unexpected)[:5])
        )
    return model, theta_halves


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
    theta_halves: dict[str, torch.Tensor] | None,
    metric_prefix: str | None = None,
    wandb_step_offset: int = 0,
    transport: bool = True,
) -> dict:
    """
    Compute E[u_t,x**2] after Cayley correction and, optionally, blockwise transport.

    For every sample and OFT parameter, the raw coordinate gradient G is first
    converted to the left-trivialized gradient W = G/2. If `transport` is set,
    each skew block is then transported as U = theta_t^(1/2) W theta_t^(-1/2)
    before being converted back to the OFT upper-triangle basis and squared;
    at the pretrained chart theta_t is the identity, so transport is skipped
    entirely rather than applying a no-op matmul. This never constructs a
    d-by-d Fisher or coordinate transport matrix.
    """
    if transport and theta_halves is None:
        raise ValueError("theta_halves must be provided when transport=True.")
    model.eval()
    model.to(device)

    named_params = {
        name: p
        for name, p in model.named_parameters()
        if p.requires_grad
    }

    if transport and set(named_params) != set(theta_halves):
        missing = set(named_params) - set(theta_halves)
        extra = set(theta_halves) - set(named_params)
        raise KeyError(
            f"theta_t^(1/2) map does not match trainable OFT parameters; "
            f"missing={sorted(missing)[:5]}, extra={sorted(extra)[:5]}."
        )

    fisher = {
        name: torch.zeros_like(parameter, dtype=torch.float32, device="cpu")
        for name, parameter in named_params.items()
    }

    def sequence_nll(logits, labels):
        """Summed target-token NLL for one sample."""
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        valid = shift_labels != -100
        safe_labels = shift_labels.masked_fill(~valid, 0)
        token_losses = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            safe_labels.view(-1),
            reduction="none",
        ).view_as(shift_labels)
        return (token_losses * valid).sum()

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
        labels = batch["labels"].to(device)

        batch_size = input_ids.shape[0]

        for b in range(batch_size):
            model.zero_grad(set_to_none=True)

            sample_input_ids = input_ids[b:b + 1]
            sample_attention_mask = attention_mask[b:b + 1]
            sample_labels = labels[b:b + 1]

            outputs = model(
                input_ids=sample_input_ids,
                attention_mask=sample_attention_mask,
            )

            loss = sequence_nll(
                logits=outputs.logits,
                labels=sample_labels,
            )

            loss.backward()

            with torch.no_grad():
                for name, p in named_params.items():
                    if p.grad is None:
                        continue

                    # d Cayley_0 = 2 id, hence W = G/2 before transport.
                    left_trivialized = oft_coordinates_to_skew(
                        0.5 * p.grad.detach().float()
                    )
                    if transport:
                        theta_half = theta_halves[name]
                        transported = (
                            theta_half
                            @ left_trivialized
                            @ theta_half.transpose(-1, -2)
                        )
                        transported = 0.5 * (
                            transported - transported.transpose(-1, -2)
                        )
                    else:
                        # theta_t is the identity at the pretrained chart.
                        transported = left_trivialized
                    transported_coordinates = skew_to_oft_coordinates(transported)
                    fisher[name].add_(transported_coordinates.square().cpu())

            num_samples += 1
        log_convergence_metrics(batch_idx, num_samples)

    if num_samples == 0:
        raise ValueError("The DataLoader produced zero samples.")

    for name in fisher:
        fisher[name] /= num_samples

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
        wandb.define_metric(f"fim/{model_state}/step")
        wandb.define_metric(f"fim/{model_state}/*", step_metric=f"fim/{model_state}/step")


def load_materialized_adapter_model(
    base_model_path: str,
    adapter_path: str,
    device: str,
    materialized_rotation_path: Path,
) -> tuple[torch.nn.Module, dict[str, torch.Tensor]]:
    """Materialize a trained adapter and attach a fresh zero OFT chart."""
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float32, device_map=None
    )
    adapter_config = PeftConfig.from_pretrained(adapter_path)

    trained_model = PeftModel.from_pretrained(
        base,
        adapter_path,
        is_trainable=False,
    )
    base, rotations_by_parameter = materialize_trained_oft(
        trained_model,
        materialized_rotation_path,
    )

    return attach_fresh_zero_oft(
        base_model=base,
        adapter_config=adapter_config,
        device=device,
        rotations_by_parameter=rotations_by_parameter,
    )


def load_pretrained_model(
    base_model_path: str,
    adapter_path: str,
    device: str,
) -> torch.nn.Module:
    """Attach a fresh zero OFT chart directly to the pretrained base model.

    The adapter's config is only used to determine which modules the OFT
    chart targets; no trained rotation is applied, so the model output
    matches the pretrained model exactly. Since theta_t is the identity here,
    gradients are used untransported (see `transport` in
    `compute_empirical_diagonal_transported_fisher`).
    """
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float32, device_map=None
    )
    adapter_config = PeftConfig.from_pretrained(adapter_path)

    model, _theta_halves = attach_fresh_zero_oft(
        base_model=base,
        adapter_config=adapter_config,
        device=device,
        rotations_by_parameter=None,
    )
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
    log_to_wandb: bool = False,
    dataset_cache_dir: str | None = None,
    wandb_step_offset: int = 0,
    materialized_rotation_path: Path | None = None,
    repeat_to_num_samples: bool = False,
    model_state: str = "finetuned",
) -> None:
    if model_state not in ("finetuned", "pretrained"):
        raise ValueError(f"Unknown model_state: {model_state}.")

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    theta_halves = None
    if model_state == "finetuned":
        if materialized_rotation_path is None:
            raise ValueError("materialized_rotation_path must be provided.")
        model, theta_halves = load_materialized_adapter_model(
            base_model_path=base_model_path,
            adapter_path=adapter_path,
            device=device,
            materialized_rotation_path=materialized_rotation_path,
        )
    else:
        model = load_pretrained_model(
            base_model_path=base_model_path,
            adapter_path=adapter_path,
            device=device,
        )

    loader = build_loader(
        dataset_path, dataset_name, split, doc_to_text,
        tokenizer, num_samples, batch_size, max_length, task=task_tag,
        cache_dir=dataset_cache_dir,
        repeat_to_num_samples=repeat_to_num_samples,
    )

    metric_prefix = f"fim/{model_state}" if log_to_wandb else None
    fisher = compute_empirical_diagonal_transported_fisher(
        model,
        loader,
        device,
        theta_halves=theta_halves,
        metric_prefix=metric_prefix,
        wandb_step_offset=wandb_step_offset,
        transport=(model_state == "finetuned"),
    )

    fisher = {name: value.clamp_min(1e-6) for name, value in fisher.items()}
    save_file(fisher, str(save_path))
    print(f"[{task_tag}] {model_state} Fisher saved to {save_path}", flush=True)


def compute_task_fisher(
    base_model_path: str,
    adapter_path: str,
    fisher_finetuned_path: str,
    fisher_pretrained_path: str,
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
    materialized_adapters_dir: Path,
    repeat_to_num_samples: bool,
    model_states: tuple[str, ...] = ("pretrained", "finetuned"),
) -> None:
    task_tag, dataset_path, dataset_name, split, doc_to_text = TASKS[task_index]
    adapter_tag = os.path.basename(adapter_path.rstrip("/"))
    output_dir = FIM_OUTPUT_ROOT / model_family / task_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    materialized_rotation_path = (
        materialized_adapters_dir
        / model_family
        / adapter_tag
        / "oft_rotations.safetensors"
    )
    fisher_paths = {
        "finetuned": fisher_finetuned_path,
        "pretrained": fisher_pretrained_path,
    }

    print(f"Dataset: {task_tag}", flush=True)
    print(f"Adapter model: {adapter_tag}", flush=True)
    print(f"Task index: {task_index}", flush=True)
    print(f"Debug: {debug}", flush=True)
    print(f"Materialized rotations: {materialized_rotation_path}", flush=True)

    maybe_init_wandb(model_family, task_tag, adapter_tag, debug, log_to_wandb)

    for model_state in model_states:
        save_path = Path(fisher_paths[model_state])
        save_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"\nComputing {model_state} FIM", flush=True)
        print(f"Save: {save_path}", flush=True)

        materialization_missing = (
            model_state == "finetuned" and not materialized_rotation_path.exists()
        )
        if save_path.exists() and not force_compute and not materialization_missing:
            print("[WARNING] Already exists, skipping.", flush=True)
            continue
        if materialization_missing and save_path.exists():
            print(
                "[WARNING] Fisher exists but materialized rotations are missing; "
                "recomputing the fine-tuned Fisher.",
                flush=True,
            )
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
            log_to_wandb=log_to_wandb,
            dataset_cache_dir=dataset_cache_dir,
            wandb_step_offset=0,
            materialized_rotation_path=materialized_rotation_path,
            repeat_to_num_samples=repeat_to_num_samples,
            model_state=model_state,
        )

        print(f"{model_state.capitalize()} FIM saved to {output_dir}/", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-family", type=str, default="llama3.1", choices=list(MODEL_FAMILIES),
        dest="model_family", help="Model family to use.",
    )
    parser.add_argument("--task-index", type=int, required=True, choices=range(len(TASKS)))
    parser.add_argument("--num-samples", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--dataset-cache-dir", type=str, default=str(Path(__file__).resolve().parents[1] / "data" / "hf_cache"))
    parser.add_argument(
        "--materialized-adapters-dir",
        type=Path,
        default=DEFAULT_MATERIALIZED_ADAPTERS_DIR,
        help="Root directory for saved materialized OFT rotation tensors.",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument(
        "--force-compute",
        action="store_true",
        help="Recompute and overwrite existing FIM files instead of skipping them.",
    )
    parser.add_argument(
        "--repeat-to-num-samples",
        action="store_true",
        help="Cycle through short datasets until exactly --num-samples examples are used.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use for FIM computation (e.g., 'gpu', 'cpu').",
    )
    parser.add_argument(
        "--model-states",
        nargs="+",
        default=["pretrained", "finetuned"],
        choices=["pretrained", "finetuned"],
        help="Which model state(s) to compute the FIM for.",
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
    fisher_finetuned_path = model_family.fisher_finetuned_paths[args.task_index]
    fisher_pretrained_path = model_family.fisher_pretrained_paths[args.task_index]

    compute_task_fisher(
        base_model_path=base_model_path,
        adapter_path=adapter_path,
        fisher_finetuned_path=fisher_finetuned_path,
        fisher_pretrained_path=fisher_pretrained_path,
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
        materialized_adapters_dir=args.materialized_adapters_dir,
        repeat_to_num_samples=args.repeat_to_num_samples,
        model_states=tuple(args.model_states),
    )
