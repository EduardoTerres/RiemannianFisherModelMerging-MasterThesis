"""Compute transported OFT Lie-basis Kronecker-factored Fisher estimates.

This keeps the CLI of compute_fim.py, but saves ``*_oft_lie_kfac.safetensors``.
For each transported skew gradient G, it accumulates row/column factors:

    A = E[G G^T], B = E[G^T G]

plus a per-block scale that matches the empirical Fisher trace when using
``scale * kron(B, A)``. This is a gradient-space KFAC approximation for the
OFT Lie coordinates, not activation/backprop KFAC for ordinary Linear layers.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file, save_file
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "OrthoFuse"))

from src.diffusion import compute_fim as fim


def kfac_output_name(output_name):
    if output_name.endswith("_oft_lie_fim.safetensors"):
        return output_name.replace("_oft_lie_fim.safetensors", "_oft_lie_kfac.safetensors")
    if output_name.endswith("_fim.safetensors"):
        return output_name.replace("_fim.safetensors", "_kfac.safetensors")
    if output_name.endswith(".safetensors"):
        return output_name.replace(".safetensors", "_kfac.safetensors")
    return f"{output_name}_kfac.safetensors"


def allocate_kfac(named_params):
    factors = {}
    traces = {}
    for name, param in named_params.items():
        n = param.shape[-1]
        factor_shape = (*param.shape[:-2], n, n)
        trace_shape = param.shape[:-2]
        factors[f"{name}.row"] = torch.zeros(factor_shape, dtype=torch.float32)
        factors[f"{name}.col"] = torch.zeros(factor_shape, dtype=torch.float32)
        traces[name] = torch.zeros(trace_shape, dtype=torch.float32)
    return factors, traces


def accumulate_kfac(factors, traces, name, transported_skew_grad):
    grad = transported_skew_grad.float()
    row = grad @ grad.transpose(-1, -2)
    col = grad.transpose(-1, -2) @ grad
    factors[f"{name}.row"] += row.cpu()
    factors[f"{name}.col"] += col.cpu()
    traces[name] += grad.square().sum(dim=(-2, -1)).cpu()


def finalize_kfac(factors, traces, count):
    saved = {}
    for key, value in factors.items():
        saved[key] = value / count

    for name, trace in traces.items():
        trace = trace / count
        row = saved[f"{name}.row"]
        col = saved[f"{name}.col"]
        row_trace = torch.diagonal(row, dim1=-2, dim2=-1).sum(dim=-1)
        col_trace = torch.diagonal(col, dim1=-2, dim2=-1).sum(dim=-1)
        scale = trace / (row_trace * col_trace).clamp_min(1e-30)
        saved[f"{name}.trace"] = trace
        saved[f"{name}.scale"] = scale
    return saved


def compute_one(args, config, base):
    args = fim.apply_entry_defaults(args)
    args.output_name = kfac_output_name(args.output_name)
    fim.log_stage(
        f"START compute_one_kfac entry_type={args.entry_type} dataset_name={args.dataset_name} "
        f"adapter_path={args.adapter_path} train_data_dir={args.train_data_dir} "
        f"output_name={args.output_name}"
    )
    if args.batch_size != 1:
        fim.log_stage("NOTE batch_size > 1 uses batch-mean gradients, matching compute_fim.py semantics.")

    wandb = fim.maybe_init_wandb(args, config)

    device = torch.device(args.device)
    seed = fim.resolve_seed(args, config)
    fim.log_stage(f"Using seed={seed}")
    torch.manual_seed(seed)
    scheduler = base.scheduler
    unet = base.unet
    vae = base.vae
    tokenizer = base.tokenizer
    tokenizer_2 = base.tokenizer_2
    text_encoder = base.text_encoder
    text_encoder_2 = base.text_encoder_2
    moft_layers = base.moft_layers

    fim.log_stage(f"START load adapter safetensors {args.adapter_path}")
    start = time.time()
    adapter_state = load_file(args.adapter_path, device=str(device))
    fim.log_stage(f"DONE load adapter safetensors tensors={len(adapter_state)} in {time.time() - start:.1f}s")

    fim.log_stage("START load adapter state into MOFT layers")
    start = time.time()
    moft_layers.load_state_dict(adapter_state)
    fim.log_stage(f"DONE load adapter state into MOFT layers in {time.time() - start:.1f}s")

    _, loader, prompt = fim.build_dataset_and_prompt(args, config, (tokenizer, tokenizer_2))
    fim.log_stage(f"START tokenize prompt {prompt!r}")
    start = time.time()
    input_ids_list = fim.tokenize_prompt((tokenizer, tokenizer_2), prompt)
    fim.log_stage(f"DONE tokenize prompt in {time.time() - start:.1f}s")

    named_params = {name: param for name, param in moft_layers.named_parameters() if param.requires_grad}
    fim.log_stage(f"Collected trainable MOFT tensors: {len(named_params)}")
    transport_rotations = fim.precompute_transport_rotations(named_params)

    fim.log_stage("START allocate KFAC factors on CPU")
    start = time.time()
    factors, traces = allocate_kfac(named_params)
    fim.log_stage(f"DONE allocate KFAC factors tensors={len(factors) + 2 * len(traces)} in {time.time() - start:.1f}s")

    fim.log_stage("START allocate diagonal Fisher logging tensors on CPU")
    start = time.time()
    fisher_for_logging = {
        name: torch.zeros(
            (*param.shape[:-2], param.shape[-1] * (param.shape[-1] - 1) // 2),
            dtype=torch.float32,
        )
        for name, param in named_params.items()
    }
    fim.log_stage(
        f"DONE allocate diagonal Fisher logging tensors count={len(fisher_for_logging)} "
        f"in {time.time() - start:.1f}s"
    )

    count = 0
    previous_fisher = None
    unet.train()
    fim.log_stage("START KFAC dataloader loop")
    for batch_idx, batch in enumerate(tqdm(loader, desc="Computing transported OFT Lie KFAC"), 1):
        if args.num_samples is not None and count >= args.num_samples:
            break
        if batch_idx == 1:
            fim.log_stage("START first KFAC batch")

        images = batch["pixel_values"].to(device) if args.entry_type == "style" else batch["image"].to(device) * 2.0 - 1.0
        original_sizes = batch["original_sizes"].to(device)
        crop_top_lefts = batch["crop_top_lefts"].to(device)
        batch_size = images.shape[0]

        with torch.no_grad():
            latents = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (batch_size,), device=device)
            target = noise if scheduler.config.prediction_type == "epsilon" else scheduler.get_velocity(latents, noise, timesteps)
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)
            encoder_hidden_states, pooled = fim.encode_tokens((text_encoder, text_encoder_2), input_ids_list)
            encoder_hidden_states = encoder_hidden_states.expand(batch_size, -1, -1)
            pooled = pooled.expand(batch_size, -1)
            add_time_ids = fim.compute_time_ids(original_sizes, crop_top_lefts, config.resolution)

        if batch_idx == 1:
            fim.log_stage("START first batch UNet forward/backward")
        unet.zero_grad(set_to_none=True)
        outputs = unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states,
            added_cond_kwargs={"time_ids": add_time_ids, "text_embeds": pooled},
        ).sample
        loss = F.mse_loss(outputs.float(), target.float(), reduction="mean")
        loss.backward()
        if batch_idx == 1:
            fim.log_stage("DONE first batch UNet forward/backward")

        with torch.no_grad():
            param_iter = named_params.items()
            if batch_idx == 1:
                param_iter = tqdm(param_iter, desc="Accumulating first-batch KFAC", unit="tensor")
            for name, param in param_iter:
                if param.grad is None:
                    continue
                grad = param.grad.detach().float()
                skew_grad = 0.5 * (grad - grad.transpose(-1, -2))
                transported = fim.transport_skew_gradient(skew_grad, transport_rotations[name])
                accumulate_kfac(factors, traces, name, transported)
                transported_grad = fim.upper_triangle_values(transported)
                fisher_for_logging[name] += transported_grad.pow(2).cpu()

        count += batch_size
        if batch_idx == 1:
            fim.log_stage("DONE first KFAC batch")
        if fim.WANDB_LOG_EVERY > 0 and batch_idx % fim.WANDB_LOG_EVERY == 0:
            if batch_idx == 1:
                fim.log_stage("START first wandb Fisher metric log")
            previous_fisher = fim.log_fisher_metrics(
                wandb,
                fisher_for_logging,
                count,
                previous_fisher,
            )
            if batch_idx == 1:
                fim.log_stage("DONE first wandb Fisher metric log")

    if count == 0:
        raise RuntimeError("No samples processed.")

    fim.log_stage(f"START finalize KFAC factors count={count}")
    start = time.time()
    saved = finalize_kfac(factors, traces, count)
    fim.log_stage(f"DONE finalize KFAC factors saved_tensors={len(saved)} in {time.time() - start:.1f}s")

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_name)
    fim.log_stage(f"START save KFAC to {output_path}")
    start = time.time()
    save_file(
        saved,
        output_path,
        metadata={
            "approximation": "transported_oft_lie_gradient_kfac",
            "num_samples": str(count),
            "reconstruct": "scale * kron(col, row)",
        },
    )
    fim.log_stage(f"DONE save KFAC in {time.time() - start:.1f}s")
    if wandb is not None and wandb.run is not None:
        fim.log_stage("START final wandb log/finish")
        wandb.log({"fim/saved": 1, "fim/num_tensors": len(saved), "fim/final_samples": count})
        wandb.finish()
        fim.log_stage("DONE final wandb log/finish")
    print(f"Saved transported OFT Lie-basis KFAC approximation to {output_path}", flush=True)


def main():
    args = fim.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    config_path = fim.resolve_repo_path(args.config_path)
    fim.log_stage(f"START load config {config_path}")
    start = time.time()
    with open(config_path, "r", encoding="utf-8") as handle:
        config = SimpleNamespace(**yaml.safe_load(handle))
    fim.log_stage(f"DONE load config in {time.time() - start:.1f}s")

    device = torch.device(args.device)
    seed = fim.resolve_seed(args, config)
    fim.log_stage(f"Using seed={seed}")
    torch.manual_seed(seed)
    base = fim.load_base_components(args, config, device)

    for entry in fim.iter_entries(args):
        run_args = argparse.Namespace(**vars(args))
        run_args.entry_type = entry["type"]
        run_args.dataset_name = entry["name"]
        compute_one(run_args, config, base)


if __name__ == "__main__":
    main()
