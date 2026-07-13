import argparse
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import yaml
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "OrthoFuse"))

from moft.data.dataset_sdxl import ImageDataset, StyleDataset, collate_fn, compute_time_ids, encode_tokens, tokenize_prompt
from moft.model.moft import MOFTCrossAttnProcessor
from src.diffusion.dataset_1 import CONCEPT_ADAPTERS, STYLE_ADAPTERS, get_entry


BASE_PROMPT = "a photo of {0}"
VAE_MODEL_PATH = "madebyollin/sdxl-vae-fp16-fix"
HF_HUB_CACHE_ENV = "HF_HUB_CACHE"
HF_HOME_ENV = "HF_HOME"
HF_HUB_SUBDIR = "hub"
SCHEDULER_SUBFOLDER = "scheduler"
UNET_SUBFOLDER = "unet"
TOKENIZER_SUBFOLDER = "tokenizer"
TOKENIZER_2_SUBFOLDER = "tokenizer_2"
TEXT_ENCODER_SUBFOLDER = "text_encoder"
TEXT_ENCODER_2_SUBFOLDER = "text_encoder_2"
WANDB_PROJECT = "fim"
WANDB_GROUP = "fim_diffusion"
WANDB_ENTITY = None
WANDB_MODE = os.environ.get("WANDB_MODE", "online")
WANDB_METRIC_MAX_ENTRIES = 200_000
WANDB_LOG_EVERY = 1


def log_stage(message):
    print(f"[fim {time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def weight_dtype(name):
    return {
        "float32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[name]


def maybe_init_wandb(args, config):
    try:
        import wandb
    except ImportError:
        log_stage("wandb not installed; continuing without wandb")
        return None
    if wandb.run is None:
        start = time.time()
        log_stage("START wandb.init")
        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            mode=WANDB_MODE,
            name=args.dataset_name or Path(args.train_data_dir).stem,
            group=WANDB_GROUP,
            config={
                "adapter_path": args.adapter_path,
                "train_data_dir": args.train_data_dir,
                "output_name": args.output_name,
                "batch_size": args.batch_size,
                "num_samples": args.num_samples,
                "repeats": args.repeats,
                "moft_nblocks": config.moft_nblocks,
                "moft_method": config.moft_method,
                "moft_scale": config.moft_scale,
            },
        )
        wandb.define_metric("fim/step")
        wandb.define_metric("fim/*", step_metric="fim/step")
        log_stage(f"DONE wandb.init in {time.time() - start:.1f}s")
    return wandb


def log_fisher_metrics(wandb, fisher, normalizer, previous):
    if wandb is None or wandb.run is None:
        return previous

    running = {
        name: (value.detach().float() / max(normalizer, 1)).cpu()
        for name, value in fisher.items()
    }
    flat = torch.cat([value.flatten() for value in running.values()])
    metric_flat = flat
    if metric_flat.numel() > WANDB_METRIC_MAX_ENTRIES:
        step = metric_flat.numel() / WANDB_METRIC_MAX_ENTRIES
        metric_flat = metric_flat[(torch.arange(WANDB_METRIC_MAX_ENTRIES) * step).long()]

    payload = {
        "fim/step": normalizer,
        "fim/num_samples": normalizer,
        "fim/mean": flat.mean().item(),
        "fim/p50": torch.quantile(metric_flat, 0.50).item(),
        "fim/p90": torch.quantile(metric_flat, 0.90).item(),
        "fim/p99": torch.quantile(metric_flat, 0.99).item(),
        "fim/max": flat.max().item(),
    }
    if previous is not None:
        prev_flat = torch.cat([previous[name].flatten() for name in running])
        diff = flat - prev_flat
        payload["fim/rel_fro_change"] = (diff.norm() / prev_flat.norm().clamp_min(1e-12)).item()
        payload["fim/cosine_to_prev"] = F.cosine_similarity(flat, prev_flat, dim=0, eps=1e-12).item()

    wandb.log(payload)
    return {name: value.clone() for name, value in running.items()}


def build_moft_processors(unet, config, device):
    processors = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        else:
            continue

        processors[name] = MOFTCrossAttnProcessor(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            nblocks=config.moft_nblocks,
            method=config.moft_method,
            scale=config.moft_scale,
            device=device,
        )
    return processors


def upper_triangle_values(tensor):
    row, col = torch.triu_indices(tensor.shape[-2], tensor.shape[-1], offset=1, device=tensor.device)
    return tensor[..., row, col]


@torch.no_grad()
def precompute_transport_rotations(named_params):
    transport_rotations = {}
    log_stage(f"START precompute blockwise transport rotations for {len(named_params)} tensors")
    start = time.time()
    for name, param in tqdm(named_params.items(), desc="Precomputing blockwise transport", unit="tensor"):
        skew_matrix = 0.5 * (param.detach().float() - param.detach().float().transpose(-1, -2))
        transport_rotations[name] = torch.matrix_exp(skew_matrix / 2).detach()
    log_stage(f"DONE precompute blockwise transport rotations in {time.time() - start:.1f}s")
    return transport_rotations


def transport_skew_gradient(skew_grad, transport_rotation):
    return transport_rotation @ skew_grad @ transport_rotation.transpose(-1, -2)


def apply_entry_defaults(args):
    if args.dataset_name is None:
        raise ValueError("dataset_name must be set by iter_entries().")
    entry = get_entry(args.entry_type, args.dataset_name)
    args.adapter_path = entry["adapter_path"]
    args.train_data_dir = entry["dataset_path"]
    args.output_name = os.path.basename(entry["fim_path"])
    args.class_name = entry["class_name"]
    args.placeholder_token = entry["placeholder_token"]
    return args


def iter_entries(args):
    if getattr(args, "entry_type", None) is not None or getattr(args, "dataset_name", None) is not None:
        if args.entry_type is None or args.dataset_name is None:
            raise ValueError("--entry_type and --dataset_name must be set together.")
        return [get_entry(args.entry_type, args.dataset_name)]

    entries = []
    selected = set(args.datasets)
    if "concepts" in selected:
        entries += [get_entry("concept", "cat2")] if args.debug else CONCEPT_ADAPTERS
    if "styles" in selected:
        entries += [get_entry("style", "01_07")] if args.debug else STYLE_ADAPTERS
    return entries


def resolve_seed(args, config):
    return args.seed if args.seed is not None else getattr(config, "seed", 8)


def resolve_repo_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def get_hf_cache_dir():
    return os.environ.get(HF_HUB_CACHE_ENV) or (
        os.path.join(os.environ[HF_HOME_ENV], HF_HUB_SUBDIR) if os.environ.get(HF_HOME_ENV) else None
    )


def pretrained_load_kwargs(args, config, *, revision=True):
    kwargs = {
        "cache_dir": get_hf_cache_dir(),
    }
    if revision:
        kwargs["revision"] = config.revision
    return kwargs


def vae_load_kwargs(args):
    return {
        "cache_dir": get_hf_cache_dir(),
    }


def get_placeholder(config, args):
    return (
        getattr(args, "placeholder_token", None)
        or getattr(config, "placeholder_token", None)
        or getattr(config, "placeholder_token_concept")
    )


def get_class_name(config, args):
    return getattr(args, "class_name", None) or config.class_name


def build_dataset_and_prompt(args, config, tokenizers):
    log_stage(f"START build dataset entry_type={args.entry_type} train_data_dir={args.train_data_dir}")
    start = time.time()
    placeholder = get_placeholder(config, args)
    class_name = get_class_name(config, args)
    if args.entry_type == "style":
        dataset = StyleDataset(
            instance_image_root=args.train_data_dir,
            tokenizers=tokenizers,
            placeholder_token=placeholder,
            class_name=class_name,
            size=config.resolution,
            repeats=args.repeats,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda examples: collate_fn(examples, False),
            num_workers=0,
        )
        log_stage(f"DONE build style dataset len={len(dataset)} prompt={dataset.instance_prompt!r} in {time.time() - start:.1f}s")
        return dataset, loader, dataset.instance_prompt

    dataset = ImageDataset(args.train_data_dir, resolution=config.resolution, repeats=args.repeats)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    prompt = BASE_PROMPT.format(f"{placeholder} {class_name}")
    log_stage(f"DONE build concept dataset len={len(dataset)} prompt={prompt!r} in {time.time() - start:.1f}s")
    return dataset, loader, prompt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="src/diffusion/config/config.yaml")
    parser.add_argument("--output_dir", default="/scratch-shared/eterres/fishers/sdxl")
    parser.add_argument("--datasets", nargs="+", choices=["concepts", "styles"], default=["concepts", "styles"])
    parser.add_argument("--entry_type", choices=["concept", "style"], default=None)
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weight_dtype", choices=["float32", "bf16", "fp16"], default="float32")
    return parser.parse_args()


def load_base_components(args, config, device):
    pretrained_model_path = config.pretrained_model_name_or_path
    vae_model_path = VAE_MODEL_PATH
    pretrained_kwargs = pretrained_load_kwargs(args, config)
    vae_kwargs = vae_load_kwargs(args)
    dtype = weight_dtype(args.weight_dtype)
    log_stage(
        f"Using device={device}; pretrained_model={config.pretrained_model_name_or_path}; "
        f"revision={config.revision}; vae_model={vae_model_path}; cache_dir={pretrained_kwargs['cache_dir']}; "
        f"local_files_only=False; weight_dtype={args.weight_dtype}"
    )

    log_stage(
        "START load scheduler "
        f"model={pretrained_model_path} subfolder={SCHEDULER_SUBFOLDER} revision={config.revision}"
    )
    start = time.time()
    scheduler = DDPMScheduler.from_pretrained(
        pretrained_model_path,
        subfolder=SCHEDULER_SUBFOLDER,
        **pretrained_kwargs,
    )
    log_stage(f"DONE load scheduler in {time.time() - start:.1f}s")

    log_stage(
        "START load UNet "
        f"model={pretrained_model_path} subfolder={UNET_SUBFOLDER} revision={config.revision}"
    )
    start = time.time()
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_path,
        subfolder=UNET_SUBFOLDER,
        torch_dtype=dtype,
        **pretrained_kwargs,
    )
    log_stage(f"DONE load UNet from_pretrained in {time.time() - start:.1f}s")
    log_stage(f"START move UNet to {device}")
    start = time.time()
    unet = unet.to(device)
    log_stage(f"DONE move UNet to {device} in {time.time() - start:.1f}s")

    log_stage(f"START load VAE model={vae_model_path}")
    start = time.time()
    vae = AutoencoderKL.from_pretrained(
        vae_model_path,
        torch_dtype=dtype,
        **vae_kwargs,
    )
    log_stage(f"DONE load VAE from_pretrained in {time.time() - start:.1f}s")
    log_stage(f"START move VAE to {device}")
    start = time.time()
    vae = vae.to(device)
    log_stage(f"DONE move VAE to {device} in {time.time() - start:.1f}s")

    log_stage(
        "START load tokenizer "
        f"model={pretrained_model_path} subfolder={TOKENIZER_SUBFOLDER} revision={config.revision}"
    )
    start = time.time()
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_path,
        subfolder=TOKENIZER_SUBFOLDER,
        **pretrained_kwargs,
    )
    log_stage(f"DONE load tokenizer in {time.time() - start:.1f}s")

    log_stage(
        "START load tokenizer_2 "
        f"model={pretrained_model_path} subfolder={TOKENIZER_2_SUBFOLDER} revision={config.revision}"
    )
    start = time.time()
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        pretrained_model_path,
        subfolder=TOKENIZER_2_SUBFOLDER,
        **pretrained_kwargs,
    )
    log_stage(f"DONE load tokenizer_2 in {time.time() - start:.1f}s")

    log_stage(
        "START load text_encoder "
        f"model={pretrained_model_path} subfolder={TEXT_ENCODER_SUBFOLDER} revision={config.revision}"
    )
    start = time.time()
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_path,
        subfolder=TEXT_ENCODER_SUBFOLDER,
        torch_dtype=dtype,
        **pretrained_kwargs,
    )
    log_stage(f"DONE load text_encoder from_pretrained in {time.time() - start:.1f}s")
    log_stage(f"START move text_encoder to {device}")
    start = time.time()
    text_encoder = text_encoder.to(device)
    log_stage(f"DONE move text_encoder to {device} in {time.time() - start:.1f}s")

    log_stage(
        "START load text_encoder_2 "
        f"model={pretrained_model_path} subfolder={TEXT_ENCODER_2_SUBFOLDER} revision={config.revision}"
    )
    start = time.time()
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        pretrained_model_path,
        subfolder=TEXT_ENCODER_2_SUBFOLDER,
        torch_dtype=dtype,
        **pretrained_kwargs,
    )
    log_stage(f"DONE load text_encoder_2 from_pretrained in {time.time() - start:.1f}s")
    log_stage(f"START move text_encoder_2 to {device}")
    start = time.time()
    text_encoder_2 = text_encoder_2.to(device)
    log_stage(f"DONE move text_encoder_2 to {device} in {time.time() - start:.1f}s")

    log_stage("START freeze base model parameters")
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    log_stage("DONE freeze base model parameters")

    log_stage("START build MOFT attention processors")
    start = time.time()
    unet.set_attn_processor(build_moft_processors(unet, config, device))
    log_stage(f"DONE build/set MOFT attention processors in {time.time() - start:.1f}s")

    log_stage("START wrap AttnProcsLayers")
    start = time.time()
    moft_layers = AttnProcsLayers(unet.attn_processors)
    log_stage(f"DONE wrap AttnProcsLayers in {time.time() - start:.1f}s")

    log_stage(f"START move MOFT layers to {device}")
    start = time.time()
    moft_layers = moft_layers.to(device)
    log_stage(f"DONE move MOFT layers to {device} in {time.time() - start:.1f}s")

    log_stage("START set trainable MOFT parameters")
    trainable_count = 0
    for name, param in moft_layers.named_parameters():
        param.requires_grad_(name.endswith((".L", ".R")))
        if param.requires_grad:
            trainable_count += 1
    log_stage(f"DONE set trainable MOFT parameters trainable_tensors={trainable_count}")

    return SimpleNamespace(
        scheduler=scheduler,
        unet=unet,
        vae=vae,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        moft_layers=moft_layers,
    )


def compute_one(args, config, base):
    args = apply_entry_defaults(args)
    log_stage(
        f"START compute_one entry_type={args.entry_type} dataset_name={args.dataset_name} "
        f"adapter_path={args.adapter_path} train_data_dir={args.train_data_dir} output_name={args.output_name}"
    )

    wandb = maybe_init_wandb(args, config)

    device = torch.device(args.device)
    seed = resolve_seed(args, config)
    log_stage(f"Using seed={seed}")
    torch.manual_seed(seed)
    scheduler = base.scheduler
    unet = base.unet
    vae = base.vae
    tokenizer = base.tokenizer
    tokenizer_2 = base.tokenizer_2
    text_encoder = base.text_encoder
    text_encoder_2 = base.text_encoder_2
    moft_layers = base.moft_layers

    log_stage(f"START load adapter safetensors {args.adapter_path}")
    start = time.time()
    adapter_state = load_file(args.adapter_path, device=str(device))
    log_stage(f"DONE load adapter safetensors tensors={len(adapter_state)} in {time.time() - start:.1f}s")

    log_stage("START load adapter state into MOFT layers")
    start = time.time()
    moft_layers.load_state_dict(adapter_state)
    log_stage(f"DONE load adapter state into MOFT layers in {time.time() - start:.1f}s")

    _, loader, prompt = build_dataset_and_prompt(args, config, (tokenizer, tokenizer_2))
    log_stage(f"START tokenize prompt {prompt!r}")
    start = time.time()
    input_ids_list = tokenize_prompt((tokenizer, tokenizer_2), prompt)
    log_stage(f"DONE tokenize prompt in {time.time() - start:.1f}s")

    log_stage("START collect named trainable parameters")
    named_params = {
        name: param
        for name, param in moft_layers.named_parameters()
        if param.requires_grad
    }
    log_stage(f"DONE collect named trainable parameters count={len(named_params)}")
    transport_rotations = precompute_transport_rotations(named_params)

    log_stage("START allocate Fisher tensors on CPU")
    start = time.time()
    fisher = {
        name: torch.zeros(
            (*param.shape[:-2], param.shape[-1] * (param.shape[-1] - 1) // 2),
            dtype=torch.float32,
        )
        for name, param in named_params.items()
    }
    log_stage(f"DONE allocate Fisher tensors count={len(fisher)} in {time.time() - start:.1f}s")

    count = 0
    previous_fisher = None
    unet.train()
    log_stage("START FIM dataloader loop")
    for batch_idx, batch in enumerate(tqdm(loader, desc="Computing transported OFT Lie-basis FIM"), 1):
        if batch_idx == 1:
            log_stage("START first FIM batch")
        if args.num_samples is not None and count >= args.num_samples:
            break

        if batch_idx == 1:
            log_stage("START first batch tensor transfer/prep")
        images = batch["pixel_values"].to(device) if args.entry_type == "style" else batch["image"].to(device) * 2.0 - 1.0
        original_sizes = batch["original_sizes"].to(device)
        crop_top_lefts = batch["crop_top_lefts"].to(device)
        batch_size = images.shape[0]
        if batch_idx == 1:
            log_stage("DONE first batch tensor transfer/prep")

        with torch.no_grad():
            if batch_idx == 1:
                log_stage("START first batch VAE/noise/text conditioning")
            latents = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                scheduler.num_train_timesteps,
                (batch_size,),
                device=device,
            )
            target = noise if scheduler.config.prediction_type == "epsilon" else scheduler.get_velocity(latents, noise, timesteps)
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)
            encoder_hidden_states, pooled = encode_tokens((text_encoder, text_encoder_2), input_ids_list)
            encoder_hidden_states = encoder_hidden_states.expand(batch_size, -1, -1)
            pooled = pooled.expand(batch_size, -1)
            add_time_ids = compute_time_ids(original_sizes, crop_top_lefts, config.resolution)
            if batch_idx == 1:
                log_stage("DONE first batch VAE/noise/text conditioning")

        if batch_idx == 1:
            log_stage("START first batch UNet forward/backward")
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
            log_stage("DONE first batch UNet forward/backward")

        with torch.no_grad():
            if batch_idx == 1:
                log_stage("START first batch gradient transport/accumulation")
            param_iter = named_params.items()
            if batch_idx == 1:
                param_iter = tqdm(param_iter, desc="Transporting gradients for first batch", unit="tensor")
            for name, param in param_iter:
                if param.grad is None:
                    continue
                skew_grad = 0.5 * (param.grad.detach().float() - param.grad.detach().float().transpose(-1, -2))
                transported_skew_grad = transport_skew_gradient(skew_grad, transport_rotations[name])
                transported_grad = upper_triangle_values(transported_skew_grad)
                fisher[name] += transported_grad.pow(2).cpu()
            if batch_idx == 1:
                log_stage("DONE first batch gradient transport/accumulation")

        count += batch_size
        if batch_idx == 1:
            log_stage("DONE first FIM batch")
        if WANDB_LOG_EVERY > 0 and batch_idx % WANDB_LOG_EVERY == 0:
            if batch_idx == 1:
                log_stage("START first wandb Fisher metric log")
            previous_fisher = log_fisher_metrics(wandb, fisher, count, previous_fisher)
            if batch_idx == 1:
                log_stage("DONE first wandb Fisher metric log")

    if count == 0:
        raise RuntimeError("No samples processed.")

    for name in fisher:
        fisher[name] /= count

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_name)
    log_stage(f"START save Fisher to {output_path}")
    start = time.time()
    save_file(fisher, output_path)
    log_stage(f"DONE save Fisher in {time.time() - start:.1f}s")
    if wandb is not None and wandb.run is not None:
        log_stage("START final wandb log/finish")
        wandb.log({"fim/saved": 1, "fim/num_tensors": len(fisher), "fim/final_samples": count})
        wandb.finish()
        log_stage("DONE final wandb log/finish")
    print(f"Saved transported OFT Lie-basis diagonal FIM to {output_path}")


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    config_path = resolve_repo_path(args.config_path)
    log_stage(f"START load config {config_path}")
    start = time.time()
    with open(config_path, "r", encoding="utf-8") as handle:
        config = SimpleNamespace(**yaml.safe_load(handle))
    log_stage(f"DONE load config in {time.time() - start:.1f}s")

    device = torch.device(args.device)
    seed = resolve_seed(args, config)
    log_stage(f"Using seed={seed}")
    torch.manual_seed(seed)
    base = load_base_components(args, config, device)

    for entry in iter_entries(args):
        run_args = argparse.Namespace(**vars(args))
        run_args.entry_type = entry["type"]
        run_args.dataset_name = entry["name"]
        compute_one(run_args, config, base)


if __name__ == "__main__":
    main()
