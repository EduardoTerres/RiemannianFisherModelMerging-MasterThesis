import argparse
import os
import sys
import time
from pathlib import Path
from types import MethodType, SimpleNamespace

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
from moft.model.monarch_orthogonal import MonarchOrthogonal
from src.diffusion.dataset_1 import CONCEPT_ADAPTERS, STYLE_ADAPTERS, get_entry


BASE_PROMPT = "a photo of a {0}"
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
    """Print a timestamped FIM progress message."""
    print(f"[fim {time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def weight_dtype(name):
    """Map a dtype CLI name to a torch dtype."""
    return {"float32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[name]


def cay(omega, s):
    """Apply the scaled Cayley transform to a batch of skew matrices."""
    n = omega.shape[-1]
    eye = torch.eye(n, dtype=omega.dtype, device=omega.device).expand_as(omega)
    return torch.linalg.solve(eye - s * omega, eye + s * omega, left=False)


def upper_triangle_values(tensor):
    """Return strict upper-triangle entries from the last two dimensions."""
    row, col = torch.triu_indices(tensor.shape[-2], tensor.shape[-1], offset=1, device=tensor.device)
    return tensor[..., row, col]


def infer_oft_cayley_scale(device="cpu"):
    """Infer the local Cayley derivative scale used by the OFT layer."""
    nb, h = 8, 1e-4
    layer = MonarchOrthogonal(nb, nblocks=1, orthogonal=True, method="cayley", device=device)
    e = torch.randn(1, nb, nb, device=device)
    e = 0.5 * (e - e.transpose(-1, -2))
    with torch.no_grad():
        fd = (layer.cayley_batch(h * e) - layer.cayley_batch(-h * e)) / (2 * h)
    return (0.5 * (fd * e).sum() / (e.square().sum().clamp_min(1e-30))).item()


def run_oft_geometry_checks(s):
    """Run sanity checks for the OFT Cayley geometry assumptions."""
    torch.manual_seed(0)
    nb, h = 8, 1e-6
    dtype = torch.float64
    eye = torch.eye(nb, dtype=dtype)
    om = torch.randn(nb, nb, dtype=dtype)
    om = 0.1 * 0.5 * (om - om.T) / om.norm()
    e = torch.randn(nb, nb, dtype=dtype)
    e = 0.5 * (e - e.T)

    theta = cay(om, s)
    assert torch.allclose(om, -om.T, atol=1e-12, rtol=0)
    assert torch.allclose(theta.T @ theta, eye, atol=1e-10, rtol=1e-10)
    assert torch.isclose(torch.linalg.det(theta), torch.tensor(1.0, dtype=dtype), atol=1e-10, rtol=1e-10)

    c = torch.randn(nb, nb, dtype=dtype)
    p = om.clone().detach().requires_grad_(True)
    loss = (cay(0.5 * (p - p.T), s) * c).sum()
    loss.backward()
    g = p.grad - p.grad.T
    fd = (((cay(om + h * e, s) * c).sum() - (cay(om - h * e, s) * c).sum()) / (2 * h)).item()
    inner = (0.5 * (g * e).sum()).item()
    assert abs(fd - inner) <= 1e-4 * max(abs(fd), abs(inner), 1.0)

    fd0 = (cay(h * e, s) - cay(-h * e, s)) / (2 * h)
    assert torch.allclose(fd0, 2 * s * e, rtol=1e-5, atol=1e-8)

    theta_half = cay(om / 2, s)
    residual = (theta_half @ theta_half - theta).norm()
    assert residual < 10.0 * om.norm().pow(3)

    g0 = torch.randn(nb, nb, dtype=dtype)
    g0 = 0.5 * (g0 - g0.T)
    w = ((eye - s * om) @ g0 @ (eye + s * om)) / (2 * s)
    u = theta_half @ w @ theta_half.T
    a = theta_half @ (eye - s * om)
    fused = a @ g0 @ a.T / (2 * s)
    assert torch.allclose(theta_half.T @ theta_half, eye, atol=1e-10, rtol=1e-10)
    assert torch.allclose(w, -w.T, atol=1e-10, rtol=1e-10)
    assert torch.allclose(u, -u.T, atol=1e-10, rtol=1e-10)
    assert torch.allclose(u, fused, atol=1e-10, rtol=1e-10)
    assert torch.isclose(u.norm(), w.norm(), atol=1e-10, rtol=1e-10)

    a0 = cay(torch.zeros_like(om), s) @ (eye - s * torch.zeros_like(om))
    assert torch.allclose(a0, eye, atol=1e-12, rtol=0)
    assert torch.allclose(a0 @ g0 @ a0.T / (2 * s), g0 / (2 * s), atol=1e-12, rtol=0)
    assert not torch.allclose(u, g0 / (2 * s), atol=1e-8, rtol=1e-5)

    i, j = 2, 5
    eij = torch.zeros(nb, nb, dtype=dtype)
    eij[i, j] = 1.0
    eij[j, i] = -1.0
    assert torch.isclose(0.5 * (eij * u).sum(), u[i, j], atol=1e-12, rtol=1e-12)


def coordinate_grad_to_skew(raw_param_grad, s):
    """Convert raw OFT coordinate gradients to skew Lie-algebra gradients."""
    return (raw_param_grad.float() - raw_param_grad.float().transpose(-1, -2)) / (2 * s)


def monarch_forward_with_delta(self, x):
    """Run MonarchOrthogonal with additive zero-initialized tangent deltas."""
    l_param = self.L + self.L_delta
    r_param = self.R + self.R_delta
    if self.orthogonal:
        if self.method == "cayley":
            L = self.cayley_batch(l_param)
            R = self.cayley_batch(r_param)
        elif self.method == "exp":
            L = self.exp_full(l_param)
            R = self.exp_full(r_param)
        elif self.method == "already_orthogonal":
            L = l_param
            R = r_param
        else:
            raise NotImplementedError("Method is not supported. Use 'cayley' or 'exp'.")
    else:
        L = l_param
        R = r_param
    return self.blockdiag_butterfly_multiply(x, R, L)


def enable_zero_oft_delta_parameters(moft_layers):
    """Freeze trained OFT tensors and add trainable zero deltas in the same chart."""
    for param in moft_layers.parameters():
        param.requires_grad_(False)
    for module in moft_layers.modules():
        if not isinstance(module, MonarchOrthogonal):
            continue
        if hasattr(module, "L_delta"):
            module.L_delta.data.zero_()
            module.R_delta.data.zero_()
            module.L_delta.requires_grad_(True)
            module.R_delta.requires_grad_(True)
        else:
            module.register_parameter("L_delta", torch.nn.Parameter(torch.zeros_like(module.L)))
            module.register_parameter("R_delta", torch.nn.Parameter(torch.zeros_like(module.R)))
        module.forward = MethodType(monarch_forward_with_delta, module)


def fisher_name_from_delta(name):
    """Map a delta parameter name back to the corresponding trained OFT name."""
    return name.removesuffix("_delta")


@torch.no_grad()
def assert_oft_parameters_zero(named_params):
    """Assert that all trainable OFT parameters are exactly zero."""
    assert named_params, "No trainable zero OFT delta parameters found before FIM."
    for name, param in named_params.items():
        assert name.endswith(("_delta")), f"Unexpected trainable non-delta OFT parameter before FIM: {name}"
        max_abs = param.detach().abs().max().item()
        assert max_abs == 0.0, f"OFT parameter {name} is not zero before FIM; max_abs={max_abs:g}"


def maybe_init_wandb(args, config):
    """Initialize wandb logging for an FIM run if available."""
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
    """Log running Fisher summary statistics to wandb."""
    if wandb is None or wandb.run is None:
        return previous
    running = {name: (value.detach().float() / max(normalizer, 1)).cpu() for name, value in fisher.items()}
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
    """Build MOFT attention processors matching the UNet attention blocks."""
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


def load_pretrained_unet(args, config, device):
    """Load the frozen pretrained UNet used for trained-adapter FIM."""
    dtype = weight_dtype(args.weight_dtype)
    kwargs = pretrained_load_kwargs(args, config)
    unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder=UNET_SUBFOLDER,
        torch_dtype=dtype,
        **kwargs,
    )
    unet.requires_grad_(False)
    return unet.to(device)

def apply_entry_defaults(args):
    """Fill dataset-specific paths and labels into parsed arguments."""
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
    """Yield dataset entries selected by the CLI arguments."""
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
    """Resolve the random seed from CLI args or config."""
    return args.seed if args.seed is not None else getattr(config, "seed", 8)


def resolve_repo_path(path):
    """Resolve a path relative to the repository root."""
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def get_hf_cache_dir():
    """Return the configured Hugging Face hub cache directory."""
    return os.environ.get(HF_HUB_CACHE_ENV) or (
        os.path.join(os.environ[HF_HOME_ENV], HF_HUB_SUBDIR) if os.environ.get(HF_HOME_ENV) else None
    )


def pretrained_load_kwargs(args, config, *, revision=True):
    """Build common from_pretrained keyword arguments for SDXL components."""
    kwargs = {"cache_dir": get_hf_cache_dir()}
    if revision:
        kwargs["revision"] = config.revision
    return kwargs


def vae_load_kwargs(args):
    """Build from_pretrained keyword arguments for the VAE."""
    return {"cache_dir": get_hf_cache_dir()}


def get_placeholder(config, args):
    """Resolve the placeholder token for the current dataset."""
    return (
        getattr(args, "placeholder_token", None)
        or getattr(config, "placeholder_token", None)
        or getattr(config, "placeholder_token_concept")
    )


def get_class_name(config, args):
    """Resolve the class name for the current dataset."""
    return getattr(args, "class_name", None) or config.class_name


def build_dataset_and_prompt(args, config, tokenizers):
    """Build the image dataset, dataloader, and conditioning prompt."""
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
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: collate_fn(x, False), num_workers=0)
        log_stage(f"DONE build style dataset len={len(dataset)} prompt={dataset.instance_prompt!r} in {time.time() - start:.1f}s")
        return dataset, loader, dataset.instance_prompt
    dataset = ImageDataset(args.train_data_dir, resolution=config.resolution, repeats=args.repeats)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    prompt = BASE_PROMPT.format(f"{placeholder} {class_name}")
    log_stage(f"DONE build concept dataset len={len(dataset)} prompt={prompt!r} in {time.time() - start:.1f}s")
    return dataset, loader, prompt


def parse_args():
    """Parse command-line arguments for diffusion FIM computation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="src/diffusion/config/config.yaml")
    parser.add_argument("--output_dir", default="/path/to/fishers/sdxl")
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
    """Load shared frozen SDXL components except the per-entry UNet."""
    pretrained_model_path = config.pretrained_model_name_or_path
    dtype = weight_dtype(args.weight_dtype)
    pretrained_kwargs = pretrained_load_kwargs(args, config)
    vae_kwargs = vae_load_kwargs(args)
    log_stage(
        f"Using device={device}; pretrained_model={pretrained_model_path}; revision={config.revision}; "
        f"vae_model={VAE_MODEL_PATH}; cache_dir={pretrained_kwargs['cache_dir']}; local_files_only=False; "
        f"weight_dtype={args.weight_dtype}"
    )

    log_stage(f"START load scheduler model={pretrained_model_path} subfolder={SCHEDULER_SUBFOLDER} revision={config.revision}")
    start = time.time()
    scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder=SCHEDULER_SUBFOLDER, **pretrained_kwargs)
    log_stage(f"DONE load scheduler in {time.time() - start:.1f}s")

    log_stage(f"START load VAE model={VAE_MODEL_PATH}")
    start = time.time()
    vae = AutoencoderKL.from_pretrained(VAE_MODEL_PATH, torch_dtype=dtype, **vae_kwargs)
    log_stage(f"DONE load VAE from_pretrained in {time.time() - start:.1f}s")
    log_stage(f"START move VAE to {device}")
    start = time.time()
    vae = vae.to(device)
    log_stage(f"DONE move VAE to {device} in {time.time() - start:.1f}s")

    log_stage(f"START load tokenizer model={pretrained_model_path} subfolder={TOKENIZER_SUBFOLDER} revision={config.revision}")
    start = time.time()
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder=TOKENIZER_SUBFOLDER, **pretrained_kwargs)
    log_stage(f"DONE load tokenizer in {time.time() - start:.1f}s")

    log_stage(f"START load tokenizer_2 model={pretrained_model_path} subfolder={TOKENIZER_2_SUBFOLDER} revision={config.revision}")
    start = time.time()
    tokenizer_2 = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder=TOKENIZER_2_SUBFOLDER, **pretrained_kwargs)
    log_stage(f"DONE load tokenizer_2 in {time.time() - start:.1f}s")

    log_stage(f"START load text_encoder model={pretrained_model_path} subfolder={TEXT_ENCODER_SUBFOLDER} revision={config.revision}")
    start = time.time()
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder=TEXT_ENCODER_SUBFOLDER, torch_dtype=dtype, **pretrained_kwargs)
    log_stage(f"DONE load text_encoder from_pretrained in {time.time() - start:.1f}s")
    log_stage(f"START move text_encoder to {device}")
    start = time.time()
    text_encoder = text_encoder.to(device)
    log_stage(f"DONE move text_encoder to {device} in {time.time() - start:.1f}s")

    log_stage(f"START load text_encoder_2 model={pretrained_model_path} subfolder={TEXT_ENCODER_2_SUBFOLDER} revision={config.revision}")
    start = time.time()
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        pretrained_model_path, subfolder=TEXT_ENCODER_2_SUBFOLDER, torch_dtype=dtype, **pretrained_kwargs
    )
    log_stage(f"DONE load text_encoder_2 from_pretrained in {time.time() - start:.1f}s")
    log_stage(f"START move text_encoder_2 to {device}")
    start = time.time()
    text_encoder_2 = text_encoder_2.to(device)
    log_stage(f"DONE move text_encoder_2 to {device} in {time.time() - start:.1f}s")

    log_stage("START freeze base model parameters")
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    log_stage("DONE freeze base model parameters")
    return SimpleNamespace(
        scheduler=scheduler,
        vae=vae,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
    )


def compute_one(args, config, base, chart_scale):
    """Compute and save the trained-GSOFT delta diagonal FIM for one dataset entry."""
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

    scheduler, vae = base.scheduler, base.vae
    tokenizer, tokenizer_2 = base.tokenizer, base.tokenizer_2
    text_encoder, text_encoder_2 = base.text_encoder, base.text_encoder_2

    log_stage(f"START load adapter safetensors {args.adapter_path}")
    start = time.time()
    adapter_state = load_file(args.adapter_path, device=str(device))
    log_stage(f"DONE load adapter safetensors tensors={len(adapter_state)} in {time.time() - start:.1f}s")

    log_stage("START load pretrained UNet for trained MOFT FIM")
    start = time.time()
    unet = load_pretrained_unet(args, config, device)
    log_stage(f"DONE load pretrained UNet in {time.time() - start:.1f}s")

    log_stage("START attach trained MOFT adapter with zero delta probes")
    start = time.time()
    unet.set_attn_processor(build_moft_processors(unet, config, device))
    moft_layers = AttnProcsLayers(unet.attn_processors).to(device)
    moft_layers.load_state_dict(adapter_state)
    enable_zero_oft_delta_parameters(moft_layers)
    log_stage(f"DONE attach trained MOFT adapter with zero deltas in {time.time() - start:.1f}s")

    _, loader, prompt = build_dataset_and_prompt(args, config, (tokenizer, tokenizer_2))
    log_stage(f"START tokenize prompt {prompt!r}")
    start = time.time()
    input_ids_list = tokenize_prompt((tokenizer, tokenizer_2), prompt)
    log_stage(f"DONE tokenize prompt in {time.time() - start:.1f}s")

    log_stage("START collect named trainable parameters")
    named_params = {name: param for name, param in moft_layers.named_parameters() if param.requires_grad}
    log_stage(f"DONE collect named trainable parameters count={len(named_params)}")
    assert_oft_parameters_zero(named_params)

    log_stage("START allocate Fisher tensors on CPU")
    start = time.time()
    fisher = {
        fisher_name_from_delta(name): torch.zeros(
            (*param.shape[:-2], param.shape[-1] * (param.shape[-1] - 1) // 2), dtype=torch.float32
        )
        for name, param in named_params.items()
    }
    log_stage(f"DONE allocate Fisher tensors count={len(fisher)} in {time.time() - start:.1f}s")

    count = 0
    previous_fisher = None
    unet.train()
    log_stage("START FIM dataloader loop")
    for batch_idx, batch in enumerate(tqdm(loader, desc="Computing trained-GSOFT delta FIM"), 1):
        if args.num_samples is not None and count >= args.num_samples:
            break
        if batch_idx == 1:
            log_stage("START first FIM batch")

        images = batch["pixel_values"].to(device) if args.entry_type == "style" else batch["image"].to(device) * 2.0 - 1.0
        original_sizes = batch["original_sizes"].to(device)
        crop_top_lefts = batch["crop_top_lefts"].to(device)
        batch_size = images.shape[0]
        take = batch_size if args.num_samples is None else min(batch_size, args.num_samples - count)

        with torch.no_grad():
            if batch_idx == 1:
                log_stage("START first batch VAE/noise/text conditioning")
            latents = vae.encode(images[:take]).latent_dist.sample() * vae.config.scaling_factor
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (take,), device=device)
            target = noise if scheduler.config.prediction_type == "epsilon" else scheduler.get_velocity(latents, noise, timesteps)
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)
            encoder_hidden_states, pooled = encode_tokens((text_encoder, text_encoder_2), input_ids_list)
            encoder_hidden_states = encoder_hidden_states.expand(take, -1, -1)
            pooled = pooled.expand(take, -1)
            add_time_ids = compute_time_ids(original_sizes[:take], crop_top_lefts[:take], config.resolution)
            if batch_idx == 1:
                log_stage("DONE first batch VAE/noise/text conditioning")

        if batch_idx == 1:
            log_stage("START first batch UNet forward/per-sample backward")
        unet.zero_grad(set_to_none=True)
        outputs = unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states,
            added_cond_kwargs={"time_ids": add_time_ids, "text_embeds": pooled},
        ).sample
        losses = F.mse_loss(outputs.float(), target.float(), reduction="none").flatten(1).mean(1)

        for sample_idx in range(take):
            unet.zero_grad(set_to_none=True)
            losses[sample_idx].backward(retain_graph=sample_idx + 1 < take)
            with torch.no_grad():
                param_iter = named_params.items()
                if batch_idx == 1 and sample_idx == 0:
                    param_iter = tqdm(param_iter, desc="Transporting first sample gradients", unit="tensor")
                for name, param in param_iter:
                    if param.grad is None:
                        continue
                    skew_grad = coordinate_grad_to_skew(param.grad.detach(), chart_scale)
                    fisher[fisher_name_from_delta(name)] += upper_triangle_values(skew_grad).pow(2).cpu()
            count += 1

        if batch_idx == 1:
            log_stage("DONE first batch UNet forward/per-sample backward")
            log_stage("DONE first FIM batch")
        if WANDB_LOG_EVERY > 0 and batch_idx % WANDB_LOG_EVERY == 0:
            previous_fisher = log_fisher_metrics(wandb, fisher, count, previous_fisher)

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
    print(f"Saved trained-GSOFT delta diagonal FIM to {output_path}")


def main():
    """Run diffusion trained-GSOFT delta FIM computation from the CLI."""
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    # Step 0: verify the OFT geometry used by the Fisher coordinates.
    chart_scale = infer_oft_cayley_scale("cpu")
    run_oft_geometry_checks(chart_scale)
    log_stage(f"OFT Cayley chart scale s={chart_scale:g}")

    # Step 1: load the shared pretrained SDXL components.
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

    # Step 2: for each adapter, keep trained MOFT attached, add zero deltas,
    # convert delta gradients to skew coordinates, and save the FIM.
    for entry in iter_entries(args):
        run_args = argparse.Namespace(**vars(args))
        run_args.entry_type = entry["type"]
        run_args.dataset_name = entry["name"]
        compute_one(run_args, config, base, chart_scale)


if __name__ == "__main__":
    main()
