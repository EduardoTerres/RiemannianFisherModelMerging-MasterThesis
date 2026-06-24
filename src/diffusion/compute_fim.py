import argparse
import os
import sys
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
WANDB_PROJECT = "fim"
WANDB_GROUP = "fim_diffusion"
WANDB_ENTITY = None
WANDB_MODE = os.environ.get("WANDB_MODE", "online")
WANDB_METRIC_MAX_ENTRIES = 200_000
WANDB_LOG_EVERY = 1


def maybe_init_wandb(args, config):
    try:
        import wandb
    except ImportError:
        return None
    if wandb.run is None:
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


def lower_triangle_values(tensor):
    row, col = torch.tril_indices(tensor.shape[-2], tensor.shape[-1], offset=-1, device=tensor.device)
    return tensor[..., row, col]


def apply_entry_defaults(args):
    if args.dataset_name is None:
        return args
    entry = get_entry(args.entry_type, args.dataset_name)
    args.adapter_path = entry["adapter_path"]
    args.train_data_dir = entry["dataset_path"]
    args.output_name = os.path.basename(entry["fim_path"])
    args.class_name = entry["class_name"]
    args.placeholder_token = entry["placeholder_token"]
    return args


def iter_entries(args):
    if args.all_dataset:
        return STYLE_ADAPTERS
    if args.dataset_name is not None:
        return [get_entry(args.entry_type, args.dataset_name)]
    return [None]


def get_placeholder(config, args):
    return (
        getattr(args, "placeholder_token", None)
        or getattr(config, "placeholder_token", None)
        or getattr(config, "placeholder_token_concept")
    )


def get_class_name(config, args):
    return getattr(args, "class_name", None) or config.class_name


def build_dataset_and_prompt(args, config, tokenizers):
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
        return dataset, loader, dataset.instance_prompt

    dataset = ImageDataset(args.train_data_dir, resolution=config.resolution, repeats=args.repeats)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    prompt = BASE_PROMPT.format(f"{placeholder} {class_name}")
    return dataset, loader, prompt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="output/concept_style/sdxl_merge/example/logs/hparams.yml")
    parser.add_argument("--adapter_path", default="/scratch-shared/eterres/SDXL/concepts/pytorch_lora_weights_cat.safetensors")
    parser.add_argument("--train_data_dir", default="/scratch-shared/eterres/SDXL/concepts/datasets/cat")
    parser.add_argument("--output_dir", default="/scratch-shared/eterres/fishers")
    parser.add_argument("--output_name", default="cat_oft_lie_fim.safetensors")
    parser.add_argument("--entry_type", choices=["concept", "style"], default="concept")
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--all_dataset", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def compute_one(args):
    args = apply_entry_defaults(args)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    with open(args.config_path, "r", encoding="utf-8") as handle:
        config = SimpleNamespace(**yaml.safe_load(handle))
    wandb = maybe_init_wandb(args, config)

    device = torch.device(args.device)
    torch.manual_seed(getattr(config, "seed", 8))

    scheduler = DDPMScheduler.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="scheduler",
        revision=config.revision,
    )
    unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="unet",
        revision=config.revision,
    ).to(device)
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix").to(device)
    tokenizer = CLIPTokenizer.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=config.revision,
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=config.revision,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=config.revision,
    ).to(device)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        revision=config.revision,
    ).to(device)

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)

    unet.set_attn_processor(build_moft_processors(unet, config, device))
    moft_layers = AttnProcsLayers(unet.attn_processors)
    moft_layers.load_state_dict(load_file(args.adapter_path, device=str(device)))
    moft_layers = moft_layers.to(device)
    for name, param in moft_layers.named_parameters():
        param.requires_grad_(name.endswith((".L", ".R")))

    _, loader, prompt = build_dataset_and_prompt(args, config, (tokenizer, tokenizer_2))
    input_ids_list = tokenize_prompt((tokenizer, tokenizer_2), prompt)

    fisher = {
        name: torch.zeros(
            (*param.shape[:-2], param.shape[-1] * (param.shape[-1] - 1) // 2),
            dtype=torch.float32,
        )
        for name, param in moft_layers.named_parameters()
        if param.requires_grad
    }

    count = 0
    previous_fisher = None
    unet.train()
    for batch_idx, batch in enumerate(tqdm(loader, desc="Computing OFT Lie-basis FIM"), 1):
        if args.num_samples is not None and count >= args.num_samples:
            break

        images = batch["pixel_values"].to(device) if args.entry_type == "style" else batch["image"].to(device) * 2.0 - 1.0
        original_sizes = batch["original_sizes"].to(device)
        crop_top_lefts = batch["crop_top_lefts"].to(device)
        batch_size = images.shape[0]

        with torch.no_grad():
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

        unet.zero_grad(set_to_none=True)
        outputs = unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states,
            added_cond_kwargs={"time_ids": add_time_ids, "text_embeds": pooled},
        ).sample
        loss = F.mse_loss(outputs.float(), target.float(), reduction="mean")
        loss.backward()

        with torch.no_grad():
            for name, param in moft_layers.named_parameters():
                if not param.requires_grad or param.grad is None:
                    continue
                skew_grad = 0.5 * (param.grad.detach().float() - param.grad.detach().float().transpose(-1, -2))
                fisher[name] += lower_triangle_values(skew_grad).cpu().pow(2)

        count += batch_size
        if WANDB_LOG_EVERY > 0 and batch_idx % WANDB_LOG_EVERY == 0:
            previous_fisher = log_fisher_metrics(wandb, fisher, count, previous_fisher)

    if count == 0:
        raise RuntimeError("No samples processed.")

    for name in fisher:
        fisher[name] /= count

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_name)
    save_file(fisher, output_path)
    if wandb is not None and wandb.run is not None:
        wandb.log({"fim/saved": 1, "fim/num_tensors": len(fisher), "fim/final_samples": count})
        wandb.finish()
    print(f"Saved OFT Lie-basis diagonal FIM to {output_path}")


def main():
    args = parse_args()
    if args.all_dataset:
        for entry in iter_entries(args):
            run_args = argparse.Namespace(**vars(args))
            run_args.entry_type = entry["type"]
            run_args.dataset_name = entry["name"]
            compute_one(run_args)
    else:
        compute_one(args)


if __name__ == "__main__":
    main()
