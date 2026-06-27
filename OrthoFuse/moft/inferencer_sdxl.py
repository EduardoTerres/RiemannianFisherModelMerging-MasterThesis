import os
import importlib.util
import sys
import types
import contextlib
import io
from collections import Counter, defaultdict
from tqdm import tqdm
import random
import numpy as np
from functools import reduce
from pathlib import Path
import torch
from diffusers import (
    StableDiffusionPipeline, AutoencoderKL, DDIMScheduler, UNet2DConditionModel, StableDiffusionXLPipeline
)
from diffusers.loaders import AttnProcsLayers
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from safetensors.torch import load_file
from peft import  LoraConfig, get_peft_model
from .model.moft import MOFTCrossAttnProcessor, DoubleMOFTCrossAttnProcessor, MOFTDoubleCrossAttnProcessor
from .model.lora import LoRACrossAttnProcessor
from .utils.registry import ClassRegistry
from .utils.gs_orthogonal import merge_inside_cayley_space, blocked_geodesic_combination, full_matrix_geodesic_combination, GSOrthogonal, postprocess_blocks, merge_inside_cayley_space_batch, postprocess_blocks_batch
from .model.monarch_orthogonal import MonarchOrthogonal
from .utils.fixed_rank_batch import FixedRankBatch
from diffusers import EulerDiscreteScheduler
import time
import json 


HF_HUB_CACHE_ENV = "HF_HUB_CACHE"
HF_HOME_ENV = "HF_HOME"
HF_HUB_SUBDIR = "hub"


def _truthy_env(name, default="1"):
    value = os.environ.get(name, default)
    return value.lower() not in {"0", "false", "no", "off"}


def _hf_cache_dir():
    return os.environ.get(HF_HUB_CACHE_ENV) or (
        os.path.join(os.environ[HF_HOME_ENV], HF_HUB_SUBDIR) if os.environ.get(HF_HOME_ENV) else None
    )


def _hf_load_kwargs():
    return {
        "cache_dir": _hf_cache_dir(),
        "local_files_only": _truthy_env("ORTHOFUSE_LOCAL_FILES_ONLY"),
    }


def _load_pretrained(component_name, loader_cls, model_name_or_path, **kwargs):
    load_kwargs = _hf_load_kwargs()
    load_kwargs.update(kwargs)
    cache_dir = load_kwargs.get("cache_dir") or "<default>"
    print(
        f"loading {component_name} from {model_name_or_path} "
        f"(cache_dir={cache_dir}, local_files_only={load_kwargs.get('local_files_only')})...",
        flush=True,
    )
    component = loader_cls.from_pretrained(model_name_or_path, **load_kwargs)
    print(f"loaded {component_name}", flush=True)
    return component


def _load_pretrained_on_device(component_name, loader_cls, model_name_or_path, device, **kwargs):
    load_kwargs = dict(kwargs)
    if str(device).startswith("cuda"):
        load_kwargs.setdefault("device_map", {"": device})
        load_kwargs.setdefault("low_cpu_mem_usage", True)
    component = _load_pretrained(component_name, loader_cls, model_name_or_path, **load_kwargs)
    if not str(device).startswith("cuda"):
        component = component.to(device)
    return component


def _load_orthofuse_adapter_info():
    module_path = Path(__file__).resolve().parents[2] / "src" / "diffusion" / "oft_adapter_info.py"
    spec = importlib.util.spec_from_file_location(
        "_orthofuse_oft_adapter_info",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_adapter_info = _load_orthofuse_adapter_info()
print_parameter_loading_summary = _adapter_info.print_parameter_loading_summary
print_state_dict_summary = _adapter_info.print_state_dict_summary
print_unet_oft_summary = _adapter_info.print_unet_oft_summary

inferencers = ClassRegistry()

_ROOT_OFT_MERGING_CLASS = None
_ROOT_OFT_GEODESIC_MERGING_CLASS = None


def _load_root_oft_merging_class():
    """Load top-level src/merging.py without import-package ambiguity."""
    global _ROOT_OFT_MERGING_CLASS
    if _ROOT_OFT_MERGING_CLASS is not None:
        return _ROOT_OFT_MERGING_CLASS

    repo_root = Path(__file__).resolve().parents[2]
    geometry_path = repo_root / "src" / "geometry.py"
    merging_path = repo_root / "src" / "merging.py"

    if "geoopt" not in sys.modules and importlib.util.find_spec("geoopt") is None:
        sys.modules["geoopt"] = types.SimpleNamespace(
            ManifoldParameter=None,
            Stiefel=None,
            optim=types.SimpleNamespace(RiemannianAdam=None),
        )

    if "src.geometry" not in sys.modules:
        geometry_spec = importlib.util.spec_from_file_location(
            "src.geometry",
            geometry_path,
        )
        geometry_module = importlib.util.module_from_spec(geometry_spec)
        sys.modules["src.geometry"] = geometry_module
        geometry_spec.loader.exec_module(geometry_module)

    merging_spec = importlib.util.spec_from_file_location(
        "_root_oft_merging",
        merging_path,
    )
    merging_module = importlib.util.module_from_spec(merging_spec)
    merging_spec.loader.exec_module(merging_module)
    _ROOT_OFT_MERGING_CLASS = merging_module.OFTMerging
    return _ROOT_OFT_MERGING_CLASS


def _load_root_oft_geodesic_merging_class():
    """Load top-level src/merging_geodesic.py without import-package ambiguity."""
    global _ROOT_OFT_GEODESIC_MERGING_CLASS
    if _ROOT_OFT_GEODESIC_MERGING_CLASS is not None:
        return _ROOT_OFT_GEODESIC_MERGING_CLASS

    repo_root = Path(__file__).resolve().parents[2]
    geodesic_path = repo_root / "src" / "merging_geodesic.py"

    if "src.geometry" not in sys.modules:
        _load_root_oft_merging_class()

    geodesic_spec = importlib.util.spec_from_file_location(
        "_root_oft_geodesic_merging",
        geodesic_path,
    )
    geodesic_module = importlib.util.module_from_spec(geodesic_spec)
    geodesic_spec.loader.exec_module(geodesic_module)
    _ROOT_OFT_GEODESIC_MERGING_CLASS = geodesic_module.OFTGeodesicMerging
    return _ROOT_OFT_GEODESIC_MERGING_CLASS


def _full_generator_to_oft_coords(merger, tensor):
    skew = 0.5 * (tensor - tensor.transpose(-1, -2))
    return merger.skew_matrix_to_oft_params(skew)


def _oft_coords_to_full_generator(merger, coords, dtype):
    return merger.oft_params_to_skew_matrix(coords, coords.shape[-1]).to(dtype=dtype)


def _skew(data):
    return 0.5 * (data - data.transpose(-1, -2))


def _cayley(data):
    skew = _skew(data)
    n = skew.shape[-1]
    eye = torch.eye(n, dtype=skew.dtype, device=skew.device).expand_as(skew)
    return torch.linalg.solve(eye - skew, eye + skew, left=False)


def _inverse_cayley(q):
    n = q.shape[-1]
    eye = torch.eye(n, dtype=q.dtype, device=q.device).expand_as(q)
    generator = torch.linalg.solve(q + eye, q - eye, left=False)
    return _skew(generator)


def _upper_coords(tensor):
    n = tensor.shape[-1]
    idx = torch.triu_indices(n, n, offset=1, device=tensor.device)
    return tensor[:, idx[0], idx[1]]


def _coords_to_skew(coords, n):
    idx = torch.triu_indices(n, n, offset=1, device=coords.device)
    skew = torch.zeros(coords.shape[0], n, n, dtype=coords.dtype, device=coords.device)
    skew[:, idx[0], idx[1]] = coords
    return skew - skew.transpose(-1, -2)


def _transport_matrix(skew_matrix):
    block_size = skew_matrix.shape[-1]
    idx = torch.triu_indices(block_size, block_size, offset=1, device=skew_matrix.device)
    r = torch.matrix_exp(skew_matrix / 2)
    i, j = idx[0], idx[1]
    return r[:, i][:, :, i] * r[:, j][:, :, j] - r[:, i][:, :, j] * r[:, j][:, :, i]


def _fisher_to_matrix(fisher, ref_coords):
    fisher = fisher.to(device=ref_coords.device, dtype=torch.float32)
    if fisher.dim() == 3 and fisher.shape[-1] == fisher.shape[-2]:
        if fisher.shape[-1] == ref_coords.shape[-1]:
            return fisher
        fisher = _upper_coords(fisher)
    if fisher.shape == ref_coords.shape:
        return torch.diag_embed(fisher.clamp(min=0.0))
    raise ValueError(f"Unsupported Fisher shape {tuple(fisher.shape)} for {tuple(ref_coords.shape)}")


def _cayley_fisher_tangent(tangent, fishers, beta):
    log_coords = _upper_coords(tangent)
    h1 = _fisher_to_matrix(fishers[0], log_coords)
    h2 = _fisher_to_matrix(fishers[1], log_coords)
    pt = _transport_matrix(tangent).to(dtype=torch.float32)
    h2 = pt @ h2 @ pt.transpose(-1, -2)

    d = log_coords.shape[-1]
    eye = torch.eye(d, dtype=torch.float32, device=log_coords.device).expand(
        log_coords.shape[0], d, d
    )
    system = (1.0 - beta) * h1 + beta * h2
    rhs = beta * (h2 @ log_coords.float().unsqueeze(-1)).squeeze(-1)
    coords = torch.linalg.solve(system + 1e-8 * eye, rhs.unsqueeze(-1)).squeeze(-1)
    return _coords_to_skew(coords, tangent.shape[-1])


def _cayley_geodesic_merge_with_generators(tensors, device, alphas=None, fishers=None):
    if len(tensors) != 2:
        raise ValueError(f"Geodesic interpolation requires exactly 2 tensors; got {len(tensors)}")
    if alphas is None:
        t = torch.tensor(0.5, dtype=torch.float32, device=device)
    else:
        alpha_tensor = torch.tensor(alphas, dtype=torch.float32, device=device).flatten()
        if alpha_tensor.numel() != 2:
            raise ValueError(f"Geodesic alphas must have length 2; got {alpha_tensor.numel()}")
        t = alpha_tensor[1] / alpha_tensor.sum().clamp_min(1e-8)

    if t <= 1e-8:
        return _skew(tensors[0]).to(dtype=tensors[0].dtype)
    if t >= 1.0 - 1e-8:
        return _skew(tensors[1]).to(dtype=tensors[0].dtype)

    q0 = _cayley(tensors[0].to(device=device, dtype=torch.float32))
    q1 = _cayley(tensors[1].to(device=device, dtype=torch.float32))
    relative = q0.transpose(-1, -2) @ q1
    tangent = _inverse_cayley(relative)
    tangent_step = (
        _cayley_fisher_tangent(tangent, fishers, t)
        if fishers is not None
        else t * tangent
    )
    qt = q0 @ _cayley(tangent_step)
    return _inverse_cayley(qt).to(dtype=tensors[0].dtype)


def _gradients_merge_with_merging_py(
    tensors,
    device,
    mode,
    fishers=None,
    merger=None,
    alphas=None,
    geodesic_backend="cayley",
):
    if mode == "geodesic":
        if geodesic_backend == "cayley":
            return _cayley_geodesic_merge_with_generators(tensors, device, alphas, fishers)
        raise ValueError(f"Unsupported geodesic_backend: {geodesic_backend!r}")

    if merger is None:
        merger_cls = _load_root_oft_merging_class()
        merger = merger_cls(device=str(device))
    coords = [
        _full_generator_to_oft_coords(merger, tensor).to(device=device)
        for tensor in tensors
    ]
    fisher_list = None
    if fishers is not None:
        fisher_list = [
            fisher.to(device=device, dtype=coords[0].dtype)
            for fisher in fishers
        ]
    merged_coords = merger.merge_formula(
        coords,
        fisher_list=fisher_list,
        mode=mode,
        alphas=None if alphas is None else torch.tensor(alphas, dtype=torch.float32, device=device),
    )
    return _oft_coords_to_full_generator(merger, merged_coords, tensors[0].dtype)


def _log_merge(message):
    print(f"[merge] {message}", flush=True)


def _adapter_key_to_fisher_key(adapter_key, processor_keys):
    for layer_idx, processor_key in enumerate(processor_keys):
        prefix = f"{processor_key}."
        if adapter_key.startswith(prefix):
            return f"layers.{layer_idx}." + adapter_key[len(prefix):]
    return adapter_key


def get_seed(prompt, i, seed):
    h = 0
    for el in prompt:
        h += ord(el)
    h += i
    return h + seed


@inferencers.add_to_registry('base')
class BaseInferencer:
    def __init__(self, config, args, context_prompts, base_prompts,
                 dtype=torch.float32, device='cuda'):
        print("BaseInferencer init")
        self.config = config
        self.args = args
        self.checkpoint_idx = args.checkpoint_idx
        self.num_images_per_context_prompt = args.num_images_per_medium_prompt
        self.num_images_per_base_prompt = args.num_images_per_base_prompt
        self.batch_size_context = args.batch_size_medium
        self.batch_size_base = args.batch_size_base

        if self.checkpoint_idx is None:
            self.checkpoint_path = config['output_dir']
        else:
            self.checkpoint_path = os.path.join(config['output_dir'], f'checkpoint-{self.checkpoint_idx}')

        self.context_prompts = context_prompts
        self.base_prompts = base_prompts

        self.replace_inference_output = self.args.replace_inference_output
        self.version = self.args.version

        self.device = device
        self.dtype = dtype

    def setup_pipe_kwargs(self):
        self.pipe_kwargs = {
            'guidance_scale': self.args.guidance_scale,
            'num_inference_steps': self.args.num_inference_steps,
        }

    def setup_base_model(self):
        # Here we create base models
        model_name_or_path = self.config['pretrained_model_name_or_path']
        self.scheduler = _load_pretrained(
            "scheduler", DDIMScheduler, model_name_or_path, subfolder="scheduler"
        )
        self.unet = _load_pretrained_on_device(
            "UNet", UNet2DConditionModel, model_name_or_path, self.device, subfolder="unet"
        )
        self.vae = _load_pretrained(
            "VAE",
            AutoencoderKL,
            model_name_or_path,
            subfolder="vae", revision=self.config['revision']
        )
        self.tokenizer = _load_pretrained(
            "tokenizer",
            CLIPTokenizer,
            model_name_or_path,
            subfolder="tokenizer", revision=self.config['revision']
        )
        self.tokenizer_2 = _load_pretrained(
            "tokenizer_2",
            CLIPTokenizer,
            model_name_or_path,
            subfolder="tokenizer_2", revision=self.config['revision']
        )
        self.text_encoder = _load_pretrained(
            "text encoder",
            CLIPTextModel,
            model_name_or_path,
            subfolder="text_encoder", revision=self.config['revision']
        )
        self.text_encoder_2 = _load_pretrained(
            "text encoder 2",
            CLIPTextModelWithProjection,
            model_name_or_path,
            subfolder="text_encoder_2", revision=self.config['revision']
        )

    def setup_model(self):
        self.unet.load_state_dict(torch.load(
            os.path.join(self.checkpoint_path, 'unet.bin')
        ))

    def setup_pipeline(self):
        print("setup_pipeline for SDXL", flush=True)
        print("building SDXL pipeline from preloaded components...", flush=True)
        self.pipe = StableDiffusionXLPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            text_encoder_2=self.text_encoder_2,
            tokenizer=self.tokenizer,
            tokenizer_2=self.tokenizer_2,
            unet=self.unet,
            scheduler=self.scheduler,
        ).to(self.device)
        print("built SDXL pipeline", flush=True)
        self.pipe.set_progress_bar_config(disable=True)

    def setup(self):
        self.setup_base_model()
        self.setup_model()
        self.setup_pipeline()
        self.setup_pipe_kwargs()
        self.create_folder_name()
        self.setup_paths()

    def prepare_prompts(self, context_prompts, base_prompts):
        return context_prompts, base_prompts

    def create_folder_name(self):
        self.inference_folder_name = f"ns{self.args.num_inference_steps}_gs{self.args.guidance_scale}"

    def setup_paths(self):
        if self.version is None:
            version = 0
            samples_path = os.path.join(
                self.checkpoint_path, 'samples', self.inference_folder_name, f'version_{version}'
            )
            print("samples_path", samples_path)
            if os.path.exists(samples_path):
                while not os.path.exists(samples_path):
                    samples_path = os.path.join(
                        self.checkpoint_path, 'samples', self.inference_folder_name, f'version_{version}'
                    )
                    version += 1
        else:
            samples_path = os.path.join(
                self.checkpoint_path, 'samples', self.inference_folder_name, f'version_{self.version}'
            )
        self.samples_path = samples_path

    def check_generation(self, path, num_images_per_prompt):
        if self.replace_inference_output:
            return True
        else:
            if os.path.exists(path) and len(os.listdir(path)) == num_images_per_prompt:
                return False
            else:
                return True

    def generate_with_prompt(self, prompt, num_images_per_prompt, batch_size):
        n_batches = (num_images_per_prompt - 1) // batch_size + 1
        images = []
        concept_token = self.config['placeholder_token_concept']
        style_token = self.config['placeholder_token_style']
        formatted_prompt = prompt.format(concept_token, style_token)
        for i in range(n_batches):
            seed = get_seed(prompt, i, self.args.seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            generator = torch.Generator(device='cuda')
            generator = generator.manual_seed(seed)
            print("Prompt before formatting:", prompt)
            print("Prompt after formatting:", formatted_prompt)
            
            images_batch = self.pipe(
                prompt=formatted_prompt,
                generator=generator, 
                num_images_per_prompt=batch_size,
                added_cond_kwargs={}, 
                **self.pipe_kwargs
            ).images
            images += images_batch
        return images

    def save_images(self, images, path):
        os.makedirs(path, exist_ok=True)
        for idx, image in enumerate(images):
            image.save(os.path.join(path, f'{idx}.png'))

    def generate_with_prompt_list(self, prompts, num_images_per_prompt, batch_size):
        print('num_images_per_prompt:',num_images_per_prompt)
        for prompt in tqdm(prompts):
            formatted_prompt = prompt.format(self.config['placeholder_token_concept'], self.config['placeholder_token_style'])
            path = os.path.join(self.samples_path, formatted_prompt)
            if self.check_generation(path, num_images_per_prompt):
                print(f"[generate] writing {num_images_per_prompt} images to {path}", flush=True)
                images = self.generate_with_prompt(
                    prompt, num_images_per_prompt, batch_size)
                self.save_images(images, path)
            else:
                print(f"[generate] skipping existing prompt output: {path}", flush=True)

    def generate(self):
        context_prompts, base_prompts = self.prepare_prompts(self.context_prompts, self.base_prompts)
        self.generate_with_prompt_list(
            context_prompts, self.num_images_per_context_prompt, self.batch_size_context)


@inferencers.add_to_registry('lora')
class LoraInferencer(BaseInferencer):
    def __init__(self, config, args, context_prompts, base_prompts,
                 dtype=torch.float32, device='cuda'):

        super().__init__(config, args, context_prompts, base_prompts, dtype, device)

    def setup_model(self,):
        lora_attn_procs = {}
        for name in self.unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]

            if cross_attention_dim:
                rank = min(cross_attention_dim, hidden_size, self.config['lora_rank'])
            else:
                rank = min(hidden_size, self.config['lora_rank'])

            lora_attn_procs[name] = LoRACrossAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=rank,
            )

        self.unet.set_attn_processor(lora_attn_procs)
        self.unet_lora_layers = AttnProcsLayers(self.unet.attn_processors)
        self.unet_lora_layers.load_state_dict(load_file(os.path.join(self.checkpoint_path, 'pytorch_lora_weights.safetensors')))


@inferencers.add_to_registry('moft')
class MOFTInferencer(BaseInferencer):
    def __init__(self, config, args, context_prompts, base_prompts,
                 dtype=torch.float32, device='cuda'):

        super().__init__(config, args, context_prompts, base_prompts, dtype, device)

    def setup_model(self,):
        moft_attn_procs = {}
        for name in self.unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]

            moft_attn_procs[name] = MOFTCrossAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, nblocks=self.config['moft_nblocks'],
                method=self.config['moft_method'], scale=self.config['moft_scale']
            )

        self.unet.set_attn_processor(moft_attn_procs)
        print_unet_oft_summary(self.unet, "single MOFT UNet after processor setup")
        self.moft_layers = AttnProcsLayers(self.unet.attn_processors)
        moft_path = os.path.join(self.checkpoint_path, 'pytorch_lora_weights.safetensors')
        moft_state = load_file(moft_path)
        print_state_dict_summary(moft_state, "single MOFT adapter", moft_path)
        self.moft_layers.load_state_dict(moft_state)
        print_parameter_loading_summary(
            self.moft_layers,
            "single MOFT adapter layers after load_state_dict",
        )


@inferencers.add_to_registry('gradients')
class GradientsMergeInferencer(MOFTInferencer):
    """Merge MOFT adapters in compact Lie coordinates, optionally using Fishers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not str(self.device).startswith("cuda"):
            raise ValueError(
                f"GradientsMerge requires a CUDA device, but received {self.device!r}."
            )
        if not torch.cuda.is_available():
            raise RuntimeError("GradientsMerge requires CUDA, but no CUDA GPU is available.")

    def create_folder_name(self):
        mode = self._merge_mode()
        backend_suffix = ""
        if mode == "geodesic":
            backend_suffix = f"_{getattr(self.args, 'geodesic_backend', 'cayley')}"
            if getattr(self.args, "geodesic_use_fishers", False):
                backend_suffix += "_fisher"
        pair_name = getattr(self.args, "dataset_pair_name", None)
        pair_suffix = f"_{pair_name}" if pair_name else ""
        self.inference_folder_name = (
            f"ns{self.args.num_inference_steps}_gs{self.args.guidance_scale}"
            f"_gradients_{mode}{backend_suffix}{pair_suffix}"
        )

    def _fisher_paths(self):
        if getattr(self.args, "merge_mode", None) == "geodesic" and not getattr(
            self.args,
            "geodesic_use_fishers",
            False,
        ):
            return None
        concept_fisher_path = getattr(self.args, "concept_fisher_path", None)
        style_fisher_path = getattr(self.args, "style_fisher_path", None)
        if concept_fisher_path is None and style_fisher_path is None:
            return None
        if concept_fisher_path is None or style_fisher_path is None:
            raise ValueError("Both concept_fisher_path and style_fisher_path are required.")
        return concept_fisher_path, style_fisher_path

    def _merge_mode(self):
        explicit_mode = getattr(self.args, "merge_mode", None)
        if explicit_mode is not None:
            return explicit_mode
        if self._fisher_paths() is not None:
            return "diagonal_fisher_rescaled" if getattr(self.args, "rescale", False) else "diagonal_fisher"
        return "standard_rescaled" if getattr(self.args, "rescale", False) else "standard"

    def setup_model(self):
        self.unet = self.unet.to(self.device)
        moft_attn_procs = {}
        for name in self.unet.attn_processors.keys():
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else self.unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]

            moft_attn_procs[name] = MOFTCrossAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                nblocks=self.config['moft_nblocks'],
                method=self.config['moft_method'],
                scale=self.config['moft_scale'],
                device=self.device,
            )

        if self.args.moft_layers_concept_path is None:
            raise ValueError("moft_layers_concept_path is not provided")
        if self.args.moft_layers_style_path is None:
            raise ValueError("moft_layers_style_path is not provided")

        pair_name = getattr(self.args, "dataset_pair_name", None) or "single"
        _log_merge(f"starting Gradients merge for pair={pair_name} on device={self.device}")
        _log_merge(f"loading concept adapter: {self.args.moft_layers_concept_path}")
        load_start = time.time()
        concept_state = load_file(
            self.args.moft_layers_concept_path,
            device=self.device,
        )
        _log_merge(f"loaded concept adapter in {time.time() - load_start:.2f}s")
        print_state_dict_summary(
            concept_state,
            "Gradients concept adapter",
            self.args.moft_layers_concept_path,
        )
        _log_merge(f"loading style adapter: {self.args.moft_layers_style_path}")
        load_start = time.time()
        style_state = load_file(
            self.args.moft_layers_style_path,
            device=self.device,
        )
        _log_merge(f"loaded style adapter in {time.time() - load_start:.2f}s")
        print_state_dict_summary(
            style_state,
            "Gradients style adapter",
            self.args.moft_layers_style_path,
        )
        if concept_state.keys() != style_state.keys():
            missing_from_style = sorted(concept_state.keys() - style_state.keys())
            missing_from_concept = sorted(style_state.keys() - concept_state.keys())
            raise ValueError(
                "The concept and style adapters have different parameter sets. "
                f"Missing from style: {missing_from_style}; "
                f"missing from concept: {missing_from_concept}."
            )

        merge_mode = self._merge_mode()
        fisher_paths = self._fisher_paths()
        concept_fisher = style_fisher = None
        if fisher_paths is not None:
            _log_merge(f"loading concept Fisher: {fisher_paths[0]}")
            load_start = time.time()
            concept_fisher = load_file(fisher_paths[0], device=self.device)
            _log_merge(f"loaded concept Fisher in {time.time() - load_start:.2f}s")
            _log_merge(f"loading style Fisher: {fisher_paths[1]}")
            load_start = time.time()
            style_fisher = load_file(fisher_paths[1], device=self.device)
            _log_merge(f"loaded style Fisher in {time.time() - load_start:.2f}s")
            print_state_dict_summary(concept_fisher, "Gradients concept Fisher", fisher_paths[0])
            print_state_dict_summary(style_fisher, "Gradients style Fisher", fisher_paths[1])

        merge_keys = [key for key in concept_state if key.endswith((".L", ".R"))]
        passthrough_keys = len(concept_state) - len(merge_keys)
        processor_keys = list(moft_attn_procs.keys())
        if concept_fisher is not None and merge_keys:
            example_adapter_key = merge_keys[0]
            example_fisher_key = _adapter_key_to_fisher_key(example_adapter_key, processor_keys)
            _log_merge(f"Fisher key naming: adapter={example_adapter_key}")
            _log_merge(f"Fisher key naming: fisher ={example_fisher_key}")
            if getattr(self.args, "fisher_min", None) is not None:
                _log_merge(f"clipping Fisher values below {self.args.fisher_min:g}")
            if getattr(self.args, "fisher_rescale", None) is not None:
                _log_merge(f"multiplying Fisher values by {self.args.fisher_rescale:g}")
        _log_merge(
            f"mode={merge_mode}; tensors={len(concept_state)}; "
            f"merge_tensors={len(merge_keys)}; passthrough_tensors={passthrough_keys}"
        )
        merge_alphas = getattr(self.args, "alphas", None)
        _log_merge(f"alphas concept/style={merge_alphas if merge_alphas is not None else [1.0, 1.0]}")
        shape_counts = Counter(tuple(concept_state[key].shape) for key in merge_keys)
        side_counts = Counter(key.rsplit(".", 1)[-1] for key in merge_keys)
        projection_counts = Counter(
            projection
            for key in merge_keys
            for projection in ("to_q_moft", "to_k_moft", "to_v_moft", "to_out_moft")
            if f".{projection}." in key
        )
        processor_count = len({
            key.split(".to_", 1)[0]
            for key in merge_keys
        })
        shape_summary = ", ".join(
            f"{shape}: {count}"
            for shape, count in sorted(shape_counts.items())
        )
        _log_merge(
            f"summary: processors={processor_count}; "
            f"projections={dict(projection_counts)}; sides={dict(side_counts)}"
        )
        _log_merge(f"summary: tensor shapes={shape_summary}")
        merged_state = {}
        geodesic_backend = getattr(self.args, "geodesic_backend", "cayley")
        if merge_mode == "geodesic" and geodesic_backend == "cayley":
            merge_engine = None
            _log_merge("using fast Cayley geodesic merge for this adapter merge")
        else:
            merger_cls = (
                _load_root_oft_geodesic_merging_class()
                if merge_mode == "geodesic"
                else _load_root_oft_merging_class()
            )
            merge_engine = merger_cls(device=str(self.device))
            _log_merge(f"initialized {merger_cls.__name__} engine once for this adapter merge")
        merge_start = time.time()
        merged_count = 0
        slowest_key = None
        slowest_seconds = 0.0
        seconds_by_shape = defaultdict(float)
        for key, concept_tensor in concept_state.items():
            style_tensor = style_state[key]
            if key.endswith((".L", ".R")):
                fishers = None
                if concept_fisher is not None:
                    fisher_key = _adapter_key_to_fisher_key(key, processor_keys)
                    fishers = [concept_fisher[fisher_key], style_fisher[fisher_key]]
                    fisher_min = getattr(self.args, "fisher_min", None)
                    if fisher_min is not None:
                        fishers = [fisher.clamp_min(fisher_min) for fisher in fishers]
                    fisher_rescale = getattr(self.args, "fisher_rescale", None)
                    if fisher_rescale is not None:
                        fishers = [fisher * fisher_rescale for fisher in fishers]
                tensor_start = time.time()
                merged_state[key] = _gradients_merge_with_merging_py(
                    [concept_tensor, style_tensor],
                    device=concept_tensor.device,
                    mode=merge_mode,
                    fishers=fishers,
                    merger=merge_engine,
                    alphas=merge_alphas,
                    geodesic_backend=geodesic_backend,
                )
                tensor_seconds = time.time() - tensor_start
                merged_count += 1
                shape = tuple(concept_tensor.shape)
                seconds_by_shape[shape] += tensor_seconds
                if tensor_seconds > slowest_seconds:
                    slowest_key = key
                    slowest_seconds = tensor_seconds
                if merged_count == 1 or merged_count % 50 == 0 or merged_count == len(merge_keys):
                    _log_merge(
                        f"progress: merged {merged_count}/{len(merge_keys)} "
                        f"elapsed={time.time() - merge_start:.1f}s "
                        f"last_shape={shape} slowest={slowest_seconds:.2f}s"
                    )
            else:
                # Gradients merging acts on the orthogonal generators. Keep auxiliary
                # parameters (for example learned output scales) from concept,
                # matching the behavior of the existing OrthoFuse mergers.
                merged_state[key] = concept_tensor
        time_summary = ", ".join(
            f"{shape}: {seconds:.1f}s"
            for shape, seconds in sorted(seconds_by_shape.items())
        )
        _log_merge(f"merge time by tensor shape: {time_summary}")
        if slowest_key is not None:
            _log_merge(f"slowest tensor: {slowest_key} ({slowest_seconds:.2f}s)")
        _log_merge(f"built merged state in {time.time() - merge_start:.2f}s")

        _log_merge("setting UNet attention processors")
        self.unet.set_attn_processor(moft_attn_procs)
        print_unet_oft_summary(self.unet, "Gradients UNet after processor setup")
        _log_merge("loading merged state into AttnProcsLayers")
        self.moft_layers = AttnProcsLayers(self.unet.attn_processors)
        self.moft_layers.load_state_dict(merged_state)
        self.moft_layers = self.moft_layers.to(self.device)
        print_parameter_loading_summary(
            self.moft_layers,
            "Gradients merged adapter layers after load_state_dict",
        )

        moft_layer_names = (
            "to_q_moft",
            "to_k_moft",
            "to_v_moft",
            "to_out_moft",
        )
        orthogonal_layers = [
            getattr(processor, layer_name).ort_monarch
            for processor in self.unet.attn_processors.values()
            for layer_name in moft_layer_names
            if hasattr(processor, layer_name)
        ]
        _log_merge(f"applying Cayley transforms to {len(orthogonal_layers)} orthogonal layers")
        cayley_start = time.time()
        with torch.no_grad():
            for layer_idx, orthogonal_layer in enumerate(tqdm(
                orthogonal_layers,
                desc=f"Applying Cayley transforms [{pair_name}]",
                unit="layer",
            ), 1):
                orthogonal_layer.L.copy_(
                    orthogonal_layer.cayley_batch(orthogonal_layer.L)
                )
                orthogonal_layer.R.copy_(
                    orthogonal_layer.cayley_batch(orthogonal_layer.R)
                )
                orthogonal_layer.method = "already_orthogonal"
                if layer_idx == 1 or layer_idx == len(orthogonal_layers) or layer_idx % 25 == 0:
                    _log_merge(f"Cayley layer {layer_idx}/{len(orthogonal_layers)} done")
        _log_merge(f"finished Cayley transforms in {time.time() - cayley_start:.2f}s")

        parameter_device = next(self.moft_layers.parameters()).device
        if parameter_device.type != "cuda":
            raise RuntimeError(
                f"Gradients adapters were expected on CUDA, but are on {parameter_device}."
            )
        _log_merge(f"Gradients merge completed on {parameter_device}")


@inferencers.add_to_registry('orthomerge')
class OrthoMergeInferencer(GradientsMergeInferencer):
    """Compatibility alias: no Fishers, optional OrthoMerge norm correction."""

    def _fisher_paths(self):
        return None

    def _merge_mode(self):
        return "standard_rescaled" if getattr(self.args, "rescale", False) else "standard"

    def create_folder_name(self):
        rescale_suffix = "rescaled" if self.args.rescale else "unscaled"
        pair_name = getattr(self.args, "dataset_pair_name", None)
        pair_suffix = f"_{pair_name}" if pair_name else ""
        self.inference_folder_name = (
            f"ns{self.args.num_inference_steps}_gs{self.args.guidance_scale}"
            f"_orthomerge_{rescale_suffix}{pair_suffix}"
        )


@inferencers.add_to_registry('moft_merge')
class MOFTMergeInferencer(BaseInferencer):
    """
    MOFTMergeInferencer performs merge of orthogonal adapters and than runs
    inference. The merge in particular is implemented in layer_merge function.
    layer_merge supports two types of merging: block-wise merging and merging
    inside Cayley space, which corresponds to weighted sum of skew-hermittian matrix.
    """
    def __init__(self, config, args, context_prompts, base_prompts,
                 dtype=torch.float32, device='cuda'):

        super().__init__(config, args, context_prompts, base_prompts, dtype, device)
     
    def create_folder_name(self):
        pair_name = getattr(self.args, "dataset_pair_name", None)
        pair_suffix = f"_{pair_name}" if pair_name else ""
        self.inference_folder_name = (
            f"ns{self.args.num_inference_steps}_gs{self.args.guidance_scale}"
            f"_orthofuse_t{self.args.t}_method_{self.args.postprocessing_method}{pair_suffix}"
        )
    
    def layer_merge(self, t, layer = "to_q_moft", merging_type="blocked", parameter=1, postprocessing_method="uniform"):
        for name, proc1 in self.unet.attn_processors.items():
            proc2 = self.unet_style.attn_processors[name]
            if hasattr(proc1, layer) and hasattr(proc2, layer):
                if merging_type == "blocked":
                    merged_gs = blocked_geodesic_combination(
                            getattr(proc1, layer).ort_monarch,
                            getattr(proc2, layer).ort_monarch,
                            t
                    )

                    assert merged_gs.method == "already_orthogonal", "merge is not already_orthogonal" # Remove it after debugging
                elif merging_type == "merge_inside_cayley_space":
                    merged_gs = merge_inside_cayley_space(
                            [getattr(proc1, layer).ort_monarch, getattr(proc2, layer).ort_monarch],
                            torch.tensor([t, 1-t], device=getattr(proc1, layer).ort_monarch.L.data.device)
                    )
                    
                    assert merged_gs.method == "already_orthogonal", "merge is not already_orthogonal" # Remove it after debugging

                # Here we modify eigenvalues of the already orthogonal matrix
                merged_gs = postprocess_blocks(merged_gs, method=postprocessing_method, parameter=parameter)
                getattr(proc1, layer).ort_monarch.L.data.copy_(merged_gs.L.data)
                getattr(proc1, layer).ort_monarch.R.data.copy_(merged_gs.R.data)
                getattr(proc1, layer).ort_monarch.method = "already_orthogonal"         

    def setup_base_model(self):
        # Here we create base models
        print("setup_base_model SDXL ...", flush=True)
        model_name_or_path = self.config['pretrained_model_name_or_path']
        self.scheduler = _load_pretrained(
            "scheduler", DDIMScheduler, model_name_or_path, subfolder="scheduler"
        )
        self.unet = _load_pretrained_on_device(
            "concept UNet", UNet2DConditionModel, model_name_or_path, self.device, subfolder="unet"
        )
        self.unet_style = _load_pretrained_on_device(
            "style UNet", UNet2DConditionModel, model_name_or_path, self.device, subfolder="unet"
        )
        self.vae = _load_pretrained(
            "VAE",
            AutoencoderKL,
            model_name_or_path,
            subfolder="vae", revision=self.config['revision']
        )
        self.tokenizer = _load_pretrained(
            "tokenizer",
            CLIPTokenizer,
            model_name_or_path,
            subfolder="tokenizer", revision=self.config['revision']
        )
        self.tokenizer_2 = _load_pretrained(
            "tokenizer_2",
            CLIPTokenizer,
            model_name_or_path,
            subfolder="tokenizer_2", revision=self.config['revision']
        )
        self.text_encoder = _load_pretrained(
            "text encoder",
            CLIPTextModel,
            model_name_or_path,
            subfolder="text_encoder", revision=self.config['revision']
        )
        self.text_encoder_2 = _load_pretrained(
            "text encoder 2",
            CLIPTextModelWithProjection,
            model_name_or_path,
            subfolder="text_encoder_2", revision=self.config['revision']
        )

    def setup_model(self,):
        print("setup_model ...")
        print("self.device:", self.device)
        moft_attn_procs = {}
        moft_attn_procs_style = {}
        for name in self.unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]

            moft_attn_procs[name] = MOFTCrossAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, nblocks=self.config['moft_nblocks'],
                method=self.config['moft_method'], scale=self.config['moft_scale']
            )
            
        for name in self.unet_style.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.unet_style.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.unet_style.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet_style.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet_style.config.block_out_channels[block_id]

            moft_attn_procs_style[name] = MOFTCrossAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, nblocks=self.config['moft_nblocks'],
                method=self.config['moft_method'], scale=self.config['moft_scale']
            )
        self.unet.set_attn_processor(moft_attn_procs)
        print_unet_oft_summary(self.unet, "concept UNet after processor setup")
        self.moft_layers = AttnProcsLayers(self.unet.attn_processors)
        # self.moft_layers.load_state_dict(load_file(os.path.join(self.checkpoint_path, 'pytorch_lora_weights_concept.safetensors')))
        if self.args.moft_layers_concept_path is not None:
            concept_state = load_file(self.args.moft_layers_concept_path)
            print_state_dict_summary(
                concept_state,
                "concept adapter",
                self.args.moft_layers_concept_path,
            )
            self.moft_layers.load_state_dict(concept_state)
            print_parameter_loading_summary(
                self.moft_layers,
                "concept adapter layers after load_state_dict",
            )
        
        else:
            raise ValueError("moft_layers_concept_path is not provided")
        
        self.unet_style.set_attn_processor(moft_attn_procs_style)
        print_unet_oft_summary(self.unet_style, "style UNet after processor setup")
        self.moft_layers_style = AttnProcsLayers(self.unet_style.attn_processors)
        if self.args.moft_layers_style_path is not None:
            style_state = load_file(self.args.moft_layers_style_path)
            print_state_dict_summary(
                style_state,
                "style adapter",
                self.args.moft_layers_style_path,
            )
            self.moft_layers_style.load_state_dict(style_state)
            print_parameter_loading_summary(
                self.moft_layers_style,
                "style adapter layers after load_state_dict",
            )
        else:
            raise ValueError("moft_layers_style_path is not provided")

        # Перенос всех весов на device
        for proc in self.unet.attn_processors.values():
            if hasattr(proc, "ort_monarch"):
                proc.ort_monarch = proc.ort_monarch.to(self.device)
        for proc in self.unet_style.attn_processors.values():
            if hasattr(proc, "ort_monarch"):
                proc.ort_monarch = proc.ort_monarch.to(self.device)

        if self.args.t is not None:
            t = self.args.t
        else:
            print("WARNING: t is not provided, using midpoint merge")
            t = 0.5  # midpoint merge
        # merging_type = "merge_inside_cayley_space"
        merging_type = "blocked"
        merging = True
        if merging == True:
            print("Starting merging with merging_type:", merging_type, "and t:", t)
            print("merging to_q_moft...")
            self.layer_merge(t, layer = "to_q_moft", merging_type=merging_type, parameter=t, postprocessing_method=self.args.postprocessing_method)
            print("merging to_k_moft...")
            self.layer_merge(t, layer = "to_k_moft", merging_type=merging_type, parameter=t, postprocessing_method=self.args.postprocessing_method)
            print("merging to_v_moft...")
            self.layer_merge(t, layer = "to_v_moft", merging_type=merging_type, parameter=t, postprocessing_method=self.args.postprocessing_method)
            print("merging to_out_moft...")
            self.layer_merge(t, layer = "to_out_moft", merging_type=merging_type, parameter=t, postprocessing_method=self.args.postprocessing_method)
            print("merging done!")


@inferencers.add_to_registry('lora_merge')
class LoraMergeInferencer(BaseInferencer):
    def __init__(self, config, args, context_prompts, base_prompts,
                 dtype=torch.float32, device='cuda'):

        super().__init__(config, args, context_prompts, base_prompts, dtype, device)
    
    def create_folder_name(self):
        self.inference_folder_name = f"ns{self.args.num_inference_steps}_gs{self.args.guidance_scale}_t{self.args.t}" 
    
    def setup_base_model(self):
        # Here we create base models
        print("setup_base_model ...")
        self.scheduler = DDIMScheduler.from_pretrained(
            self.config['pretrained_model_name_or_path'], subfolder="scheduler"
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            self.config['pretrained_model_name_or_path'], subfolder="unet"
        )
        self.unet_style = UNet2DConditionModel.from_pretrained(
            self.config['pretrained_model_name_or_path'], subfolder="unet"
        )
        self.vae = AutoencoderKL.from_pretrained(
            self.config['pretrained_model_name_or_path'],
            subfolder="vae", revision=self.config['revision']
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.config['pretrained_model_name_or_path'],
            subfolder="tokenizer", revision=self.config['revision']
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.config['pretrained_model_name_or_path'],
            subfolder="text_encoder", revision=self.config['revision']
        )
        
    def setup_model(self,):
        lora_attn_procs = {}
        lora_attn_procs_style = {}
        for name in self.unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]

            if cross_attention_dim:
                rank = min(cross_attention_dim, hidden_size, self.config['lora_rank'])
            else:
                rank = min(hidden_size, self.config['lora_rank'])

            lora_attn_procs[name] = LoRACrossAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=rank,
            )
            
        for name in self.unet_style.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.unet_style.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.unet_style.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet_style.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet_style.config.block_out_channels[block_id]

            if cross_attention_dim:
                rank = min(cross_attention_dim, hidden_size, self.config['lora_rank'])
            else:
                rank = min(hidden_size, self.config['lora_rank'])

            lora_attn_procs_style[name] = LoRACrossAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=rank,
            )
            
        self.unet.set_attn_processor(lora_attn_procs)
        self.lora_layers = AttnProcsLayers(self.unet.attn_processors)
        self.lora_layers.load_state_dict(load_file(os.path.join(self.checkpoint_path, 'pytorch_lora_weights_concept.safetensors')))
        
        self.unet_style.set_attn_processor(lora_attn_procs_style)
        self.lora_layers_style = AttnProcsLayers(self.unet_style.attn_processors)
        self.lora_layers_style.load_state_dict(load_file(os.path.join(self.checkpoint_path, 'pytorch_lora_weights_style.safetensors')))

        if self.args.t is not None:
            t = self.args.t
        else:
            print("WARNING: t is not provided, using midpoint merge")
            t = 0.5  # midpoint merge

        merging_type = "frb"
        merging = True
        
        if merging == True:
            print("Starting merging with merging_type:", merging_type, "and t:", t)
            print("merging to_q_lora...")
            self.layer_merge(t, layer = "to_q_lora", merging_type=merging_type)
            print("merging to_k_lora...")
            self.layer_merge(t, layer = "to_k_lora", merging_type=merging_type)
            print("merging to_v_lora...")
            self.layer_merge(t, layer = "to_v_lora", merging_type=merging_type)
            print("merging to_out_lora...")
            self.layer_merge(t, layer = "to_out_lora", merging_type=merging_type)
            print("merging done!")
        print("Merged L norm:", self.unet.attn_processors['down_blocks.1.attentions.0.transformer_blocks.0.attn1.processor'].to_q_lora.down.weight.norm())
         
    def layer_merge(self, t, layer = "to_q_lora", merging_type="blocked"):
        # Create a directory for saving matrices if it doesn't exist
        matrices_dir = os.path.join(self.checkpoint_path, 'matrices')
        os.makedirs(matrices_dir, exist_ok=True)
        
        for name, proc1 in self.unet.attn_processors.items():
            proc2 = self.unet_style.attn_processors[name]
            if hasattr(proc1, layer) and hasattr(proc2, layer):
                                   
                if merging_type == "frb":
                    if hasattr(getattr(proc1, layer), 'down'):
                        A_proc1 = getattr(proc1, layer).down.weight
                        B_proc1 = getattr(proc1, layer).up.weight
                        A_proc2 = getattr(proc2, layer).down.weight
                        B_proc2 = getattr(proc2, layer).up.weight
                        fr_batch = [
                            (A_proc1.T, B_proc1),
                            (A_proc2.T, B_proc2),
                        ] 

                        conv_coef = torch.tensor([t, 1-t])

                        frb = FixedRankBatch(fr_batch, conv_coef)
                        approx = frb.riemannian_barycenter_approximation()
                        new_matrix = approx.to_dense()
                        with torch.no_grad():
                            getattr(proc1, layer).down.weight.copy_(approx.U.T)
                            getattr(proc1, layer).up.weight.copy_(approx.V)
 
                else:
                    raise ValueError(f"Invalid merging type: {merging_type}")


@inferencers.add_to_registry('moft_merge_fast')
class MOFTMergeFastInferencer(BaseInferencer):
    """
    MOFTMergeFastInferencer is the accelerated version of MOFTMergeInferencer. 
    """
    def __init__(self, config, args, context_prompts, base_prompts,
                 dtype=torch.float32, device='cuda'):

        super().__init__(config, args, context_prompts, base_prompts, dtype, device)
     
    def create_folder_name(self):
        self.inference_folder_name = f"ns{self.args.num_inference_steps}_gs{self.args.guidance_scale}_t{self.args.t}_method_{self.args.postprocessing_method}_ultra_fast" 
    
    def layer_merge(self, t, layers = ["to_q_moft", "to_k_moft", "to_v_moft", "to_out_moft"], merging_type="blocked", parameter=1, postprocessing_method="uniform"):
      
        filename = "layers_distribution.json"
        with open(filename, 'r') as f:
            layers_distribution = json.load(f)

        # Локальные ссылки на словари процессоров, чтобы не дергать
        # атрибуты self.unet / self.unet_style в каждом шаге цикла.
        
        unet_attn_procs = self.unet.attn_processors
        unet_style_attn_procs = self.unet_style.attn_processors
        layer_size_list = [20, 40, 64]   # 3 размера слоев

        for layer_size in layer_size_list:
            layer_q_moft = layers_distribution['to_q_moft'][str(layer_size)] # берем названия слоев to_q_moft с конкретным размером
            layer_k_moft = layers_distribution['to_k_moft'][str(layer_size)]
            layer_v_moft = layers_distribution['to_v_moft'][str(layer_size)]
            layer_out_moft = layers_distribution['to_out_moft'][str(layer_size)]
            p1_list, p2_list = [], []
            for layer in layer_q_moft:
                proc1 = unet_attn_procs[layer]
                proc2 = unet_style_attn_procs[layer]
                p1_monarch = getattr(proc1, "to_q_moft").ort_monarch
                p2_monarch = getattr(proc2, "to_q_moft").ort_monarch
                p1_list.append(p1_monarch)
                p2_list.append(p2_monarch)
            
            for layer in layer_k_moft:
                proc1 = unet_attn_procs[layer]
                proc2 = unet_style_attn_procs[layer]
                p1_monarch = getattr(proc1, "to_k_moft").ort_monarch
                p2_monarch = getattr(proc2, "to_k_moft").ort_monarch
                p1_list.append(p1_monarch)
                p2_list.append(p2_monarch)

            for layer in layer_v_moft:
                proc1 = unet_attn_procs[layer]
                proc2 = unet_style_attn_procs[layer]
                p1_monarch = getattr(proc1, "to_v_moft").ort_monarch
                p2_monarch = getattr(proc2, "to_v_moft").ort_monarch
                p1_list.append(p1_monarch)
                p2_list.append(p2_monarch)

            for layer in layer_out_moft:
                proc1 = unet_attn_procs[layer]
                proc2 = unet_style_attn_procs[layer]
                p1_monarch = getattr(proc1, "to_out_moft").ort_monarch
                p2_monarch = getattr(proc2, "to_out_moft").ort_monarch
                p1_list.append(p1_monarch)
                p2_list.append(p2_monarch)

            L1 = torch.stack([p.L.data for p in p1_list], dim=0)  # [B, nblocks, d, d]
            R1 = torch.stack([p.R.data for p in p1_list], dim=0)
            L2 = torch.stack([p.L.data for p in p2_list], dim=0)
            R2 = torch.stack([p.R.data for p in p2_list], dim=0)
            L_ortho, R_ortho = merge_inside_cayley_space_batch(
                L1,
                R1,
                L2,
                R2,
                torch.tensor([t, 1-t], device=p1_list[0].L.data.device)
            )

            # batch post-processing of orthogonal matrices
            L_ortho, R_ortho = postprocess_blocks_batch(
                L_ortho, R_ortho, method=postprocessing_method, parameter=parameter
            )

            # write back results into p1_list
            B = L_ortho.shape[0]
            assert B == len(p1_list), "Batch size mismatch in layer_merge"
            for i in range(B):
                p1_list[i].L.data.copy_(L_ortho[i])
                p1_list[i].R.data.copy_(R_ortho[i])
                p1_list[i].method = "already_orthogonal"

    def setup_base_model(self):
        # Here we create base models
        print("setup_base_model SDXL ...")
        model_name_or_path = self.config['pretrained_model_name_or_path']
        self.scheduler = _load_pretrained(
            "fast scheduler", DDIMScheduler, model_name_or_path, subfolder="scheduler"
        )
        self.unet = _load_pretrained_on_device(
            "fast concept UNet", UNet2DConditionModel, model_name_or_path, self.device, subfolder="unet"
        )
        self.unet_style = _load_pretrained_on_device(
            "fast style UNet", UNet2DConditionModel, model_name_or_path, self.device, subfolder="unet"
        )
        self.vae = _load_pretrained(
            "fast VAE",
            AutoencoderKL,
            model_name_or_path,
            subfolder="vae", revision=self.config['revision']
        )
        self.tokenizer = _load_pretrained(
            "fast tokenizer",
            CLIPTokenizer,
            model_name_or_path,
            subfolder="tokenizer", revision=self.config['revision']
        )
        self.tokenizer_2 = _load_pretrained(
            "fast tokenizer_2",
            CLIPTokenizer,
            model_name_or_path,
            subfolder="tokenizer_2", revision=self.config['revision']
        )
        self.text_encoder = _load_pretrained(
            "fast text encoder",
            CLIPTextModel,
            model_name_or_path,
            subfolder="text_encoder", revision=self.config['revision']
        )
        self.text_encoder_2 = _load_pretrained(
            "fast text encoder 2",
            CLIPTextModelWithProjection,
            model_name_or_path,
            subfolder="text_encoder_2", revision=self.config['revision']
        )

    def setup_model(self,):
        print("setup_model ...")
        print("self.device:", self.device)
        moft_attn_procs = {}
        moft_attn_procs_style = {}
        for name in self.unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]

            moft_attn_procs[name] = MOFTCrossAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, nblocks=self.config['moft_nblocks'],
                method=self.config['moft_method'], scale=self.config['moft_scale']
            )
            
        for name in self.unet_style.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.unet_style.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.unet_style.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet_style.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet_style.config.block_out_channels[block_id]

            moft_attn_procs_style[name] = MOFTCrossAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, nblocks=self.config['moft_nblocks'],
                method=self.config['moft_method'], scale=self.config['moft_scale']
            )
        self.unet.set_attn_processor(moft_attn_procs)
        print_unet_oft_summary(self.unet, "fast concept UNet after processor setup")
        self.moft_layers = AttnProcsLayers(self.unet.attn_processors)
        if self.args.moft_layers_concept_path is not None:
            concept_state = load_file(self.args.moft_layers_concept_path)
            print_state_dict_summary(
                concept_state,
                "fast concept adapter",
                self.args.moft_layers_concept_path,
            )
            self.moft_layers.load_state_dict(concept_state)
            print_parameter_loading_summary(
                self.moft_layers,
                "fast concept adapter layers after load_state_dict",
            )
        else:
            raise ValueError("moft_layers_concept_path is not provided")

        self.unet_style.set_attn_processor(moft_attn_procs_style)
        print_unet_oft_summary(self.unet_style, "fast style UNet after processor setup")
        self.moft_layers_style = AttnProcsLayers(self.unet_style.attn_processors)
        if self.args.moft_layers_style_path is not None:
            style_state = load_file(self.args.moft_layers_style_path)
            print_state_dict_summary(
                style_state,
                "fast style adapter",
                self.args.moft_layers_style_path,
            )
            self.moft_layers_style.load_state_dict(style_state)
            print_parameter_loading_summary(
                self.moft_layers_style,
                "fast style adapter layers after load_state_dict",
            )
        else:
            raise ValueError("moft_layers_style_path is not provided")

        # Перенос всех весов на device
        for proc in self.unet.attn_processors.values():
            if hasattr(proc, "ort_monarch"):
                proc.ort_monarch = proc.ort_monarch.to(self.device)
        for proc in self.unet_style.attn_processors.values():
            if hasattr(proc, "ort_monarch"):
                proc.ort_monarch = proc.ort_monarch.to(self.device)

        if self.args.t is not None:
            t = 1 - self.args.t # так сделано потому что в MOFTMergeInferencer t=0 это адаптер концепта, а t=1 это адаптер стиля, а в MOFTMergeFastInferencer порядок противоположный
        else:
            print("WARNING: t is not provided, using midpoint merge")
            t = 0.5  # midpoint merge
        merging_type = "merge_inside_cayley_space"
        # merging_type = "blocked"
        merging = True
        if merging == True:
            print("Starting merging with merging_type:", merging_type, "and t:", 1-t)
            print("merging...")
            self.layer_merge(t, layers = ["to_q_moft", "to_k_moft", "to_v_moft", "to_out_moft"], merging_type=merging_type, parameter=t, postprocessing_method=self.args.postprocessing_method)
            print("merging done!")
