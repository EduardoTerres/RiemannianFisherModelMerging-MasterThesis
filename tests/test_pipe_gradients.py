import argparse
import sys
import types
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "OrthoFuse"))


def stub_module(name, **attrs):
    module = types.ModuleType(name)
    for attr_name, attr_value in attrs.items():
        setattr(module, attr_name, attr_value)
    sys.modules[name] = module
    return module


class DummyExternal:
    pass


stub_module(
    "diffusers",
    StableDiffusionPipeline=DummyExternal,
    AutoencoderKL=DummyExternal,
    DDIMScheduler=DummyExternal,
    UNet2DConditionModel=DummyExternal,
    StableDiffusionXLPipeline=DummyExternal,
    EulerDiscreteScheduler=DummyExternal,
)
stub_module("diffusers.loaders", AttnProcsLayers=DummyExternal)
stub_module(
    "transformers",
    CLIPTextModel=DummyExternal,
    CLIPTextModelWithProjection=DummyExternal,
    CLIPTokenizer=DummyExternal,
    PretrainedConfig=DummyExternal,
)
stub_module(
    "peft",
    LoraConfig=DummyExternal,
    get_peft_model=lambda *args, **kwargs: None,
)

from moft.inferencer_sdxl import (  # noqa: E402
    _diagonal_fisher_correction,
    _gradients_merge_with_merging_py,
)
from src.diffusion import pipe_gradients  # noqa: E402
from src.merging import OFTMerging  # noqa: E402


def make_args(**overrides):
    defaults = {
        "merge_mode": None,
        "rescale": False,
        "concept_fisher_path": None,
        "style_fisher_path": None,
        "fisher_min": None,
        "fisher_rescale": None,
        "fisher_backend": "diagonal",
        "fisher_correction_mu": None,
        "diagonal_fisher_correction_mu": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def make_merger():
    return OFTMerging(lam=1.0, device="cpu")


def make_generators(merger):
    concept_coords = torch.tensor([[0.10, -0.20, 0.30]], dtype=torch.float32)
    style_coords = torch.tensor([[-0.40, 0.50, 0.20]], dtype=torch.float32)
    return [
        merger.oft_params_to_skew_matrix(concept_coords, son_dimension=3),
        merger.oft_params_to_skew_matrix(style_coords, son_dimension=3),
    ]


def generator_to_coords(merger, generator):
    skew = 0.5 * (generator - generator.transpose(-1, -2))
    return merger.skew_matrix_to_oft_params(skew)


def coords_to_generator(merger, coords, dtype=torch.float32):
    return merger.oft_params_to_skew_matrix(coords, son_dimension=3).to(dtype=dtype)


def test_apply_pair_uses_kfac_paths_when_requested():
    pair = {
        "name": "cat__style",
        "concept": {
            "adapter_path": "/tmp/concept_adapter.safetensors",
            "fim_path": "/tmp/cat_oft_lie_fim.safetensors",
            "class_name": "cat",
            "placeholder_token": "<cat>",
        },
        "style": {
            "adapter_path": "/tmp/style_adapter.safetensors",
            "fim_path": "/tmp/style_oft_lie_fim.safetensors",
            "placeholder_token": "<style>",
        },
    }
    args = pipe_gradients.apply_pair(make_args(fisher_backend="kfac"), pair)

    assert args.concept_fisher_path == "/tmp/cat_oft_lie_kfac.safetensors"
    assert args.style_fisher_path == "/tmp/style_oft_lie_kfac.safetensors"


def test_without_fishers_matches_standard_gradients_merge():
    merger = make_merger()
    tensors = make_generators(merger)
    alphas = [0.25, 0.75]
    args = pipe_gradients.prepare_merge_args(make_args())

    actual = _gradients_merge_with_merging_py(
        tensors,
        device=torch.device("cpu"),
        mode=args.merge_mode,
        fishers=None,
        merger=merger,
        alphas=alphas,
    )
    expected_coords = merger.merge_formula(
        [generator_to_coords(merger, tensor) for tensor in tensors],
        fisher_list=None,
        mode="standard",
        alphas=torch.tensor(alphas),
    )

    assert args.merge_mode == "standard"
    assert torch.allclose(actual, coords_to_generator(merger, expected_coords), atol=1e-6)


def test_without_fishers_allows_fisher_mu_correction_on_standard_merge():
    merger = make_merger()
    tensors = make_generators(merger)
    alphas = [0.25, 0.75]
    args = pipe_gradients.prepare_merge_args(make_args(fisher_correction_mu=4.0))

    merged = _gradients_merge_with_merging_py(
        tensors,
        device=torch.device("cpu"),
        mode=args.merge_mode,
        fishers=None,
        merger=merger,
        alphas=alphas,
    )
    corrected = _diagonal_fisher_correction(
        merged,
        alphas=alphas,
        mu=args.diagonal_fisher_correction_mu,
    )

    assert args.merge_mode == "standard"
    assert torch.allclose(corrected, merged * 1.75, atol=1e-6)


def test_with_fishers_matches_fisher_merge():
    merger = make_merger()
    tensors = make_generators(merger)
    fishers = [
        torch.tensor([[1.0, 3.0, 2.0]], dtype=torch.float32),
        torch.tensor([[4.0, 0.5, 1.5]], dtype=torch.float32),
    ]
    alphas = [0.25, 0.75]
    args = pipe_gradients.prepare_merge_args(
        make_args(
            concept_fisher_path="/tmp/concept.safetensors",
            style_fisher_path="/tmp/style.safetensors",
            fisher_min=0.0,
            fisher_rescale=1.0,
        )
    )

    actual = _gradients_merge_with_merging_py(
        tensors,
        device=torch.device("cpu"),
        mode=args.merge_mode,
        fishers=fishers,
        merger=merger,
        alphas=alphas,
    )
    expected_coords = merger.merge_formula(
        [generator_to_coords(merger, tensor) for tensor in tensors],
        fisher_list=fishers,
        mode="diagonal_fisher",
        alphas=torch.tensor(alphas),
    )

    assert args.merge_mode == "fisher"
    assert torch.allclose(actual, coords_to_generator(merger, expected_coords), atol=1e-6)


def test_with_fishers_and_rescale_matches_fisher_rescaled_merge():
    merger = make_merger()
    tensors = make_generators(merger)
    fishers = [
        torch.tensor([[1.0, 3.0, 2.0]], dtype=torch.float32),
        torch.tensor([[4.0, 0.5, 1.5]], dtype=torch.float32),
    ]
    alphas = [0.25, 0.75]
    args = pipe_gradients.prepare_merge_args(
        make_args(
            rescale=True,
            concept_fisher_path="/tmp/concept.safetensors",
            style_fisher_path="/tmp/style.safetensors",
            fisher_min=0.0,
            fisher_rescale=1.0,
        )
    )

    actual = _gradients_merge_with_merging_py(
        tensors,
        device=torch.device("cpu"),
        mode=args.merge_mode,
        fishers=fishers,
        merger=merger,
        alphas=alphas,
    )
    expected_coords = merger.merge_formula(
        [generator_to_coords(merger, tensor) for tensor in tensors],
        fisher_list=fishers,
        mode="diagonal_fisher_rescaled",
        alphas=torch.tensor(alphas),
    )

    assert args.merge_mode == "fisher_rescaled"
    assert torch.allclose(actual, coords_to_generator(merger, expected_coords), atol=1e-6)
