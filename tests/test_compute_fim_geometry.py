import torch
import torch.nn.functional as F
from peft import OFTConfig, get_peft_model
from torch import nn

from src.compute_fim import (
    attach_fresh_zero_oft,
    materialize_trained_oft,
    oft_coordinates_to_skew,
    principal_rotation_sqrt,
    skew_to_oft_coordinates,
    unnormalized_cayley,
)


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3, bias=False)

    def forward(self, inputs):
        return self.linear(inputs)


def _toy_oft_config():
    return OFTConfig(
        r=0,
        oft_block_size=2,
        target_modules=["linear"],
        module_dropout=0.0,
        use_cayley_neumann=True,
    )


def test_blockwise_materialization_then_fresh_zero_adapter(tmp_path):
    config = _toy_oft_config()
    base = _ToyModel()
    original_weight = base.linear.weight.detach().clone()
    trained = get_peft_model(base, config)
    coordinate_name, coordinates = next(
        (name, parameter)
        for name, parameter in trained.named_parameters()
        if parameter.requires_grad
    )
    with torch.no_grad():
        coordinates.copy_(torch.tensor([[0.2], [-0.1]]))

    rotation = unnormalized_cayley(oft_coordinates_to_skew(coordinates.detach()))
    inputs = torch.randn(5, 4)
    rotated_inputs = torch.einsum(
        "...rk,rkc->...rc",
        inputs.reshape(5, 2, 2),
        rotation,
    ).reshape_as(inputs)
    expected = F.linear(rotated_inputs, original_weight)

    materialized, rotations = materialize_trained_oft(
        trained,
        tmp_path / "oft_rotations.safetensors",
    )
    assert torch.allclose(materialized(inputs), expected, atol=1e-6, rtol=1e-6)

    fresh, theta_halves = attach_fresh_zero_oft(
        materialized,
        config,
        "cpu",
        rotations,
    )
    assert set(theta_halves) == {coordinate_name}
    assert torch.allclose(fresh(inputs), expected, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        theta_halves[coordinate_name] @ theta_halves[coordinate_name],
        rotation,
        atol=1e-5,
        rtol=1e-5,
    )

    fresh_coordinates = dict(fresh.named_parameters())[coordinate_name]
    with torch.no_grad():
        fresh_coordinates.copy_(torch.tensor([[0.05], [-0.15]]))
    rotation_delta = unnormalized_cayley(
        oft_coordinates_to_skew(fresh_coordinates.detach())
    )
    expected_local_inputs = torch.einsum(
        "...rk,rkc->...rc",
        inputs.reshape(5, 2, 2),
        rotation @ rotation_delta,
    ).reshape_as(inputs)
    expected_local = F.linear(expected_local_inputs, original_weight)
    assert torch.allclose(
        fresh(inputs),
        expected_local,
        atol=1e-6,
        rtol=1e-6,
    )


def test_cayley_correction_precedes_blockwise_transport():
    trained_coordinates = torch.tensor([[0.2, -0.1, 0.3]])
    rotation = unnormalized_cayley(
        oft_coordinates_to_skew(trained_coordinates)
    )
    rotation_half = principal_rotation_sqrt(rotation)
    raw_coordinate_gradient = torch.tensor([[3.0, -2.0, 1.0]])

    left_trivialized = oft_coordinates_to_skew(raw_coordinate_gradient / 2)
    transported = (
        rotation_half
        @ left_trivialized
        @ rotation_half.transpose(-1, -2)
    )
    transported_coordinates = skew_to_oft_coordinates(transported)

    assert torch.equal(
        skew_to_oft_coordinates(left_trivialized),
        raw_coordinate_gradient / 2,
    )
    assert not torch.allclose(
        transported_coordinates.square(),
        raw_coordinate_gradient.square() / 4,
    )
