import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.merging_geodesic import OFTGeodesicMerging


BLOCK_SIZE = 3
SON_DIM = 3
NUM_BLOCKS = 1


def make_oft_params(values: list[float]) -> torch.Tensor:
    return torch.tensor([values], dtype=torch.float32)


def make_identity_kfac(scale: float = 1.0) -> dict[str, torch.Tensor]:
    eye = torch.eye(BLOCK_SIZE, dtype=torch.float32).unsqueeze(0)
    return {
        "row": eye,
        "col": eye,
        "scale": torch.full((NUM_BLOCKS,), scale, dtype=torch.float32),
    }


def test_geodesic_kfac_reconstructs_compact_skew_metric():
    merger = OFTGeodesicMerging(device="cpu", fisher_backend="kfac")
    matrix = merger._fisher_matrix(make_identity_kfac(), make_oft_params([0.1, 0.2, -0.1]))
    expected = 2.0 * torch.eye(SON_DIM).unsqueeze(0)

    assert matrix.shape == (NUM_BLOCKS, SON_DIM, SON_DIM)
    assert torch.allclose(matrix, expected, atol=1e-6)


def test_geodesic_fisher_normalization_divides_by_trace():
    merger = OFTGeodesicMerging(device="cpu")
    fisher = torch.tensor([[2.0, 3.0, 5.0]], dtype=torch.float32)
    matrix = merger._fisher_matrix(fisher, make_oft_params([0.1, 0.2, -0.1]))

    actual = merger._normalize_fisher_matrix(matrix)
    expected = torch.diag_embed(fisher / fisher.sum(dim=-1, keepdim=True))

    assert torch.allclose(actual, expected, atol=1e-6)
    assert torch.allclose(actual.diagonal(dim1=-2, dim2=-1).sum(dim=-1), torch.ones(NUM_BLOCKS))


def test_geodesic_kfac_merge_matches_equivalent_full_fisher():
    weights = [
        make_oft_params([0.1, 0.2, -0.1]),
        make_oft_params([-0.2, 0.1, 0.3]),
    ]
    alphas = torch.tensor([0.25, 0.75], dtype=torch.float32)

    kfac_merger = OFTGeodesicMerging(device="cpu", fisher_backend="kfac")
    full_merger = OFTGeodesicMerging(device="cpu")

    kfac_fishers = [make_identity_kfac(), make_identity_kfac()]
    full_fishers = [2.0 * torch.eye(SON_DIM).unsqueeze(0) for _ in range(2)]

    actual = kfac_merger.merge_formula(
        weights,
        fisher_list=kfac_fishers,
        mode="standard",
        alphas=alphas,
    )
    expected = full_merger.merge_formula(
        weights,
        fisher_list=full_fishers,
        mode="standard",
        alphas=alphas,
    )

    assert torch.allclose(actual, expected, atol=1e-6)
