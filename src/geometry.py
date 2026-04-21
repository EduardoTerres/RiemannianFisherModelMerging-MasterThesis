"""Riemannian geometry primitives for SO(n)-based model merging."""

from __future__ import annotations
from abc import ABC, abstractmethod
import torch
from torch import Tensor
from safetensors import safe_open


class Manifold(ABC):
    """Abstract Riemannian manifold with log, exp, and geodesic distance."""

    @abstractmethod
    def log(self, base: Tensor, point: Tensor) -> Tensor:
        """
        Riemannian logarithmic map Log_base(point).

        Args:
            base: (..., n, n) base point on the manifold.
            point: (..., n, n) point on the manifold.

        Returns:
            (..., n, n) tangent vector at base pointing toward point.
        """

    @abstractmethod
    def exp(self, base: Tensor, tangent: Tensor) -> Tensor:
        """
        Riemannian exponential map Exp_base(tangent).

        Args:
            base: (..., n, n) base point on the manifold.
            tangent: (..., n, n) tangent vector at base.

        Returns:
            (..., n, n) point on the manifold reached by following tangent.
        """

    @abstractmethod
    def dist_sq(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Squared geodesic distance d^2(x, y).

        Args:
            x: (..., n, n) point on the manifold.
            y: (..., n, n) point on the manifold.

        Returns:
            (...,) scalar squared distance for each batch element.
        """

    def geodesic_interpolate(
        self, start_point: Tensor, end_point: Tensor, s: float | Tensor,
    ) -> Tensor:
        """
        Interpolate along the geodesic from start_point to end_point.

        Args:
            start_point: (..., n, n) starting point on the manifold.
            end_point: (..., n, n) end point on the manifold.
            s: scalar in [0, 1], interpolation parameter.

        Returns:
            (..., n, n) point on the geodesic at fraction s.
        """
        tangent = self.log(start_point, end_point)
        return self.exp(start_point, s * tangent)


def invert_diagonal_fisher_sum(
    paths: list[str],
    eps: float = 1e-6,
    device: str = "cpu",
) -> dict[str, Tensor]:
    """
    Compute the element-wise inverse of the sum of diagonal Fisher matrices.

    Loads per-task diagonal Fishers F_1, ..., F_T from safetensors files and
    returns 1 / (F_1 + ... + F_T + eps) for each parameter.

    Args:
        paths: safetensors paths, one file per task Fisher.
        eps: small constant added before inversion for numerical stability.
        device: torch device for the returned tensors.

    Returns:
        Mapping from parameter name to its inverse-Fisher diagonal tensor.
    """
    fisher_sum: dict[str, Tensor] = {}

    for path in paths:
        with safe_open(path, framework="pt", device=device) as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                if key in fisher_sum:
                    fisher_sum[key] += tensor
                else:
                    fisher_sum[key] = tensor.clone()

    return {name: 1.0 / (val + eps) for name, val in fisher_sum.items()}


class SOnManifold(Manifold):
    """
    SO(n) with bi-invariant metric g_Q(A, B) = (1/2) * tr(A^T @ B).

    Points are (..., n, n) orthogonal matrices; tangent vectors and Lie algebra
    elements are (..., n, n) skew-symmetric matrices (Omega in so(n)).

    Two backends are provided for log/exp:
      - Cayley (default): rational approximation, cheap but only first-order exact.
      - Exact (exact_log / exact_exp): true matrix exp/log via torch.matrix_exp
        and eigendecomposition; preferred when accuracy matters.
    """

    def __init__(self):
        super().__init__()

    def cayley_exp(self, A: Tensor) -> Tensor:
        """
        Cayley map so(n) -> SO(n): Cay(A) = (I - A/2)^{-1} (I + A/2).

        Args:
            A: (..., n, n) skew-symmetric matrix.

        Returns:
            (..., n, n) orthogonal matrix in SO(n).
        """
        n = A.shape[-1]
        Id = torch.eye(n, dtype=A.dtype, device=A.device)
        if A.ndim > 2:
            Id = Id.expand(*A.shape[:-2], n, n)

        A = 0.5 * (A - A.transpose(-1, -2))  # enforce skew-symmetry
        return torch.linalg.solve(Id - 0.5 * A, Id + 0.5 * A)

    def cayley_inverse_log(self, A: Tensor) -> Tensor:
        """
        Inverse Cayley map SO(n) -> so(n): Cay^{-1}(A) = 2(A - I)(A + I)^{-1}.

        Args:
            A: (..., n, n) orthogonal matrix in SO(n).

        Returns:
            (..., n, n) skew-symmetric matrix in so(n).
        """
        n = A.shape[-1]
        Id = torch.eye(n, dtype=A.dtype, device=A.device)
        if A.ndim > 2:
            Id = Id.expand(*A.shape[:-2], n, n)

        omega = 2.0 * ((A - Id) @ torch.linalg.inv(A + Id))
        return 0.5 * (omega - omega.transpose(-1, -2))

    def exact_exp(self, base: Tensor, tangent: Tensor) -> Tensor:
        """
        Exact Exp_base(tangent) using torch.matrix_exp.

        More accurate than the Cayley-based exp for large Omega. Computes the
        body-frame Omega = base^T @ tangent and applies the true matrix exponential.

        Args:
            base: (..., n, n) base point in SO(n).
            tangent: (..., n, n) tangent vector at base.

        Returns:
            (..., n, n) point on SO(n).
        """
        omega = base.transpose(-1, -2) @ tangent
        return base @ torch.matrix_exp(omega)

    def exact_log(self, base: Tensor, point: Tensor) -> Tensor:
        """
        Exact Log_base(point) via eigendecomposition.

        More accurate than the Cayley-based log for large rotations. Computes
        log(base^T @ point) using torch.linalg.eig and complex torch.log.

        Args:
            base: (..., n, n) base point in SO(n).
            point: (..., n, n) point in SO(n).

        Returns:
            (..., n, n) tangent vector at base pointing toward point.
        """
        relative = base.transpose(-1, -2) @ point
        vals, vecs = torch.linalg.eig(relative)
        omega = (vecs @ torch.diag_embed(torch.log(vals)) @ torch.linalg.inv(vecs)).real
        omega = 0.5 * (omega - omega.transpose(-1, -2))  # enforce skew-symmetry numerically
        return base @ omega

    def log(self, base: Tensor, point: Tensor) -> Tensor:
        """
        Log_base(point) = base @ Cay^{-1}(base^T @ point).

        Args:
            base: (..., n, n) base point in SO(n).
            point: (..., n, n) point in SO(n).

        Returns:
            (..., n, n) tangent vector at base.
        """
        relative = base.transpose(-1, -2) @ point
        omega = self.cayley_inverse_log(relative)
        return base @ omega

    def exp(self, base: Tensor, tangent: Tensor) -> Tensor:
        """
        Exp_base(tangent) = base @ Cay(base^T @ tangent).

        Args:
            base: (..., n, n) base point in SO(n).
            tangent: (..., n, n) tangent vector at base.

        Returns:
            (..., n, n) point on SO(n).
        """
        omega = base.transpose(-1, -2) @ tangent
        return base @ self.cayley_exp(omega)

    def compute_Pt(self, skew_matrix: torch.Tensor, block_size: int) -> torch.Tensor:
        """
        Parallel transport matrix P_{theta_t -> theta_LLM} in the upper-triangle basis.

        Computes R = exp(Omega/2) = theta_t^{1/2} and assembles the (d x d) transport
        matrix from 2x2 minors of R, where d = n*(n-1)/2 is the manifold dimension.

        Args:
            skew_matrix: (num_blocks, n, n) skew-symmetric Omega matrices.
            block_size: n, the orthogonal-block size.

        Returns:
            Pt: (num_blocks, d, d) parallel transport matrices.
        """
        idx = torch.triu_indices(block_size, block_size, offset=1, device=skew_matrix.device)

        # R = theta_t^{1/2} = exp(skew_matrix/2)
        R = torch.matrix_exp(skew_matrix / 2)  # (num_blocks, n, n)

        i, j = idx[0], idx[1]  # both index sets are the same upper-triangle pairs

        Rik = R[:, i][:, :, i]  # (num_blocks, d, d)
        Rjl = R[:, j][:, :, j]
        Ril = R[:, i][:, :, j]
        Rjk = R[:, j][:, :, i]

        # Pt[b, k, a] = M_{ka} = matrix of P_{theta_t -> theta_LLM} in upper-triangle basis
        Pt = Rik * Rjl - Ril * Rjk  # (num_blocks, d, d)
        return Pt

    def dist_sq(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Squared geodesic distance d^2(x, y) = -(1/2) * tr(Omega^2).

        Uses the identity ||Omega||_F^2 = -tr(Omega^2) for skew-symmetric Omega,
        where Omega = Cay^{-1}(x^T @ y).

        Args:
            x: (..., n, n) point in SO(n).
            y: (..., n, n) point in SO(n).

        Returns:
            (...,) scalar squared geodesic distance for each batch element.
        """
        omega = self.cayley_inverse_log(x.transpose(-1, -2) @ y)
        return -0.5 * torch.diagonal(omega @ omega, dim1=-2, dim2=-1).sum(-1)
