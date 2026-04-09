"""
Riemannian geometry primitives for model merging.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import torch
from torch import Tensor
from safetensors import safe_open


class Manifold(ABC):
    """Abstract Riemannian manifold interface."""

    @abstractmethod
    def log(self, base: Tensor, point: Tensor) -> Tensor:
        """Riemannian logarithmic map: Log_base(point) -> tangent vector at base."""

    @abstractmethod
    def exp(self, base: Tensor, tangent: Tensor) -> Tensor:
        """Riemannian exponential map: Exp_base(tangent) -> point on manifold."""

    @abstractmethod
    def parallel_transport(self, base: Tensor, tangent_vec: Tensor, target: Tensor) -> Tensor:
        """Parallel transport tangent_vec from T_base(M) to T_target(M) along the geodesic."""

    @abstractmethod
    def transported_hessian(self, hessian_fn, base: Tensor, target_base: Tensor, omega_t: Tensor) -> callable:
        """
        Return transported Hessian operator on T_target_base(M).
        hessian_fn: V -> Hess(loss at base)[V], acts on tangent vectors at base.
        Returns callable V -> H_tilde_t(V) as in Eq. 4.19 / SO(n) Eq. 10.
        """

    @abstractmethod
    def dist_sq(self, x: Tensor, y: Tensor) -> Tensor:
        """Squared geodesic distance d^2(x, y)."""


def invert_diagonal_fisher_sum(
    paths: list[str],
    eps: float = 1e-6,
    device: str = "cpu",
) -> dict[str, Tensor]:
    """
    Compute the element-wise inverse of the sum of diagonal Fisher matrices.

    For diagonal F = diag(f_1, ..., f_n), the inverse is diag(1/f_1, ..., 1/f_n).
    Given multiple diagonal Fishers F_1, ..., F_T (one per task), this returns
    (F_1 + ... + F_T)^{-1}, i.e. element-wise 1 / (sum of diagonals).

    Args:
        paths: list of safetensors paths, one per task Fisher.
        eps: small constant added before inversion for numerical stability.
        device: torch device for the returned tensors.

    Returns:
        dict mapping parameter name -> 1-D (or same-shape) inverse-Fisher tensor.
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

    All inputs are either:
      - Q-matrices:  (..., n, n) orthogonal
      - Omega (Lie algebra so(n)):  (..., n, n) skew-symmetric

    For OFT adapters the "base" theta_LLM is implicitly I (identity), so
    Omega_t = log(theta_LLM^T @ theta_t) = log(theta_t) directly.
    """

    def __init__(self, use_scipy_logm: bool = False):
        self.use_scipy_logm = use_scipy_logm

    def _matrix_exp(self, A: Tensor) -> Tensor:
        """
        Cayley map on so(n) -> SO(n):
            Cay(A) = (I - A/2)^{-1} (I + A/2)
        """
        n = A.shape[-1]
        Id = torch.eye(n, dtype=A.dtype, device=A.device)
        if A.ndim > 2:
            Id = Id.expand(*A.shape[:-2], n, n)

        A = 0.5 * (A - A.transpose(-1, -2))  # enforce skew-symmetry
        return torch.linalg.solve(Id - 0.5 * A, Id + 0.5 * A)

    def _matrix_log(self, A: Tensor) -> Tensor:
        """
        Cayley inverse on SO(n)
        """
        n = A.shape[-1]
        Id = torch.eye(n, dtype=A.dtype, device=A.device)
        if A.ndim > 2:
            Id = Id.expand(*A.shape[:-2], n, n)

        omega = 2.0 * ((A - Id) @ torch.linalg.inv(A + Id))
        return 0.5 * (omega - omega.transpose(-1, -2))

    def log(self, base: Tensor, point: Tensor) -> Tensor:
        """
        Log_base(point) = base @ log(base^T @ point)  [Eq. 3]
        Returns tangent vector at base (element of T_base SO(n)).
        """
        relative = base.transpose(-1, -2) @ point   # base^T @ point, in SO(n)
        omega = self._matrix_log(relative)           # in so(n)
        return base @ omega                          # tangent vector at base

    def exp(self, base: Tensor, tangent: Tensor) -> Tensor:
        """
        Exp_base(base @ Omega) = base @ exp(Omega)  [Eq. 2]
        tangent is a tangent vector at base, i.e. tangent = base @ Omega.
        """
        omega = base.transpose(-1, -2) @ tangent    # body-frame Omega
        return base @ self._matrix_exp(omega)

    def task_vector(self, theta_llm: Tensor, theta_t: Tensor) -> Tensor:
        """
        Omega_t = log(theta_LLM^T @ theta_t) in so(n)  [Eq. 4]
        Returns the Lie-algebra task vector in the body frame.
        """
        return self._matrix_log(theta_llm.transpose(-1, -2) @ theta_t)

    def parallel_transport(self, base: Tensor, tangent_vec: Tensor, target: Tensor) -> Tensor:
        """
        Transport tangent_vec (Lie-algebra component V in so(n)) from theta_t back to theta_LLM.
        Formula: P_{theta_t -> theta_LLM}(theta_t @ V) = exp(Omega_t/2) @ V @ exp(-Omega_t/2)
        where Omega_t = log(theta_LLM^T @ theta_t)  [Eq. 6]

        Args:
            base: theta_t
            tangent_vec: V in so(n) at theta_t
            target: theta_LLM
        """
        omega_t = self.task_vector(target, base)
        half_exp     = self._matrix_exp( omega_t / 2)
        half_exp_inv = self._matrix_exp(-omega_t / 2)
        return half_exp @ tangent_vec @ half_exp_inv

    def transported_hessian(self, hessian_fn, base: Tensor, target_base: Tensor, omega_t: Tensor):
        """
        Transported Hessian H_tilde_t as an endomorphism of so(n)  [Eq. 10]:
          H_tilde_t(V) = exp(Omega_t/2) @ Hess(theta_t)[exp(-Omega_t/2) @ V @ exp(Omega_t/2)] @ exp(-Omega_t/2)

        Args:
            hessian_fn: callable V -> Hess(loss at theta_t)[V]
            omega_t: Lie-algebra task vector log(theta_LLM^T @ theta_t) in so(n)
        Returns:
            callable V -> H_tilde_t(V)
        """
        half_exp     = self._matrix_exp( omega_t / 2)
        half_exp_inv = self._matrix_exp(-omega_t / 2)

        def transported(V: Tensor) -> Tensor:
            # Pull V back: exp(-Omega/2) @ V @ exp(Omega/2)
            V_back = half_exp_inv @ V @ half_exp
            HV = hessian_fn(V_back)
            # Push forward: exp(Omega/2) @ HV @ exp(-Omega/2)
            return half_exp @ HV @ half_exp_inv

        return transported

    def dist_sq(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Squared geodesic distance d^2(Q, R) = -(1/2) * tr(log(Q^T @ R)^2)  [Eq. 14]
        Uses skew-symmetry: ||Omega||_F^2 = -tr(Omega^2).
        """
        omega = self._matrix_log(x.transpose(-1, -2) @ y)
        return -0.5 * torch.diagonal(omega @ omega, dim1=-2, dim2=-1).sum(-1)
