from __future__ import annotations

import os
from abc import ABC, abstractmethod
from itertools import cycle
from typing import Dict, List, Literal, Optional

import torch
import wandb
from safetensors.torch import load_file
from torch import Tensor
from torch.func import functional_call
import matplotlib.pyplot as plt
from tqdm import tqdm
try:
    import geoopt
except ModuleNotFoundError:
    geoopt = None
import re
from collections import defaultdict

from src.geometry import Manifold, SOnManifold


MergeMode = Literal[
    "standard",
    "standard_rescaled",
    "diagonal_fisher",
    "diagonal_fisher_avg",
    "diagonal_fisher_rescaled",
    "diagonal_fisher_max_rescaled",
    "diagonal_fisher_std_rescaled",
    "diagonal_fisher_kl_rescaled",
    "fisher",
]

class RiemannianMerging(ABC):
    """
    Abstract Riemannian merging interface.

    Subclasses implement three concerns separately:
      1. load_adapters  - I/O: open files and return raw weight dicts.
      2. to_task_vectors - geometry: convert weights to Lie-algebra task vectors.
      3. merge_formula  - math: combine task vectors (with optional curvature).

    The public entry-point `merge` wires them together.
    """

    def __init__(
        self,
        manifold: Manifold,
        lam: float = 1.0,
        alphas: Optional[List[float]] = None,
    ):
        self.manifold = manifold
        self.lam = lam
        self.alphas = alphas  # per-task weights; ones if None

    def _resolve_alphas(self, T: int) -> Tensor:
        if self.alphas is not None:
            return torch.tensor(self.alphas, dtype=torch.float32)
        return torch.ones(T, dtype=torch.float32)

    @abstractmethod
    def load_weights(self, adapter_paths: List[str]) -> List[Dict[str, Tensor]]:
        """Load adapter weights from disk. Returns one state-dict per path."""

    @abstractmethod
    def load_fishers(self, paths: List[str]) -> List[Dict[str, Tensor]]:
        """Load diagonal Fisher dicts from disk. Returns one dict per path."""

    @abstractmethod
    def merge_formula(
        self,
        weights_list: List[Tensor],
        fisher_list: Optional[List[Dict[str, Tensor]]] = None,
        mode: MergeMode = "standard",
        alphas: Optional[Tensor] = None,
    ) -> Tensor:
        """Merge task vectors into one (B, n, n) Omega.

        Modes: "standard" or "diagonal_fisher"
        Fisher data is required for Fisher-based modes.
        """

    @abstractmethod
    def merge(
        self,
        adapter_paths: List[str],
        fisher_paths: Optional[List[str]] = None,
        mode: MergeMode = "standard",
    ) -> Dict[str, Tensor]:
        """
        Full merging pipeline
        """


class OFTMerging(RiemannianMerging):
    """
    Riemannian merging for OFT adapters on SO(n).

    Three merge modes
    -----------------
    standard           Weighted average of Lie-algebra task vectors. No Fisher.
    diagonal_fisher Element-wise Fisher weighting in the vectorised so(n) basis.
                    Each component j of the merged vector is
                        Omega*_j = (sum_t alpha_t f_(t,j) Omega_(t,j)) / (lambda + sum_t alpha_t f_(t,j))
    """

    def __init__(
        self,
        lam: float = 0.0,
        alphas: Optional[List[float]] = None,
        device: str = "cpu",
    ):
        super().__init__(manifold=SOnManifold(), lam=lam, alphas=alphas)
        self.manifold: SOnManifold
        self.device = device

    def oft_params_to_skew_matrix(
            self, oft_params: torch.Tensor, son_dimension: int,
        ) -> torch.Tensor:
        """
        Convert OFT parameters (num_blocks, num_params_per_block)
        to skew-symmetric matrices (num_blocks, block_size, block_size).

        Taken from OrthoMerge_OFT_models.py.
        """
        num_blocks, num_params = oft_params.shape
        block_size = int((1 + (1 + 8 * son_dimension) ** 0.5) / 2)
        if num_params != son_dimension:
            raise ValueError(f"num_params_per_block={num_params}  block_size={block_size} ")

        indices = torch.triu_indices(block_size, block_size, offset=1, device=oft_params.device)
        rows, cols = indices[0], indices[1]
        S = torch.zeros(
            num_blocks, block_size, block_size, dtype=oft_params.dtype, device=oft_params.device
        )

        S[:, rows, cols] = oft_params
        S = S - S.transpose(-2, -1)

        return S


    def skew_matrix_to_oft_params(self, S: torch.Tensor) -> torch.Tensor:
        """
        Convert skew-symmetric matrices (num_blocks, block_size, block_size)
        to OFT parameters (num_blocks, num_params_per_block).

        Taken from OrthoMerge_OFT_models.py.
        """

        num_blocks, block_size, _ = S.shape

        indices = torch.triu_indices(block_size, block_size, offset=1, device=S.device)
        rows, cols = indices[0], indices[1]
        oft_params = S[:, rows, cols]  # (num_blocks, num_params_per_block)

        return oft_params

    def load_weights(self, adapter_paths: List[str]) -> List[Dict[str, Tensor]]:
        """
        Load OFT adapters

        Code taken from OrthoMerge_OFT_models.py
        """
        all_weights = []
        for adapter_path in adapter_paths:
            model_path = os.path.join(adapter_path, "adapter_model.safetensors")
            if not os.path.exists(model_path):
                model_path = os.path.join(adapter_path, "adapter_model.bin")
                if os.path.exists(model_path):
                    weights = torch.load(model_path, map_location=self.device)
                else:
                    print(f"[WARNING]: No adapter weights found at {adapter_path}, skipping...")
                    continue
            else:
                weights = load_file(model_path, device=self.device)

            all_weights.append(weights)
            print(f"Loaded adapters: {adapter_path}")

        return all_weights

    def load_fishers(
        self,
        paths: List[str],
        remove_default_ettiquete: bool = True,
    ) -> List[Dict[str, Tensor]]:
        all_fishers = []
        for path in paths:
            if not os.path.exists(path):
                print(f"[WARNING]: No Fisher file found at {path}, skipping...")
                continue
            fisher = load_file(path, device=self.device)
            all_fishers.append(fisher)
            print(f"Loaded Fisher: {path}")

        # Remove .default from all keys
        if remove_default_ettiquete:
            for fisher in all_fishers:
                for key in list(fisher.keys()):
                    if ".default" in key:
                        new_key = key.replace(".default", "")
                        fisher[new_key] = fisher.pop(key)
        return all_fishers

    def merge_formula(
        self,
        weights_list: List[Tensor],
        fisher_list: Optional[List[Dict[str, Tensor]]] = None,
        mode: MergeMode = "standard",
        alphas: Optional[Tensor] = None,
    ) -> Tensor:
        """Merge OFT task vectors using the specified mode.

        Args:
            weights_list: T tensors of shape (num_blocks, d), skew-symmetric so(n) parameters per task.
            fisher_list: Per-task Fisher info — diagonal (num_blocks, d) for ``"diagonal_fisher"``,
                full (num_blocks, d, d) for ``"fisher"``. Unused in ``"standard"`` mode.
            mode: ``"standard"`` (alpha-weighted avg), ``"diagonal_fisher"``, or ``"fisher"``
                (both Fisher modes transport to a common tangent space and solve a linear system).
            alphas: Optional (T,) tensor overriding self.alphas. Kept differentiable when provided.

        Returns:
            Merged OFT parameters of shape (num_blocks, d).

        Raises:
            ValueError: If ``fisher_list`` is None for a Fisher mode, or ``mode`` is unsupported.
        """
        T = len(weights_list)
        if alphas is None:
            alphas = self._resolve_alphas(T).to(self.device)

        if mode == "standard":
            return len(weights_list) * self._standard_merging(weights_list, alphas)

        if mode == "standard_rescaled":
            # assumes alpha_t = 1 and then applies orthomerge correction
            merged = self._standard_merging(weights_list, alphas)
            return self._orthomerge_rescale(weights_list, merged, alphas)

        if mode == "diagonal_fisher":
            if fisher_list is None:
                raise ValueError(f"Fisher data is required for mode {mode!r}")
            return self._diagonal_fisher_merging(weights_list, fisher_list, alphas)

        if mode == "diagonal_fisher_avg":
            if fisher_list is None:
                raise ValueError(f"Fisher data is required for mode {mode!r}")
            avg_alphas = torch.full_like(alphas, 1.0 / len(weights_list))
            return self._diagonal_fisher_merging(
                weights_list,
                fisher_list,
                avg_alphas,
            )

        if mode == "diagonal_fisher_rescaled":
            if fisher_list is None:
                raise ValueError(f"Fisher data is required for mode {mode!r}")
            merged = self._diagonal_fisher_merging(weights_list, fisher_list, alphas)
            return self._fisher_norm_rescale(weights_list, fisher_list, merged, alphas)

        if mode == "diagonal_fisher_max_rescaled":
            if fisher_list is None:
                raise ValueError(f"Fisher data is required for mode {mode!r}")
            merged = self._diagonal_fisher_merging(weights_list, fisher_list, alphas)
            return self._fisher_norm_max_rescale(weights_list, fisher_list, merged, alphas)

        if mode == "diagonal_fisher_std_rescaled":
            if fisher_list is None:
                raise ValueError(f"Fisher data is required for mode {mode!r}")
            merged = self._diagonal_fisher_merging(weights_list, fisher_list, alphas)
            return self._standard_norm_rescale(weights_list, merged, alphas)

        if mode == "diagonal_fisher_kl_rescaled":
            if fisher_list is None:
                raise ValueError(f"Fisher data is required for mode {mode!r}")
            return self._diagonal_fisher_merging_kl_rescaled(
                weights_list=weights_list,
                fisher_list=fisher_list,
                alphas=alphas,
            )

        if mode == "fisher":
            if fisher_list is None:
                raise ValueError(f"Fisher data is required for mode {mode!r}")
            return self._fisher_merging(weights_list, fisher_list, alphas)

        raise ValueError(f"Unsupported merge mode: {mode!r}")

    def apply_alphas(self, weights_list: List[Tensor], alphas: Tensor) -> Tensor:
        """Differentiable weighted sum of task vectors; alphas is (T,) or (T, L) broadcast to (T,).

        Used by _AlphaOptimizerAdaMerging to apply a learned alpha vector for a single key.
        Subclasses can override for non-Euclidean interpolation (e.g. Karcher mean).
        """
        stacked = torch.stack(weights_list, dim=0).float()  # (T, ...)
        a = alphas.to(stacked)
        if a.dim() == 0:
            a = a.unsqueeze(0).expand(stacked.shape[0])
        return torch.einsum("t,t...->...", a, stacked)

    def _standard_merging(self, weights_list: List[Tensor], alphas) -> Tensor:
        """Compute a weighted average of OFT task vectors in so(n).

        Args:
            weights_list: T tensors of shape (num_blocks, d), the skew-symmetric parameters per task.
            alphas: T scalars or a (T,) tensor, one mixing coefficient per task.

        Returns:
            Merged parameters of shape (num_blocks, d).
        """
        stacked = torch.stack(weights_list, dim=0)  # (T, num_blocks, n, n)
        if isinstance(alphas, torch.Tensor):
            a = alphas.to(dtype=stacked.dtype, device=stacked.device)
        else:
            a = torch.tensor(alphas, dtype=stacked.dtype, device=stacked.device)
        return torch.einsum("t,t...->...", a, stacked)

    def _orthomerge_rescale(
        self,
        weights_list: List[Tensor],
        merged: Tensor,
        alphas,
    ) -> Tensor:
        """Apply the OrthoMerge Frobenius-norm correction.

        Args:
            weights_list: T tensors of shape (num_blocks, d), OFT parameters per task.
            merged: Merged OFT parameters of shape (num_blocks, d).
            alphas: T scalars or a (T,) tensor, one mixing coefficient per task.

        Returns:
            Rescaled merged parameters of shape (num_blocks, d).
        """
        stacked = torch.stack(weights_list, dim=0).float().to(merged.device)
        if isinstance(alphas, torch.Tensor):
            a = alphas.to(dtype=stacked.dtype, device=stacked.device)
        else:
            a = torch.tensor(alphas, dtype=stacked.dtype, device=stacked.device)

        while a.dim() < stacked.dim():
            a = a.unsqueeze(-1)

        weighted = a * stacked
        sum_of_norms = torch.norm(weighted.flatten(1), p="fro", dim=1).sum()
        norm_of_merged = torch.norm(merged.float(), p="fro")
        correction = sum_of_norms / norm_of_merged.clamp(min=1e-8)
        print("[orthomerge_rescale] " f"sum_of_norms={sum_of_norms.item():.6g}, " f"norm_of_sum={norm_of_merged.item():.6g}, " f"correction={correction.item():.6g}")
        return (correction * merged).to(dtype=weights_list[0].dtype)

    def _fisher_norm_rescale(
            self,
            weights_list: List[Tensor],
            fisher_list: List[Tensor],
            merged: Tensor,
            alphas,
        ) -> Tensor:
            """Apply the OrthoMerge norm correction with diagonal Fisher metrics.

            Args:
                weights_list: T tensors of shape (num_blocks, d), OFT parameters per task.
                fisher_list: T tensors of shape (num_blocks, d), diagonal Fisher estimates per task.
                merged: Merged OFT parameters of shape (num_blocks, d).
                alphas: T scalars or a (T,) tensor, one mixing coefficient per task.

            Returns:
                Rescaled merged parameters of shape (num_blocks, d).
            """
            stacked = torch.stack(weights_list, dim=0).float().to(merged.device)
            fishers = torch.stack(
                [f.float().to(merged.device).clamp(min=0.0) for f in fisher_list],
                dim=0,
            )

            if isinstance(alphas, torch.Tensor):
                a = alphas.to(dtype=stacked.dtype, device=stacked.device)
            else:
                a = torch.tensor(alphas, dtype=stacked.dtype, device=stacked.device)

            while a.dim() < stacked.dim():
                a = a.unsqueeze(-1)

            individual_norms = torch.sqrt(
                (fishers * stacked.pow(2)).flatten(1).sum(dim=1).clamp(min=0.0)
            )

            a_flat = a.reshape(a.shape[0], -1)[:, 0]
            sum_of_norms = (a_flat.abs() * individual_norms).sum()

            fisher_sum = (a.abs() * fishers).sum(dim=0)

            norm_of_merged = torch.sqrt(
                (fisher_sum * merged.float().pow(2)).sum().clamp(min=0.0)
            )

            correction = sum_of_norms / norm_of_merged.clamp(min=1e-8)
            norm_rescaled = correction * norm_of_merged

            print(
                "[fisher_norm_rescale] "
                f"sum_of_norms={sum_of_norms.item():.6g}, "
                f"norm_of_sum={norm_of_merged.item():.6g}, "
                f"norm_rescaled={norm_rescaled.item():.6g}, "
                f"correction={correction.item():.6g}"
            )

            new_merged = (correction * merged).to(dtype=weights_list[0].dtype)
            return new_merged

    def _fisher_norm_max_rescale(
            self,
            weights_list: List[Tensor],
            fisher_list: List[Tensor],
            merged: Tensor,
            alphas,
        ) -> Tensor:
            """Apply Fisher-energy lower-bound correction with diagonal Fisher metrics.

            Args:
                weights_list: T tensors of shape (num_blocks, d), OFT parameters per task.
                fisher_list: T tensors of shape (num_blocks, d), diagonal Fisher estimates per task.
                merged: Merged OFT parameters of shape (num_blocks, d).
                alphas: T scalars or a (T,) tensor, one mixing coefficient per task.

            Returns:
                Rescaled merged parameters of shape (num_blocks, d).
            """
            stacked = torch.stack(weights_list, dim=0).float().to(merged.device)
            fishers = torch.stack(
                [f.float().to(merged.device).clamp(min=0.0) for f in fisher_list],
                dim=0,
            )

            if isinstance(alphas, torch.Tensor):
                a = alphas.to(dtype=stacked.dtype, device=stacked.device)
            else:
                a = torch.tensor(alphas, dtype=stacked.dtype, device=stacked.device)

            while a.dim() < stacked.dim():
                a = a.unsqueeze(-1)

            a_flat = a.reshape(a.shape[0], -1)[:, 0].abs()

            target_energies = (fishers * stacked.pow(2)).flatten(1).sum(dim=1)
            target_energies = a_flat * target_energies

            merged_energies = (fishers * merged.float().pow(2)).flatten(1).sum(dim=1)

            ratios = target_energies / merged_energies.clamp(min=1e-8)
            correction = torch.sqrt(ratios.clamp(min=0.0)).max()

            print(
                "[fisher_norm_max_rescale] "
                f"max_target={target_energies.max().item():.6g}, "
                f"min_merged={merged_energies.min().item():.6g}, "
                f"correction={correction.item():.6g}"
            )

            new_merged = (correction * merged).to(dtype=weights_list[0].dtype)
            return new_merged

    def _standard_norm_rescale(
        self,
        weights_list: List[Tensor],
        merged: Tensor,
        alphas,
    ) -> Tensor:
        stacked = torch.stack(weights_list, dim=0).float().to(merged.device)

        if isinstance(alphas, torch.Tensor):
            a = alphas.to(dtype=stacked.dtype, device=stacked.device)
        else:
            a = torch.tensor(alphas, dtype=stacked.dtype, device=stacked.device)

        while a.dim() < stacked.dim():
            a = a.unsqueeze(-1)

        individual_norms = torch.norm(stacked.flatten(1), p="fro", dim=1)
        a_flat = a.reshape(a.shape[0], -1)[:, 0].abs()
        alpha_bar_t = a_flat / a_flat.sum().clamp(min=1e-8)  # normalized alphas
        target_norm = (alpha_bar_t * individual_norms).sum()
        norm_of_merged = torch.norm(merged.float(), p="fro")
        correction = target_norm / norm_of_merged.clamp(min=1e-8)

        print(
            "[standard_norm_rescale] "
            f"target_norm={target_norm.item():.6g}, "
            f"norm_of_merged={norm_of_merged.item():.6g}, "
            f"correction={correction.item():.6g}"
        )

        return (correction * merged).to(dtype=weights_list[0].dtype)

    def _fisher_diagonal_alphas(self, fisher_list: List[Tensor]) -> Tensor:
        """Compute per-task alphas as alpha_t = sum_s <f_t, f_s> / Z.

        For diagonal Fishers, trace(F_t F_s) = dot(f_t, f_s) (summed over all
        blocks and coordinates). Tasks whose Fisher is most aligned with the
        others receive higher weight. Alphas are L1-normalised to sum to 1.

        Args:
            fisher_list: T tensors of shape (num_blocks, d), diagonal Fisher estimates.

        Returns:
            Normalised alpha tensor of shape (T,).
        """
        flat = [f.flatten().float() for f in fisher_list]        # T x (num_blocks*d,)
        F = torch.stack(flat)                                     # (T, D)
        F = F / F.norm(dim=1, keepdim=True).clamp(min=1e-8)      # normalise by Frobenius norm
        scores = (F @ F.T).sum(dim=1)                            # (T,), alpha_t = sum_s <f_t, f_s>
        alphas = scores / scores.sum().clamp(min=1e-8)
        return 1 - alphas.to(fisher_list[0].device)

    def _diagonal_fisher_merging(
        self,
        weights_list: List[Tensor],
        fisher_list: List[Tensor],
        alphas,
    ) -> Tensor:
        """Merge OFT adapters using diagonal Fisher information as per-parameter weights.

        Since F_tilde = diag(fisher_t), the full matrix system A @ omega = b reduces to
        an element-wise division: omega_i = b_i / (lam + A_i), where
            A_i = sum_t alpha_t * fisher_t_i
            b_i = sum_t alpha_t * (lam + fisher_t_i) * omega_t_i

        Args:
            weights_list: T tensors of shape (num_blocks, d).
            fisher_list: T tensors of shape (num_blocks, d), diagonal Fisher estimates.
            alphas: T scalars or (T,) tensor.

        Returns:
            Merged parameters of shape (num_blocks, d).
        """
        ref = weights_list[0]
        A = torch.zeros_like(ref, dtype=torch.float32)  # sum_t alpha_t * f_t
        b = torch.zeros_like(ref, dtype=torch.float32)  # sum_t alpha_t * (lam + f_t) * omega_t

        for alpha_t, omega_t, f_t in zip(alphas, weights_list, fisher_list):
            ft = f_t.float().to(ref.device)
            ft = ft / torch.norm(ft, p="fro").clamp(min=1e-8)
            A = A + alpha_t * ft
            b = b + alpha_t * (self.lam + ft) * omega_t.float().to(ref.device)

        return (b / (self.lam + A).clamp(min=1e-8)).to(ref.dtype)

    def _diagonal_fisher_merging_kl_rescaled(
        self,
        weights_list: List[Tensor],
        fisher_list: List[Tensor],
        alphas,
        base_fisher_list: Optional[List[Tensor]] = None,
        eps: float = 1e-8,
        tol: float = 1e-6,
        max_outer: int = 20,
        max_newton: int = 50,
        damping: float = 1e-4,
    ) -> Tensor:
        """
        Diagonal Fisher OFT merge with task-wise KL/Fisher lower-bound constraints.

        Solves, on the positive-definite KKT branch,

            min_w  1/2 w^T diag(D) w - b^T w

        subject to, for each dataset/task k,

            1/2 w^T diag(H_k) w >= 1/2 xi_k^T diag(H_k) xi_k.

        where

            D = lambda + sum_t alpha_t f_t
            b = sum_t alpha_t (lambda + f_t) * xi_t

        and H_k is the pretrained/base Fisher on dataset k. If base_fisher_list is
        not provided, fisher_list is used as a surrogate.

        Args:
            weights_list: T tensors of shape (num_blocks, d), OFT coordinates xi_t.
            fisher_list: T tensors of shape (num_blocks, d), diagonal transported Fishers f_t.
            alphas: T coefficients.
            base_fisher_list: optional T tensors of shape (num_blocks, d), base Fishers H_k.
            eps: numerical floor.
            tol: feasibility/KKT tolerance.
            max_outer: active-set iterations.
            max_newton: Newton iterations per active set.
            damping: diagonal damping added to the small Newton system.

        Returns:
            Merged OFT parameters of shape (num_blocks, d).
        """
        ref = weights_list[0]
        device = ref.device

        if isinstance(alphas, torch.Tensor):
            alphas_t = alphas.to(device=device, dtype=torch.float32)
        else:
            alphas_t = torch.tensor(alphas, device=device, dtype=torch.float32)

        weights = [
            w.to(device=device, dtype=torch.float32)
            for w in weights_list
        ]

        fishers = [
            f.to(device=device, dtype=torch.float32).clamp(min=0.0)
            for f in fisher_list
        ]

        if base_fisher_list is None:
            # Surrogate: use the task Fisher as the dataset-wise KL metric.
            H_list = fishers
        else:
            # Preferred: H_k = Fisher of pretrained/base model on dataset k.
            H_list = [
                h.to(device=device, dtype=torch.float32).clamp(min=0.0)
                for h in base_fisher_list
            ]

        T = len(weights)

        # Build the diagonal linear system:
        #   D * omega = b
        # where D = lambda + sum_t alpha_t f_t.
        A_diag = torch.zeros_like(ref, dtype=torch.float32, device=device)
        b = torch.zeros_like(ref, dtype=torch.float32, device=device)

        for alpha_t, omega_t, f_t in zip(alphas_t, weights, fishers):
            A_diag = A_diag + alpha_t * f_t
            b = b + alpha_t * (self.lam + f_t) * omega_t

        D = (self.lam + A_diag).clamp(min=eps)

        # Unconstrained diagonal Fisher merge.
        omega_hat = b / D

        H_stack = torch.stack(H_list, dim=0)  # (T, num_blocks, d)

        # Dataset-wise thresholds:
        #   delta_k = 1/2 xi_k^T H_k xi_k.
        deltas = torch.stack([
            0.5 * (H_k * omega_k.pow(2)).sum()
            for H_k, omega_k in zip(H_list, weights)
        ])

        # Check whether the unconstrained merge already satisfies every constraint.
        infos_hat = 0.5 * (
            H_stack * omega_hat.pow(2).unsqueeze(0)
        ).flatten(1).sum(dim=1)

        if torch.all(infos_hat >= deltas - tol):
            return omega_hat.to(dtype=ref.dtype)

        # Active-set initialization: constraints violated by the unconstrained merge.
        active = infos_hat < deltas - tol
        mu_full = torch.zeros(T, device=device, dtype=torch.float32)

        omega = omega_hat

        for _ in range(max_outer):
            active_idx = torch.nonzero(active, as_tuple=False).flatten()

            if active_idx.numel() == 0:
                omega = b / D
                break

            mu = mu_full[active_idx].clone()
            H_A = H_stack[active_idx]  # (m, num_blocks, d)

            # Newton solve for active multipliers.
            for _ in range(max_newton):
                denom = D - torch.einsum("m,m...->...", mu, H_A)

                # Stay on the positive-definite branch.
                if denom.min() <= eps:
                    mu = 0.5 * mu
                    continue

                omega = b / denom

                infos_A = 0.5 * (
                    H_A * omega.pow(2).unsqueeze(0)
                ).flatten(1).sum(dim=1)

                r = infos_A - deltas[active_idx]

                if r.abs().max() < tol:
                    break

                m = active_idx.numel()
                J = torch.empty((m, m), device=device, dtype=torch.float32)

                # Jacobian:
                #   J[k,j] = sum_i h_k_i h_j_i b_i^2 / denom_i^3.
                denom3 = denom.pow(3).clamp(min=eps)
                b2 = b.pow(2)

                for a in range(m):
                    for c in range(m):
                        J[a, c] = (H_A[a] * H_A[c] * b2 / denom3).sum()

                J = J + damping * torch.eye(m, device=device, dtype=torch.float32)

                try:
                    step = torch.linalg.solve(J, -r)
                except RuntimeError:
                    step = torch.linalg.lstsq(J, -r.unsqueeze(-1)).solution.squeeze(-1)

                # Backtracking to preserve mu >= 0 and denom > 0.
                step_scale = 1.0
                accepted = False

                for _bt in range(30):
                    mu_new = (mu + step_scale * step).clamp(min=0.0)
                    denom_new = D - torch.einsum("m,m...->...", mu_new, H_A)

                    if denom_new.min() > eps:
                        omega_new = b / denom_new

                        infos_new = 0.5 * (
                            H_A * omega_new.pow(2).unsqueeze(0)
                        ).flatten(1).sum(dim=1)

                        r_new = infos_new - deltas[active_idx]

                        if r_new.abs().sum() <= r.abs().sum() or step_scale < 1e-4:
                            mu = mu_new
                            accepted = True
                            break

                    step_scale *= 0.5

                if not accepted:
                    break

            # Write active multipliers back into the full vector.
            mu_full[active_idx] = mu

            denom = D - torch.einsum("t,t...->...", mu_full, H_stack)

            if denom.min() <= eps:
                denom = denom.clamp(min=eps)

            omega = b / denom

            all_infos = 0.5 * (
                H_stack * omega.pow(2).unsqueeze(0)
            ).flatten(1).sum(dim=1)

            slack = all_infos - deltas

            feasible = torch.all(slack >= -tol)
            comp_ok = torch.all((mu_full <= tol) | (slack.abs() <= tol))

            if feasible and comp_ok:
                return omega.to(dtype=ref.dtype)

            # Add newly violated constraints.
            active = active | (slack < -tol)

            # Remove constraints with nearly zero multiplier and positive slack.
            active = active & ~((mu_full <= tol) & (slack > tol))

        # Final feasibility check.
        final_infos = 0.5 * (
            H_stack * omega.pow(2).unsqueeze(0)
        ).flatten(1).sum(dim=1)

        if torch.any(final_infos < deltas - tol):
            print("[WARNING] Multi-KL Lagrange solver returned an infeasible point.")

        return omega.to(dtype=ref.dtype)

    def _fisher_merging(
        self,
        weights_list: List[Tensor],
        fisher_list: List[Tensor],
        alphas: List[float],
    ) -> Tensor:
        """Merge OFT adapters using full Fisher information matrices as per-parameter weights.

        Args:
            weights_list: T tensors of shape (num_blocks, d), skew-symmetric parameters per task.
            fisher_list: T tensors of shape (num_blocks, d, d), full Fisher matrices per task.
            alphas: T scalars, one mixing coefficient per task.

        Returns:
            Merged parameters of shape (num_blocks, d).
        """
        raise NotImplementedError("Full Fisher merging is not implemented yet.")
        num_blocks, son_dimension = weights_list[0].shape
        dtype, device = weights_list[0].dtype, weights_list[0].device
        block_size = int((1 + (1 + 8 * son_dimension) ** 0.5) / 2)

        Id = torch.eye(son_dimension, dtype=dtype, device=device).unsqueeze(0)

        A = self.lam * Id.expand(num_blocks, -1, -1).clone()
        b = torch.zeros(num_blocks, son_dimension, dtype=dtype, device=device)

        for alpha_t, oft_params_t, fisher_t in zip(alphas, weights_list, fisher_list):
            # Reconstruct skew-symmetric matrices
            skew_matrix = self.oft_params_to_skew_matrix(oft_params_t, son_dimension)  # (num_blocks, n, n)
            Pt = self.manifold.compute_Pt(skew_matrix, block_size)  # (num_blocks, d, d)

            F_tilde = Pt @ fisher_t @ Pt.transpose(-2, -1)

            A += alpha_t * F_tilde
            rhs = (self.lam * Id.squeeze(0) + F_tilde) @ oft_params_t.unsqueeze(-1)
            b += alpha_t * rhs.squeeze(-1)

        return torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)

    def fisher_full_task_vectors(
        self,
        weights_list: List[Tensor],
        fisher_list: List[Tensor],
        alphas: List[float],
    ) -> List[Tensor]:
        """Return the Fisher-corrected task vectors

            v_t = (lambda I_d + sum_j alpha_j F_tilde_j)^(-1) (lambda I_d + F_tilde_t) xi_t

        for each task t.

        Args:
            weights_list: T tensors of shape (num_blocks, d), skew-symmetric parameters per task.
            fisher_list: T tensors of shape (num_blocks, d), diagonal Fisher estimates per task.
            alphas: T scalars, one mixing coefficient per task.

        Returns:
            List of T tensors, each of shape (num_blocks, d).
        """
        num_blocks, son_dimension = weights_list[0].shape
        dtype, device = weights_list[0].dtype, weights_list[0].device
        block_size = int((1 + (1 + 8 * son_dimension) ** 0.5) / 2)

        Id = torch.eye(son_dimension, dtype=dtype, device=device).unsqueeze(0)  # (1, d, d)

        # First build all transported Fishers
        F_tilde_list = []
        for oft_params_t, fisher_t in zip(weights_list, fisher_list):
            skew_matrix = self.oft_params_to_skew_matrix(oft_params_t, son_dimension)  # (num_blocks, n, n)
            # Pt = self.manifold.compute_Pt(skew_matrix=skew_matrix, block_size=block_size)  # (num_blocks, d, d)
            # F_tilde = Pt @ torch.diag_embed(fisher_t) @ Pt.transpose(-2, -1)  # (num_blocks, d, d)
            F_tilde = torch.diag_embed(fisher_t)  # (num_blocks, d, d), no transport for simplicity
            # F_tilde = torch.diag_embed(fisher_t)  # (num_blocks, d, d), no transport for simplicity
            F_tilde_list.append(F_tilde)

        # Shared inverse term: (lambda I_d + sum_t alpha_t F_tilde_t)^(-1)
        A = self.lam * Id.expand(num_blocks, -1, -1).clone()  # (num_blocks, d, d)
        for alpha_t, F_tilde in zip(alphas, F_tilde_list):
            A += alpha_t * F_tilde

        fisher_task_vectors = []
        for oft_params_t, F_tilde_t in zip(weights_list, F_tilde_list):
            rhs = ((self.lam * Id.squeeze(0) + F_tilde_t) @ oft_params_t.unsqueeze(-1)).squeeze(-1)
            v_t = torch.linalg.solve(A, rhs.unsqueeze(-1)).squeeze(-1)
            fisher_task_vectors.append(v_t)

        return fisher_task_vectors

    def fisher_task_vectors(
        self,
        weights_list: List[Tensor],
        fisher_list: List[Tensor],
        alphas: List[float] = None,
    ) -> List[Tensor]:
        """Return the Fisher-weighted task vectors

            v_t = (lambda I_d + F_tilde_t) xi_t

        for each task t.

        Args:
            weights_list: T tensors of shape (num_blocks, d), skew-symmetric parameters per task.
            fisher_list: T tensors of shape (num_blocks, d), diagonal Fisher estimates per task.

        Returns:
            List of T tensors, each of shape (num_blocks, d).
        """
        num_blocks, son_dimension = weights_list[0].shape
        dtype, device = weights_list[0].dtype, weights_list[0].device
        block_size = int((1 + (1 + 8 * son_dimension) ** 0.5) / 2)

        Id = torch.eye(son_dimension, dtype=dtype, device=device).unsqueeze(0)  # (1, d, d)

        fisher_task_vectors = []
        for oft_params_t, fisher_t in zip(weights_list, fisher_list):
            skew_matrix = self.oft_params_to_skew_matrix(oft_params_t, son_dimension)  # (num_blocks, n, n)
            Pt = self.manifold.compute_Pt(skew_matrix=skew_matrix, block_size=block_size)  # (num_blocks, d, d)

            F_tilde_t = Pt @ torch.diag_embed(fisher_t) @ Pt.transpose(-2, -1)  # (num_blocks, d, d)

            v_t = ((self.lam * Id.squeeze(0) + F_tilde_t) @ oft_params_t.unsqueeze(-1)).squeeze(-1)
            fisher_task_vectors.append(v_t)

        return fisher_task_vectors

    def merge(
        self,
        adapter_paths: List[str],
        fisher_paths: Optional[List[str]] = None,
        mode: MergeMode = "standard",
        optimize_alphas: Optional[str] = None,
        model=None,
        task_loaders: Optional[List] = None,
        task_names: Optional[List[str]] = None,
    ):
        if optimize_alphas is not None:
            valid_alpha_opts = ("adamerging", "adamergingpp", "adamerging_equal", "adamergingpp_equal")
            if optimize_alphas not in valid_alpha_opts:
                raise ValueError(
                    f"optimize_alphas must be one of {valid_alpha_opts},"
                    f" got {optimize_alphas!r}"
                )
            if model is None or task_loaders is None or task_names is None:
                raise ValueError("optimize_alphas requires model, task_loaders, and task_names")
            opt = _AlphaOptimizerAdaMerging(
                merger=self,
                equal_alphas=optimize_alphas.endswith("_equal"),
                device=self.device,
            )
            opt.use_pp = optimize_alphas.startswith("adamergingpp")
            return opt.optimize(adapter_paths, model, task_loaders, task_names, fisher_paths, mode)

        print(f"\nMerging {len(adapter_paths)} OFT adapters with {mode} mode...")

        # Load weights
        all_weights = self.load_weights(adapter_paths)
        if not all_weights:
            raise ValueError("No valid adapter weights found!")

        # Load Fishers
        all_fishers = self.load_fishers(fisher_paths) if fisher_paths else None

        merged_weights = {}

        # Merge per key
        for key in all_weights[0].keys():
            if "oft_r" in key or ("oft_" in key.lower() and "classifier" not in key.lower()):
                print(f"  Processing key: {key}")

                weights_layer = [weights[key] for weights in all_weights]

                fishers_layer = None
                if all_fishers:
                    fishers_layer = [fisher[key] for fisher in all_fishers]

                avg_weight = self.merge_formula(
                    weights_list=weights_layer,
                    fisher_list=fishers_layer,
                    mode=mode,
                )

                merged_weights[key] = avg_weight

            else:
                print("[WARNING] Non-OFT weight detected.")

        print(f"  Merged {len(merged_weights)} weight tensors")
        return merged_weights


class OFTKarcherMerging(OFTMerging):
    """
    Karcher (Frechet) mean merging for OFT adapters on SO(n).

    standard:        isotropic Karcher mean via Riemannian gradient descent.
    diagonal_fisher: Fisher-weighted Karcher mean via gradient descent on so(n) coords.
    """

    def __init__(
        self,
        n_steps: int = 200,
        lr: float = 0.1,
        lam: float = 0.0,
        alphas: Optional[List[float]] = None,
        device: str = "cpu",
    ):
        super().__init__(lam=lam, alphas=alphas, device=device)
        self.n_steps = n_steps
        self.lr = lr

    def _matrix_log_skew(self, R: Tensor) -> Tensor:
        """Stable SO(n) log via Cayley inverse: 2(R-I)(R+I)^{-1}.

        Avoids eig (unstable gradients for near-degenerate eigenvalues). Valid
        for all rotations except R ~= -I (rotation angle ~= pi).
        """
        return self.manifold.cayley_inverse_log(R)

    def merge_formula(
        self,
        weights_list: List[Tensor],
        fisher_list: Optional[List[Dict[str, Tensor]]] = None,
        mode: MergeMode = "standard",
        alphas: Optional[Tensor] = None,
    ) -> Tensor:
        if alphas is None:
            alphas = (
                torch.tensor(self.alphas, dtype=torch.float32)
                if self.alphas is not None
                else torch.ones(1, dtype=torch.float32)
            ).to(self.device)

        if mode == "standard":
            result = self._karcher_merging(weights_list, alphas)
        elif mode == "diagonal_fisher":
            if fisher_list is None:
                raise ValueError(f"Fisher data required for mode {mode!r}")
            result = self._fisher_karcher_merging(weights_list, fisher_list, alphas)
        else:
            raise ValueError(f"Unsupported merge mode: {mode!r}")

        return result

    def _riemannian_karcher(
        self,
        weights_list: List[Tensor],
        alphas: Tensor,
        fisher_list: Optional[List[Tensor]] = None,
    ) -> Tensor:
        """Karcher mean on SO(n) via RiemannianAdam (geoopt).

        Standard:  minimises sum_t alpha_t/2 * ||Omega_t||_F^2,  Omega_t = log(R_t^T R_m).
        Fisher:    minimises sum_t alpha_t/2 * <eta_t, f_t * eta_t>,
                   eta_t = coord(Omega_t) in the upper-tri so(n) basis.
                   f_t is the diagonal parameter-space Fisher -- an approximation of the full
                   coordinate Fisher F_t^SO = (E_ij theta_0)^T R_t^T F_t R_t (E_kl theta_0).
        R_m lives on the Stiefel manifold; RiemannianAdam handles gradient projection + retraction.
        """
        if geoopt is None:
            raise ImportError("OFTKarcherMerging requires geoopt to be installed.")

        son_dimension = weights_list[0].shape[1]
        R_list = [
            torch.matrix_exp(self.oft_params_to_skew_matrix(w.float(), son_dimension)).detach()
            for w in weights_list
        ]

        alphas_d = alphas.detach().cpu()
        mean_w = torch.einsum(
            "t,tbd->bd", alphas_d,
            torch.stack([w.float().cpu() for w in weights_list]),
        ).to(self.device)
        R_m = geoopt.ManifoldParameter(
            torch.matrix_exp(self.oft_params_to_skew_matrix(mean_w.detach(), son_dimension)),
            manifold=geoopt.Stiefel(),
        )
        inner_opt = geoopt.optim.RiemannianAdam([R_m], lr=self.lr)

        alphas_d_dev = alphas_d.to(self.device)
        convergence_interval = max(1, self.n_steps // 10)
        for step in range(self.n_steps):
            should_track_convergence = (
                wandb.run is not None
                and hasattr(self, "_karcher_convergence")
                and (
                    step == 0
                    or (step + 1) % convergence_interval == 0
                    or step + 1 == self.n_steps
                )
            )
            previous_R_m = R_m.detach().clone() if should_track_convergence else None
            inner_opt.zero_grad()
            loss: Tensor = torch.zeros(1, device=self.device)
            fisher_iter = iter(fisher_list) if fisher_list is not None else None
            for alpha_t, R_t in zip(alphas_d_dev, R_list):
                omega_t = self._matrix_log_skew(R_t.transpose(-1, -2) @ R_m)
                if fisher_iter is not None:
                    eta = self.skew_matrix_to_oft_params(omega_t)
                    loss = loss + alpha_t * 0.5 * (next(fisher_iter).float() * eta.pow(2)).sum()
                else:
                    loss = loss + alpha_t * 0.5 * omega_t.pow(2).sum()
            loss.backward()
            inner_opt.step()
            if previous_R_m is not None:
                with torch.no_grad():
                    step_omega = self._matrix_log_skew(
                        previous_R_m.transpose(-1, -2) @ R_m.detach()
                    )
                    current_omega = self._matrix_log_skew(R_m.detach())
                    step_delta = step_omega.norm().item()
                    current_norm = current_omega.norm().item()
                    self._karcher_convergence[step + 1]["delta"].append(step_delta)
                    self._karcher_convergence[step + 1]["relative_delta"].append(
                        step_delta / (current_norm + 1e-12)
                    )
                    self._karcher_convergence[step + 1]["loss"].append(loss.item())

        self._last_loss: float = loss.item()
        with torch.no_grad():
            merged = self.skew_matrix_to_oft_params(self._matrix_log_skew(R_m.detach()))
            t_norms = [w.float().norm().item() for w in weights_list]
            # print(
            #     f"    ||omega_m||={merged.float().norm().item():.4f}  "
            #     f"sum||xi_t||={sum(t_norms):.4f}  "
            #     f"mean||xi_t||={sum(t_norms)/len(t_norms):.4f}  "
            #     f"loss={self._last_loss:.6f}"
            # )
        return merged.to(weights_list[0].dtype)

    def _karcher_merging(self, weights_list: List[Tensor], alphas: Tensor) -> Tensor:
        """Isotropic Karcher mean: arg min_R sum_t alpha_t/2 * ||log(R_t^T R)||_F^2."""
        return self._riemannian_karcher(weights_list, alphas)

    def _fisher_karcher_merging(
        self,
        weights_list: List[Tensor],
        fisher_list: List[Tensor],
        alphas: Tensor,
    ) -> Tensor:
        """Fisher-weighted Karcher mean: arg min_R sum_t alpha_t/2 * f_t * eta_t^2."""
        return self._riemannian_karcher(weights_list, alphas, fisher_list)

    def merge(
        self,
        adapter_paths: List[str],
        fisher_paths: Optional[List[str]] = None,
        mode: MergeMode = "standard",
        optimize_alphas: Optional[str] = None,
        model=None,
        task_loaders: Optional[List] = None,
        task_names: Optional[List[str]] = None,
    ) -> Dict[str, Tensor]:
        if optimize_alphas is not None:
            valid_alpha_opts = ("adamerging", "adamergingpp", "adamerging_equal", "adamergingpp_equal")
            if optimize_alphas not in valid_alpha_opts:
                raise ValueError(
                    f"optimize_alphas must be one of {valid_alpha_opts},"
                    f" got {optimize_alphas!r}"
                )
            if model is None or task_loaders is None or task_names is None:
                raise ValueError("optimize_alphas requires model, task_loaders, and task_names")
            opt = _AlphaOptimizerAdaMerging(
                merger=self,
                equal_alphas=optimize_alphas.endswith("_equal"),
                device=self.device,
            )
            opt.use_pp = optimize_alphas.startswith("adamergingpp")
            return opt.optimize(adapter_paths, model, task_loaders, task_names, fisher_paths, mode)

        print(f"\nMerging {len(adapter_paths)} OFT adapters with Karcher mean ({mode})...")
        all_weights = self.load_weights(adapter_paths)
        if not all_weights:
            raise ValueError("No valid adapter weights found!")
        all_fishers = self.load_fishers(fisher_paths) if fisher_paths else None

        merged_weights: Dict[str, Tensor] = {}
        key_losses: Dict[str, float] = {}
        self._karcher_convergence = defaultdict(lambda: defaultdict(list))

        for key in all_weights[0].keys():
            if "oft_r" in key or ("oft_" in key.lower() and "classifier" not in key.lower()):
                print(f"  Processing key: {key}")
                weights_layer = [w[key] for w in all_weights]
                fishers_layer = [f[key] for f in all_fishers] if all_fishers else None
                merged_weights[key] = self.merge_formula(weights_layer, fishers_layer, mode)
                key_losses[key] = self._last_loss
            else:
                print("[WARNING] Non-OFT weight detected.")

        print(f"  Merged {len(merged_weights)} weight tensors")

        if wandb.run is not None and key_losses:
            by_num: dict = defaultdict(list)
            by_type: dict = defaultdict(list)
            for key, v in key_losses.items():
                m = re.search(r'(\d+)\.(.+)$', key)
                by_num[m.group(1) if m else "0"].append(v)
                by_type[m.group(2) if m else key].append(v)
            wandb.log({
                **{f"karcher/layer/{n}": sum(vs) / len(vs) for n, vs in by_num.items()},
                **{f"karcher/type/{t}": sum(vs) / len(vs) for t, vs in by_type.items()},
            })
            for iteration, metrics in sorted(self._karcher_convergence.items()):
                wandb.log({
                    "karcher/convergence/iteration": iteration,
                    "karcher/convergence/mean_delta": (
                        sum(metrics["delta"]) / len(metrics["delta"])
                    ),
                    "karcher/convergence/mean_relative_delta": (
                        sum(metrics["relative_delta"]) / len(metrics["relative_delta"])
                    ),
                    "karcher/convergence/mean_loss": (
                        sum(metrics["loss"]) / len(metrics["loss"])
                    ),
                })

        return merged_weights


class WudiOFTMerging(OFTMerging):
    """
    WUDI merging for OFT on SO(n).

    Optimizes merged Lie-algebra coordinates via Adam to minimise a
    direction-alignment loss across tasks, then returns the result in
    the same oft_params format as the other merging methods.
    """

    def __init__(self, n_steps: int = 200, lr: float = 1e-3, device: str = "cpu"):
        super().__init__(lam=1.0, alphas=None, device=device)
        self.n_steps = n_steps
        self.lr = lr

    def merge_formula(
        self,
        weights_list: List[Tensor],
        fisher_list: Optional[List[Dict[str, Tensor]]] = None,
        mode: MergeMode = "standard",
    ) -> Tensor:
        """Merge OFT task vectors via WUDI optimisation in so(n) coordinate space.

        Args:
            weights_list: T tensors of shape (num_blocks, d), skew-symmetric so(n) parameters per task.
            fisher_list: Unused; kept for interface compatibility.
            mode: Unused; kept for interface compatibility.

        Returns:
            Merged OFT parameters of shape (num_blocks, d).
        """
        task_vectors = None
        if "fisher" in mode:
            task_vectors = self.fisher_full_task_vectors(weights_list, fisher_list, self.alphas)
        elif mode == "standard":
            task_vectors = weights_list
        else:
            raise ValueError(f"Unsupported merge mode for WUDI: {mode!r}")
        return self._wudi_merging(task_vectors)

    def _wudi_merging(self, weights_list: List[Tensor]) -> Tensor:
        """
        Algorithm 1: WUDI-Merging adapted for OFT on SO(n).

        Operates in skew-symmetric (Lie algebra) space.
        The task vectors here are the skew matrices themselves.

        Loss (Eq. 21):
            L = sum_i (1 / ||tau_i||^2_F) * ||(tau_m - tau_i)(tau_i)^T||^2_F

        Args:
            weights_list: T tensors of shape (num_blocks, d),
                        each being OFT upper-tri coordinates (task vectors).
        Returns:
            Merged coordinates of shape (num_blocks, d).
        """
        T = len(weights_list)
        _, son_dimension = weights_list[0].shape

        # Convert task vectors from upper-tri coordinates to skew matrices
        # tau_{i,l} in paper corresponds to skew matrices here
        tau_list = []
        tau_norm_sq_list = []
        for w in weights_list:
            tau = self.oft_params_to_skew_matrix(w.float(), son_dimension).detach()
            tau_list.append(tau)
            # ||tau_{i,l}||^2_F per block
            norm_sq = tau.pow(2).sum(dim=(-2, -1)).clamp(min=1e-8)
            tau_norm_sq_list.append(norm_sq)

        # Precompute tau_i^T for each task (for skew matrices, tau^T = -tau,
        # but we keep it explicit to match the paper exactly)
        tau_T_list = [tau.transpose(-2, -1) for tau in tau_list]

        # Step 1: Initialize tau^0_{m,l} = sum_i tau_{i,l} (Algorithm 1, line 3)
        tau_m_init = torch.stack(tau_list).sum(dim=0)

        # Optimise in skew-matrix space directly
        tau_m = tau_m_init.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([tau_m], lr=self.lr)

        # Step 2: Optimise (Algorithm 1, lines 5-9)
        for step in range(self.n_steps):
            optimizer.zero_grad()

            loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
            for t in range(T):
                # delta_i = tau_m - tau_i (Eq. 2)
                delta = tau_m - tau_list[t]

                # ||delta_i * (tau_i)^T||^2_F (Eq. 21)
                prod = delta @ tau_T_list[t]                    # (num_blocks, n, n)
                frob_sq = prod.pow(2).sum(dim=(-2, -1))         # (num_blocks,)

                # Weighted by 1/||tau_i||^2_F (balanced weighting, Eq. 21)
                loss = loss + (frob_sq / tau_norm_sq_list[t]).sum()

            loss.backward()
            optimizer.step()

        # Convert back from skew matrix to upper-tri OFT coordinates
        merged_params = self.skew_matrix_to_oft_params(tau_m.detach())
        return merged_params.to(weights_list[0].dtype)


class AdaMergingPP:
    """AdaMerging++ for OFT adapters: TIES preprocessing + entropy-based alpha optimization.

    Wraps _AlphaOptimizerAdaMerging(OFTMerging) and applies TIES trimming to task vectors before
    delegating the entropy loop to _AlphaOptimizerAdaMerging.optimize_from_vecs.
    """

    def __init__(
        self,
        n_iters: int = 100,
        lr: float = 1e-2,
        max_batch_size: int = 64,
        ties_trim_ratio: float = 0.2,
        init_lambda: float = 0.3,
        log_wandb: bool = True,
        device: str = "cpu",
    ) -> None:
        self.ties_trim_ratio = ties_trim_ratio
        self.device = device
        self._opt = _AlphaOptimizerAdaMerging(
            merger=OFTMerging(device=device),
            n_iters=n_iters,
            lr=lr,
            max_batch_size=max_batch_size,
            init_alpha=init_lambda,
            log_wandb=log_wandb,
            device=device,
        )

    @staticmethod
    def _ties(vecs: List[Tensor], trim_ratio: float) -> List[Tensor]:
        """TIES: trim, elect sign, disjoint mask."""
        T, shape = len(vecs), vecs[0].shape
        flat = torch.stack(vecs).float().view(T, -1)
        D = flat.shape[1]
        k = max(1, int(trim_ratio * D))
        thresh = flat.abs().kthvalue(D - k + 1, dim=1, keepdim=True).values
        trimmed = flat * (flat.abs() >= thresh)
        elected = torch.where(
            (trimmed * (trimmed > 0)).sum(0) >= (trimmed.abs() * (trimmed < 0)).sum(0),
            torch.ones(D, device=flat.device), -torch.ones(D, device=flat.device),
        )
        mask = (trimmed.sign() == elected.unsqueeze(0)) | (trimmed == 0)
        phi = trimmed * mask
        return [phi[t].view(shape) for t in range(T)]

    def merge(self, adapter_paths: List[str], model, task_loaders: List,
              task_names: List[str]) -> Dict[str, Tensor]:
        T = len(adapter_paths)
        assert len(task_loaders) == T == len(task_names)

        all_w = self._opt.merger.load_weights(adapter_paths)
        oft_keys = sorted(k for k in all_w[0]
                          if "oft_r" in k or ("oft_" in k.lower() and "classifier" not in k.lower()))
        assert oft_keys, "No OFT keys found"

        # TIES preprocessing
        task_vecs = {
            k: self._ties([w[k].float().to(self.device) for w in all_w], self.ties_trim_ratio)
            for k in oft_keys
        }

        params = dict(model.named_parameters())
        key_map = {}
        for k in oft_keys:
            for c in [k, k.replace(".weight", ".default.weight")]:
                if c in params:
                    key_map[k] = c
                    break
        keys = [k for k in oft_keys if k in key_map]
        assert keys, "No keys mapped"

        base = {k: params[key_map[k]].data.clone().float().to(self.device) for k in keys}
        task_vecs = {k: task_vecs[k] for k in keys}

        return self._opt.optimize_from_vecs(
            task_vecs, base, key_map, keys, all_w[0], model, task_loaders, task_names,
            mode="standard", fisher_list=None,
        )


class _AlphaOptimizerAdaMerging:
    """Entropy-based alpha optimizer for any RiemannianMerging subclass.

    Wraps a merger and learns per-task, per-layer mixing coefficients via
    entropy minimization on unlabeled data, exactly as AdaMerging++ does but
    delegating the actual interpolation to `merger.merge_formula`.

    Usage::

        merger = OFTKarcherMerging(lam=1.0, device="cuda")
        optimizer = _AlphaOptimizerAdaMerging(merger, n_iters=500, lr=1e-2, device="cuda")
        merged = optimizer.optimize(adapter_paths, model, task_loaders, task_names)
    """

    def __init__(
        self,
        merger: RiemannianMerging,
        n_iters: int = 100,
        lr: float = 1e-3,
        max_batch_size: int = 64,
        init_alpha: float = 1,  # 1 is equal_alphas
        equal_alphas: bool = False,
        log_wandb: bool = True,
        device: str = "cpu",
    ) -> None:
        self.merger = merger
        self.n_iters = n_iters
        self.lr = lr
        self.max_batch_size = max_batch_size
        self.init_alpha = init_alpha
        self.equal_alphas = equal_alphas
        self.log_wandb = log_wandb
        self.device = device
        self.use_pp = False
        self.ties_trim_ratio = 0.2

    @staticmethod
    def _entropy(logits: Tensor, mask: Tensor) -> Tensor:
        p = logits.softmax(-1)
        h = -(p * p.clamp(min=1e-8).log()).sum(-1)
        return (h * mask.float()).sum() / mask.float().sum().clamp(min=1.0)

    @staticmethod
    def _ties(vecs: List[Tensor], trim_ratio: float) -> List[Tensor]:
        """AdaMerging++ TIES: trim, elect sign, disjoint merge mask."""
        T, shape = len(vecs), vecs[0].shape
        flat = torch.stack(vecs).float().view(T, -1)
        D = flat.shape[1]
        k = max(1, int(trim_ratio * D))
        thresh = flat.abs().kthvalue(D - k + 1, dim=1, keepdim=True).values
        trimmed = flat * (flat.abs() >= thresh)
        elected = torch.where(
            (trimmed * (trimmed > 0)).sum(0) >= (trimmed.abs() * (trimmed < 0)).sum(0),
            torch.ones(D, device=flat.device), -torch.ones(D, device=flat.device),
        )
        mask = (trimmed.sign() == elected.unsqueeze(0)) | (trimmed == 0)
        phi = trimmed * mask
        return [phi[t].view(shape) for t in range(T)]

    def optimize_from_vecs(
        self,
        task_vecs: Dict[str, List[Tensor]],
        base: Dict[str, Tensor],
        key_map: Dict[str, str],
        keys: List[str],
        ref_w: Dict[str, Tensor],
        model,
        task_loaders: List,
        task_names: List[str],
        mode: MergeMode = "standard",
        fisher_list: Optional[Dict[str, List[Tensor]]] = None,
    ) -> Dict[str, Tensor]:
        """Entropy loop given pre-built task vectors and base weights.

        Useful when a caller (e.g. AdaMergingPP) needs to preprocess task vectors
        (e.g. TIES) before optimization.  `ref_w` is the first adapter's state-dict,
        used only to recover the original dtype for the final cast.
        """
        T = len(task_names)
        L = len(keys)

        alpha_shape = (1, 1) if self.equal_alphas else (T, L)
        alphas = torch.full(
            alpha_shape,
            float(self.init_alpha),
            device=self.device,
            dtype=torch.float32,
            requires_grad=True,
        )
        opt = torch.optim.Adam([alphas], lr=self.lr, betas=(0.9, 0.999))

        cyc = [cycle(dl) for dl in task_loaders]
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        for step in tqdm(range(self.n_iters), desc=f"AlphaOptimizer({type(self.merger).__name__})"):
            opt.zero_grad()
            ents: List[Tensor] = []

            clamped_alphas = torch.clamp(alphas, 0.0, 1.0).expand(T, L)

            merged_params = {
                key_map[k]: base[k] + self.merger.merge_formula(
                    task_vecs[k],
                    fisher_list[k] if fisher_list else None,
                    mode,
                    alphas=clamped_alphas[:, i],
                )
                for i, k in enumerate(keys)
            }

            ce_vals: List[float] = []
            for t in range(T):
                batch = next(cyc[t])
                ids = batch["input_ids"][:self.max_batch_size].to(self.device)
                attn = batch["attention_mask"][:self.max_batch_size].to(self.device)
                labels = batch["labels"][:self.max_batch_size].to(self.device)

                out = functional_call(
                    model, merged_params, (),
                    {"input_ids": ids, "attention_mask": attn},
                )
                logits = out.logits.float()
                assert logits.dim() == 3
                ents.append(self._entropy(logits, attn))

                with torch.no_grad():
                    ce_out = functional_call(
                        model, merged_params, (),
                        {"input_ids": ids, "attention_mask": attn, "labels": labels},
                    )
                    ce_vals.append(ce_out.loss.item())

            loss: Tensor = sum(ents)  # type: ignore[assignment]
            loss.backward()
            opt.step()

            if self.log_wandb:
                log = {"entropy/total": loss.item(), "ce/total": sum(ce_vals) / len(ce_vals)}
                for t, name in enumerate(task_names):
                    log[f"entropy/{name}"] = ents[t].item()
                    log[f"ce/{name}"] = ce_vals[t]
                    log[f"alpha_mean/{name}"] = clamped_alphas[t].mean().item()
                wandb.log(log, step=step)  # type: ignore[attr-defined]

            del merged_params, ents, loss, ce_vals
            torch.cuda.empty_cache()

        with torch.no_grad():
            final_alphas = alphas.clamp(0.0, 1.0).expand(T, L)
            merged: Dict[str, Tensor] = {}
            for i, k in enumerate(keys):
                delta = self.merger.merge_formula(
                    task_vecs[k],
                    fisher_list[k] if fisher_list else None,
                    mode,
                    alphas=final_alphas[:, i],
                )
                merged[k] = (base[k] + delta).to(ref_w[k].dtype)

        if self.log_wandb:
            fig, ax = plt.subplots(figsize=(max(6, L // 4), max(3, T)))
            ax.imshow(final_alphas.detach().cpu().numpy(), aspect="auto", cmap="Blues")
            ax.set_yticks(range(T)); ax.set_yticklabels(task_names)
            ax.set_xlabel("Layer"); ax.set_ylabel("Task")
            plt.colorbar(ax.images[0], ax=ax); plt.tight_layout()
            wandb.log({"alpha/heatmap": wandb.Image(fig)})  # type: ignore[attr-defined]
            plt.close(fig)

        return merged

    def optimize(
        self,
        adapter_paths: List[str],
        model,
        task_loaders: List,
        task_names: List[str],
        fisher_paths: Optional[List[str]] = None,
        mode: MergeMode = "standard",
    ) -> Dict[str, Tensor]:
        T = len(adapter_paths)
        assert T == len(task_loaders) == len(task_names)

        all_w = self.merger.load_weights(adapter_paths)
        oft_keys = sorted(k for k in all_w[0]
                          if "oft_r" in k or ("oft_" in k.lower() and "classifier" not in k.lower()))
        assert oft_keys, "No OFT keys found"

        params = dict(model.named_parameters())
        key_map = {}
        for k in oft_keys:
            for c in [k, k.replace(".weight", ".default.weight")]:
                if c in params:
                    key_map[k] = c
                    break
        keys = [k for k in oft_keys if k in key_map]
        assert keys, "No adapter keys mapped to model parameters"

        task_vecs = {}
        for k in keys:
            vecs = [all_w[t][k].float().to(self.device) for t in range(T)]
            task_vecs[k] = self._ties(vecs, self.ties_trim_ratio) if self.use_pp else vecs
        base = {k: params[key_map[k]].data.clone().float().to(self.device) for k in keys}

        fisher_list: Optional[Dict[str, List[Tensor]]] = None
        if fisher_paths is not None:
            all_f = self.merger.load_fishers(fisher_paths)
            fisher_list = {k: [all_f[t][k].float().to(self.device) for t in range(T)] for k in keys}

        return self.optimize_from_vecs(
            task_vecs, base, key_map, keys, all_w[0], model, task_loaders, task_names,
            mode=mode, fisher_list=fisher_list,
        )
