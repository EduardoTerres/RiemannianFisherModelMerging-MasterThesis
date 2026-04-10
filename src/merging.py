from __future__ import annotations
from abc import ABC, abstractmethod
import os
from typing import Dict, List, Literal, Optional
from OrthoMerge.merge.OrthoMerge_C_TA import merge_cayley_Q_list
import torch
from torch import Tensor
from safetensors import safe_open
from safetensors.torch import load_file

from src.utils.path import OFT_LLAMA_MODELS_DIR

from .geometry import Manifold, SOnManifold
from OrthoMerge.merge.OrthoMerge_OFT_models import (
    oft_params_to_skew_matrix,
    skew_matrix_to_oft_params,
    merge_cayley_Q_list,
)


MergeMode = Literal["plain", "diagonal_fisher", "linear_system"]


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
        self.alphas = alphas  # per-task weights; uniform if None

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
        mode: MergeMode = "plain",
    ) -> Tensor:
        """Merge task vectors into one (B, n, n) Omega.

        Modes: "plain", "diagonal_fisher", or "linear_system".
        Fisher data is required for Fisher-based modes.
        """

    @abstractmethod
    def merge(
        self,
        adapter_paths: List[str],
        fisher_paths: Optional[List[str]] = None,
        mode: MergeMode = "plain",
    ) -> Dict[str, Tensor]:
        """
        Full merging pipeline
        """


class OFTMerging(RiemannianMerging):
    """
    Riemannian merging for OFT adapters on SO(n).

    Three merge modes
    -----------------
    plain           Weighted average of Lie-algebra task vectors. No Fisher.
    diagonal_fisher Element-wise Fisher weighting in the vectorised so(n) basis.
                    Each component j of the merged vector is
                        Omega*_j = (sum_t alpha_t f_(t,j) Omega_(t,j)) / (lambda + sum_t alpha_t f_(t,j))
                    where f_(t,j) is the j-th diagonal Fisher entry for task t.
    linear_system   Full transported-Fisher solve (Eq. 4.26 / SO(n) Eq. 11):
                        (lambda I + sum_t alpha_t I_hat_t) Omega* = sum_t alpha_t (lambda I + I_hat_t) Omega_t
                    where I_hat_t is the exp-conjugated Fisher matrix.
    """

    def __init__(
        self,
        lam: float = 1.0,
        alphas: Optional[List[float]] = None,
        device: str = "cpu",
    ):
        super().__init__(manifold=SOnManifold(), lam=lam, alphas=alphas)
        self.device = device

    def _vec(self, omega: Tensor) -> Tensor:
        """Skew matrix to upper-triangular vector."""
        n = omega.shape[-1]
        idx = torch.triu_indices(n, n, offset=1, device=omega.device)
        return omega[..., idx[0], idx[1]]

    def _unvec(self, v: Tensor, n: int) -> Tensor:
        """Upper-triangular vector to skew matrix."""
        *batch, _ = v.shape
        idx = torch.triu_indices(n, n, offset=1, device=v.device)
        omega = torch.zeros(*batch, n, n, dtype=v.dtype, device=v.device)
        omega[..., idx[0], idx[1]] = v
        return omega - omega.transpose(-1, -2)

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
                    weights = torch.load(model_path, map_location="cpu")
                else:
                    print(f"[WARNING]: No adapter weights found at {adapter_path}, skipping...")
                    continue
            else:
                weights = load_file(model_path)

            all_weights.append(weights)
            print(f"Loaded adapters: {adapter_path}")

        return all_weights

    def load_fishers(self, paths: List[str]) -> List[Dict[str, Tensor]]:
        all_fishers = []
        for path in paths:
            if not os.path.exists(path):
                print(f"[WARNING]: No Fisher file found at {path}, skipping...")
                continue
            fisher = load_file(path, device=self.device)
            all_fishers.append(fisher)
            print(f"Loaded Fisher: {path}")
        return all_fishers

    def merge_formula(
        self,
        weights_list: List[Tensor],
        fisher_list: Optional[List[Dict[str, Tensor]]] = None,
        mode: MergeMode = "plain",
    ) -> Tensor:
        T = len(weights_list)
        alphas = self.alphas if self.alphas is not None else [1.0 / T] * T

        if mode == "plain":
            return self._plain(weights_list, alphas)

        if mode == "diagonal_fisher":
            if fisher_list is None:
                raise ValueError(f"Fisher data is required for mode {mode!r}")
            return self._diagonal_fisher(weights_list, alphas, fisher_list)

    def _plain(self, weights_list: List[Tensor], alphas: List[float]) -> Tensor:
        """Weighted average of task vectors in so(n)."""
        stacked = torch.stack(weights_list, dim=0)  # (T, B, n, n)
        a = torch.tensor(alphas, dtype=stacked.dtype, device=stacked.device)
        return torch.einsum("t,t...->...", a, stacked)

    def _diagonal_fisher(
        self,
        weights_list: List[Tensor],
        alphas: List[float],
        fisher_list: List[Dict[str, Tensor]],
    ) -> Tensor:
        """
        Element-wise Fisher-weighted merge in the vectorised so(n) basis.
        No matrix solve; each component is an independent scalar problem.
        """
        B, n, _ = weights_list[0].shape
        d = n * (n - 1) // 2

        numer = torch.zeros(B, d, dtype=weights_list[0].dtype, device=weights_list[0].device)
        denom = torch.full_like(numer, self.lam)

        for a, omega, fisher in zip(weights_list, alphas, fisher_list):
            v = self._vec(omega)      # (B, d)
            f = fisher.to(v.device)   # (d,) or broadcastable
            af = a * f               # (d,)
            numer = numer + af * v
            denom = denom + af

        merged_vec = numer / denom  # (B, d)
        return self._unvec(merged_vec, n)

    def merge(
        self,
        adapter_paths: List[str],
        fisher_paths: Optional[List[str]] = None,
        mode: MergeMode = "plain",
    ):
        print(f"\nMerging {len(adapter_paths)} OFT adapters...")

        # Load weights
        all_weights = self.load_weights(adapter_paths)
        if not all_weights:
            raise ValueError("No valid adapter weights found!")

        # Load Fishers
        all_fishers = self.load_fishers(fisher_paths) if fisher_paths else None

        merged_weights = {}

        # Merge per key
        for key in all_weights[0].keys():
            print(f"  Processing key: {key}")

            if "oft_r" in key or ("oft_" in key.lower() and "classifier" not in key.lower()):
                weights_list = [
                    oft_params_to_skew_matrix(weights[key])
                    for weights in all_weights
                ]

                fishers_list = None
                if all_fishers:
                    fishers_list = [
                        fisher[key] for fisher in all_fishers
                    ]

                avg_weight = self.merge_formula(
                    weights_list=weights_list,
                    fisher_list=fishers_list,
                    mode=mode,
                )

                avg_weight = skew_matrix_to_oft_params(avg_weight)
                merged_weights[key] = avg_weight

            else:
                print("[WARNING] Non-OFT weight detected.")

        print(f"  Merged {len(merged_weights)} weight tensors")
        return merged_weights

if __name__ == "__main__":
    # Example usage
    merging = OFTMerging(lam=1.0, alphas=[0.5, 0.5], device="cuda")
    merged_weights = merging.merge(
        adapter_paths=[
            f"{OFT_LLAMA_MODELS_DIR}/Llama-3.1-8B_OFT_adapters/llama3-1_8b_finetune_magicoder",
            f"{OFT_LLAMA_MODELS_DIR}/Llama-3.1-8B_OFT_adapters/llama3-1_8b_finetune_numinamath",
            f"{OFT_LLAMA_MODELS_DIR}/Llama-3.1-8B_OFT_adapters/llama3-1_8b_finetune_commonsense",
            f"{OFT_LLAMA_MODELS_DIR}/Llama-3.1-8B_OFT_adapters/llama3-1_8b_finetune_socialiqa",
            f"{OFT_LLAMA_MODELS_DIR}/Llama-3.1-8B_OFT_adapters/llama3-1_8b_finetune_scienceqa",
        ],
        fisher_paths=[
            "data"
        ],
        mode="diagonal_fisher",
    )
    print(merged_weights.keys())


class Unused:
    def oft_to_skew_matrices(
        self, weights_list: List[Dict[str, Tensor]]
    ) -> Dict[str, List[Tensor]]:
        """Convert OFT raw params to skew-matrix task vectors."""
        task_vectors: Dict[str, List[Tensor]] = {}
        for w in weights_list:
            for key, val in w.items():
                if "oft_r" in key:
                    omega = oft_params_to_skew_matrix(val)  # (B, n, n)
                    task_vectors.setdefault(key, []).append(omega)
        return task_vectors

    def from_task_vector(self, key: str, omega: Tensor) -> Tensor:
        omega = 0.5 * (omega - omega.transpose(-1, -2))  # enforce skew
        return skew_matrix_to_oft_params(omega)

    def merge_formula(
        self,
        omegas: List[Tensor],
        mode: MergeMode,
        fisher_list: Optional[List[Dict[str, Tensor]]] = None,
        key: Optional[str] = None,
    ) -> Tensor:
        T = len(omegas)
        alphas = self.alphas if self.alphas is not None else [1.0 / T] * T

        if mode == "plain":
            return self._plain(omegas, alphas)

        if mode == "diagonal_fisher":
            fisher_diags = [f[key] for f in fisher_list]
            return self._diagonal_fisher(omegas, alphas, fisher_diags)

        if mode == "linear_system":
            fisher_diags = [f[key] for f in fisher_list]
            return self._linear_system(omegas, alphas, fisher_diags)

        raise ValueError(f"Unknown mode: {mode!r}")

    def _plain(self, omegas: List[Tensor], alphas: List[float]) -> Tensor:
        """Weighted average of task vectors in so(n)."""
        return sum(a * om for a, om in zip(alphas, omegas))

    def _diagonal_fisher(
        self,
        omegas: List[Tensor],
        alphas: List[float],
        fisher_diags: List[Tensor],
    ) -> Tensor:
        """
        Element-wise Fisher-weighted merge in the vectorised so(n) basis.
        No matrix solve; each component is an independent scalar problem.
        """
        B, n, _ = omegas[0].shape

        numer = torch.zeros(B, n * (n - 1) // 2, dtype=omegas[0].dtype, device=omegas[0].device)
        denom = torch.full_like(numer, self.lam)

        for a, omega, f in zip(alphas, omegas, fisher_diags):
            v = self._vec(omega)      # (B, d)
            # f is (d,) or broadcastable; broadcast over B
            af = a * f.to(v.device)  # (d,)
            numer = numer + af * v
            denom = denom + af

        merged_vec = numer / denom  # (B, d)
        return self._unvec(merged_vec, n)

    def _linear_system(
        self,
        omegas: List[Tensor],
        alphas: List[float],
        fisher_diags: List[Tensor],
    ) -> Tensor:
        """
        Full transported-Fisher linear system (Eq. 4.26).
        Builds the (B, d, d) transported Fisher matrix per task and solves.
        """
        B, n, _ = omegas[0].shape
        d = n * (n - 1) // 2
        dtype, device = omegas[0].dtype, omegas[0].device

        lhs = self.lam * torch.eye(d, dtype=dtype, device=device).unsqueeze(0).expand(B, -1, -1).clone()
        rhs = torch.zeros(B, d, dtype=dtype, device=device)

        for a, omega, f in zip(alphas, omegas, fisher_diags):
            It = self._build_transported_fisher(omega, f.to(device))  # (B, d, d)
            v  = self._vec(omega)                                           # (B, d)
            lhs = lhs + a * It
            rhs = rhs + a * (self.lam * v + torch.einsum("bij,bj->bi", It, v))

        merged_vec = torch.linalg.solve(lhs, rhs)  # (B, d)
        return self.    _unvec(merged_vec, n)

    def _build_transported_fisher(self, omega_t: Tensor, fisher_diag: Tensor) -> Tensor:
        """
        Build (B, d, d) transported Fisher matrix I_hat_t via exp-conjugation.
        I_hat_t(V) = exp(Omega/2) I_t(exp(-Omega/2) V exp(Omega/2)) exp(-Omega/2)
        where I_t acts as diagonal scaling by fisher_diag in the vec basis.
        """
        B, n, _ = omega_t.shape
        d = n * (n - 1) // 2
        half_exp     = self.manifold._matrix_exp( omega_t / 2)
        half_exp_inv = self.manifold._matrix_exp(-omega_t / 2)

        eye_d = torch.eye(d, dtype=omega_t.dtype, device=omega_t.device)
        cols  = []
        for i in range(d):
            v_vec = eye_d[i].unsqueeze(0).expand(B, -1)
            V = self._unvec(v_vec, n)
            FV = fisher_diag[i] * V
            transported = half_exp @ FV @ half_exp_inv
            cols.append(self._vec(transported))

        return torch.stack(cols, dim=-1)  # (B, d, d)

    def merge(
        self,
        adapter_paths: List[str],
        fisher_paths: Optional[List[str]] = None,
        mode: MergeMode = "plain",
    ) -> Dict[str, Tensor]:
        """
        Full pipeline
        """
        weights_list = self.load_weights(adapter_paths)
        if not weights_list:
            raise ValueError("No adapter weights loaded; cannot merge.")
        fisher_list  = self.load_fishers(fisher_paths) if fisher_paths else None

        task_vectors = self.oft_to_skew_matrices(weights_list)

        merged = {}
        for key, omegas in task_vectors.items():
            merged_omega = self.merge_formula(omegas, mode, fisher_list, key)
            merged[key]  = self.from_task_vector(key, merged_omega)

        return merged
