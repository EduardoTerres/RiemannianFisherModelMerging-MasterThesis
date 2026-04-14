from __future__ import annotations

from abc import ABC, abstractmethod
import os
from typing import Dict, List, Literal, Optional
import torch
from torch import Tensor
from safetensors.torch import load_file

from src.utils.path import OFT_LLAMA_MODELS_DIR, ROOTDIR

from src.geometry import Manifold, SOnManifold


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
                        Omega*_j = (sum_t alpha_t f_(t,j) Omega_(t,j)) / (lama + sum_t alpha_t f_(t,j))
    """

    def __init__(
        self,
        lam: float = 1.0,
        alphas: Optional[List[float]] = None,
        device: str = "cpu",
    ):
        super().__init__(manifold=SOnManifold(), lam=lam, alphas=alphas)
        self.device = device

    def oft_params_to_skew_matrix(
            self, oft_params: torch.Tensor, block_size: int,
        ) -> torch.Tensor:
        """
        Convert OFT parameters (num_blocks, num_params_per_block)
        to skew-symmetric matrices (num_blocks, block_size, block_size).

        Taken from OrthoMerge_OFT_models.py.
        """
        num_blocks, num_params = oft_params.shape
        expected = block_size * (block_size - 1) // 2
        if num_params != expected:
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

    def load_fishers(self, paths: List[str]) -> List[Dict[str, Tensor]]:
        all_fishers = []
        for path in paths:
            if not os.path.exists(path):
                print(f"[WARNING]: No Fisher file found at {path}, skipping...")
                continue
            fisher = load_file(path, device=self.device)
            all_fishers.append(fisher)
            print(f"Loaded Fisher: {path}")

        # Remove .default from all keys
        for fisher in all_fishers:
            for key in list(fisher.keys()):
                if ".default" in key:
                    new_key = key.replace(".default", "")
                    fisher[new_key] = fisher.pop(key)
        return all_fishers

    def compute_Pt(self, oft_params: torch.Tensor, block_size: int) -> torch.Tensor:
        """
        Args:
            oft_params: (num_blocks, d) OFT parameters, upper-triangle entries of skew-symmetric matrices.
            block_size: n, the size of each orthogonal block.
        Returns:
            Pt: (num_blocks, d, d) parallel transport matrices.
        """
        idx = torch.triu_indices(block_size, block_size, offset=1, device=oft_params.device)

        # Reconstruct skew-symmetric matrices
        S = self.oft_params_to_skew_matrix(oft_params, block_size)  # (num_blocks, n, n)

        # R = theta_t^{1/2} = exp(S/2)
        R = torch.matrix_exp(S / 2)  # (num_blocks, n, n)

        i, j = idx[0], idx[1]  # both index sets are the same upper-triangle pairs

        Rik = R[:, i][:, :, i]  # (num_blocks, d, d)
        Rjl = R[:, j][:, :, j]
        Ril = R[:, i][:, :, j]
        Rjk = R[:, j][:, :, i]

        Pt = (Rik * Rjl - Ril * Rjk).transpose(1, 2)  # (num_blocks, d, d)
        return Pt

    def merge_formula(
        self,
        weights_list: List[Tensor],
        fisher_list: Optional[List[Dict[str, Tensor]]] = None,
        mode: MergeMode = "plain",
    ) -> Tensor:
        T = len(weights_list)
        alphas = (
            torch.tensor(self.alphas, dtype=torch.float32)
            if self.alphas is not None
            else torch.tensor([1.0 / T] * T, dtype=torch.float32)
        )
        alphas = alphas.to(self.device)

        if mode == "plain":
            return self._plain(weights_list, alphas)

        if mode == "diagonal_fisher":
            if fisher_list is None:
                raise ValueError(f"Fisher data is required for mode {mode!r}")
            return self._diagonal_fisher(weights_list, fisher_list, alphas)

    def _plain(self, weights_list: List[Tensor], alphas: List[float]) -> Tensor:
        """Weighted average of task vectors in so(n)."""
        stacked = torch.stack(weights_list, dim=0)  # (T, B, n, n)
        a = torch.tensor(alphas, dtype=stacked.dtype, device=stacked.device)
        return torch.einsum("t,t...->...", a, stacked)

    def _diagonal_fisher(
        self,
        weights_list: List[Tensor],
        fisher_list: List[Dict[str, Tensor]],
        alphas: List[float],
    ) -> Tensor:
        num_blocks, son_dimension = weights_list[0].shape
        dtype, device = weights_list[0].dtype, weights_list[0].device
        block_size = int((1 + (1 + 8 * son_dimension) ** 0.5) / 2)

        denom = torch.zeros(num_blocks, son_dimension, dtype=dtype, device=device)
        numer = torch.zeros(num_blocks, son_dimension, dtype=dtype, device=device)

        for alpha_t, oft_params_t, fisher_t in zip(alphas, weights_list, fisher_list):
            Pt = self.compute_Pt(oft_params=oft_params_t, block_size=block_size)
            F_tilde = ((Pt ** 2) @ fisher_t.unsqueeze(-1)).squeeze(-1)

            denom += alpha_t * F_tilde
            numer += alpha_t * (self.lam + F_tilde) * oft_params_t

        denom += self.lam

        return numer / denom

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

        # print(list(iter(all_fishers[0].keys())))
        # print(list(iter(all_weights[0].keys())))

        merged_weights = {}

        # Merge per key
        for key in all_weights[0].keys():
            print(f"  Processing key: {key}")

            if "oft_r" in key or ("oft_" in key.lower() and "classifier" not in key.lower()):
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

# if __name__ == "__main__":
    # Example usage
    # merging = OFTMerging(lam=1.0, alphas=[0.5, 0.5], device="cuda")
    # merged_weights = merging.merge(
    #     adapter_paths=[
    #         # f"{OFT_LLAMA_MODELS_DIR}/Llama-3.1-8B_OFT_adapters/llama3-1_8b_finetune_magicoder",
    #         # f"{OFT_LLAMA_MODELS_DIR}/Llama-3.1-8B_OFT_adapters/llama3-1_8b_finetune_numinamath",
    #         # f"{OFT_LLAMA_MODELS_DIR}/Llama-3.1-8B_OFT_adapters/llama3-1_8b_finetune_commonsense",
    #         f"{OFT_LLAMA_MODELS_DIR}/Llama-3.1-8B_OFT_adapters/llama3-1_8b_finetune_socialiqa",
    #         # f"{OFT_LLAMA_MODELS_DIR}/Llama-3.1-8B_OFT_adapters/llama3-1_8b_finetune_scienceqa",
    #     ],
    #     fisher_paths=[
    #         f"{ROOTDIR}/data/fishers/llama3-1_8b_finetune_socialiqa.safetensors"
    #     ],
    #     mode="diagonal_fisher",
    # )
    # print(merged_weights.keys())
