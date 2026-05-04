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
import geoopt

from src.geometry import Manifold, SOnManifold
from src.paths import ROOTDIR


MergeMode = Literal["standard", "diagonal_fisher"]

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
        return 2 * torch.ones(T, dtype=torch.float32)

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
                        Omega*_j = (sum_t alpha_t f_(t,j) Omega_(t,j)) / (lama + sum_t alpha_t f_(t,j))
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
            return self._standard_merging(weights_list, alphas)

        if mode == "diagonal_fisher":
            if fisher_list is None:
                raise ValueError(f"Fisher data is required for mode {mode!r}")
            return self._diagonal_fisher_merging(weights_list, fisher_list, alphas)

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
        alphas: List[float],
    ) -> Tensor:
        """Merge OFT adapters using diagonal Fisher information as per-parameter weights.

        Args:
            weights_list: T tensors of shape (num_blocks, d), skew-symmetric parameters per task.
            fisher_list: T tensors of shape (num_blocks, d), diagonal Fisher estimates per task.
            alphas: T scalars, one mixing coefficient per task.

        Returns:
            Merged parameters of shape (num_blocks, d).
        """
        # alphas = self._fisher_diagonal_alphas(fisher_list)
        # print(alphas)
        num_blocks, son_dimension = weights_list[0].shape
        dtype, device = weights_list[0].dtype, weights_list[0].device
        block_size = int((1 + (1 + 8 * son_dimension) ** 0.5) / 2)

        Id = torch.eye(son_dimension, dtype=dtype, device=device).unsqueeze(0)  # (1, d, d)

        A = self.lam * Id.expand(num_blocks, -1, -1).clone()  # (num_blocks, d, d)
        b = torch.zeros(num_blocks, son_dimension, dtype=dtype, device=device)

        for alpha_t, oft_params_t, fisher_t in zip(alphas, weights_list, fisher_list):
            # skew_matrix = self.oft_params_to_skew_matrix(oft_params_t, son_dimension)  # (num_blocks, n, n)
            # Pt = self.manifold.compute_Pt(skew_matrix=skew_matrix, block_size=block_size)  # (num_blocks, d, d)

            # Full transported Fisher: (num_blocks, d, d)
            # F_tilde = Pt @ torch.diag_embed(fisher_t) @ Pt.transpose(-2, -1)
            F_tilde = torch.diag_embed(fisher_t)  # (num_blocks, d, d), no transport for simplicity

            A += alpha_t * F_tilde
            b += alpha_t * ((self.lam * Id.squeeze(0) + F_tilde) @ oft_params_t.unsqueeze(-1)).squeeze(-1)

        # Solve A @ omega = b per block
        merged_oft = torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)

        return merged_oft

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
            if optimize_alphas not in ("adamerging", "adamergingpp"):
                raise ValueError(
                    "optimize_alphas must be 'adamerging' or 'adamergingpp',"
                    f" got {optimize_alphas!r}"
                )
            if model is None or task_loaders is None or task_names is None:
                raise ValueError("optimize_alphas requires model, task_loaders, and task_names")
            opt = _AlphaOptimizerAdaMerging(merger=self, device=self.device)
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
        n_steps: int = 50,
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
            return self._karcher_merging(weights_list, alphas)
        if mode == "diagonal_fisher":
            if fisher_list is None:
                raise ValueError(f"Fisher data required for mode {mode!r}")
            return self._fisher_karcher_merging(weights_list, fisher_list, alphas)
        raise ValueError(f"Unsupported merge mode: {mode!r}")

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
                   coordinate Fisher F_t^SO = (E_ij theta_0)^T R_t^T F_t R_t (E_kl theta_0)
                   from the PDF, which requires theta_0 and is not computed here.
        R_m lives on the Stiefel manifold; RiemannianAdam handles gradient projection + retraction.
        """
        son_dimension = weights_list[0].shape[1]
        R_list = [
            torch.matrix_exp(self.oft_params_to_skew_matrix(w.float(), son_dimension)).detach()
            for w in weights_list
        ]

        mean_w = torch.einsum(
            "t,tbd->bd", alphas.cpu(),
            torch.stack([w.float().cpu() for w in weights_list]),
        ).to(self.device)
        R_m = geoopt.ManifoldParameter(
            torch.matrix_exp(self.oft_params_to_skew_matrix(mean_w, son_dimension)),
            manifold=geoopt.Stiefel(),
        )
        opt = geoopt.optim.RiemannianAdam([R_m], lr=self.lr)

        for _ in range(self.n_steps):
            opt.zero_grad()
            loss: Tensor = torch.zeros(1, device=self.device)
            for t, (alpha_t, R_t) in enumerate(zip(alphas, R_list)):
                omega_t = self._matrix_log_skew(R_t.transpose(-1, -2) @ R_m)
                if fisher_list is not None:
                    # upper-tri coords weighted by diagonal Fisher
                    eta = self.skew_matrix_to_oft_params(omega_t)
                    loss = loss + alpha_t * 0.5 * (fisher_list[t].float() * eta.pow(2)).sum()
                else:
                    # full Frobenius: ||Omega_t||_F^2 = sum_{i,j} Omega_{t,ij}^2
                    loss = loss + alpha_t * 0.5 * omega_t.pow(2).sum()
            loss.backward()
            opt.step()

        self._last_loss: float = loss.item()
        with torch.no_grad():
            merged = self.skew_matrix_to_oft_params(self._matrix_log_skew(R_m.detach()))
            t_norms = [w.float().norm().item() for w in weights_list]
            print(
                f"    ||omega_m||={merged.float().norm().item():.4f}  "
                f"sum||xi_t||={sum(t_norms):.4f}  "
                f"mean||xi_t||={sum(t_norms)/len(t_norms):.4f}  "
                f"loss={self._last_loss:.6f}"
            )
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
            if model is None or task_loaders is None or task_names is None:
                raise ValueError("optimize_alphas requires model, task_loaders, and task_names")
            opt = _AlphaOptimizerAdaMerging(merger=self, device=self.device)
            return opt.optimize(adapter_paths, model, task_loaders, task_names, fisher_paths, mode)

        import re
        from collections import defaultdict

        print(f"\nMerging {len(adapter_paths)} OFT adapters with Karcher mean ({mode})...")
        all_weights = self.load_weights(adapter_paths)
        if not all_weights:
            raise ValueError("No valid adapter weights found!")
        all_fishers = self.load_fishers(fisher_paths) if fisher_paths else None

        merged_weights: Dict[str, Tensor] = {}
        key_losses: Dict[str, float] = {}

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
            alphas = torch.tensor(self.alphas, dtype=torch.float32) if self.alphas is not None else torch.tensor([0.5] * len(weights_list), dtype=torch.float32)
            task_vectors = self.fisher_full_task_vectors(weights_list, fisher_list, alphas)
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
        batch_size: int = 64,
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
            batch_size=batch_size,
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
        n_iters: int = 500,
        lr: float = 1e-2,
        batch_size: int = 64,
        init_alpha: float = 0.3,
        log_wandb: bool = True,
        device: str = "cpu",
    ) -> None:
        self.merger = merger
        self.n_iters = n_iters
        self.lr = lr
        self.batch_size = batch_size
        self.init_alpha = init_alpha
        self.log_wandb = log_wandb
        self.device = device

    @staticmethod
    def _entropy(logits: Tensor, mask: Tensor) -> Tensor:
        p = logits.softmax(-1)
        h = -(p * p.clamp(min=1e-8).log()).sum(-1)
        return (h * mask.float()).sum() / mask.float().sum().clamp(min=1.0)

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

        alphas = torch.full((T, L), self.init_alpha, device=self.device, requires_grad=True)
        opt = torch.optim.Adam([alphas], lr=self.lr, betas=(0.9, 0.999))

        cyc = [cycle(dl) for dl in task_loaders]
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        for step in tqdm(range(self.n_iters), desc=f"AlphaOptimizer({type(self.merger).__name__})"):
            opt.zero_grad()
            ents: List[Tensor] = []

            clamped_alphas = torch.clamp(alphas, 0.0, 1.0)

            merged_params = {
                key_map[k]: base[k] + self.merger.merge_formula(
                    task_vecs[k],
                    fisher_list[k] if fisher_list else None,
                    mode,
                    alphas=clamped_alphas[:, i],
                )
                for i, k in enumerate(keys)
            }

            for t in range(T):
                batch = next(cyc[t])
                ids = batch["input_ids"][:self.batch_size].to(self.device)
                attn = batch["attention_mask"][:self.batch_size].to(self.device)

                logits = functional_call(
                    model, merged_params, (), {"input_ids": ids, "attention_mask": attn},
                ).logits.float()
                assert logits.dim() == 3
                ents.append(self._entropy(logits, attn))

            loss: Tensor = sum(ents)  # type: ignore[assignment]
            loss.backward()
            opt.step()

            if self.log_wandb:
                log = {"entropy/total": loss.item()}
                for t, name in enumerate(task_names):
                    log[f"entropy/{name}"] = ents[t].item()
                    log[f"alpha_mean/{name}"] = clamped_alphas[t].mean().item()
                wandb.log(log, step=step)  # type: ignore[attr-defined]

        with torch.no_grad():
            final_alphas = alphas.clamp(0.0, 1.0)
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

        task_vecs = {k: [all_w[t][k].float().to(self.device) for t in range(T)] for k in keys}
        base = {k: params[key_map[k]].data.clone().float().to(self.device) for k in keys}

        fisher_list: Optional[Dict[str, List[Tensor]]] = None
        if fisher_paths is not None:
            all_f = self.merger.load_fishers(fisher_paths)
            fisher_list = {k: [all_f[t][k].float().to(self.device) for t in range(T)] for k in keys}

        return self.optimize_from_vecs(
            task_vecs, base, key_map, keys, all_w[0], model, task_loaders, task_names,
            mode=mode, fisher_list=fisher_list,
        )


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
