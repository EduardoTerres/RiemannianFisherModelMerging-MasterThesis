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
        mode: MergeMode = "standard",
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
        lam: float = 1.0,
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
    ) -> Tensor:
        """Merge OFT task vectors using the specified mode.

        Args:
            weights_list: T tensors of shape (num_blocks, d), skew-symmetric so(n) parameters per task.
            fisher_list: Per-task Fisher info — diagonal (num_blocks, d) for ``"diagonal_fisher"``,
                full (num_blocks, d, d) for ``"fisher"``. Unused in ``"standard"`` mode.
            mode: ``"standard"`` (alpha-weighted avg), ``"diagonal_fisher"``, or ``"fisher"``
                (both Fisher modes transport to a common tangent space and solve a linear system).

        Returns:
            Merged OFT parameters of shape (num_blocks, d).

        Raises:
            ValueError: If ``fisher_list`` is None for a Fisher mode, or ``mode`` is unsupported.
        """
        T = len(weights_list)
        alphas = (
            torch.tensor(self.alphas, dtype=torch.float32)
            if self.alphas is not None
            else torch.tensor([0.5] * T, dtype=torch.float32)
        )
        alphas = alphas.to(self.device)

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

    def _standard_merging(self, weights_list: List[Tensor], alphas: List[float]) -> Tensor:
        """Compute a weighted average of OFT task vectors in so(n).

        Args:
            weights_list: T tensors of shape (num_blocks, d), the skew-symmetric parameters per task.
            alphas: T scalars, one mixing coefficient per task.

        Returns:
            Merged parameters of shape (num_blocks, d).
        """
        stacked = torch.stack(weights_list, dim=0)  # (T, num_blocks, n, n)
        a = torch.tensor(alphas, dtype=stacked.dtype, device=stacked.device)
        return torch.einsum("t,t...->...", a, stacked)


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
        num_blocks, son_dimension = weights_list[0].shape
        dtype, device = weights_list[0].dtype, weights_list[0].device
        block_size = int((1 + (1 + 8 * son_dimension) ** 0.5) / 2)

        Id = torch.eye(son_dimension, dtype=dtype, device=device).unsqueeze(0)  # (1, d, d)

        A = self.lam * Id.expand(num_blocks, -1, -1).clone()  # (num_blocks, d, d)
        b = torch.zeros(num_blocks, son_dimension, dtype=dtype, device=device)

        for alpha_t, oft_params_t, fisher_t in zip(alphas, weights_list, fisher_list):
            skew_matrix = self.oft_params_to_skew_matrix(oft_params_t, son_dimension)  # (num_blocks, n, n)
            Pt = self.manifold.compute_Pt(skew_matrix=skew_matrix, block_size=block_size)  # (num_blocks, d, d)

            # Full transported Fisher: (num_blocks, d, d)
            F_tilde = Pt @ torch.diag_embed(fisher_t) @ Pt.transpose(-2, -1)

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
            Pt = self.manifold.compute_Pt(skew_matrix=skew_matrix, block_size=block_size)  # (num_blocks, d, d)
            F_tilde = Pt @ torch.diag_embed(fisher_t) @ Pt.transpose(-2, -1)  # (num_blocks, d, d)
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
    ):
        print(f"\nMerging {len(adapter_paths)} OFT adapters with {mode} mode...")

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

    def __init__(
        self,
        n_iters: int = 500,
        lr: float = 1e-2,
        batch_size: int = 64,
        ties_trim_ratio: float = 0.2,
        init_lambda: float = 0.3,
        log_wandb: bool = True,
        device: str = "cpu",
    ) -> None:
        self.n_iters, self.lr, self.batch_size = n_iters, lr, batch_size
        self.ties_trim_ratio, self.init_lambda, self.device = ties_trim_ratio, init_lambda, device
        self.log_wandb = log_wandb

    def _load(self, path: str) -> Dict[str, Tensor]:
        sf = os.path.join(path, "adapter_model.safetensors")
        bn = os.path.join(path, "adapter_model.bin")
        if os.path.exists(sf):
            return load_file(sf, device=self.device)
        if os.path.exists(bn):
            return torch.load(bn, map_location=self.device)
        raise FileNotFoundError(f"No adapter at {path}")

    @staticmethod
    def _ties(vecs: List[Tensor], trim_ratio: float) -> List[Tensor]:
        """Ties-Merging: trim, elect sign, disjoint mask. vecs: T tensors of same shape."""
        T, shape = len(vecs), vecs[0].shape
        flat = torch.stack(vecs).float().view(T, -1)  # (T, D)
        D = flat.shape[1]

        k = max(1, int(trim_ratio * D))
        thresh = flat.abs().kthvalue(D - k + 1, dim=1, keepdim=True).values  # (T, 1)
        trimmed = flat * (flat.abs() >= thresh)  # (T, D)

        elected = torch.where(
            (trimmed * (trimmed > 0)).sum(0) >= (trimmed.abs() * (trimmed < 0)).sum(0),
            torch.ones(D, device=flat.device), -torch.ones(D, device=flat.device),
        )  # (D,)

        mask = (trimmed.sign() == elected.unsqueeze(0)) | (trimmed == 0)  # (T, D)
        phi = trimmed * mask  # (T, D)
        return [phi[t].view(shape) for t in range(T)]

    @staticmethod
    def _entropy(logits: Tensor, mask: Tensor) -> Tensor:
        """Mean token entropy. logits: (B,S,V), mask: (B,S) -> scalar."""
        p = logits.softmax(-1)  # (B, S, V)
        h = -(p * p.clamp(min=1e-8).log()).sum(-1)  # (B, S)
        return (h * mask.float()).sum() / mask.float().sum().clamp(min=1.0)

    def merge(self, adapter_paths: List[str], model, task_loaders: List,
              task_names: List[str]) -> Dict[str, Tensor]:

        T = len(adapter_paths)
        assert len(task_loaders) == T == len(task_names)

        all_w = [self._load(p) for p in adapter_paths]
        oft_keys = sorted(k for k in all_w[0]
                          if "oft_r" in k or ("oft_" in k.lower() and "classifier" not in k.lower()))
        L = len(oft_keys)
        assert L > 0, "No OFT keys found"

        # Ties preprocess and stack: phi[key] = (T, *param_shape)
        phi = {k: torch.stack(self._ties([w[k].float().to(self.device) for w in all_w],
                                         self.ties_trim_ratio)).to(self.device)
               for k in oft_keys}

        # Map adapter keys -> model param keys
        params = dict(model.named_parameters())
        key_map = {}
        for k in oft_keys:
            for c in [k, k.replace(".weight", ".default.weight")]:
                if c in params:
                    key_map[k] = c
                    break
        keys = [k for k in oft_keys if k in key_map]
        L_m = len(keys)
        assert L_m > 0, "No keys mapped"

        # Snapshot pretrained base weights (theta_pre^l)
        base = {k: params[key_map[k]].data.clone().float() for k in keys}

        # Learnable coefficients lambda_k^l: (T, L_m)
        lam = torch.full((T, L_m), self.init_lambda, device=self.device, requires_grad=True)
        opt = torch.optim.Adam([lam], lr=self.lr, betas=(0.9, 0.999))

        cyc = [cycle(dl) for dl in task_loaders]
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        for step in tqdm(range(self.n_iters), desc="AdaMerging++"):
            opt.zero_grad()
            ents: List[Tensor] = []

            for t in range(T):
                batch = next(cyc[t])
                ids = batch["input_ids"][:self.batch_size].to(self.device)
                attn = batch["attention_mask"][:self.batch_size].to(self.device)

                # theta_MTL^l = theta_pre^l + sum_k lambda_k^l * Phi(T_k^l)
                override = {
                    key_map[k]: base[k] + torch.einsum("t,t...->...", lam[:, i], phi[k])
                    for i, k in enumerate(keys)
                }

                logits = functional_call(
                    model, override, (), {"input_ids": ids, "attention_mask": attn},
                ).logits.float()
                assert logits.dim() == 3  # (B, S, V)

                ents.append(self._entropy(logits, attn))

            loss = sum(ents)
            loss.backward()
            opt.step()

            if self.log_wandb:
                log = {"entropy/total": loss.item()}
                for t, name in enumerate(task_names):
                    log[f"entropy/{name}"] = ents[t].item()
                    log[f"lambda_mean/{name}"] = lam[t].mean().item()
                wandb.log(log, step=step)

        # Final merged weights
        with torch.no_grad():
            merged = {}
            for i, k in enumerate(keys):
                w = base[k] + torch.einsum("t,t...->...", lam[:, i], phi[k])
                merged[k] = w.to(all_w[0][k].dtype)

        if self.log_wandb:
            fig, ax = plt.subplots(figsize=(max(6, L_m // 4), max(3, T)))
            ax.imshow(lam.detach().cpu().numpy(), aspect="auto", cmap="Blues")
            ax.set_yticks(range(T)); ax.set_yticklabels(task_names)
            ax.set_xlabel("Layer"); ax.set_ylabel("Task")
            plt.colorbar(ax.images[0], ax=ax); plt.tight_layout()
            wandb.log({"lambda/heatmap": wandb.Image(fig)})
            plt.close(fig)

        return merged

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
