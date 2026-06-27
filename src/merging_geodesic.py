from __future__ import annotations

import os
from typing import Dict, List, Optional

import torch
from safetensors.torch import load_file
from torch import Tensor

from src.geometry import SOnManifold
from src.merging import MergeMode, RiemannianMerging


class OFTGeodesicMerging(RiemannianMerging):
    """
    Geodesic merging for exactly two OFT adapters on SO(n).

    A geodesic is a two-point curve, so this merger rejects any call with a
    number of adapters other than two. The output keeps the same adapter
    coordinate format as OFTMerging: upper-triangle coordinates of the
    identity-based Lie-algebra log of the interpolated rotation.
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

    def _ensure_two(self, items: List, name: str) -> None:
        if len(items) != 2:
            raise ValueError(
                f"OFTGeodesicMerging requires exactly 2 {name}; got {len(items)}"
            )

    def _geodesic_fraction(self, alphas: Optional[Tensor] = None) -> Tensor:
        """Resolve alphas to the geodesic fraction s from model 0 to model 1."""
        if alphas is None:
            if self.alphas is None:
                return torch.tensor(0.5, dtype=torch.float32, device=self.device)
            alphas = torch.tensor(self.alphas, dtype=torch.float32, device=self.device)
        else:
            alphas = alphas.to(dtype=torch.float32, device=self.device)

        if alphas.ndim == 0 or alphas.numel() == 1:
            s = alphas.reshape(()).to(device=self.device)
        elif alphas.numel() == 2:
            a = alphas.flatten()
            total = a.sum()
            if total.abs() < 1e-8:
                raise ValueError("Geodesic alphas must not sum to zero")
            s = a[1] / total
        else:
            raise ValueError(
                f"Geodesic merging accepts either 1 interpolation value or 2 alphas; "
                f"got {alphas.numel()}"
            )

        if not torch.isfinite(s):
            raise ValueError("Geodesic interpolation fraction must be finite")
        if s < 0 or s > 1:
            raise ValueError(
                f"Geodesic interpolation fraction must be in [0, 1]; got {s.item():.6g}"
            )
        return s

    def oft_params_to_skew_matrix(
        self,
        oft_params: Tensor,
        son_dimension: int,
    ) -> Tensor:
        """
        Convert OFT parameters (num_blocks, num_params_per_block)
        to skew-symmetric matrices (num_blocks, block_size, block_size).
        """
        num_blocks, num_params = oft_params.shape
        block_size = int((1 + (1 + 8 * son_dimension) ** 0.5) / 2)
        if num_params != son_dimension:
            raise ValueError(
                f"num_params_per_block={num_params}  block_size={block_size}"
            )

        indices = torch.triu_indices(
            block_size,
            block_size,
            offset=1,
            device=oft_params.device,
        )
        rows, cols = indices[0], indices[1]
        skew = torch.zeros(
            num_blocks,
            block_size,
            block_size,
            dtype=oft_params.dtype,
            device=oft_params.device,
        )

        skew[:, rows, cols] = oft_params
        return skew - skew.transpose(-2, -1)

    def skew_matrix_to_oft_params(self, skew: Tensor) -> Tensor:
        """
        Convert skew-symmetric matrices (num_blocks, block_size, block_size)
        to OFT parameters (num_blocks, num_params_per_block).
        """
        _, block_size, _ = skew.shape
        indices = torch.triu_indices(
            block_size,
            block_size,
            offset=1,
            device=skew.device,
        )
        rows, cols = indices[0], indices[1]
        return skew[:, rows, cols]

    def _fisher_matrix(self, fisher: Tensor, ref: Tensor) -> Tensor:
        fisher = fisher.to(device=ref.device, dtype=torch.float32)
        if fisher.dim() == ref.dim():
            return torch.diag_embed(fisher.clamp(min=0.0))
        if fisher.dim() == ref.dim() + 1:
            return fisher
        raise ValueError(
            f"Fisher tensor must be diagonal {tuple(ref.shape)} or full "
            f"{tuple(ref.shape) + (ref.shape[-1],)}; got {tuple(fisher.shape)}"
        )

    def _fisher_geodesic_tangent(
        self,
        log_coords: Tensor,
        relative_omega: Tensor,
        fisher_list: List[Tensor],
        beta: Tensor,
    ) -> Tensor:
        """Compute the Fisher-weighted tangent step at ``theta_1``.

        This implements the tangent-coordinate part of

            Exp_{theta_1}[
                (lambda I + (1-beta) H_1 + beta H_2)^-1
                beta (lambda I + H_2) Log_{theta_1}(theta_2)
            ].

        ``log_coords`` is ``Log_{theta_1}(theta_2)`` expressed in the OFT
        upper-triangular Lie-algebra basis, shape ``(num_blocks, d)``.
        Fishers may be diagonal ``(num_blocks, d)`` or full
        ``(num_blocks, d, d)``. The second Fisher is transported into the
        tangent space at ``theta_1`` before solving the blockwise linear
        systems.
        """
        self._ensure_two(fisher_list, "Fisher tensors")

        num_blocks, son_dimension = log_coords.shape
        block_size = relative_omega.shape[-1]
        dtype, device = log_coords.dtype, log_coords.device

        h1 = self._fisher_matrix(fisher_list[0], log_coords)
        h2 = self._fisher_matrix(fisher_list[1], log_coords)

        if h2.dim() == 3:
            transport = self.manifold.compute_Pt(relative_omega, block_size).to(
                device=device,
                dtype=torch.float32,
            )
            h2 = transport @ h2 @ transport.transpose(-1, -2)

        eye = torch.eye(son_dimension, device=device, dtype=torch.float32)
        eye = eye.unsqueeze(0).expand(num_blocks, -1, -1)
        beta = beta.to(device=device, dtype=torch.float32)

        system = self.lam * eye + (1.0 - beta) * h1 + beta * h2
        rhs_matrix = self.lam * eye + h2
        rhs = beta * (rhs_matrix @ log_coords.float().unsqueeze(-1)).squeeze(-1)

        try:
            return torch.linalg.solve(system, rhs.unsqueeze(-1)).squeeze(-1).to(dtype)
        except RuntimeError:
            return torch.linalg.lstsq(system, rhs.unsqueeze(-1)).solution.squeeze(-1).to(dtype)

    def load_weights(self, adapter_paths: List[str]) -> List[Dict[str, Tensor]]:
        self._ensure_two(adapter_paths, "adapter paths")

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

        self._ensure_two(all_weights, "valid adapter weight dicts")
        return all_weights

    def load_fishers(
        self,
        paths: List[str],
        remove_default_ettiquete: bool = True,
    ) -> List[Dict[str, Tensor]]:
        self._ensure_two(paths, "Fisher paths")

        all_fishers = []
        for path in paths:
            if not os.path.exists(path):
                print(f"[WARNING]: No Fisher file found at {path}, skipping...")
                continue
            fisher = load_file(path, device=self.device)
            all_fishers.append(fisher)
            print(f"Loaded Fisher: {path}")

        if remove_default_ettiquete:
            for fisher in all_fishers:
                for key in list(fisher.keys()):
                    if ".default" in key:
                        new_key = key.replace(".default", "")
                        fisher[new_key] = fisher.pop(key)

        self._ensure_two(all_fishers, "valid Fisher dicts")
        return all_fishers

    def merge_formula(
        self,
        weights_list: List[Tensor],
        fisher_list: Optional[List[Dict[str, Tensor]]] = None,
        mode: MergeMode = "standard",
        alphas: Optional[Tensor] = None,
    ) -> Tensor:
        """Merge two OFT tensors by interpolation along their SO(n) geodesic."""
        self._ensure_two(weights_list, "weight tensors")
        if mode != "standard":
            raise ValueError(
                f"OFTGeodesicMerging only supports mode 'standard'; got {mode!r}"
            )
        if fisher_list is not None:
            self._ensure_two(fisher_list, "Fisher tensors")

        s = self._geodesic_fraction(alphas).to(self.device)
        start_params, end_params = weights_list
        _, son_dimension = start_params.shape
        start_skew = self.oft_params_to_skew_matrix(
            start_params.float().to(self.device),
            son_dimension,
        )
        end_skew = self.oft_params_to_skew_matrix(
            end_params.float().to(self.device),
            son_dimension,
        )

        start_rot = torch.matrix_exp(start_skew)
        end_rot = torch.matrix_exp(end_skew)

        tangent = self.manifold.exact_log(start_rot, end_rot)
        relative_omega = start_rot.transpose(-1, -2) @ tangent

        if fisher_list is None:
            merged_tangent = s * tangent
        else:
            log_coords = self.skew_matrix_to_oft_params(relative_omega)
            merged_coords = self._fisher_geodesic_tangent(
                log_coords=log_coords,
                relative_omega=relative_omega,
                fisher_list=fisher_list,
                beta=s,
            )
            merged_body_omega = self.oft_params_to_skew_matrix(
                merged_coords,
                son_dimension,
            )
            merged_tangent = start_rot @ merged_body_omega

        merged_rot = self.manifold.exact_exp(start_rot, merged_tangent)

        block_size = merged_rot.shape[-1]
        identity = torch.eye(
            block_size,
            dtype=merged_rot.dtype,
            device=merged_rot.device,
        ).expand_as(merged_rot)
        merged_skew = self.manifold.exact_log(identity, merged_rot)
        merged_params = self.skew_matrix_to_oft_params(merged_skew)
        return merged_params.to(dtype=start_params.dtype, device=start_params.device)

    def merge(
        self,
        adapter_paths: List[str],
        fisher_paths: Optional[List[str]] = None,
        mode: MergeMode = "standard",
    ) -> Dict[str, Tensor]:
        """
        Full geodesic merging pipeline for exactly two OFT adapters.
        """
        self._ensure_two(adapter_paths, "adapter paths")
        if fisher_paths is not None:
            self._ensure_two(fisher_paths, "Fisher paths")

        print(f"\nMerging 2 OFT adapters with geodesic interpolation ({mode})...")

        all_weights = self.load_weights(adapter_paths)
        all_fishers = self.load_fishers(fisher_paths) if fisher_paths else None

        merged_weights: Dict[str, Tensor] = {}

        for key in all_weights[0].keys():
            if "oft_r" in key or ("oft_" in key.lower() and "classifier" not in key.lower()):
                print(f"  Processing key: {key}")
                weights_layer = [weights[key] for weights in all_weights]
                fishers_layer = [fisher[key] for fisher in all_fishers] if all_fishers else None
                merged_weights[key] = self.merge_formula(
                    weights_list=weights_layer,
                    fisher_list=fishers_layer,
                    mode=mode,
                )
            else:
                print("[WARNING] Non-OFT weight detected.")

        print(f"  Merged {len(merged_weights)} weight tensors")
        return merged_weights


GeodesicMerging = OFTGeodesicMerging
