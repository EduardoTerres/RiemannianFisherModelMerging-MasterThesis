from typing import Dict, List
from src.merging import OFTMerging
import torch
from torch import Tensor

def _is_oft_param_name(name: str) -> bool:
    """Return True if a parameter name belongs to OFT adapters."""
    n = name.lower()
    return ("oft_r" in n or "oft_" in n) and "classifier" not in n

def fisher_task_vectors(
    oft_params_per_task: List[Dict[str, Tensor]],
    fisher_per_task: List[Dict[str, Tensor]],
    alphas: List[float],
    merging: OFTMerging,
    lam: float = 1.0,
) -> List[Dict[str, Tensor]]:
    """
    Diagonal-Fisher task vectors across T tasks (eq. 11), per layer key:

        xi_t* = (lam*I + sum_{t'} alpha_{t'} * F_tilde_{t'})^{-1} (lam*I + F_tilde_t) xi_t

    where F_tilde_t = P_t @ diag(f_t) @ P_t^T is the transported diagonal Fisher.
    The shared inverse denominator couples all tasks together.

    Args:
        oft_params_per_task: T dicts mapping layer key -> (num_blocks, d) OFT params.
        fisher_per_task:     T dicts mapping layer key -> (num_blocks, d) diagonal Fisher.
        alphas:              T scalars alpha_t (e.g. uniform 1/T).
        merging:             OFTMerging instance (used for compute_Pt).
        lam:                 regularisation scalar lambda.

    Returns:
        T dicts mapping layer key -> (num_blocks, d) Fisher-scaled task vector.
    """
    # Infer block size
    son_dimension = oft_params_per_task[0][list(oft_params_per_task[0].keys())[0]].shape[-1]
    block_size = int((1 + (1 + 8 * son_dimension) ** 0.5) / 2)
    assert block_size == 32  # TODO: remove

    keys = list(oft_params_per_task[0].keys())
    result: List[Dict[str, Tensor]] = [{} for _ in oft_params_per_task]

    for key in keys:
        print(f"Processing key: {key}...")
        # Transported Fisher matrices for every task at this layer
        F_tildes: List[Tensor] = []
        for t_params, t_fishers in zip(oft_params_per_task, fisher_per_task):
            xi = t_params[key]
            Pt = merging.compute_Pt(xi, block_size)                    # (num_blocks, d, d)
            f = t_fishers[key].to(dtype=xi.dtype, device=xi.device)
            F_tildes.append(Pt @ torch.diag_embed(f) @ Pt.transpose(-2, -1))

        _, d = oft_params_per_task[0][key].shape
        dtype, device = oft_params_per_task[0][key].dtype, oft_params_per_task[0][key].device
        Id = torch.eye(d, dtype=dtype, device=device).unsqueeze(0)    # (1, d, d)

        # Shared inverse denominator: (lam*I + sum_t alpha_t * F_tilde_t)^{-1}
        denom_inv = torch.linalg.inv(
            lam * Id + sum(a * F for a, F in zip(alphas, F_tildes))
        )                                                              # (num_blocks, d, d)

        for i, (t_params, F_tilde) in enumerate(zip(oft_params_per_task, F_tildes)):
            xi = t_params[key]
            numer = (lam * Id + F_tilde) @ xi.unsqueeze(-1)           # (num_blocks, d, 1)
            result[i][key] = (denom_inv @ numer).squeeze(-1)          # (num_blocks, d)

    return result
