import math
import os
import sys
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from safetensors.torch import load_file, save_file
import shutil
from tqdm import tqdm

from torch import Tensor
from typing import Dict, List, Optional
import itertools

def merge_cayley_Q_list(
    weights_list: List[torch.Tensor],
    correction: Optional[bool] = True,
) -> torch.Tensor:
    print("product C")
    assert len(weights_list) > 0, "weights_list none"
    n_task = len(weights_list)

    Q_stack = torch.stack(weights_list, dim=0)

    merged_sum = Q_stack.sum(dim=0)

    if correction:
        # sum_{i=0}^{N-1} |delta_i|_F
        N = Q_stack.shape[0]
        norms = torch.norm(Q_stack.view(N, -1), p="fro", dim=1)
        sum_of_norms = norms.sum()

        # |sum_{i=0}^{N-1} delta_i|_F
        norm_of_sum = torch.norm(merged_sum, p="fro")

        c = sum_of_norms / (norm_of_sum)
    else:
        c = 1.0

    merged = (1 / n_task) * c * merged_sum

    merged = 0.5 * (merged - merged.transpose(-1, -2))

    return merged


def oft_params_to_skew_matrix(oft_params: torch.Tensor, block_size: int = 32) -> torch.Tensor:

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


def skew_matrix_to_oft_params(S: torch.Tensor) -> torch.Tensor:

    num_blocks, block_size, _ = S.shape

    indices = torch.triu_indices(block_size, block_size, offset=1, device=S.device)
    rows, cols = indices[0], indices[1]
    oft_params = S[:, rows, cols]  # (num_blocks, num_params_per_block)

    return oft_params


cache_dir = None


parser = argparse.ArgumentParser(
    "Interface for merging LLMs with multiple OFT adapters (no evaluation)"
)
parser.add_argument(
    "--language_model_name",
    type=str,
    required=True,
    help="Base LLM name or path, e.g., 'meta-llama/Llama-2-7b-hf' or local path",
)
parser.add_argument("--gpu", type=int, default=0, help="GPU id to use, -1 for CPU")
parser.add_argument(
    "--adapter_paths",
    type=str,
    nargs="+",
    required=True,
    help="Paths to the saved OFT adapter directories (can provide multiple)",
)
parser.add_argument(
    "--output_merged_adapter_dir",
    type=str,
    default="./merged_oft_adapter",
    help="Where to save merged adapter and/or merged base model",
)
parser.add_argument(
    "--save_merged_model",
    action="store_true",
    help="If set, will save base model with merged weights (merge_and_unload) to output_merged_adapter_dir/model",
)
parser.add_argument(
    "--just_merge_adapter",
    action="store_true",
    help="If set, only merge adapter weights and save a merged adapter folder, without loading base model",
)
parser.add_argument(
    "--fisher_paths",
    type=str,
    nargs="+",
    required=True,
    help="Paths to Fisher information safetensors files, one per adapter (in matching order)",
)


args = parser.parse_args()
if torch.cuda.is_available() and args.gpu >= 0:
    args.device = f"cuda:{args.gpu}"
else:
    args.device = "cpu"


def merge_oft_adapter_weights_jacobi(adapter_paths, fishers):
    print(f"\nMerging {len(adapter_paths)} OFT adapters...")

    all_weights = []
    for adapter_path in adapter_paths:
        model_path = os.path.join(adapter_path, "adapter_model.safetensors")
        if not os.path.exists(model_path):
            model_path = os.path.join(adapter_path, "adapter_model.bin")
            if os.path.exists(model_path):
                weights = torch.load(model_path, map_location="cpu")
            else:
                print(f"Warning: No adapter weights found at {adapter_path}, skipping...")
                continue
        else:
            weights = load_file(model_path)

        all_weights.append(weights)
        print(f"  Loaded: {adapter_path}")

    if not all_weights:
        raise ValueError("No valid adapter weights found!")

    merged_weights = {}
    first_weights = all_weights[0]

    device = next(iter(fishers[0].values())).device

    for key in tqdm(first_weights.keys(), desc="Merging adapter keys"):
        print(f"  Processing key: {key}")

        if "oft_r" in key or ("oft_" in key.lower() and "classifier" not in key.lower()):
            weights_list = []
            for weights in all_weights:
                if key in weights:
                    w = weights[key]
                    # w = oft_params_to_skew_matrix(w)
                    weights_list.append(w.to(device))

            def compute_Pt(oft_params: torch.Tensor, block_size: int) -> torch.Tensor:
                """
                Args:
                    oft_params: (num_blocks, d) OFT parameters, upper-triangle entries of skew-symmetric matrices.
                    block_size: n, the size of each orthogonal block.
                Returns:
                    Pt: (num_blocks, d, d) parallel transport matrices.
                """
                idx = torch.triu_indices(block_size, block_size, offset=1, device=oft_params.device)

                # Reconstruct skew-symmetric matrices
                S = oft_params_to_skew_matrix(oft_params, block_size)  # (num_blocks, n, n)

                # R = theta_t^{1/2} = exp(S/2)
                R = torch.matrix_exp(S / 2)  # (num_blocks, n, n)

                i, j = idx[0], idx[1]  # both index sets are the same upper-triangle pairs

                Rik = R[:, i][:, :, i]  # (num_blocks, d, d)
                Rjl = R[:, j][:, :, j]
                Ril = R[:, i][:, :, j]
                Rjk = R[:, j][:, :, i]

                # Pt[b, k, a] = M_{ka} = matrix of P_{θ_t → θ_LLM} in upper-triangle basis
                Pt = Rik * Rjl - Ril * Rjk  # (num_blocks, d, d)
                return Pt

            alphas = torch.tensor([0.5 for _ in weights_list], dtype=torch.float32)
            fisher_list = [fisher[key] for fisher in fishers if key in fisher]

            # Parallel transport Fisher
            fisher_transported_list = []
            for alpha_t, oft_params_t, fisher_t in zip(alphas, weights_list, fisher_list):
                Pt = compute_Pt(oft_params=oft_params_t, block_size=32)
                F_tilde = ((Pt ** 2) @ fisher_t.unsqueeze(-1)).squeeze(-1)
                fisher_transported_list.append(F_tilde)

            fisher_list = fisher_transported_list
            # Transform to fisher task vector
            lam = 1
            denom = 0.0

            denom = torch.zeros(fisher_list[0].shape).to(device)
            for alpha_t, fisher_t in zip(alphas, fisher_list):
                denom += alpha_t * fisher_t
            denom += lam

            weights = [
                (lam + fisher_t) * oft_params_t / denom
                for oft_params_t, fisher_t in zip(weights_list, fisher_list)
            ]

            weights_list = [oft_params_to_skew_matrix(w) for w in weights]

            if weights_list:
                avg_weight = merge_cayley_Q_list(weights_list, correction=False)

                # Apply fixed-point iterations
                def lie_bracket(A, B):
                    return A @ B - B @ A

                A_iter = avg_weight
                N = 20  # truncation order (Jacobi energy objective)
                T = len(weights_list)
                tol = 1e-10
                for _ in tqdm(range(5), desc="Fixed-point iterations"):
                    # Per-task correction: C_t = sum_{m=1}^{N} 1/(2m+1)! * ad_{A_t}^{2m}(A_iter)
                    C_list = []
                    for A_t in weights_list:
                        B = A_iter
                        C_t = torch.zeros_like(A_iter)
                        for step in range(2 * N):
                            B = lie_bracket(A_t, B)
                            if (step + 1) % 2 == 0:  # collect even powers B_{2m}
                                m = (step + 1) // 2   # m = 1, 2, ..., N
                                coef = 10.0 / math.factorial(2 * m + 1)
                                C_t = C_t + coef * B
                        C_list.append(C_t)

                    A_iter_prev = A_iter.clone()
                    # A^(k+1) = A_bar - (1/T) * sum_t C_t  (uniform weights)
                    A_iter = avg_weight - (1.0 / T) * torch.stack(C_list).sum(dim=0)

                    # Print the difference in norm in scientific notation
                    diff_norm = torch.norm(A_iter - A_iter_prev, p="fro").item()
                    print(f"    Iteration diff norm: {diff_norm:.2e}")

                    if diff_norm < tol:
                        print(f"    Converged (tol={tol:.1e}), stopping early.")
                        break

                # difference in A_iter and avg_weight
                final_diff_norm = torch.norm(A_iter - avg_weight, p="fro").item()
                print(f"    Final diff norm from avg_weight: {final_diff_norm:.4f}")

                # average of per-task Frobenius norms
                avg_norms = sum(torch.norm(w, p="fro").item() for w in weights_list) / len(weights_list)
                print(f"    Sum of norms: {avg_norms:.4f}")

                # Frobenius norm of merged tensor
                norm_of_sum = torch.norm(A_iter, p="fro").item()
                print(f"    Norm of sums: {norm_of_sum:.4f}")

                # ratio = (previous line 1) / (previous line 2)
                ratio = avg_norms / norm_of_sum if norm_of_sum != 0 else float("inf")
                print(f"    Ratio (sum norms / norm of sum): {ratio:.4f}")


                # Iso-energy rescaling (Theorem 3, eq. 8 from iso-energy PDF):
                # xi_iso^(N) = sqrt( sum_t alpha_t ||A_t||_F^2
                #                  / sum_t alpha_t sum_{m=0}^{N} 1/(2m+1)! ||ad_{A_t}^m B||_F^2 )
                B = A_iter
                numerator = 0.0
                denominator = 0.0
                for A_t in weights_list:
                    numerator += torch.sum(A_t * A_t).item()  # ||A_t||_F^2
                    Bm = B.clone()
                    for m in range(N + 1):
                        coef = 1.0 / math.factorial(2 * m + 1)
                        denominator += coef * torch.sum(Bm * Bm).item()  # ||ad_{A_t}^m B||_F^2
                        if m < N:
                            Bm = lie_bracket(A_t, Bm)

                print(f"    iso-energy num = {numerator:.4f}, den = {denominator:.4f}")
                xi_star = math.sqrt(numerator / denominator) if denominator > 1e-12 else 1.0
                print(f"  Iso-energy rescaling xi_iso = {xi_star:.4f}")

                A_iter = xi_star * A_iter

                A_iter = 0.5 * (A_iter - A_iter.transpose(-1, -2))

                avg_weight = skew_matrix_to_oft_params(A_iter)
                merged_weights[key] = avg_weight
                print(f"    Merged shape: {avg_weight.shape}")
            else:
                print(f"    Warning: No weights found for key {key}")

        else:
            weights_list = [weights[key] for weights in all_weights if key in weights]
            if len(weights_list) > 1:
                avg_weight = torch.stack(weights_list).mean(dim=0)
                merged_weights[key] = avg_weight
                print(f"    Averaged non-OFT weight, shape: {avg_weight.shape}")
            else:
                merged_weights[key] = first_weights[key].clone()
                print(f"    Copied from first adapter, shape: {merged_weights[key].shape}")

    print(f"  Merged {len(merged_weights)} weight tensors")
    return merged_weights


def print_groupwise_avg_correction(
    weights_list: List[torch.Tensor],
    max_groups_per_k: Optional[int] = None,
) -> None:
    """
    Print avg correction coefficient c for group sizes k=2..N:
      c = (sum_i ||Q_i||_F) / ||sum_i Q_i||_F
    Also prints avg ||sum_i Q_i||_F to monitor growth of the sum.
    """
    assert len(weights_list) >= 2, "Need at least 2 tensors"
    Q_stack = torch.stack(weights_list, dim=0)
    N = Q_stack.shape[0]

    print("\nGroup-wise average correction growth:")
    for k in range(2, N + 1):
        c_vals, sum_norm_vals = [], []
        groups = itertools.combinations(range(N), k)
        if max_groups_per_k is not None:
            groups = itertools.islice(groups, max_groups_per_k)

        for idx in groups:
            G = Q_stack[list(idx)]                  # (k, ...)
            sum_q = G.sum(dim=0)
            norm_sum = torch.norm(sum_q, p="fro")
            if norm_sum == 0:
                continue
            sum_norms = torch.norm(G.reshape(k, -1), p="fro", dim=1).sum()
            c_vals.append((sum_norms / norm_sum).item())
            sum_norm_vals.append(norm_sum.item())

        if c_vals:
            print(
                f"  k={k:2d} | avg_c={sum(c_vals)/len(c_vals):.6f} "
                f"| avg_||sumQ||={sum(sum_norm_vals)/len(sum_norm_vals):.6f} "
                f"| groups={len(c_vals)}"
            )
        else:
            print(f"  k={k:2d} | no valid groups")


def load_fishers(paths: List[str], device: str) -> List[Dict[str, Tensor]]:
    all_fishers = []
    for path in paths:
        if not os.path.exists(path):
            print(f"[WARNING]: No Fisher file found at {path}, skipping...")
            continue
        fisher = load_file(path, device=device)
        all_fishers.append(fisher)
        print(f"Loaded Fisher: {path}")

    # Remove .default from all keys
    for fisher in all_fishers:
        for key in list(fisher.keys()):
            if ".default" in key:
                new_key = key.replace(".default", "")
                fisher[new_key] = fisher.pop(key)
    return all_fishers


def create_merged_adapter_with_oft_for_llm(
    base_model_name,
    adapter_paths,
    fisher_paths,
    output_merged_adapter_dir,
    save_merged_model: bool = False,
    device: str = "cpu",
):
    print(f"\n{'=' * 80}")
    print("Creating merged OFT adapter for LLM")
    print(f"Base model: {base_model_name}")
    print(f"{'=' * 80}")

    config_path = os.path.join(adapter_paths[0], "adapter_config.json")

    fishers = load_fishers(fisher_paths, device=device)

    merged_weights = merge_oft_adapter_weights_jacobi(adapter_paths, fishers)

    os.makedirs(output_merged_adapter_dir, exist_ok=True)
    merged_adapter_path = os.path.join(output_merged_adapter_dir, "merged_adapter")
    os.makedirs(merged_adapter_path, exist_ok=True)

    merged_weights_path = os.path.join(merged_adapter_path, "adapter_model.safetensors")
    save_file(merged_weights, merged_weights_path)
    shutil.copy(config_path, os.path.join(merged_adapter_path, "adapter_config.json"))

    print(f"  Saved merged adapter to: {merged_adapter_path}")

    if not save_merged_model:
        print("  [INFO] save_merged_model = False, stop after saving merged adapter.")
        print(f"{'=' * 80}\n")
        return None, merged_adapter_path

    print("\nLoading base LLM (causal LM)...")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=(
                os.path.join(cache_dir, base_model_name) if cache_dir else base_model_name
            ),
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
            device_map=None,
        )
    except Exception as e:
        print(
            f"  Failed to load from cache_dir, fallback to {base_model_name} directly. Error: {e}"
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=base_model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
            device_map=None,
        )

    base_model.to(device)
    print(f"  Base model loaded on {device}")

    print("  Loading merged adapter with PeftModel and merging into base model...")
    peft_model = PeftModel.from_pretrained(base_model, merged_adapter_path)
    merged_model = peft_model.merge_and_unload()
    merged_model.to(device)

    print("  Successfully merged encoder weights into LLM (merge_and_unload finished).")

    model_save_dir = os.path.join(output_merged_adapter_dir, "merged_model")
    os.makedirs(model_save_dir, exist_ok=True)
    merged_model.save_pretrained(model_save_dir)
    print(f"  Saved merged base model to: {model_save_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.language_model_name)
    tokenizer.save_pretrained(model_save_dir)

    del peft_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"{'=' * 80}\n")
    return merged_model, merged_adapter_path


if __name__ == "__main__":
    print("=" * 80)
    print(f"Processing {len(args.adapter_paths)} adapters:")
    for i, path in enumerate(args.adapter_paths, 1):
        print(f"  {i}. {path}")
    print(f"\nBase Model: {args.language_model_name}")
    print(f"Device: {args.device}")
    print(f"Output Dir: {args.output_merged_adapter_dir}")
    print("=" * 80)
    print()

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=(
                os.path.join(cache_dir, args.language_model_name)
                if cache_dir
                else args.language_model_name
            ),
            cache_dir=cache_dir,
        )
        print("Tokenizer loaded.")
    except Exception as e:
        print(f"Failed to load tokenizer, error: {e}")
        tokenizer = None

    merged_model, merged_adapter_path = create_merged_adapter_with_oft_for_llm(
        base_model_name=args.language_model_name,
        adapter_paths=args.adapter_paths,
        fisher_paths=args.fisher_paths,
        output_merged_adapter_dir=args.output_merged_adapter_dir,
        save_merged_model=not args.just_merge_adapter,
        device=args.device,
    )

    sys.exit()
