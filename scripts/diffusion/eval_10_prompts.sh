#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=00:20:00
#SBATCH --output=outputs/diffusion/slurms/eval_%A.out

set -e

REPO_ROOT="/gpfs/home6/eterres/MasterThesis"
OUTPUT_DIR="${REPO_ROOT}/outputs/diffusion"
SAMPLES_DIR="${OUTPUT_DIR}/samples_10_prompts"
METHOD_SAMPLES_ROOT="${OUTPUT_DIR}/samples_10_prompts"
EVAL_ROOT="${OUTPUT_DIR}/eval_runs"
TABLES_DIR="${OUTPUT_DIR}/tables"

METHODS=(
  orthofuse
  fisher_geodesic_diagonal_corr_2_fim_frobenius
  fisher_geodesic_diagonal_corr_2_fim_trace
  # fisher_geodesic_diagonal_corr_2_fim_kl
  fisher_geodesic_diagonal_corr_2_fim_none
  # fisher_geodesic_diagonal_corr_2_fim_layer-trace
  # diagonal_fisher_mu_4_fim_frobenius
  # diagonal_fisher_mu_4_fim_trace
  # diagonal_fisher_mu_4_fim_layer-trace
)

# METHODS=(
#   orthofuse
#   diagonal_fisher_mu_4_fim_frobenius
#   diagonal_fisher_mu_4_fim_trace
# )

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate orthofuse_env

export HF_HOME=/scratch-shared/eterres/huggingface-cache
export HF_HUB_CACHE="${HF_HOME}/hub"
export TORCH_HOME="${HF_HOME}/torch"
export HF_HUB_OFFLINE="${EVAL_OFFLINE:-0}"

python -m src.diffusion.eval.table \
  --output_dir "${OUTPUT_DIR}" \
  --samples_dir "${SAMPLES_DIR}" \
  --method_samples_root "${METHOD_SAMPLES_ROOT}" \
  --eval_root "${EVAL_ROOT}" \
  --tables_dir "${TABLES_DIR}" \
  --methods "${METHODS[@]}" \
  --aggregate-over-prompts
  "$@"
