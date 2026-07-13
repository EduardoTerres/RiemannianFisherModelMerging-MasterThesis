#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=00:10:00
#SBATCH --output=outputs/diffusion/slurms/eval_%A.out

set -e

REPO_ROOT="/gpfs/home6/eterres/MasterThesis"
OUTPUT_DIR="${REPO_ROOT}/outputs/diffusion"
SAMPLES_DIR="${OUTPUT_DIR}/samples"
METHOD_SAMPLES_ROOT="${OUTPUT_DIR}/samples"
EVAL_ROOT="${OUTPUT_DIR}/eval_runs"
TABLES_DIR="${OUTPUT_DIR}/tables"

# METHODS=(
#   diagonal_fisher_mu_3
#   fisher_kfac_mu_2
#   orthofuse
#   standard_rescaled
#   fisher_geodesic_kfac_mu_4
#   fisher_geodesic_kfac_mu_3
# )


METHODS=(
  orthofuse
  standard_rescaled
  diagonal_fisher_mu_0
  diagonal_fisher_mu_3
  diagonal_fisher_mu_4
)

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
  --methods "${METHODS[@]}"
