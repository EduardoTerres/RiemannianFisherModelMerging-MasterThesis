#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=diffusion_eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=02:00:00
#SBATCH --output=outputs/diffusion/slurms/eval_%A.out

set -e

REPO_ROOT="/gpfs/home6/eterres/MasterThesis"
OUTPUT_DIR="${REPO_ROOT}/outputs/diffusion"
SAMPLES_DIR="${OUTPUT_DIR}/samples"
EVAL_ROOT="${OUTPUT_DIR}/eval_runs"
TABLES_DIR="${OUTPUT_DIR}/tables"
DIAGONAL_FISHER_MU=4

METHODS=(
  diagonal_fisher
  orthofuse
)

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate orthofuse_env

export HF_HOME=/scratch-shared/eterres/huggingface-cache
export HF_HUB_CACHE="${HF_HOME}/hub"
export TORCH_HOME="${HF_HOME}/torch"
export EVAL_OFFLINE="${EVAL_OFFLINE:-0}"
export HF_HUB_OFFLINE="${EVAL_OFFLINE}"
export TRANSFORMERS_OFFLINE="${EVAL_OFFLINE}"
export DIFFUSERS_OFFLINE="${EVAL_OFFLINE}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/matplotlib-${USER}}"
export PYTHONPATH="${REPO_ROOT}/OrthoFuse:${REPO_ROOT}:${PYTHONPATH:-}"

python -m src.diffusion.eval.table \
  --output_dir "${OUTPUT_DIR}" \
  --samples_dir "${SAMPLES_DIR}" \
  --eval_root "${EVAL_ROOT}" \
  --tables_dir "${TABLES_DIR}" \
  --diagonal_fisher_mu "${DIAGONAL_FISHER_MU}" \
  --methods "${METHODS[@]}"
