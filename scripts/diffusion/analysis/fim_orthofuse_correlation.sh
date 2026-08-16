#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=fim_ortho_corr
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=outputs/diffusion/slurms/fim_orthofuse_correlation_%A.out

set -e

REPO_ROOT="/path/to/MasterThesis"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate orthofuse_env

export CUDA_VISIBLE_DEVICES=0
export MPLCONFIGDIR="/tmp/matplotlib-${USER}"
export XDG_CACHE_HOME="/tmp/cache-${USER}"
export PYTHONPATH="${REPO_ROOT}/OrthoFuse:${REPO_ROOT}:${PYTHONPATH:-}"

mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

python "${REPO_ROOT}/src/diffusion/analysis/fim_orthofuse_correlation.py" \
  --samples_root "${REPO_ROOT}/outputs/diffusion/samples_10_prompts" \
  --method_dirs orthofuse \
  --samples all_dataset_pairs \
  --output_dir "${REPO_ROOT}/outputs/diffusion/analysis/fim_orthofuse_correlation" \
  --device cuda \
  "$@"
