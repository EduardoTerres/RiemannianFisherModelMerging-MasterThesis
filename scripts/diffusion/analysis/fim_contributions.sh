#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=fim_contrib
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=outputs/diffusion/slurms/fim_contributions_%A.out

set -e

REPO_ROOT="/gpfs/home6/eterres/MasterThesis"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate orthofuse_env

export CUDA_VISIBLE_DEVICES=0
export MPLCONFIGDIR="/tmp/matplotlib-${USER}"
export PYTHONPATH="${REPO_ROOT}/OrthoFuse:${REPO_ROOT}:${PYTHONPATH:-}"

mkdir -p "${MPLCONFIGDIR}"

# --samples all_dataset_pairs
python "${REPO_ROOT}/src/diffusion/analysis/fim_contributions.py" \
  --samples dog2:dolina \
  --output_dir="${REPO_ROOT}/outputs/diffusion/analysis/fim_contributions" \
  --device=cpu \
  "$@"
