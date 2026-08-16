#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=stats_sdxl
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=outputs/diffusion/slurms/stats_sdxl_%A.out

set -e

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate orthofuse_env

export HF_HOME=/path/to/hf_cache
export HF_HUB_CACHE="${HF_HOME}/hub"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER}}"

mkdir -p "${MPLCONFIGDIR}" outputs/diffusion/slurms outputs/diffusion/analysis

python src/diffusion/analysis/sdxl_merge_stats.py \
  --output_dir="${OUTPUT_DIR:-outputs/diffusion/analysis}" \
  --device="${DEVICE:-cpu}" \
  --progress_every="${PROGRESS_EVERY:-200}" \
  --all_dataset \
  "$@"
