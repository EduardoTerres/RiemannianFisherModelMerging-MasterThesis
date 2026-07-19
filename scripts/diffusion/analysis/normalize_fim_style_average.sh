#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=fim_style_avg
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --array=0-11
#SBATCH --output=outputs/diffusion/slurms/fim_style_avg_%A_%a.out

set -e

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate orthofuse_env

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER}}"
mkdir -p "${MPLCONFIGDIR}" outputs/diffusion/slurms outputs/diffusion/analysis/fim/sdxl_uncharted

MODE="${MODE:-batch}"
STYLE_AVERAGE_MODE="${STYLE_AVERAGE_MODE:-raw}"

if [[ "${MODE}" == "render" ]]; then
  python src/diffusion/analysis/normalize_fim_stats.py \
    --render_style_average_raw_fim \
    --style_average_mode="${STYLE_AVERAGE_MODE}"
else
  python src/diffusion/analysis/normalize_fim_stats.py \
    --device="${DEVICE:-cpu}" \
    --only_style_average_raw_fim \
    --style_average_mode="${STYLE_AVERAGE_MODE}" \
    --style_batch_start="${SLURM_ARRAY_TASK_ID:-0}" \
    --style_batch_size="${STYLE_BATCH_SIZE:-1}" \
    --no_render
fi
