#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=fim_style_avg
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=outputs/diffusion/slurms/fim_style_avg_%j.out

set -e

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate orthofuse_env

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER}}"
mkdir -p "${MPLCONFIGDIR}" outputs/diffusion/slurms outputs/diffusion/analysis/fim/sdxl_uncharted

STYLE_AVERAGE_MODES="${STYLE_AVERAGE_MODES:-raw trace frobenius}"

for STYLE_AVERAGE_MODE in ${STYLE_AVERAGE_MODES}; do
  python src/diffusion/analysis/normalize_fim_stats.py \
    --device="${DEVICE:-cpu}" \
    --only_style_average_raw_fim \
    --style_average_mode="${STYLE_AVERAGE_MODE}"
done

python src/diffusion/analysis/normalize_fim_stats.py \
  --render_style_average_selected_mode_stack
