#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=corr_rescale_curve
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=outputs/diffusion/slurms/corr_rescale_curve_%A.out

set -e

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate orthofuse_env

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER}}"

mkdir -p "${MPLCONFIGDIR}" outputs/diffusion/slurms outputs/diffusion/analysis

OUTPUT_DIR="outputs/diffusion/analysis"
DEVICE="cuda"
NUM_POINTS=21
FISHER_MIN="1e-14"
FISHER_RESCALE="1e10"

PAIRS=(
  "cat:01_01"
  "dog2:dolina"
  "dog6:gondoliers"
  "cat2:pots"
  "dog:03_04"
)

for pair in "${PAIRS[@]}"; do
  IFS=":" read -r concept_name style_name <<< "${pair}"
  echo "[correction_rescale_curve] ${concept_name}__${style_name}"
  python src/diffusion/analysis/rescaling_curve_hyperparam_corrections.py \
    --concept_name="${concept_name}" \
    --style_name="${style_name}" \
    --output_dir="${OUTPUT_DIR}" \
    --device="${DEVICE}" \
    --num_points="${NUM_POINTS}" \
    --fisher_min="${FISHER_MIN}" \
    --fisher_rescale="${FISHER_RESCALE}"
done
