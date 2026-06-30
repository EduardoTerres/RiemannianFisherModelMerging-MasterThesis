#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=rescale_curve
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=outputs/diffusion/slurms/rescale_curve_%A.out

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
METHODS=(
  "orthofuse__curve_over_id"
  "diagonal_fisher__rescaled"
  "diagonal_fisher__max_rescaled"
  "diagonal_fisher__std_rescaled"
  "standard__rescaled"
)

PAIRS=(
  "dog6:01_07"
  "cat:01_08"
  "cat:01_03"
  "dog:etsy"
  "cat2:dolina"
  "dog3:pots"
)

for pair in "${PAIRS[@]}"; do
  IFS=":" read -r concept_name style_name <<< "${pair}"
  echo "[rescale_curve] ${concept_name}__${style_name}"
  python src/diffusion/analysis/compare_rescaling_curve.py \
    --concept_name="${concept_name}" \
    --style_name="${style_name}" \
    --output_dir="${OUTPUT_DIR}" \
    --device="${DEVICE}" \
    --num_points="${NUM_POINTS}" \
    --fisher_min="${FISHER_MIN}" \
    --fisher_rescale="${FISHER_RESCALE}" \
    --methods "${METHODS[@]}"
done
