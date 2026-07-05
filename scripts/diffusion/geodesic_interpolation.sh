#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=geodesic_interp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=04:00:00
#SBATCH --output=outputs/diffusion/slurms/geodesic_interpolation_%A.out

set -e

REPO_ROOT="/gpfs/home6/eterres/MasterThesis"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate orthofuse_env

export HF_HOME=/scratch-shared/eterres/huggingface-cache
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${REPO_ROOT}/OrthoFuse:${REPO_ROOT}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/matplotlib-${USER}}"

PAIRS=(
  "cat:01_01"
  "dog2:dolina"
  "dog6:gondoliers"
  "cat2:pots"
  "dog:03_04"
)

METHODS=(
  orthofuse_geodesic_curve_over_id
)

# METHODS=(
#   standard_rescaled
#   fisher
#   fisher_rescaled
#   standard_geodesic
#   fisher_geodesic
#   standard_rescaled
#   orthofuse_geodesic
#   orthofuse_geodesic_curve_over_id
# )

for PAIR in "${PAIRS[@]}"; do
  python "${REPO_ROOT}/src/diffusion/geodesic_interpolation.py" \
    --config_path="${REPO_ROOT}/src/diffusion/config/config.yaml" \
    --output_dir="${REPO_ROOT}/outputs/diffusion" \
    --samples "${PAIR}" \
    --methods "${METHODS[@]}" \
    --geodesic_backend "cayley" \
    --fisher_min 1e-14 \
    --fisher_rescale 1e10 \
    --num_points 10 \
    --num_images_per_medium_prompt 1 \
    --replace_inference_output
done
