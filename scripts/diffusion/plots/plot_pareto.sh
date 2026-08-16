#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=pareto_curves
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=01:00:00
#SBATCH --output=outputs/diffusion/slurms/pareto_curves_%A.out

set -e

REPO_ROOT="/path/to/MasterThesis"
OUTPUT_DIR="${REPO_ROOT}/outputs/diffusion"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate orthofuse_env

export HF_HOME=/path/to/hf_cache
export HF_HUB_CACHE="${HF_HOME}/hub"
export TORCH_HOME="${HF_HOME}/torch"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${REPO_ROOT}/OrthoFuse:${REPO_ROOT}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/matplotlib-${USER}}"

PAIRS=(
  cat:01_01
  dog2:dolina
  dog6:gondoliers
  cat2:pots
  dog:03_04
)

METHODS=(
  standard_geodesic
  fisher_geodesic
  standard_rescaled
  fisher
  fisher_rescaled
  orthofuse_geodesic
  orthofuse_geodesic_curve_over_id
  orthofuse_geodesic_rotation
)

python "${REPO_ROOT}/src/diffusion/eval/pareto_curves.py" \
  --output_dir "${OUTPUT_DIR}" \
  --samples "${PAIRS[@]}" \
  --methods "${METHODS[@]}" \
  --geodesic_backend cayley \
  --num_points 10 \
  --num_inference_steps 50 \
  --guidance_scale 5.0 \
  --montage_prompt "a {0} in {1} style" \
  --save_path "${OUTPUT_DIR}/geodesic_interpolation/pareto_curves.png"
