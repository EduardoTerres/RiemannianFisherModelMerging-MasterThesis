#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=correction_tangent
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=outputs/diffusion/slurms/correction_tangent_visualization_%A.out

set -e

REPO_ROOT="/gpfs/home6/eterres/MasterThesis"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate orthofuse_env

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/cache-${USER}}"
export PYTHONPATH="${REPO_ROOT}/OrthoFuse:${REPO_ROOT}:${PYTHONPATH:-}"

mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}" "${REPO_ROOT}/outputs/diffusion/slurms"

CONCEPT_NAME="${CONCEPT_NAME:-cat}"
STYLE_NAME="${STYLE_NAME:-01_01}"

python "${REPO_ROOT}/src/diffusion/analysis/correction_tangent_visualization.py" \
  --concept_name="${CONCEPT_NAME}" \
  --style_name="${STYLE_NAME}" \
  --plot_dir="${REPO_ROOT}/outputs/diffusion/analysis/correction_tangent_visualization" \
  --device="${DEVICE:-cpu}" \
  "$@"
