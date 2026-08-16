#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=plot_interp_search
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=04:00:00
#SBATCH --output=outputs/diffusion/slurms/plot_interpolation_hyperparam_search_%A.out

set -e

REPO_ROOT="/path/to/MasterThesis"
OUTPUT_DIR="${REPO_ROOT}/outputs/diffusion"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate orthofuse_env

export HF_HOME=/path/to/hf_cache
export HF_HUB_CACHE="${HF_HOME}/hub"
export TORCH_HOME="${HF_HOME}/torch"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${REPO_ROOT}/OrthoFuse:${REPO_ROOT}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/matplotlib-${USER}}"

mkdir -p "${MPLCONFIGDIR}" "${OUTPUT_DIR}/slurms"

MU=3
RESULTS_FOLDER="interpolation_hyperparam_search_mu${MU}"

python "${REPO_ROOT}/src/diffusion/results/interpolation_hyperparam_search_results.py" \
  --output_dir "${OUTPUT_DIR}" \
  --results_folder "${RESULTS_FOLDER}" \
  --output_prefix "all_styles" \
  --mu "${MU}"
