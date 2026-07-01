#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=plot_correction
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=04:00:00
#SBATCH --output=outputs/diffusion/slurms/plot_correction_hyperparam_search_%A.out

set -e

REPO_ROOT="/gpfs/home6/eterres/MasterThesis"
OUTPUT_DIR="${REPO_ROOT}/outputs/diffusion"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate orthofuse_env

export HF_HOME=/scratch-shared/eterres/huggingface-cache
export HF_HUB_CACHE="${HF_HOME}/hub"
export TORCH_HOME="${HF_HOME}/torch"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${REPO_ROOT}/OrthoFuse:${REPO_ROOT}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/matplotlib-${USER}}"

PAIRS=(
  cat:pots
  cat2:01_07
  dog2:dolina
)

plot_samples() {
  python "${REPO_ROOT}/src/diffusion/correction_hyperparam_search.py" \
    --fisher_min 1e-14 \
    --fisher_rescale 1e10 \
    --num_images_per_medium_prompt 1 \
    --plot-only \
    --config_path "${REPO_ROOT}/src/diffusion/config/config.yaml" \
    --output_dir "${OUTPUT_DIR}" \
    --samples "$@" \
    "${EXTRA_ARGS[@]}"
}

EXTRA_ARGS=("$@")

for pair in "${PAIRS[@]}"; do
  plot_samples "${pair}"
done

plot_samples "${PAIRS[@]}"
