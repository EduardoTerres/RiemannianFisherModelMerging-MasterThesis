#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=correction_search
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=08:00:00
#SBATCH --output=outputs/diffusion/slurms/correction_hyperparam_search_%A.out

set -e

REPO_ROOT="/gpfs/home6/eterres/MasterThesis"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate orthofuse_env

export HF_HOME=/scratch-shared/eterres/huggingface-cache
export HF_HUB_CACHE="${HF_HOME}/hub"
export TORCH_HOME="${HF_HOME}/torch"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export MPLCONFIGDIR="/tmp/matplotlib-${USER}"
export PYTHONPATH="${REPO_ROOT}/OrthoFuse:${REPO_ROOT}:${PYTHONPATH:-}"

mkdir -p "${MPLCONFIGDIR}"

PAIRS=(
  "cat:01_01"
  "dog2:dolina"
  "dog6:gondoliers"
  "cat2:pots"
  "dog:03_04"
)

PAIRS=(
  "cat2:pots"
)

python "${REPO_ROOT}/src/diffusion/correction_hyperparam_search.py" \
  --fisher_min 1e-14 \
  --fisher_rescale 1e10 \
  --num_points 10 \
  --num_images_per_medium_prompt 1 \
  --config_path="${REPO_ROOT}/src/diffusion/config/config.yaml" \
  --output_dir="${REPO_ROOT}/outputs/diffusion" \
  --samples "${PAIRS[@]}"
