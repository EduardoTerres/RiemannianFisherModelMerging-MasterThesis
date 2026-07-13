#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=interp_search
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=08:00:00
#SBATCH --array=0-11
#SBATCH --output=outputs/diffusion/slurms/interpolation_hyperparam_search_%A_%a.out

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

RESULTS_FOLDER="interpolation_hyperparam_search_mu4"

STYLES=(
  "01_01"
  "01_02"
  "01_03"
  "01_07"
  "01_08"
  "02_03"
  "03_04"
  "dolina"
  "etsy"
  "gondoliers"
  "image_scan"
  "pots"
)

STYLE="${STYLES[${SLURM_ARRAY_TASK_ID:-0}]}"
echo "[interpolation-search] array_task=${SLURM_ARRAY_TASK_ID:-0} style=${STYLE}"

python "${REPO_ROOT}/src/diffusion/interpolation_hyperparam_search.py" \
  --fisher_min 1e-14 \
  --fisher_rescale 1e10 \
  --mu "${MU}" \
  --num_points 6 \
  --num_images_per_medium_prompt 1 \
  --config_path="${REPO_ROOT}/src/diffusion/config/config.yaml" \
  --output_dir="${REPO_ROOT}/outputs/diffusion" \
  --results_folder "${RESULTS_FOLDER}" \
  --samples "style:${STYLE}"
