#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=kl
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=02:00:00
#SBATCH --output=outputs/analysis/slurm/kl_%A_%a.out

set -euo pipefail

SCRIPT_DIR="/home/eterres/MasterThesis/scripts/analysis"
REPO_ROOT="/home/eterres/MasterThesis"
cd "${REPO_ROOT}"

PLOT_DIR="${REPO_ROOT}/outputs/analysis"
DATA_DIR="${REPO_ROOT}/outputs/analysis/data_2"
EVAL_DIR="${REPO_ROOT}/outputs/evaluation"
PIPE_MERGE_ROOT="/scratch-shared/eterres"
MERGED_MODELS_DIR="${PIPE_MERGE_ROOT}/models"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate merge

python -m src.analysis.compare_kl_geodist \
    --plot_dir "${PLOT_DIR}" \
    --data-dir "${DATA_DIR}" \
    --eval-dir "${EVAL_DIR}" \
    --merged-models-dir "${MERGED_MODELS_DIR}" \
    --force-compute
