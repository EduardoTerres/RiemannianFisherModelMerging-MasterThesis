#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=fim
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=02:00:00
#SBATCH --array=6-14%5
#SBATCH --output=outputs/fim/slurm/finetune_%A_%a.out

set -e

REPO_ROOT="/gpfs/home6/eterres/MasterThesis"
DATASET_CACHE_DIR="${REPO_ROOT}/data/hf_cache"
mkdir -p "${DATASET_CACHE_DIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate merge

TASK_INDEX="${SLURM_ARRAY_TASK_ID:-0}"

python "${REPO_ROOT}/src/compute_fim.py" \
    --model-family llama3.1 \
    --task-index "${TASK_INDEX}" \
    --dataset-cache-dir "${DATASET_CACHE_DIR}" \
    --force-compute

# python src/compute_fim.py \
#     --model-family qwen2.5 \
#     --task-index "${TASK_INDEX}"
