#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=eval_loss
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=01:00:00
#SBATCH --array=1,3
#SBATCH --output=/gpfs/home6/eterres/MasterThesis/outputs/eval_loss/slurm/eval_loss_%A_%a.out

set -e

REPO_ROOT="/gpfs/home6/eterres/MasterThesis"
OUTPUT_ROOT="${REPO_ROOT}/outputs/eval_loss"
DATASET_CACHE_DIR="${REPO_ROOT}/data/hf_cache"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
MODEL="llama"
LOSS_BATCH_SIZE=4
EVAL_TYPES=(pretrained finetunes)

mkdir -p "${OUTPUT_ROOT}/slurm"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate merge

for EVAL_TYPE in "${EVAL_TYPES[@]}"; do
    python "${REPO_ROOT}/src/loss/compute_eval_loss.py" \
        --eval-type "${EVAL_TYPE}" \
        --model "${MODEL}" \
        --output-root "${OUTPUT_ROOT}" \
        --dataset-cache-dir "${DATASET_CACHE_DIR}" \
        --task-id "${TASK_ID}" \
        --batch-size "${LOSS_BATCH_SIZE}"
done
