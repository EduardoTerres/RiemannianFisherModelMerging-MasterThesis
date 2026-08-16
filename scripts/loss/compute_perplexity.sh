#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=perplexity
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=01:00:00
#SBATCH --array=0-11%4
#SBATCH --output=/path/to/MasterThesis/outputs/perplexity/slurm/perplexity_%A_%a.out

set -e
REPO_ROOT="/path/to/MasterThesis"
OUTPUT_ROOT="${REPO_ROOT}/outputs/perplexity"
DATASET_CACHE_DIR="${REPO_ROOT}/data/hf_cache"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
MODEL="llama3.1"  # "llama3.1" or "qwen2.5"
LOSS_BATCH_SIZE=4
MODEL_SOURCE="adapter"
MERGE_MODE="standard_rescaled"
PEFT_MODEL="${REPO_ROOT}/outputs/models/${MERGE_MODE}/${MODEL}/merged_model/merged_adapter"
RUN_NAME="${MERGE_MODE}"

mkdir -p "${OUTPUT_ROOT}/slurm"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate merge

# # Pretrained baseline:
# python "${REPO_ROOT}/src/loss/compute_eval_loss.py" \
#     --model-source "pretrained" \
#     --model "${MODEL}" \
#     --output-root "${OUTPUT_ROOT}" \
#     --dataset-cache-dir "${DATASET_CACHE_DIR}" \
#     --task-id "${TASK_ID}" \
#     --batch-size "${LOSS_BATCH_SIZE}"

# # Per-task finetuned adapters
# python "${REPO_ROOT}/src/loss/compute_eval_loss.py" \
#     --model-source "finetunes" \
#     --model "${MODEL}" \
#     --output-root "${OUTPUT_ROOT}" \
#     --dataset-cache-dir "${DATASET_CACHE_DIR}" \
#     --task-id "${TASK_ID}" \
#     --batch-size "${LOSS_BATCH_SIZE}"

# Merged model
python "${REPO_ROOT}/src/loss/compute_eval_loss.py" \
    --model-source "${MODEL_SOURCE}" \
    --model "${MODEL}" \
    --output-root "${OUTPUT_ROOT}" \
    --dataset-cache-dir "${DATASET_CACHE_DIR}" \
    --task-id "${TASK_ID}" \
    --batch-size "${LOSS_BATCH_SIZE}" \
    --peft-model "${PEFT_MODEL}" \
    --run-name "${RUN_NAME}"

