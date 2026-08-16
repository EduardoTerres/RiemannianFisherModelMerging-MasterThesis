#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=fim
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=04:00:00
#SBATCH --array=5
#SBATCH --output=outputs/fim/slurm/finetune_%A_%a.out

set -e

REPO_ROOT="/path/to/MasterThesis"
DATASET_CACHE_DIR="${REPO_ROOT}/data/hf_cache"
MATERIALIZED_ADAPTERS_DIR="/path/to/materialized_adapters"
mkdir -p "${DATASET_CACHE_DIR}"
mkdir -p "${MATERIALIZED_ADAPTERS_DIR}"

MODEL_NAME="qwen2.5"  # "llama3.1" or "qwen2.5"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate merge

TASK_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
EXTRA_ARGS=()
if [[ "${TASK_INDEX}" == "5" ]]; then
    EXTRA_ARGS+=(--repeat-to-num-samples)
fi

python "${REPO_ROOT}/src/compute_fim.py" \
    --model-family "${MODEL_NAME}" \
    --task-index "${TASK_INDEX}" \
    --num-samples 4098 \
    --batch-size 1 \
    --dataset-cache-dir "${DATASET_CACHE_DIR}" \
    --materialized-adapters-dir "${MATERIALIZED_ADAPTERS_DIR}" \
    --force-compute \
    "${EXTRA_ARGS[@]}"
