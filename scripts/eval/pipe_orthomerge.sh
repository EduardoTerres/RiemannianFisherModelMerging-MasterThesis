#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=pipe_orthomerge
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=00:10:00
#SBATCH --output=/home/eterres/MasterThesis/outputs/evaluation/standard_rescaled/slurm/pipe_orthomerge_%A.out

set -e

MODEL_NAME="llama3.1"
# MODEL_NAME="qwen2.5"

MERGE_METHOD="gradients"
MERGE_MODE="standard_rescaled"
SCRIPT_DIR="/home/eterres/MasterThesis/scripts/eval"
REPO_ROOT="/home/eterres/MasterThesis"

if [[ "${MODEL_NAME}" == "llama3.1" ]]; then
    BASE_MODEL_PATH="${REPO_ROOT}/data/models/Llama-3.1-8B"
elif [[ "${MODEL_NAME}" == "qwen2.5" ]]; then
    BASE_MODEL_PATH="${REPO_ROOT}/data/models/Qwen-2.5-3B"
else
    echo "Unknown model name: ${MODEL_NAME}"
    exit 1
fi

MODEL_DIR="${REPO_ROOT}/outputs/models/${MERGE_MODE}/${MODEL_NAME}/merged_model"
MERGED_ADAPTER_PATH="${MODEL_DIR}/merged_adapter"
EVAL_DIR="${REPO_ROOT}/outputs/evaluation/${MERGE_MODE}/${MODEL_NAME}"
SLURM_DIR="${REPO_ROOT}/outputs/evaluation/${MERGE_MODE}/slurm"

source "$(conda info --base)/etc/profile.d/conda.sh"

mkdir -p "${SLURM_DIR}"

conda activate merge

python "${REPO_ROOT}/src/scripts/perform_merging.py" \
    --model_family "${MODEL_NAME}" \
    --merge_method "${MERGE_METHOD}" \
    --merge_mode "${MERGE_MODE}" \
    --output_dir "${MODEL_DIR}" \
    --save_merged_model \
    --lam 0.0 \
    --device gpu

# sbatch \
#     --output="${SLURM_DIR}/eval_model_%A_%a.out" \
#     "${REPO_ROOT}/scripts/eval/eval.sh" \
#     "${BASE_MODEL_PATH}" \
#     "${EVAL_DIR}" \
#     "${MERGED_ADAPTER_PATH}"
