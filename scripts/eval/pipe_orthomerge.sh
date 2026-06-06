#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=pipe_orthomerge
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=02:00:00
#SBATCH --output=pipe_orthomerge_%A.out

set -e

MODEL_NAME="llama3.1"
MERGE_METHOD="gradients"
MERGE_MODE="standard_rescaled"
METHOD_NAME="orthomerge_rescaled"
NUM_RUNS=1
SCRIPT_DIR="/home/eterres/MasterThesis/scripts/eval"
REPO_ROOT="/home/eterres/MasterThesis"

source "$(conda info --base)/etc/profile.d/conda.sh"

for ((RUN_ID = 0; RUN_ID < NUM_RUNS; RUN_ID++)); do
    RUN_NAME="run_${RUN_ID}"
    MODEL_DIR="${REPO_ROOT}/outputs/models/${MODEL_NAME}-${MERGE_MODE}/${RUN_NAME}/merged_model"
    MODEL_PATH="${MODEL_DIR}/merged_model"
    EVAL_DIR="${REPO_ROOT}/outputs/evaluation/${METHOD_NAME}/${RUN_NAME}"

    conda activate merge

    python "${REPO_ROOT}/src/scripts/perform_merging.py" \
        --model_family "${MODEL_NAME}" \
        --merge_method "${MERGE_METHOD}" \
        --merge_mode "${MERGE_MODE}" \
        --output_dir "${MODEL_DIR}" \
        --save_merged_model \
        --lam 0.0 \
        --device gpu

    sbatch --array=0-14 "${REPO_ROOT}/scripts/eval/eval.sh" \
        "${MODEL_PATH}" \
        "${EVAL_DIR}"
done
