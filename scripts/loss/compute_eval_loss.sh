#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=eval_loss
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=05:00:00
#SBATCH --array=0-14
#SBATCH --output=/gpfs/home6/eterres/MasterThesis/outputs/eval_loss/slurm/eval_loss_%A_%a.out

set -e
export PYTHONNOUSERSITE=1

REPO_ROOT="/gpfs/home6/eterres/MasterThesis"
MODEL_PATH="${1:-${REPO_ROOT}/data/models/Llama-3.1-8B}"
RUN_NAME="${2:-pretrained}"
MODEL_NAME="${3:-llama3.1}"
PEFT_MODEL="${4:-}"
OUTPUT_ROOT="${REPO_ROOT}/outputs/eval_loss"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
LOSS_BATCH_SIZE=4
LOSS_ARGS=()

mkdir -p "${OUTPUT_ROOT}/slurm"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate merge

if [ -z "${PEFT_MODEL}" ] && [ "${RUN_NAME}" = "finetunes" ]; then
    PEFT_MODEL="$(
        cd "${REPO_ROOT}"
        python -c "from src.paths import MODEL_FAMILIES_D2; print(MODEL_FAMILIES_D2['${MODEL_NAME}'].adapter_paths[${TASK_ID}])"
    )"
fi

if [ -n "${PEFT_MODEL}" ]; then
    LOSS_ARGS+=(--peft-model "${PEFT_MODEL}")
fi

if [ "${TASK_ID}" = "0" ] || [ "${TASK_ID}" = "1" ]; then
    LOSS_BATCH_SIZE=16
elif [ "${TASK_ID}" = "6" ] || [ "${TASK_ID}" = "8" ]; then
    LOSS_BATCH_SIZE=2
elif [ "${TASK_ID}" = "11" ]; then
    LOSS_BATCH_SIZE=1
fi

python "${REPO_ROOT}/src/scripts/compute_eval_loss.py" \
    --model-path "${MODEL_PATH}" \
    --output-root "${OUTPUT_ROOT}" \
    --run-name "${RUN_NAME}" \
    --model-name "${MODEL_NAME}" \
    --task-id "${TASK_ID}" \
    --batch-size "${LOSS_BATCH_SIZE}" \
    "${LOSS_ARGS[@]}"
