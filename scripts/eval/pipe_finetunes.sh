#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=pipe_finetunes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=00:10:00
#SBATCH --output=/dev/null

set -e

REPO_ROOT="/path/to/MasterThesis"
MODELS_ROOT="/path/to/models"

MODEL_NAME="llama3.1"
MODEL_NAME="qwen2.5"

if [ ${MODEL_NAME} == "llama3.1" ]; then
    MODEL_PATH="${MODELS_ROOT}/Llama-3.1-8B"
    ADAPTER_ROOT="${MODELS_ROOT}/Llama-3.1-8B_OFT_dataset3_adapters"
    MODEL_NAME_ADAPTERS_PREFIX="llama3-1_8b_finetune"
elif [ ${MODEL_NAME} == "qwen2.5" ]; then
    MODEL_PATH="${MODELS_ROOT}/Qwen-2.5-3B"
    ADAPTER_ROOT="${MODELS_ROOT}/Qwen-2.5-3B_OFT_dataset3_adapters"
    MODEL_NAME_ADAPTERS_PREFIX="qwen2.5_3b_finetune"
else
    echo "Unsupported model family: ${MODEL_NAME}"
    exit 1
fi

EVAL_ROOT="${REPO_ROOT}/outputs/evaluation/finetunes/${MODEL_NAME}"
SLURM_DIR="${REPO_ROOT}/outputs/evaluation/finetunes/slurm_${MODEL_NAME}"

# TASK_IDS=(0 1 2 3 4 5 6 7 8 9 10 11)
TASK_IDS=(9)
ADAPTER_TASKS=(coqa nq_open triviaqa meddialog_qsumm wmt16-en-de wikitext cnn_dailymail xsum babi squadv2 mbpp numinamath)


mkdir -p "${SLURM_DIR}"

for IDX in "${!TASK_IDS[@]}"; do
    TASK_ID="${TASK_IDS[$IDX]}"
    ADAPTER_TASK="${ADAPTER_TASKS[$TASK_ID]}"
    ADAPTER_PATH="${ADAPTER_ROOT}/${MODEL_NAME_ADAPTERS_PREFIX}_${ADAPTER_TASK}"
    EVAL_DIR="${EVAL_ROOT}"

    sbatch \
        --array="${TASK_ID}" \
        --output="${SLURM_DIR}/eval_model_%A_%a.out" \
        "${REPO_ROOT}/scripts/eval/eval.sh" \
        "${MODEL_PATH}" \
        "${EVAL_DIR}" \
        "${ADAPTER_PATH}"
done
