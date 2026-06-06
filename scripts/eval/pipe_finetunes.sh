#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=pipe_finetunes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=00:10:00
#SBATCH --output=/dev/null

set -e

MODEL_NAME="llama3.1"
REPO_ROOT="/home/eterres/MasterThesis"
MODEL_PATH="${REPO_ROOT}/data/models/Llama-3.1-8B"
ADAPTER_ROOT="${REPO_ROOT}/data/models/Llama-3.1-8B_OFT_dataset2_adapters"
EVAL_ROOT="${REPO_ROOT}/outputs/evaluation/finetunes/${MODEL_NAME}"
SLURM_DIR="${REPO_ROOT}/outputs/evaluation/finetunes/slurm"

# TASK_IDS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14)
TASK_IDS=(1 4 5 9 10 14)
ADAPTER_TASKS=(coqa drop nq_open triviaqa meddialog_qsumm wmt16-en-de wikitext cnn_dailymail xsum gsm8k babi squadv2 mbpp numinamath magicoder)


mkdir -p "${SLURM_DIR}"

for IDX in "${!TASK_IDS[@]}"; do
    TASK_ID="${TASK_IDS[$IDX]}"
    ADAPTER_TASK="${ADAPTER_TASKS[$TASK_ID]}"
    ADAPTER_PATH="${ADAPTER_ROOT}/llama3-1_8b_finetune_${ADAPTER_TASK}"
    EVAL_DIR="${EVAL_ROOT}"

    sbatch \
        --array="${TASK_ID}" \
        --output="${SLURM_DIR}/eval_model_%A_%a.out" \
        "${REPO_ROOT}/scripts/eval/eval.sh" \
        "${MODEL_PATH}" \
        "${EVAL_DIR}" \
        "${ADAPTER_PATH}"
done
