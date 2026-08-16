#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=08:00:00
#SBATCH --array=0-11%4
#SBATCH --output=eval_model_%A_%a.out

set -e

SCRIPT_DIR="/path/to/MasterThesis/scripts/eval"
REPO_ROOT="/path/to/MasterThesis"
MODEL_PATH="$1"
OUTPUT_ROOT="$2"
PEFT_MODEL="$3"
PEFT_ARGS=()

if [ -n "${PEFT_MODEL}" ]; then
    PEFT_ARGS=(--peft_model "${PEFT_MODEL}")
fi

DATASETS=(
    "coqa"
    "nq_open"
    "triviaqa"
    "meddialog_qsumm"
    "wmt16-en-de"
    "wikitext"
    "cnn_dailymail"
    "xsum"
    "babi"
    "squadv2"
    "mbpp"
    "math500"
)

LM_EVAL_TASKS=(
    "coqa"
    "nq_open"
    "triviaqa"
    "meddialog_qsumm"
    "wmt16-en-de"
    "wikitext"
    "cnn_dailymail_abisee"
    "xsum"
    "babi"
    "squadv2"
    "mbpp"
    "minerva_math500"
)

TASK_ID="${SLURM_ARRAY_TASK_ID}"

TASK_NAME="${DATASETS[$TASK_ID]}"
LM_EVAL_TASK="${LM_EVAL_TASKS[$TASK_ID]}"
EVAL_BACKEND="${EVAL_BACKENDS[$TASK_ID]}"
LM_EVAL_BATCH_SIZE=64
LM_EVAL_LIMIT_ARGS=()

# For A100 these batch sizes dont give CUDA memory error
if [ "${TASK_NAME}" = "squadv2" ]; then
    LM_EVAL_BATCH_SIZE=1
elif [ "${TASK_NAME}" = "wikitext" ] || [ "${TASK_NAME}" = "xsum" ]; then
    LM_EVAL_BATCH_SIZE=2
elif [ "${TASK_NAME}" = "meddialog_qsumm" ] || [ "${TASK_NAME}" = "cnn_dailymail" ] || [ "${TASK_NAME}" = "babi" ] || [ "${TASK_NAME}" = "mbpp" ] || [ "${TASK_NAME}" = "math500" ]; then
    LM_EVAL_BATCH_SIZE=4
fi

# If using finetune, we have less CUDA memory available
if [ -n "${PEFT_MODEL}" ]; then
    if [ "${TASK_ID}" = "0" ] || [ "${TASK_ID}" = "1" ]; then
        LM_EVAL_BATCH_SIZE=16
    fi
fi

if [ "${TASK_NAME}" = "meddialog_qsumm" ] || [ "${TASK_NAME}" = "cnn_dailymail" ] || [ "${TASK_NAME}" = "xsum" ] || [ "${TASK_NAME}" = "babi" ] || [ "${TASK_NAME}" = "squadv2" ]; then
    LM_EVAL_LIMIT_ARGS=(--limit 1000)
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

mkdir -p "${OUTPUT_ROOT}/${TASK_NAME}"


conda activate lm-eval
cd "${REPO_ROOT}/OrthoMerge/eval/lm-evaluation-harness"

echo "Evaluating ${MODEL_PATH} on ${TASK_NAME} with lm_eval task ${LM_EVAL_TASK}${PEFT_MODEL:+ and PEFT adapter ${PEFT_MODEL}}"

if [ "${TASK_NAME}" = "mbpp" ]; then
    export HF_ALLOW_CODE_EVAL=1
fi

# --gen_kwargs max_gen_toks=128 \

lm_eval --model hf \
    --tasks "${LM_EVAL_TASK}" \
    --model_args "pretrained=${MODEL_PATH}${PEFT_MODEL:+,peft=${PEFT_MODEL}}" \
    --device cuda:0 \
    --batch_size "${LM_EVAL_BATCH_SIZE}" \
    --output_path "${OUTPUT_ROOT}/${TASK_NAME}" \
    "${LM_EVAL_LIMIT_ARGS[@]}" \
    --confirm_run_unsafe_code \
    --trust_remote_code
