#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=02:00:00
#SBATCH --array=0-14%5
#SBATCH --output=eval_model_%A_%a.out

set -e

SCRIPT_DIR="/home/eterres/MasterThesis/scripts/eval"
REPO_ROOT="/home/eterres/MasterThesis"
MODEL_PATH="$1"
OUTPUT_ROOT="$2"
PEFT_MODEL="$3"
PEFT_ARGS=()

if [ -n "${PEFT_MODEL}" ]; then
    PEFT_ARGS=(--peft_model "${PEFT_MODEL}")
fi

DATASETS=(
    "coqa"
    "drop"
    "nq_open"
    "triviaqa"
    "meddialog_qsumm"
    "wmt16-en-de"
    "wikitext"
    "cnn_dailymail"
    "xsum"
    "gsm8k"
    "babi"
    "squadv2"
    "mbpp"
    "math500"
    "humanevalplus"
)

LM_EVAL_TASKS=(
    "coqa"
    "drop"
    "nq_open"
    "triviaqa"
    "meddialog_qsumm"
    "wmt16-en-de"
    "wikitext"
    "cnn_dailymail_abisee"
    "xsum"
    "gsm8k"
    "babi"
    "squadv2"
    "mbpp"
    "minerva_math500"
    "humanevalplus"
)

EVAL_BACKENDS=(
    "lm_eval"
    "lm_eval"
    "lm_eval"
    "lm_eval"
    "lm_eval"
    "lm_eval"
    "lm_eval"
    "lm_eval"
    "lm_eval"
    "lm_eval"
    "lm_eval"
    "lm_eval"
    "lm_eval"
    "lm_eval"
    "bigcode"
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
elif [ "${TASK_NAME}" = "meddialog_qsumm" ] || [ "${TASK_NAME}" = "cnn_dailymail" ] || [ "${TASK_NAME}" = "gsm8k" ] || [ "${TASK_NAME}" = "babi" ] || [ "${TASK_NAME}" = "mbpp" ] || [ "${TASK_NAME}" = "math500" ]; then
    LM_EVAL_BATCH_SIZE=4
fi

# If using finetune, we have less CUDA memory available
if [ -n "${PEFT_MODEL}" ]; then
    if [ "${TASK_ID}" = "0" ] || [ "${TASK_ID}" = "1" ]; then
        LM_EVAL_BATCH_SIZE=16
    fi
fi

# Dev-only shortcut. remember to remove/empty this block for final benchmark numbers.
# 1 4 7 8
if [ "${TASK_ID}" = "1" ] || [ "${TASK_ID}" = "4" ] || [ "${TASK_ID}" = "7" ] || [ "${TASK_ID}" = "8" ] ; then
    LM_EVAL_LIMIT_ARGS=(--limit 500)
fi

# 10 11
if [ "${TASK_ID}" = "10" ] || [ "${TASK_ID}" = "11" ]; then
    LM_EVAL_LIMIT_ARGS=(--limit 5000)
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

mkdir -p "${OUTPUT_ROOT}/${TASK_NAME}"

if [ "${EVAL_BACKEND}" = "bigcode" ]; then
    conda activate bigcode
    cd "${REPO_ROOT}/OrthoMerge/eval/bigcode-evaluation-harness"

    echo "Evaluating ${MODEL_PATH} on ${TASK_NAME} with BigCode task ${LM_EVAL_TASK}${PEFT_MODEL:+ and PEFT adapter ${PEFT_MODEL}}"

    accelerate launch main.py \
        --model "${MODEL_PATH}" \
        "${PEFT_ARGS[@]}" \
        --max_length_generation 4096 \
        --precision bf16 \
        --tasks "${LM_EVAL_TASK}" \
        --temperature 0.2 \
        --n_samples 10 \
        --batch_size 6 \
        --metric_output_path "${OUTPUT_ROOT}/${TASK_NAME}/metrics.json" \
        --allow_code_execution \
        --use_auth_token
else
    conda activate lm-eval
    cd "${REPO_ROOT}/OrthoMerge/eval/lm-evaluation-harness"

    echo "Evaluating ${MODEL_PATH} on ${TASK_NAME} with lm_eval task ${LM_EVAL_TASK}${PEFT_MODEL:+ and PEFT adapter ${PEFT_MODEL}}"

    # otherwise complains
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
fi
