#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=eval_model
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=04:00:00
#SBATCH --array=0-14
#SBATCH --output=eval_model_%A_%a.out

set -e

SCRIPT_DIR="/home/eterres/MasterThesis/scripts/eval"
REPO_ROOT="/home/eterres/MasterThesis"
MODEL_PATH="$1"
OUTPUT_ROOT="$2"

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

source "$(conda info --base)/etc/profile.d/conda.sh"

mkdir -p "${OUTPUT_ROOT}/${TASK_NAME}"

if [ "${EVAL_BACKEND}" = "bigcode" ]; then
    conda activate bigcode
    cd "${REPO_ROOT}/OrthoMerge/eval/bigcode-evaluation-harness"

    echo "Evaluating ${MODEL_PATH} on ${TASK_NAME} with BigCode task ${LM_EVAL_TASK}"

    accelerate launch main.py \
        --model "${MODEL_PATH}" \
        --max_length_generation 4096 \
        --precision bf16 \
        --tasks "${LM_EVAL_TASK}" \
        --temperature 0.2 \
        --n_samples 10 \
        --batch_size 10 \
        --metric_output_path "${OUTPUT_ROOT}/${TASK_NAME}/metrics.json" \
        --allow_code_execution \
        --use_auth_token
else
    conda activate lm-eval
    cd "${REPO_ROOT}/OrthoMerge/eval/lm-evaluation-harness"

    echo "Evaluating ${MODEL_PATH} on ${TASK_NAME} with lm_eval task ${LM_EVAL_TASK}"

    lm_eval --model hf \
        --tasks "${LM_EVAL_TASK}" \
        --model_args pretrained="${MODEL_PATH}" \
        --device cuda:0 \
        --batch_size 64 \
        --output_path "${OUTPUT_ROOT}/${TASK_NAME}" \
        --confirm_run_unsafe_code \
        --trust_remote_code
fi
