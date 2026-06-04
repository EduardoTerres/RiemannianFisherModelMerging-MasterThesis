#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=eval_dataset_2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=04:00:00
#SBATCH --array=0-13
#SBATCH --output=eval_dataset_2_%A_%a.out

MODEL_PATH="outputs/models/qwen2.5-diagonal_fisher/merged_model/merged_model"
RELATIVE_PATH="../../.."
OUTPUT_ROOT="outputs/evaluation/${METHOD_NAME}"

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
    "bigcode"
)

set -e

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    SLURM_ARRAY_TASK_ID=0
fi

TASK_NAME="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
LM_EVAL_TASK="${LM_EVAL_TASKS[$SLURM_ARRAY_TASK_ID]}"
EVAL_BACKEND="${EVAL_BACKENDS[$SLURM_ARRAY_TASK_ID]}"

source "$(conda info --base)/etc/profile.d/conda.sh"

mkdir -p "${OUTPUT_ROOT}/${TASK_NAME}"

if [ "${EVAL_BACKEND}" = "bigcode" ]; then
    conda activate bigcode
    cd OrthoMerge/eval/bigcode-evaluation-harness

    echo "Evaluating ${TASK_NAME} with BigCode task ${LM_EVAL_TASK}"

    accelerate launch main.py \
        --model "${RELATIVE_PATH}/${MODEL_PATH}" \
        --max_length_generation 4096 \
        --precision bf16 \
        --tasks "${LM_EVAL_TASK}" \
        --temperature 0.2 \
        --n_samples 10 \
        --batch_size 10 \
        --metric_output_path "${RELATIVE_PATH}/${OUTPUT_ROOT}/${TASK_NAME}/metrics.json" \
        --allow_code_execution \
        --use_auth_token
else
    conda activate lm-eval
    cd OrthoMerge/eval/lm-evaluation-harness

    echo "Evaluating ${TASK_NAME} with lm_eval task ${LM_EVAL_TASK}"

    lm_eval --model hf \
        --tasks "${LM_EVAL_TASK}" \
        --model_args pretrained=${RELATIVE_PATH}/${MODEL_PATH} \
        --device cuda:0 \
        --batch_size 64 \
        --output_path "${RELATIVE_PATH}/${OUTPUT_ROOT}/${TASK_NAME}" \
        --confirm_run_unsafe_code \
        --trust_remote_code
fi
