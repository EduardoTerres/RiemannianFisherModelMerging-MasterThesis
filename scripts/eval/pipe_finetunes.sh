#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=eval_finetunes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=04:00:00
#SBATCH --array=0-8
#SBATCH --output=eval_finetunes_%A_%a.out

set -e

MODEL_NAME="llama3.1"
REPO_ROOT="/home/eterres/MasterThesis"
BASE_MODEL_PATH="${REPO_ROOT}/data/models/Llama-3.1-8B"
ADAPTER_ROOT="${REPO_ROOT}/data/models/Llama-3.1-8B_OFT_dataset2_adapters"
OUTPUT_ROOT="${REPO_ROOT}/outputs/evaluation/finetunes/${MODEL_NAME}"

TASK_IDS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14)
TASKS=(coqa drop nq_open triviaqa meddialog_qsumm wmt16-en-de wikitext cnn_dailymail xsum gsm8k babi squadv2 mbpp numinamath magicoder)

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

TASK="${TASKS[$SLURM_ARRAY_TASK_ID]}"
TASK_ID="${TASK_IDS[$SLURM_ARRAY_TASK_ID]}"
DATASET="${DATASETS[$TASK_ID]}"
LM_EVAL_TASK="${LM_EVAL_TASKS[$TASK_ID]}"
EVAL_BACKEND="${EVAL_BACKENDS[$TASK_ID]}"
ADAPTER_PATH="${ADAPTER_ROOT}/llama3-1_8b_finetune_${TASK}"
EVAL_DIR="${OUTPUT_ROOT}/${TASK}"

source "$(conda info --base)/etc/profile.d/conda.sh"

mkdir -p "${EVAL_DIR}/${DATASET}"

if [ "${EVAL_BACKEND}" = "bigcode" ]; then
    conda activate bigcode
    cd "${REPO_ROOT}/OrthoMerge/eval/bigcode-evaluation-harness"

    accelerate launch main.py \
        --model "${BASE_MODEL_PATH}" \
        --peft_model "${ADAPTER_PATH}" \
        --max_length_generation 4096 \
        --precision bf16 \
        --tasks "${LM_EVAL_TASK}" \
        --temperature 0.2 \
        --n_samples 10 \
        --batch_size 10 \
        --metric_output_path "${EVAL_DIR}/${DATASET}/metrics.json" \
        --allow_code_execution \
        --use_auth_token
else
    conda activate lm-eval
    cd "${REPO_ROOT}/OrthoMerge/eval/lm-evaluation-harness"

    lm_eval --model hf \
        --tasks "${LM_EVAL_TASK}" \
        --model_args "pretrained=${BASE_MODEL_PATH},peft=${ADAPTER_PATH}" \
        --device cuda:0 \
        --batch_size 64 \
        --output_path "${EVAL_DIR}/${DATASET}" \
        --confirm_run_unsafe_code \
        --trust_remote_code
fi
