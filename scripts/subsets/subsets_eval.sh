#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=subset_eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=02:00:00
#SBATCH --array=0-11%12
#SBATCH --output=outputs/evaluation_subsets/slurm/subset_eval_%A_%a.out

set -e

SEED="${1:?Usage: subset_eval.sh SEED MODEL_FAMILY MODELS_DIR}"
MODEL_FAMILY="${2:?Usage: subset_eval.sh SEED MODEL_FAMILY MODELS_DIR}"
MODELS_DIR="${3:?Usage: subset_eval.sh SEED MODEL_FAMILY MODELS_DIR}"
REPO_ROOT="/home/eterres/MasterThesis"
OUTPUT_ROOT="${REPO_ROOT}/outputs/evaluation_subsets/${MODEL_FAMILY}"

DATASETS=(coqa nq_open triviaqa meddialog_qsumm wmt16-en-de wikitext cnn_dailymail xsum babi squadv2 mbpp math500)
TASKS=(coqa nq_open triviaqa meddialog_qsumm wmt16-en-de wikitext cnn_dailymail_abisee xsum babi squadv2 mbpp minerva_math500)
TASK_NAME="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
LM_EVAL_TASK="${TASKS[$SLURM_ARRAY_TASK_ID]}"

if [ "${MODEL_FAMILY}" = "qwen2.5" ]; then
    BASE_MODEL="${REPO_ROOT}/data/models/Qwen-2.5-3B"
elif [ "${MODEL_FAMILY}" = "llama3.1" ]; then
    BASE_MODEL="${REPO_ROOT}/data/models/Llama-3.1-8B"
else
    echo "Unknown model family: ${MODEL_FAMILY}"
    exit 1
fi

BATCH_SIZE=64
case "${TASK_NAME}" in
    squadv2) BATCH_SIZE=1 ;;
    wikitext|xsum) BATCH_SIZE=2 ;;
    meddialog_qsumm|cnn_dailymail|babi|mbpp|math500) BATCH_SIZE=4 ;;
esac

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lm-eval
cd "${REPO_ROOT}/OrthoMerge/eval/lm-evaluation-harness"

for SIZE in 2 4 6 8 10 12; do
    for MODE in standard standard_rescaled diagonal_fisher; do
        ADAPTER="${MODELS_DIR}/${MODE}_${SIZE}/merged_adapter"
        OUTPUT="${OUTPUT_ROOT}/${SIZE}/run_${SEED}/${MODE}/${TASK_NAME}"
        mkdir -p "${OUTPUT}"
        [ "${TASK_NAME}" = "mbpp" ] && export HF_ALLOW_CODE_EVAL=1
        lm_eval --model hf \
            --tasks "${LM_EVAL_TASK}" \
            --model_args "pretrained=${BASE_MODEL},peft=${ADAPTER}" \
            --device cuda:0 \
            --batch_size "${BATCH_SIZE}" \
            --limit 32 \
            --seed "${SEED}" \
            --output_path "${OUTPUT}" \
            --confirm_run_unsafe_code \
            --trust_remote_code
    done
done
