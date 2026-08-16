#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=mergeschema_fim
#SBATCH --array=0-4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=03:00:00
#SBATCH --chdir=/gpfs/home6/eterres/MasterThesis
#SBATCH --output=/gpfs/home6/eterres/MasterThesis/outputs/mergeschema_fim/slurm/mergeschema_fim_%A_%a.out

set -e

REPO_ROOT="/gpfs/home6/eterres/MasterThesis"
OUTPUT_DIR="${REPO_ROOT}/outputs/mergeschema_fim"
export MPLCONFIGDIR="${REPO_ROOT}/.matplotlib"
export XDG_CACHE_HOME="${REPO_ROOT}/.cache"

mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}" "${OUTPUT_DIR}" "${OUTPUT_DIR}/slurm"

TASK_A_LIST=(coqa nq_open meddialog_qsumm cnn_dailymail squadv2)
TASK_B_LIST=(triviaqa xsum wmt16-en-de babi mbpp)

EXTRA_ARGS=("$@")

if [[ "$#" -ge 2 && "$1" != --* && "$2" != --* ]]; then
    TASK_A="$1"
    TASK_B="$2"
    EXTRA_ARGS=("${@:3}")
else
    IDX="${SLURM_ARRAY_TASK_ID:-0}"
    TASK_A="${TASK_A_LIST[${IDX}]:-}"
    TASK_B="${TASK_B_LIST[${IDX}]:-}"
    if [[ -z "${TASK_A}" || -z "${TASK_B}" ]]; then
        echo "No task pair configured for array index ${IDX}." >&2
        exit 1
    fi
fi

PAIR_NAME="${TASK_A}__${TASK_B}"
CACHE_PATH="${OUTPUT_DIR}/${PAIR_NAME}_fim.npz"
PLOT_PATH="${OUTPUT_DIR}/merge_schema_fim_${PAIR_NAME}.png"
PDF_PATH="${OUTPUT_DIR}/merge_schema_fim_${PAIR_NAME}.pdf"

echo "Running FIM merge schema for tasks: ${TASK_A}, ${TASK_B}"
echo "Cache: ${CACHE_PATH}"
echo "Plot: ${PLOT_PATH}"
echo "PDF: ${PDF_PATH}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate merge
python "${REPO_ROOT}/src/plots/plot_mergeschema_fim.py" \
    --repo-root "${REPO_ROOT}" \
    --tasks "${TASK_A}" "${TASK_B}" \
    --cache "${CACHE_PATH}" \
    --output "${PLOT_PATH}" \
    "${EXTRA_ARGS[@]}"
