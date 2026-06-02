#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=oft_ft
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=08:00:00
#SBATCH --array=0-11%4
#SBATCH --output=outputs/slurm/finetune_%A_%a.out

set -e

DEBUG=0
MODEL_FAMILY="llama3.1"

mkdir -p outputs/slurm
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate merge

if [[ "$DEBUG" == "1" && "$SLURM_ARRAY_TASK_ID" != "0" ]]; then
    exit 0
fi

EXTRA_ARGS=""
if [[ "$DEBUG" == "1" ]]; then
    EXTRA_ARGS="--debug"
fi

python src/finetune/finetune.py \
    --model-family "$MODEL_FAMILY" \
    --task-index "$SLURM_ARRAY_TASK_ID"
