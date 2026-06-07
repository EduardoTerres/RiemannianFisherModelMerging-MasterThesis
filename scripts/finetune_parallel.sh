#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=oft_ft
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=10:00:00
#SBATCH --array=12
#SBATCH --output=outputs/slurm/finetune_%A_%a.out

set -e

MODEL_FAMILY="llama3.1"

mkdir -p outputs/slurm
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate merge

python src/finetune/finetune.py \
    --model-family "$MODEL_FAMILY" \
    --task-index "$SLURM_ARRAY_TASK_ID"