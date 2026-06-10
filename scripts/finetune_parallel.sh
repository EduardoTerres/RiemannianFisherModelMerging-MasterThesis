#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=oft_ft
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=10:00:00
#SBATCH --array=0-14%5
#SBATCH --output=outputs/finetunes/slurm/finetune_%A_%a.out

set -e

# MODEL_FAMILY="llama3.1"
MODEL_FAMILY="qwen2.5"

mkdir -p outputs/slurm_${MODEL_FAMILY}
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate merge

python src/finetune/finetune.py \
    --model-family "$MODEL_FAMILY" \
    --task-index "$SLURM_ARRAY_TASK_ID"