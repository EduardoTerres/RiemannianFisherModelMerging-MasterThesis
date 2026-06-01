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

mkdir -p outputs/slurm
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate OrthoMerge

# make sure debug uses only 1 run
python src/finetune/finetune.py \
    --model-family "llama3.1" \
    --debug
