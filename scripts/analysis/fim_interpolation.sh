#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=fim_interpolation
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=010:00:00
#SBATCH --output=fim_interpolation_%A.out

set -e

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate merge

python src/analysis/fim_interpolation.py \
    --num-points 30 \
    --interpolation-start 0 \
    --interpolation-end 2 \
    --model-family qwen2.5 llama3.1 \
    --fisher-state finetuned \
    --save-path outputs/fim_interpolation \
    --plots \
    "$@"
