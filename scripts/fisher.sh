#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=fisher
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=02:00:00
#SBATCH --output=fisher_%A.out

set -e

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate OrthoMerge

python src/fisher.py \
    --num-samples 2048 \
    --batch-size 64 \
    --max-length 64 \
    --model-family llama3.1 \
    --output-dir data/diagonal_fishers/llama3.1

python src/fisher.py \
    --num-samples 2048 \
    --batch-size 64 \
    --max-length 64 \
    --model-family qwen2.5 \
    --output-dir data/diagonal_fishers/qwen2.5
