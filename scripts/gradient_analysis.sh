#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=gradient_analysis
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=04:00:00
#SBATCH --output=gradient_analysis_%A.out

set -e

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate OrthoMerge

python -m src.analysis.gradients \
    --family_name llama3.1 \
    --num_samples 256 \
    --batch_size 4 \
    --max_length 512 \
    --save_path outputs/gradient_analysis

python -m src.analysis.gradients \
    --family_name qwen2.5 \
    --num_samples 256 \
    --batch_size 4 \
    --max_length 512 \
    --save_path outputs/gradient_analysis
