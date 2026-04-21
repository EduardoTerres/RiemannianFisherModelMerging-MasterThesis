#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=interpolation
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=01:00:00
#SBATCH --output=interpolation_%A.out

set -e

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate OrthoMerge

python src/analysis/interpolation.py \
    --num-points 4 \
    --num-samples 64 \
    --batch-size 64 \
    --max-length 256 \
    --model-family qwen2.5 \
    --save-path outputs/interpolation