#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=loss_interpolation
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=010:00:00
#SBATCH --output=loss_interpolation_%A.out

set -e

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate merge

DATASET_CACHE_DIR="${PWD}/data/hf_cache"

python src/analysis/loss_interpolation.py \
    --num-points 30 \
    --interpolation-start 0 \
    --interpolation-end 2 \
    --num-samples 256 \
    --batch-size 64 \
    --max-length 512 \
    --model-family qwen2.5 llama3.1 \
    --dataset-split test \
    --dataset-cache-dir "${DATASET_CACHE_DIR}" \
    --save-path outputs/interpolation \
    --plots \
    "$@"
