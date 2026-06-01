#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=interp_merge
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=01:00:00
#SBATCH --output=interpolation_merge_%A.out

set -e

MODEL_NAME="qwen2.5"
MERGE_METHOD="gradients"
MERGE_MODE="diagonal_fisher"
NUM_POINTS=5
NUM_SAMPLES=64
BATCH_SIZE=32
MAX_LENGTH=256

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate OrthoMerge

python src/analysis/interpolation_merge.py \
    --num-points ${NUM_POINTS} \
    --num-samples ${NUM_SAMPLES} \
    --batch-size ${BATCH_SIZE} \
    --max-length ${MAX_LENGTH} \
    --model-family ${MODEL_NAME} \
    --merge-method ${MERGE_METHOD} \
    --merge-mode ${MERGE_MODE} \
    --lam 0.0 \
    --n 3 \
    --save-path outputs/interpolation_merge \
    --device cuda
