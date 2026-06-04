#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=merging
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=01:00:00
#SBATCH --output=merging_%A.out

MODEL_NAME="llama3.1"
# MODEL_NAME="qwen2.5"
MERGE_MODE="diagonal_fisher_kl_rescaled"
MODEL_FOLDER="outputs/models/${MODEL_NAME}-${MERGE_MODE}/merged_model"
MODEL_PATH="${MODEL_FOLDER}/merged_model"
RELATIVE_PATH="../../.."

set -e 

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate merge

python src/scripts/perform_merging.py \
    --model_family ${MODEL_NAME} \
    --merge_method "gradients" \
    --merge_mode ${MERGE_MODE} \
    --output_dir $MODEL_FOLDER \
    --save_merged_model \
    --lam 0.0 \
    --device gpu
