#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=gradient_analysis
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=04:00:00
#SBATCH --output=outputs/analysis/slurm/gradient_analysis_%A.out

set -e

REPO_ROOT="/gpfs/home6/eterres/MasterThesis"
OUTPUT_ROOT="${REPO_ROOT}/outputs/gradient_analysis"
DATASET_CACHE_DIR="${REPO_ROOT}/data/hf_cache"

mkdir -p "${REPO_ROOT}/outputs/analysis/slurm" "${OUTPUT_ROOT}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate merge

python "${REPO_ROOT}/src/analysis/gradients.py" \
    --family_name llama3.1 \
    --num_samples 256 \
    --batch_size 4 \
    --max_length 512 \
    --save_path "${OUTPUT_ROOT}" \
    --dataset-cache-dir "${DATASET_CACHE_DIR}"

python "${REPO_ROOT}/src/analysis/gradients.py" \
    --family_name qwen2.5 \
    --num_samples 256 \
    --batch_size 4 \
    --max_length 512 \
    --save_path "${OUTPUT_ROOT}" \
    --dataset-cache-dir "${DATASET_CACHE_DIR}"
