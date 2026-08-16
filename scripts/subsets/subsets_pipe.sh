#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=subset_merge
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=00:05:00
#SBATCH --output=outputs/evaluation_subsets/subset_merge_%A.out

set -e

REPO_ROOT="/path/to/MasterThesis"
MODEL_FAMILY="qwen2.5"  # llama3.1 or qwen2.5
MODELS_DIR="${REPO_ROOT}/outputs/models/subsets/${MODEL_FAMILY}"
SLURM_DIR="${REPO_ROOT}/outputs/evaluation_subsets/slurm"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate merge
cd "${REPO_ROOT}"
mkdir -p "${SLURM_DIR}"

# python -m src.subsets.subsets_merge \
#     --phase merge \
#     --model-family "${MODEL_FAMILY}" \
#     --models-dir "${MODELS_DIR}"

# for SEED in 1 2 3; do
for SEED in 1; do
    sbatch \
        scripts/subsets/subsets_eval.sh \
        "${SEED}" \
        "${MODEL_FAMILY}" \
        "${MODELS_DIR}" &
done

