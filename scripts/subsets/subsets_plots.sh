#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=subset_plots
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=00:05:00
#SBATCH --output=outputs/evaluation_subsets/subset_plots_%A.out

set -e

REPO_ROOT="/home/eterres/MasterThesis"
MODEL_FAMILIES=(llama3.1 qwen2.5)  # Use either one family or both.
SLURM_DIR="${REPO_ROOT}/outputs/evaluation_subsets/slurm"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate merge
cd "${REPO_ROOT}"

python -m src.subsets.subsets_merge \
    --phase results \
    --model-family "${MODEL_FAMILIES[@]}"
