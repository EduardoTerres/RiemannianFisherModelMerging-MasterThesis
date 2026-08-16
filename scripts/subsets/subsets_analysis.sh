#!/bin/bash
#SBATCH --partition=thin
#SBATCH --job-name=subset_analysis
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=outputs/analysis/subsets/subset_analysis_%A.out

set -e

REPO_ROOT="/path/to/MasterThesis"
MODEL_FAMILY="llama3.1"  # llama3.1 or qwen2.5

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate merge
cd "${REPO_ROOT}"
mkdir -p outputs/analysis/subsets

python -m src.subsets.subsets_analysis --model-family "${MODEL_FAMILY}"
