#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=fisher_analysis
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=01:00:00
#SBATCH --output=fisher_analysis_%A.out

set -e

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate OrthoMerge

python -m src.analysis.fisher_vectors \
    --family_name llama3.1 \
    --save_path outputs/fisher_analysis

python -m src.analysis.fisher_vectors \
    --family_name qwen2.5 \
    --save_path outputs/fisher_analysis
