#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=visualization
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=01:00:00
#SBATCH --output=visualization_%A.out

set -e

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate OrthoMerge

python -m src.analysis.visualization \
    --task-names magicoder numinamath commonsense socialiqa scienceqa merged \
    --model-family llama3.1 qwen2.5 \
    --save-path outputs/visualization
