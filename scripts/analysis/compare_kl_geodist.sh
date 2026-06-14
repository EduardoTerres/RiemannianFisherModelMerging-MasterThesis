#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=fim
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=02:00:00
#SBATCH --array=0-5
#SBATCH --output=outputs/fim/slurm/finetune_%A_%a.out

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate merge

python src/analysis/compare_kl_geodist.py
