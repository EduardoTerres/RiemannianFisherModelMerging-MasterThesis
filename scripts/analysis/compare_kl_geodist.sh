#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=kl
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=02:00:00
#SBATCH --output=outputs/analysis/slurm/kl_%A_%a.out

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate merge

python src/analysis/compare_kl_geodist.py \
    # --force-compute
