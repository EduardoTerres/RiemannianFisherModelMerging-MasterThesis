#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=loss_interpolation
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=010:00:00
#SBATCH --output=outputs/oft_neighborhood_certificate/oft_check_%A.out

set -e

REPO_ROOT="/home/eterres/MasterThesis"
cd "${REPO_ROOT}"

mkdir -p outputs/oft_neighborhood_certificate/slurm

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate merge

python -m src.analysis.oft_neighborhood_certificate \
    --model-family llama3.1 qwen2.5 \
    --output-dir outputs/oft_neighborhood_certificate

# Heavier audit:
# add --pairwise to independently check theta_max(A_m^T B_m) < pi for
# every finetune-to-finetune block geodesic.
