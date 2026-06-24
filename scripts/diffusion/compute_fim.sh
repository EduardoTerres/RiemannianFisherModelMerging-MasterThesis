#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=fim
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=outputs/diffusion/slurms/fim_%A.out

set -e

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate orthofuse_env

export HF_HOME=/scratch-shared/eterres/huggingface-cache
export HF_HUB_CACHE="${HF_HOME}/hub"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd /home/eterres/MasterThesis/OrthoFuse

python ../src/diffusion/compute_fim.py \
  --config_path=output/concept_style/sdxl_merge/example/logs/hparams.yml \
  --output_dir=/scratch-shared/eterres/fishers \
  --all_dataset \
  --batch_size=1 \
  --repeats=500 \
  --device=cuda
