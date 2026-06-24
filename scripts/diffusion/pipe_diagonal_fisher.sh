#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=pipe_fisher
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=02:00:00
#SBATCH --output=outputs/diffusion/slurms/pipe_fisher_%A.out

set -e

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate orthofuse_env

export HF_HOME=/scratch-shared/eterres/huggingface-cache
export HF_HUB_CACHE="${HF_HOME}/hub"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd /home/eterres/MasterThesis/OrthoFuse

python ../src/diffusion/pipe_diagonal_fisher.py \
  --config_path=output/concept_style/sdxl_merge/example/logs/hparams.yml \
  --output_dir=/home/eterres/MasterThesis/outputs/diffusion \
  --all_dataset \
  --merge_mode=diagonal_fisher_rescaled \
  --alphas 1.0 1.0 \
  --num_images_per_medium_prompt=2 \
  --replace_inference_output \
  --fisher_min=0 \
  --fisher_rescale=1e10 \
  --debug
