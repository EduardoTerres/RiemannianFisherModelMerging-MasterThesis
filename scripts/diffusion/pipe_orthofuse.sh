#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=pipe_orthofuse
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=02:00:00
#SBATCH --output=outputs/diffusion/slurms/pipe_orthofuse_%A.out

set -e

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate orthofuse_env

export HF_HOME=/scratch-shared/eterres/huggingface-cache
export HF_HUB_CACHE="${HF_HOME}/hub"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd /home/eterres/MasterThesis/OrthoFuse

python ../src/diffusion/pipe_orthofuse.py \
  --config_path=output/concept_style/sdxl_merge/example/logs/hparams.yml \
  --output_dir=/home/eterres/MasterThesis/outputs/diffusion \
  --all_dataset \
  --t=0.6 \
  --num_images_per_medium_prompt=2 \
  --replace_inference_output \
  --debug
