#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=pipe_orthomerge
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=02:00:00
#SBATCH --output=outputs/diffusion/slurms/pipe_orthomerge_%A.out

set -e

REPO_ROOT="/gpfs/home6/eterres/MasterThesis"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate orthofuse_env

export HF_HOME=/scratch-shared/eterres/huggingface-cache
export HF_HUB_CACHE="${HF_HOME}/hub"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

ALPHA_CONCEPT=1.0  # concept alpha
ALPHA_STYLE=1.0    # style alpha

python "${REPO_ROOT}/src/diffusion/pipe_gradients.py" \
  --config_path="${REPO_ROOT}/src/diffusion/config/config.yaml" \
  --output_dir="${REPO_ROOT}/outputs/diffusion" \
  --samples all_dataset_pairs \
  --merge_mode=standard_rescaled \
  --alphas "${ALPHA_CONCEPT}" "${ALPHA_STYLE}" \
  --num_images_per_medium_prompt=2 \
  --replace_inference_output
