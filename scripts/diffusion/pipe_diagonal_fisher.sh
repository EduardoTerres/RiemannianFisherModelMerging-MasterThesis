#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=pipe_fisher
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=05:00:00
#SBATCH --array=0-5
#SBATCH --output=outputs/diffusion/slurms/pipe_fisher_%A_%a.out

set -e

REPO_ROOT="/gpfs/home6/eterres/MasterThesis"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate orthofuse_env

export HF_HOME=/scratch-shared/eterres/huggingface-cache
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CONCEPTS=(
  "cat"
  "cat2"
  "dog"
  "dog2"
  "dog3"
  "dog6"
)

CONCEPT="${CONCEPTS[${SLURM_ARRAY_TASK_ID:-0}]}"
ALPHA_CONCEPT=0.4  # concept alpha
ALPHA_STYLE=0.6    # style alpha
echo "[pipe_fisher] array_task=${SLURM_ARRAY_TASK_ID:-0} concept=${CONCEPT}"

python "${REPO_ROOT}/src/diffusion/pipe_gradients.py" \
  --config_path="${REPO_ROOT}/src/diffusion/config/config.yaml" \
  --output_dir="${REPO_ROOT}/outputs/diffusion" \
  --concept_name="${CONCEPT}" \
  --merge_mode=diagonal_fisher \
  --alphas "${ALPHA_CONCEPT}" "${ALPHA_STYLE}" \
  --num_images_per_medium_prompt=10 \
  --diagonal_fisher_correction_mu=4 \
  --replace_inference_output
  
  
# --fisher_min=1e-14  
# --fisher_rescale=1e10
