#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=pipe_orthofuse
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=00:30:00
#SBATCH --array=0-5
#SBATCH --output=outputs/diffusion/slurms/pipe_orthofuse_%A_%a.out

set -e

REPO_ROOT="/path/to/MasterThesis"
OUTPUT_DIR="${REPO_ROOT}/outputs/diffusion"
METHOD_OUTPUT_DIR="${OUTPUT_DIR}/samples_10_prompts/orthofuse"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate orthofuse_env

export HF_HOME=/path/to/hf_cache
export HF_HUB_CACHE="${HF_HOME}/hub"
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
echo "[pipe_orthofuse] array_task=${SLURM_ARRAY_TASK_ID:-0} concept=${CONCEPT}"

# curve over id
python "${REPO_ROOT}/src/diffusion/pipe_orthofuse.py" \
  --config_path="${REPO_ROOT}/src/diffusion/config/config.yaml" \
  --output_dir="${METHOD_OUTPUT_DIR}" \
  --concept_name="${CONCEPT}" \
  --t=0.6 \
  --postprocessing_method=curve_over_id \
  --num_images_per_medium_prompt=5 \
  --replace_inference_output
  # Add --style_name "pots" above this line to restrict the run to one style.

# eigenvalue rotation
# python "${REPO_ROOT}/src/diffusion/pipe_orthofuse.py" \
#   --config_path="${REPO_ROOT}/src/diffusion/config/config.yaml" \
#   --output_dir="${METHOD_OUTPUT_DIR}" \
#   --concept_name="${CONCEPT}" \
#   --t=0.6 \
#   --postprocessing_method=rotation \
#   --num_images_per_medium_prompt=5 \
#   --replace_inference_output
