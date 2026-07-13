#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=fisher_geodesic
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=03:00:00
#SBATCH --array=0-5
#SBATCH --output=outputs/diffusion/slurms/pipe_fisher_geodesic_%A_%a.out

set -e

MU=4  # correction
FISHER_BACKEND="diagonal"  # "diagonal" or "kfac"

REPO_ROOT="/gpfs/home6/eterres/MasterThesis"
OUTPUT_DIR="${REPO_ROOT}/outputs/diffusion"
METHOD_OUTPUT_DIR="${OUTPUT_DIR}/samples/fisher_geodesic_${FISHER_BACKEND}_mu_${MU}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate orthofuse_env

export HF_HOME=/scratch-shared/eterres/huggingface-cache
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${REPO_ROOT}/OrthoFuse:${REPO_ROOT}:${PYTHONPATH:-}"

CONCEPTS=(
  "cat"
  "cat2"
  "dog"
  "dog2"
  "dog3"
  "dog6"
)

CONCEPT="${CONCEPTS[${SLURM_ARRAY_TASK_ID:-0}]}"

echo "[pipe_fisher_geodesic] array_task=${SLURM_ARRAY_TASK_ID:-0} concept=${CONCEPT}"
echo "[pipe_fisher_geodesic] fisher_backend=${FISHER_BACKEND} output_dir=${METHOD_OUTPUT_DIR}"

python "${REPO_ROOT}/src/diffusion/pipe_fisher_geodesic.py" \
  --config_path="${REPO_ROOT}/src/diffusion/config/config.yaml" \
  --output_dir="${METHOD_OUTPUT_DIR}" \
  --concept_name="${CONCEPT}" \
  --t=0.6 \
  --geodesic_backend=cayley \
  --fisher_backend="${FISHER_BACKEND}" \
  --num_images_per_medium_prompt=10 \
  --replace_inference_output \
  --fisher_correction_mu=${MU}

  # --fisher_min=1e-14 \
  # --fisher_rescale=1e10 \
