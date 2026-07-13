#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=pipe_fisher
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=02:00:00
#SBATCH --array=0-5
#SBATCH --output=outputs/diffusion/slurms/pipe_fisher_%A_%a.out

set -e

MU=0  # Fisher correction mu
FISHER_BACKEND="diagonal"  # diagonal or kfac

if [[ "${FISHER_BACKEND}" == "diagonal" ]]; then
  METHOD_NAME="diagonal_fisher_mu_${MU}"
elif [[ "${FISHER_BACKEND}" == "kfac" ]]; then
  METHOD_NAME="fisher_kfac_mu_${MU}"
else
  echo "Unsupported FISHER_BACKEND=${FISHER_BACKEND}" >&2
  exit 1
fi

REPO_ROOT="/gpfs/home6/eterres/MasterThesis"
OUTPUT_DIR="${REPO_ROOT}/outputs/diffusion"
METHOD_OUTPUT_DIR="${OUTPUT_DIR}/samples/${METHOD_NAME}"

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
ALPHA_CONCEPT=0.3  # concept alpha
ALPHA_STYLE=0.7    # style alpha
echo "[pipe_fisher] array_task=${SLURM_ARRAY_TASK_ID:-0} concept=${CONCEPT} method=${METHOD_NAME} backend=${FISHER_BACKEND}"

python "${REPO_ROOT}/src/diffusion/pipe_gradients.py" \
  --config_path="${REPO_ROOT}/src/diffusion/config/config.yaml" \
  --output_dir="${METHOD_OUTPUT_DIR}" \
  --concept_name="${CONCEPT}" \
  --merge_mode=fisher \
  --alphas "${ALPHA_CONCEPT}" "${ALPHA_STYLE}" \
  --num_images_per_medium_prompt=10 \
  --fisher_correction_mu=${MU} \
  --replace_inference_output \
  --fisher_backend "${FISHER_BACKEND}"
  
  
# --fisher_min=1e-14  
# --fisher_rescale=1e10
