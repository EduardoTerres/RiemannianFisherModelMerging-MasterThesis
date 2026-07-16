#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=pipe_fisher
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=02:00:00
#SBATCH --array=0-4
#SBATCH --output=outputs/diffusion/slurms/pipe_fisher_%A_%a.out

set -e

MU=4  # Fisher correction mu
FISHER_BACKEND="diagonal"  # diagonal or kfac
FIM_NORMALIZATION="frobenius"  # "none", "trace", "frobenius", or "kl"

if [[ "${FISHER_BACKEND}" == "diagonal" ]]; then
  METHOD_NAME="diagonal_fisher_mu_${MU}_fim_${FIM_NORMALIZATION}"
elif [[ "${FISHER_BACKEND}" == "kfac" ]]; then
  METHOD_NAME="fisher_kfac_mu_${MU}_fim_${FIM_NORMALIZATION}"
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

STYLES=(
  # "01_01"
  # "01_02"
  # "01_03"
  # "01_07"
  # "01_08"
  # "02_03"
  # "03_04"
  "dolina"
  "etsy"
  "gondoliers"
  "image_scan"
  "pots"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
STYLE="${STYLES[${TASK_ID}]}"
ALPHA_CONCEPT=0.4  # concept alpha
ALPHA_STYLE=0.6   # style alpha
echo "[pipe_fisher] array_task=${TASK_ID} style=${STYLE} method=${METHOD_NAME} backend=${FISHER_BACKEND}"
echo "[pipe_fisher] fim_normalization=${FIM_NORMALIZATION} output_dir=${METHOD_OUTPUT_DIR}"
echo "[pipe_fisher] paths are printed by pipe_gradients after resolving style:${STYLE}"

python "${REPO_ROOT}/src/diffusion/pipe_gradients.py" \
  --config_path="${REPO_ROOT}/src/diffusion/config/config.yaml" \
  --output_dir="${METHOD_OUTPUT_DIR}" \
  --style_name="${STYLE}" \
  --merge_mode=fisher \
  --alphas "${ALPHA_CONCEPT}" "${ALPHA_STYLE}" \
  --num_images_per_medium_prompt=5 \
  --fisher_correction_mu=${MU} \
  --replace_inference_output \
  --fisher_backend "${FISHER_BACKEND}" \
  --fim_normalization="${FIM_NORMALIZATION}" \
  --fisher_rescale=1e10

  # --fisher_min=1e-14 \
