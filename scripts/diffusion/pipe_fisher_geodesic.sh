#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=fisher_geodesic
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=06:00:00
#SBATCH --array=0-11
#SBATCH --output=outputs/diffusion/slurms/pipe_fisher_geodesic_%A_%a.out

set -e

FISHER_BACKEND="diagonal"  # "diagonal" or "kfac"
CORRECTION_MU=2
FIM_NORMALIZATION="none"  # "none", "trace", "frobenius", "layer-trace", "layer-frobenius", or "kl"

REPO_ROOT="/gpfs/home6/eterres/MasterThesis"
OUTPUT_DIR="${REPO_ROOT}/outputs/diffusion"
METHOD_OUTPUT_DIR="${OUTPUT_DIR}/samples_10_prompts/fisher_geodesic_${FISHER_BACKEND}_corr_${CORRECTION_MU}_fim_${FIM_NORMALIZATION}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate orthofuse_env

export HF_HOME=/scratch-shared/eterres/huggingface-cache
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${REPO_ROOT}/OrthoFuse:${REPO_ROOT}:${PYTHONPATH:-}"

STYLES=(
  "01_01"
  "01_02"
  "01_03"
  "01_07"
  "01_08"
  "02_03"
  "03_04"
  "dolina"
  "etsy"
  "gondoliers"
  "image_scan"
  "pots"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
STYLE="${STYLES[${TASK_ID}]}"

echo "[pipe_fisher_geodesic] array_task=${TASK_ID} style=${STYLE}"
echo "[pipe_fisher_geodesic] fisher_backend=${FISHER_BACKEND} correction_mu=${CORRECTION_MU} fim_normalization=${FIM_NORMALIZATION} output_dir=${METHOD_OUTPUT_DIR}"
echo "[pipe_fisher_geodesic] paths are printed by pipe_gradients after resolving style:${STYLE}"

python "${REPO_ROOT}/src/diffusion/pipe_fisher_geodesic.py" \
  --config_path="${REPO_ROOT}/src/diffusion/config/config.yaml" \
  --output_dir="${METHOD_OUTPUT_DIR}" \
  --samples="style:${STYLE}" \
  --t=0.6 \
  --geodesic_backend=cayley \
  --fisher_backend="${FISHER_BACKEND}" \
  --correction_mu="${CORRECTION_MU}" \
  --fim_normalization="${FIM_NORMALIZATION}" \
  --num_images_per_medium_prompt=5 \
  --replace_inference_output

  # --fisher_min=1e-14 \
  # --fisher_rescale=1e10 \
