#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=interp_geodesic
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=08:00:00
#SBATCH --array=0-5
#SBATCH --output=outputs/diffusion/slurms/interpolation_geodesic_%A_%a.out

set -e

FISHER_BACKEND="${FISHER_BACKEND:-diagonal}"
CORRECTION_MU="${CORRECTION_MU:-2}"
FIM_NORMALIZATION="${FIM_NORMALIZATION:-trace}"
ORTHOFUSE_POSTPROCESSING="${ORTHOFUSE_POSTPROCESSING:-curve_over_id}"

REPO_ROOT="/gpfs/home6/eterres/MasterThesis"
OUTPUT_DIR="${REPO_ROOT}/outputs/diffusion"
METHOD_OUTPUT_ROOT="${OUTPUT_DIR}/samples_10_prompts"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate orthofuse_env

export HF_HOME=/scratch-shared/eterres/huggingface-cache
export HF_HUB_CACHE="${HF_HOME}/hub"
export TORCH_HOME="${HF_HOME}/torch"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${REPO_ROOT}/OrthoFuse:${REPO_ROOT}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/matplotlib-${USER}}"

mkdir -p "${MPLCONFIGDIR}"

CONCEPTS=(
  "cat"
  "cat2"
  "dog"
  "dog2"
  "dog3"
  "dog6"
)

T_VALUES=(
  "0.0"
  "0.2"
  "0.4"
  "0.5"
  "0.55"
  "0.6"
  "0.65"
  "0.7"
  "0.75"
  "0.8"
  "1.0"
)

METHODS=(
  "fisher_geodesic"
  "orthofuse"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
CONCEPT="${CONCEPTS[${TASK_ID}]}"

echo "[interpolation-geodesic] array_task=${TASK_ID} concept=${CONCEPT}"
echo "[interpolation-geodesic] fisher_backend=${FISHER_BACKEND} correction_mu=${CORRECTION_MU} fim_normalization=${FIM_NORMALIZATION}"
echo "[interpolation-geodesic] t_values=${T_VALUES[*]}"

python "${REPO_ROOT}/src/diffusion/geodesic_interpolation.py" \
  --config_path="${REPO_ROOT}/src/diffusion/config/config.yaml" \
  --output_dir="${OUTPUT_DIR}" \
  --method_output_root="${METHOD_OUTPUT_ROOT}" \
  --concept_name="${CONCEPT}" \
  --methods "${METHODS[@]}" \
  --t_values "${T_VALUES[@]}" \
  --geodesic_backend=cayley \
  --fisher_backend="${FISHER_BACKEND}" \
  --correction_mu="${CORRECTION_MU}" \
  --fim_normalization="${FIM_NORMALIZATION}" \
  --orthofuse_postprocessing_method="${ORTHOFUSE_POSTPROCESSING}" \
  --num_images_per_medium_prompt=2 \
  --batch_size_medium=2 \
  --replace_inference_output
