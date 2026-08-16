#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=interp_geodesic
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=05:00:00
#SBATCH --array=0-11
#SBATCH --output=outputs/diffusion/slurms/geo_int_%A_%a.out

set -e

# CONCEPT_NAME="dog3"

REPO_ROOT="/gpfs/home6/eterres/MasterThesis"
OUTPUT_DIR="${REPO_ROOT}/outputs/diffusion"
OUTPUT_DIR="/scratch-shared/eterres/MasterThesis/outputs/diffusion"
mkdir -p "${OUTPUT_DIR}/slurms" "${OUTPUT_DIR}/samples_interpolation_geodesic"
METHOD_OUTPUT_ROOT="${OUTPUT_DIR}/samples_interpolation_geodesic"

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
  # "fisher_geodesic:diagonal::0:trace"
  # "fisher_geodesic:diagonal::4:trace"
  # "fisher:diagonal:4::trace"
  # "fisher:diagonal:0::trace"
  "orthofuse_geodesic"
  # "orthofuse_geodesic_curve_over_id"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
METHOD_COUNT="${#METHODS[@]}"
STYLE_INDEX=$((TASK_ID / METHOD_COUNT))
METHOD_INDEX=$((TASK_ID % METHOD_COUNT))
STYLE="${STYLES[${STYLE_INDEX}]}"
METHOD_SPEC="${METHODS[${METHOD_INDEX}]}"
IFS=":" read -r METHOD FISHER_BACKEND FISHER_MU CORRECTION_MU FIM_NORMALIZATION <<< "${METHOD_SPEC}"

METHOD_FLAGS=()
[[ -n "${FISHER_BACKEND}" ]] && METHOD_FLAGS+=(--fisher_backend="${FISHER_BACKEND}")
[[ -n "${FISHER_MU}" ]] && METHOD_FLAGS+=(--fisher_correction_mu="${FISHER_MU}")
[[ -n "${CORRECTION_MU}" ]] && METHOD_FLAGS+=(--correction_mu="${CORRECTION_MU}")
[[ -n "${FIM_NORMALIZATION}" ]] && METHOD_FLAGS+=(--fim_normalization="${FIM_NORMALIZATION}")

echo "[interpolation-geodesic] array_task=${TASK_ID} style=${STYLE} method=${METHOD} spec=${METHOD_SPEC}"
echo "[interpolation-geodesic] t_values=${T_VALUES[*]}"

python "${REPO_ROOT}/src/diffusion/geodesic_interpolation.py" \
  --config_path="${REPO_ROOT}/src/diffusion/config/config.yaml" \
  --output_dir="${OUTPUT_DIR}" \
  --method_output_root="${METHOD_OUTPUT_ROOT}" \
  --style_name="${STYLE}" \
  --methods "${METHOD}" \
  --prompt_templates "a {0} in {1} style" \
  --t_values "${T_VALUES[@]}" \
  --geodesic_backend=cayley \
  --num_images_per_medium_prompt=2 \
  --batch_size_medium=2 \
  --replace_inference_output \
  "${METHOD_FLAGS[@]}"

  # --concept_name="${CONCEPT_NAME}" \
