#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=plot_interp_geodesic
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=04:00:00
#SBATCH --output=outputs/diffusion/slurms/plot_interpolation_geodesic_%A.out

set -e

FISHER_BACKEND="${FISHER_BACKEND:-diagonal}"
CORRECTION_MU="${CORRECTION_MU:-2}"
FISHER_MU="${FISHER_MU:-4}"
FIM_NORMALIZATION="${FIM_NORMALIZATION:-trace}"
ORTHOFUSE_POSTPROCESSING="${ORTHOFUSE_POSTPROCESSING:-curve_over_id}"
REPLACE_SCORES_CACHE="${REPLACE_SCORES_CACHE:-0}"

REPO_ROOT="/gpfs/home6/eterres/MasterThesis"
OUTPUT_DIR="${REPO_ROOT}/outputs/diffusion"
METHOD_OUTPUT_ROOT="${OUTPUT_DIR}/samples_interpolation_geodesic"

METHODS=(
  "fisher_geodesic"
  "fisher"
  "orthofuse"
)

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

mkdir -p "${MPLCONFIGDIR}" "${OUTPUT_DIR}/slurms" "${OUTPUT_DIR}/tables"

REPLACE_FLAGS=()
[[ "${REPLACE_SCORES_CACHE}" == "1" ]] && REPLACE_FLAGS+=(--replace_scores_cache)

python "${REPO_ROOT}/src/diffusion/results/interpolation_geodesic_results.py" \
  --config_path="${REPO_ROOT}/src/diffusion/config/config.yaml" \
  --output_dir="${OUTPUT_DIR}" \
  --method_output_root="${METHOD_OUTPUT_ROOT}" \
  --plot_dir="${OUTPUT_DIR}/tables" \
  --output_prefix="interpolation_geodesic" \
  --fisher_backend="${FISHER_BACKEND}" \
  --fisher_correction_mu="${FISHER_MU}" \
  --correction_mu="${CORRECTION_MU}" \
  --fim_normalization="${FIM_NORMALIZATION}" \
  --orthofuse_postprocessing_method="${ORTHOFUSE_POSTPROCESSING}" \
  --methods "${METHODS[@]}" \
  --prompt_templates "a {0} in {1} style" \
  --num_images_per_medium_prompt=5 \
  "${REPLACE_FLAGS[@]}"
