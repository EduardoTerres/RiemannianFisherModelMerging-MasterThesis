#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=plot_interp_geodesic
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=04:00:00
#SBATCH --output=outputs/diffusion/slurms/plot_interpolation_geodesic_%A.out

set -e

REPLACE_SCORES_CACHE="${REPLACE_SCORES_CACHE:-0}"

REPO_ROOT="/path/to/MasterThesis"
OUTPUT_DIR="${REPO_ROOT}/outputs/diffusion"
OUTPUT_DIR="/path/to/MasterThesis/outputs/diffusion"
METHOD_OUTPUT_ROOT="${OUTPUT_DIR}/samples_interpolation_geodesic"

METHOD_SPECS=(
  # Format for fisher methods: "method:fisher_backend:fisher_mu:correction_mu:fim_normalization"
  "fisher_geodesic:diagonal::0:trace"
  "fisher_geodesic:diagonal::2:trace"
  "fisher:diagonal:0::trace"
  "fisher:diagonal:4::trace"
  "orthofuse_geodesic"
  "orthofuse"
)

LEGEND_NAMES=(
  "Fisher geodesic (no corr.)"
  "Fisher geodesic (corr.)"
  "Fisher (no corr.)"
  "Fisher (corr.)"
  "OrthoFuse (no corr.)"
  "OrthoFuse (corr.)"
)

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate orthofuse_env

export HF_HOME=/path/to/hf_cache
export HF_HUB_CACHE="${HF_HOME}/hub"
export TORCH_HOME="${HF_HOME}/torch"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${REPO_ROOT}/OrthoFuse:${REPO_ROOT}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/matplotlib-${USER}}"

mkdir -p "${MPLCONFIGDIR}" "${OUTPUT_DIR}/slurms" "${OUTPUT_DIR}/tables"

prepare_plot_args() {
  # patch
  REPLACE_FLAGS=()
  [[ "${REPLACE_SCORES_CACHE}" == "1" ]] && REPLACE_FLAGS+=(--replace_scores_cache)
  LEGEND_FLAGS=()
  ((${#LEGEND_NAMES[@]} > 0)) && LEGEND_FLAGS+=(--legend_names "${LEGEND_NAMES[@]}")

  METHODS=()
  FISHER_BACKEND=""
  CORRECTION_MU=""
  FIM_NORMALIZATION=""

  for METHOD_SPEC in "${METHOD_SPECS[@]}"; do
    IFS=":" read -r METHOD SPEC_FISHER_BACKEND SPEC_FISHER_MU SPEC_CORRECTION_MU SPEC_FIM_NORMALIZATION <<< "${METHOD_SPEC}"

    if [[ "${METHOD}" == "fisher_geodesic" && -n "${SPEC_CORRECTION_MU}" ]]; then
      PLOT_CORRECTION_MU="${SPEC_CORRECTION_MU}"
      [[ "${PLOT_CORRECTION_MU}" == *.0 ]] && PLOT_CORRECTION_MU="${PLOT_CORRECTION_MU%.0}"
      METHODS+=("fisher_geodesic_corr${PLOT_CORRECTION_MU}")
    elif [[ "${METHOD}" == "fisher" && -n "${SPEC_FISHER_MU}" ]]; then
      PLOT_FISHER_MU="${SPEC_FISHER_MU}"
      [[ "${PLOT_FISHER_MU}" == *.0 ]] && PLOT_FISHER_MU="${PLOT_FISHER_MU%.0}"
      METHODS+=("fisher_mu${PLOT_FISHER_MU}")
    else
      METHODS+=("${METHOD}")
    fi

    [[ -z "${FISHER_BACKEND}" && -n "${SPEC_FISHER_BACKEND}" ]] && FISHER_BACKEND="${SPEC_FISHER_BACKEND}"
    [[ -z "${CORRECTION_MU}" && -n "${SPEC_CORRECTION_MU}" ]] && CORRECTION_MU="${SPEC_CORRECTION_MU}"
    [[ -z "${FIM_NORMALIZATION}" && -n "${SPEC_FIM_NORMALIZATION}" ]] && FIM_NORMALIZATION="${SPEC_FIM_NORMALIZATION}"
  done

  FISHER_BACKEND="${FISHER_BACKEND:-diagonal}"
  CORRECTION_MU="${CORRECTION_MU:-2}"
  FIM_NORMALIZATION="${FIM_NORMALIZATION:-trace}"
}

prepare_plot_args

echo "[plot-interpolation-geodesic] method_specs=${METHOD_SPECS[*]}"
echo "[plot-interpolation-geodesic] methods=${METHODS[*]}"
echo "[plot-interpolation-geodesic] fisher_backend=${FISHER_BACKEND} correction_mu=${CORRECTION_MU} fim_normalization=${FIM_NORMALIZATION}"

python "${REPO_ROOT}/src/diffusion/results/interpolation_geodesic_results.py" \
  --config_path="${REPO_ROOT}/src/diffusion/config/config.yaml" \
  --output_dir="${OUTPUT_DIR}" \
  --method_output_root="${METHOD_OUTPUT_ROOT}" \
  --plot_dir="${OUTPUT_DIR}/tables" \
  --output_prefix="interpolation_geodesic" \
  --fisher_backend="${FISHER_BACKEND}" \
  --correction_mu="${CORRECTION_MU}" \
  --fim_normalization="${FIM_NORMALIZATION}" \
  --methods "${METHODS[@]}" \
  "${LEGEND_FLAGS[@]}" \
  --prompt_templates "a {0} in {1} style" \
  --num_images_per_medium_prompt=2 \
  "${REPLACE_FLAGS[@]}"
