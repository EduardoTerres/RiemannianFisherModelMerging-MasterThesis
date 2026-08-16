#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=pipe_baselines
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=02:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/path/to/MasterThesis/outputs/evaluation/baselines/slurm/pipe_baselines_%A_%a.out

set -euo pipefail

REPO_ROOT="/path/to/MasterThesis"
SLURM_DIR="${REPO_ROOT}/outputs/evaluation/baselines/slurm"

MODEL_NAMES=(
    # "llama3.1" "qwen2.5"
    # "llama3.1" "qwen2.5"
    # "llama3.1" "qwen2.5"
    "llama3.1" "qwen2.5"
)
MERGE_METHODS=(
    # "orthomerge_c_ties" "orthomerge_c_ties"
    # "orthomerge_c_tsvm" "orthomerge_c_tsvm"
    "adamerging" "adamerging"
    # "wudi" "wudi"
)
mkdir -p "${SLURM_DIR}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

run_one() {
    local idx="$1"
    local model_name="${MODEL_NAMES[$idx]}"
    local merge_method="${MERGE_METHODS[$idx]}"
    local merge_mode="standard"
    local base_model_path

    if [[ "${model_name}" == "llama3.1" ]]; then
        base_model_path="/path/to/models/Llama-3.1-8B"
    elif [[ "${model_name}" == "qwen2.5" ]]; then
        base_model_path="/path/to/models/Qwen-2.5-3B"
    else
        echo "Unknown model name: ${model_name}"
        exit 1
    fi

    local model_dir="${REPO_ROOT}/outputs/models/${merge_method}/${model_name}/merged_model"
    local merged_adapter_path="${model_dir}/merged_adapter"
    local eval_dir="${REPO_ROOT}/outputs/evaluation/${merge_method}/${model_name}"

    log "Starting baseline job ${idx}: method=${merge_method}, mode=${merge_mode}, model=${model_name}"
    log "Repo root: ${REPO_ROOT}"
    log "Base model: ${base_model_path}"
    log "Output model dir: ${model_dir}"
    log "Eval dir: ${eval_dir}"

    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate merge

    log "Running merge: ${merge_method} on ${model_name}"
    python "${REPO_ROOT}/src/scripts/perform_merging.py" \
        --model_family "${model_name}" \
        --merge_method "${merge_method}" \
        --merge_mode "${merge_mode}" \
        --output_dir "${model_dir}" \
        --save_merged_model \
        --lam 0.0 \
        --device gpu
    log "Finished merge: ${merge_method} on ${model_name}"

    log "Submitting eval array for method=${merge_method}, model=${model_name}"
    eval_job_id=$(sbatch --parsable \
        --array=0-11%4 \
        --output="${SLURM_DIR}/eval_${merge_method}_${model_name}_%A_%a.out" \
        "${REPO_ROOT}/scripts/eval/eval.sh" \
        "${base_model_path}" \
        "${eval_dir}" \
        "${merged_adapter_path}")
    log "Submitted eval job ${eval_job_id} for method=${merge_method}, model=${model_name}"
}

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    log "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
    run_one "${SLURM_ARRAY_TASK_ID}"
else
    for idx in "${!MODEL_NAMES[@]}"; do
        run_one "${idx}"
    done
fi
