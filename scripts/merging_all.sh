#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=merging_all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=02:00:00
#SBATCH --array=0-3%4
#SBATCH --output=merging_all_%A_%a.out

set -e

REPO_DIR="$(pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate OrthoMerge

MODELS=("llama3.1" "llama3.1" "qwen2.5" "qwen2.5")
MODES=("standard" "diagonal_fisher" "standard" "diagonal_fisher")

run_merge() {
    local idx="$1"
    local model_name="${MODELS[$idx]}"
    local merge_mode="${MODES[$idx]}"
    local model_folder="outputs/models/${model_name}-${merge_mode}/merged_model"
    local merged_model_dir="${model_folder}/merged_model"

    python src/scripts/perform_merging.py \
        --model_family "${model_name}" \
        --merge_method "gradients" \
        --merge_mode "${merge_mode}" \
        --output_dir "${model_folder}" \
        --save_merged_model \
        --lam 0.0 \
        --optimize_alphas "adamerging_equal" \
        --device "gpu"

    conda activate lm-eval
    cd "${REPO_DIR}/OrthoMerge/eval/lm-evaluation-harness"

    lm_eval --model hf \
        --tasks social_iqa,commonsense_qa,minerva_math500 \
        --model_args pretrained="${REPO_DIR}/${merged_model_dir}" \
        --device cuda:0 \
        --batch_size 64 \
        --confirm_run_unsafe_code \
        --trust_remote_code

    cd "${REPO_DIR}"
    conda activate OrthoMerge
    python OrthoMerge/eval/eval_scienceqa.py \
        --merged_dir "${REPO_DIR}/${merged_model_dir}"
}

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    run_merge "${SLURM_ARRAY_TASK_ID}"
else
    for idx in "${!MODELS[@]}"; do
        run_merge "${idx}"
    done
fi
