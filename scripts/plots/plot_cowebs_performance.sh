#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=pipe_pretrained
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=00:02:00
#SBATCH --output=/dev/null

set -e

MODEL_FAMIILY="llama3.1"  # "llama3.1" or "qwen2.5"
MODEL_FAMIILY="qwen2.5"
REPO_ROOT="/path/to/MasterThesis"
# MODELS=(pretrained finetunes standard_rescaled diagonal_fisher diagonal_fisher_std_rescaled)
MODELS=(finetunes standard_rescaled diagonal_fisher)
MODELS_LEGEND_NAMES=("Finetuned" "OrthoMerge" "Diagonal Fisher (Ours)")

python "${REPO_ROOT}/src/plots/plot_cowebs.py" \
    --model-family ${MODEL_FAMIILY} \
    --models "${MODELS[@]}" \
    --legend-names "${MODELS_LEGEND_NAMES[@]}" \
    --repo-root "${REPO_ROOT}" \
    --ordering alphabet \
    --radial-max 1 \
    # --log-scale \

    # --only-well-finetuned \
    # --well-finetuned-tolerance 0.1


MODEL_FAMIILY="qwen2.5"  # "llama3.1" or "qwen2.5"

python "${REPO_ROOT}/src/plots/plot_cowebs.py" \
    --model-family ${MODEL_FAMIILY} \
    --models "${MODELS[@]}" \
    --legend-names "${MODELS_LEGEND_NAMES[@]}" \
    --repo-root "${REPO_ROOT}" \
    --ordering alphabet \
    --radial-max 1 \
    --no-legend
    # --log-scale \
