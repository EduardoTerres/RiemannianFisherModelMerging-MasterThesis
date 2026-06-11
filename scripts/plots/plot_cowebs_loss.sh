#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=plot_cowebs_loss
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=00:02:00
#SBATCH --output=/dev/null

set -e

REPO_ROOT="/home/eterres/MasterThesis"
MODEL_NAME="llama3.1"  # "llama3.1" or "qwen2.5"

python "${REPO_ROOT}/src/plots/plot_cowebs.py" \
    --model-family "${MODEL_NAME}" \
    --models pretrained finetunes standard standard_rescaled diagonal_fisher\
    --repo-root "${REPO_ROOT}" \
    --plot-mode eval_loss \
    --log-scale \
    --ordering alphabet
