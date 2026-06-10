#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=pipe_pretrained
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=00:02:00
#SBATCH --output=/dev/null

set -e

MODEL_FAMIILY="qwen2.5"
REPO_ROOT="/home/eterres/MasterThesis"
MODELS=(pretrained finetunes standard_rescaled)
MODELS=(pretrained finetunes)

python "${REPO_ROOT}/src/plots/plot_cowebs.py" \
    --model-family ${MODEL_FAMIILY} \
    --models "${MODELS[@]}" \
    --repo-root "${REPO_ROOT}" \
    --ordering alphabet \
    --log-scale \
    --well-finetuned-tolerance 0.1

# --only-well-finetuned \
