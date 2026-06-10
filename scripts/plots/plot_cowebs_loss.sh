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

python "${REPO_ROOT}/src/plots/plot_cowebs.py" \
    --model-family llama \
    --models pretrained finetunes standard_rescaled \
    --repo-root "${REPO_ROOT}" \
    --plot-mode eval_loss \
    --log-scale \
    --ordering alphabet
