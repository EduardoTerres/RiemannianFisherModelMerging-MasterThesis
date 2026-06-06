#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=pipe_pretrained
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=00:02:00
#SBATCH --output=/dev/null

set -e

REPO_ROOT="/home/eterres/MasterThesis"

if [ "$#" -eq 0 ]; then
    set -- --model-family llama --models pretrained finetunes
fi

python "${REPO_ROOT}/src/plots/plot_cowebs.py" --repo-root "${REPO_ROOT}" "$@"
