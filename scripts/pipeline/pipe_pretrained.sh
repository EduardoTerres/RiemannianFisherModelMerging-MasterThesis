#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=pipe_pretrained
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=00:10:00
#SBATCH --output=pipe_pretrained_%A.out

set -e

MODEL_NAME="llama3.1"
SCRIPT_DIR="/home/eterres/MasterThesis/scripts/eval"
REPO_ROOT="/home/eterres/MasterThesis"
MODEL_PATH="${REPO_ROOT}/data/models/Llama-3.1-8B"
EVAL_DIR="${REPO_ROOT}/outputs/evaluation/pretrained/${MODEL_NAME}"

sbatch --array=0-14 "${REPO_ROOT}/scripts/eval/eval.sh" \
    "${MODEL_PATH}" \
    "${EVAL_DIR}"
