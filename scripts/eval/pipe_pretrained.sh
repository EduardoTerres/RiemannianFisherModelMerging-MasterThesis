#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=pipe_pretrained
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=00:10:00
#SBATCH --output=/dev/null

set -e

MODEL_NAME="llama3.1"
MODEL_NAME="qwen2.5"

SCRIPT_DIR="/home/eterres/MasterThesis/scripts/eval"
REPO_ROOT="/home/eterres/MasterThesis"

MODEL_PATH="${REPO_ROOT}/data/models/Llama-3.1-8B"
MODEL_PATH="${REPO_ROOT}/data/models/Qwen-2.5-3B"

EVAL_DIR="${REPO_ROOT}/outputs/evaluation/pretrained/${MODEL_NAME}"
SLURM_DIR="${REPO_ROOT}/outputs/evaluation/pretrained/slurm_${MODEL_NAME}"



mkdir -p "${SLURM_DIR}"

sbatch \
    --output="${SLURM_DIR}/eval_model_%A_%a.out" \
    "${REPO_ROOT}/scripts/eval/eval.sh" \
    "${MODEL_PATH}" \
    "${EVAL_DIR}"
