#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=merging
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=01:00:00
#SBATCH --output=merging_%A.out

# MODEL_NAME="llama3.1"
MODEL_NAME="qwen2.5"
MERGE_MODE="diagonal_fisher"
MODEL_FOLDER="outputs/models/${MODEL_NAME}-${MERGE_MODE}/merged_model"
MODEL_PATH="${MODEL_FOLDER}/merged_model"
RELATIVE_PATH="../../.."

set -e 

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate OrthoMerge

python src/scripts/perform_merging.py \
    --model_family ${MODEL_NAME} \
    --merge_method "gradients" \
    --merge_mode ${MERGE_MODE} \
    --output_dir $MODEL_FOLDER \
    --save_merged_model \
    --lam 0.0 \
    --optimize_alphas adamerging \
    --device gpu


# EVAL
#
set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lm-eval
cd OrthoMerge/eval/lm-evaluation-harness

# SocialIQA, CommonsenseQA, and Minerva Math500
lm_eval --model hf \
    --tasks social_iqa,commonsense_qa,minerva_math500 \
    --model_args pretrained=${RELATIVE_PATH}/${MODEL_PATH} \
    --device cuda:0 \
    --batch_size 64 \
    --confirm_run_unsafe_code \
    --trust_remote_code


# Qwen
# python src/scripts/perform_merging.py \
#     --model_family qwen2.5 \
#     --merge_mode "standard" \
#     --output_dir outputs/models/Qwen-2.5-3B-merged-std \
#     --save_merged_model \
#     --device gpu

# bash scripts/eval.sh
