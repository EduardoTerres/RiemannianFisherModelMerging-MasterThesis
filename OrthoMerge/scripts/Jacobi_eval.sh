#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=01:00:00
#SBATCH --output=eval_%A.out

set -e 

source "$(conda info --base)/etc/profile.d/conda.sh"

# ScienceQA
# MODEL_PATH="outputs/OrthoMerge_Llama-3.1-8B-jacobi/merged_model"
# conda activate OrthoMerge
# python eval/eval_scienceqa.py --merged_dir ${MODEL_PATH}

# SocialIQA, CommonsenseQA, and Minerva Math500
# RELATIVE_PATH="../.."
# MODEL_PATH="outputs/OrthoMerge_Llama-3.1-8B-jacobi/merged_model"
# conda activate lm-eval; cd eval/lm-evaluation-harness
# lm_eval --model hf \
#     --tasks social_iqa,commonsense_qa,minerva_math500 \
#     --model_args pretrained=${RELATIVE_PATH}/${MODEL_PATH} \
#     --device cuda:0 \
#     --batch_size 64 \
#     --confirm_run_unsafe_code \
#     --trust_remote_code


# HumanEval+ (code generation)
conda activate bigcode; cd OrthoMerge/eval/bigcode-evaluation-harness
accelerate launch  main.py \
  --model ../../outputs/OrthoMerge_Llama-3.1-8B-jacobi/merged_model \
  --max_length_generation 4096 \
  --precision bf16 \
  --tasks humanevalplus \
  --temperature 0.2 \
  --n_samples 10 \
  --batch_size 10 \
  --allow_code_execution \
  --use_auth_token