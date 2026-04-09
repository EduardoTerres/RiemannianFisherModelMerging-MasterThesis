#!/bin/bash

set -e 

source "$(conda info --base)/etc/profile.d/conda.sh"

# ScienceQA
MODEL_PATH="outputs/OrthoMerge_Llama-3.1-8B-jacobi-2/merged_model"
conda activate OrthoMerge
python eval/eval_scienceqa.py --merged_dir ${MODEL_PATH}

# SocialIQA, CommonsenseQA, and Minerva Math500
RELATIVE_PATH="../.."
MODEL_PATH="outputs/OrthoMerge_Llama-3.1-8B-jacobi-2/merged_model"
conda activate lm-eval; cd eval/lm-evaluation-harness
lm_eval --model hf \
    --tasks social_iqa,commonsense_qa,minerva_math500 \
    --model_args pretrained=${RELATIVE_PATH}/${MODEL_PATH} \
    --device cuda:0 \
    --batch_size 8 \
    --confirm_run_unsafe_code \
    --trust_remote_code


# HumanEval+ (code generation)
conda activate bigcode; cd eval/bigcode-evaluation-harness
accelerate launch  main.py \
  --model ../../models/OrthoMerge_Llama-3.1-8B_OFT_5_task \
  --max_length_generation 4096 \
  --precision bf16 \
  --tasks humanevalplus \
  --temperature 0.2 \
  --n_samples 10 \
  --batch_size 10 \
  --allow_code_execution \
  --use_auth_token