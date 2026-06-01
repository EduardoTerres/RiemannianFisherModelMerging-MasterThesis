#!/bin/bash

set -e 

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate OrthoMerge
python merge/Jacobi_merge.py \
  --language_model_name ../data/models/Llama-3.1-8B/ \
  --adapter_paths \
      ../data/models/Llama-3.1-8B_OFT_adapters/llama3-1_8b_finetune_magicoder/ \
      ../data/models/Llama-3.1-8B_OFT_adapters/llama3-1_8b_finetune_numinamath/ \
      ../data/models/Llama-3.1-8B_OFT_adapters/llama3-1_8b_finetune_commonsense/ \
      ../data/models/Llama-3.1-8B_OFT_adapters/llama3-1_8b_finetune_socialiqa/ \
      ../data/models/Llama-3.1-8B_OFT_adapters/llama3-1_8b_finetune_scienceqa/ \
  --fisher_paths \
      ../data/diagonal_transported_fishers/llama3.1/llama3-1_8b_finetune_magicoder.safetensors \
      ../data/diagonal_transported_fishers/llama3.1/llama3-1_8b_finetune_numinamath.safetensors \
      ../data/diagonal_transported_fishers/llama3.1/llama3-1_8b_finetune_commonsense.safetensors \
      ../data/diagonal_transported_fishers/llama3.1/llama3-1_8b_finetune_socialiqa.safetensors \
      ../data/diagonal_transported_fishers/llama3.1/llama3-1_8b_finetune_scienceqa.safetensors \
  --output_merged_adapter_dir outputs/OrthoMerge_Llama-3.1-8B-jacobi \
  --save_merged_model

bash scripts/Jacobi_eval.sh