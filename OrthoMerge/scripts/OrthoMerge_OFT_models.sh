#!/bin/bash

set -e 

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate OrthoMerge
python merge/OrthoMerge_OFT_models.py \
  --language_model_name ../data/models/Llama-3.1-8B/ \
  --adapter_paths \
      ../data/models/Llama-3.1-8B_OFT_adapters/llama3-1_8b_finetune_magicoder/ \
      ../data/models/Llama-3.1-8B_OFT_adapters/llama3-1_8b_finetune_numinamath/ \
      ../data/models/Llama-3.1-8B_OFT_adapters/llama3-1_8b_finetune_commonsense/ \
      ../data/models/Llama-3.1-8B_OFT_adapters/llama3-1_8b_finetune_socialiqa/ \
      ../data/models/Llama-3.1-8B_OFT_adapters/llama3-1_8b_finetune_scienceqa/ \
  --output_merged_adapter_dir outputs/OrthoMerge_Llama-3.1-8B-jacobi-2 \
  --save_merged_model
