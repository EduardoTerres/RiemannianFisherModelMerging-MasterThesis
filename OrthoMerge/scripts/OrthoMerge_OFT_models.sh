#!/bin/bash

set -e 

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate OrthoMerge
python merge/OrthoMerge_OFT_models.py \
  --language_model_name models/Llama-3.1-8B-hf/ \
  --adapter_paths \
      models/Llama-3.1-8B_OFT_adapters/llama3-1_8b_finetune_magicoder/ \
      models/Llama-3.1-8B_OFT_adapters/llama3-1_8b_finetune_numinamath/ \
      models/Llama-3.1-8B_OFT_adapters/llama3-1_8b_finetune_commonsense/ \
      models/Llama-3.1-8B_OFT_adapters/llama3-1_8b_finetune_socialiqa/ \
      models/Llama-3.1-8B_OFT_adapters/llama3-1_8b_finetune_scienceqa/ \
  --output_merged_adapter_dir outputs/OrthoMerge_Llama-3.1-8B \
  --save_merged_model
