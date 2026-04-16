#!/bin/bash

python src/scripts/perform_merging.py \
    --language_model_name "OrthoMerge/models/Llama-3.1-8B/" \
    --adapter_paths \
        OrthoMerge/models/Llama-3.1-8B_OFT_adapters/llama3-1_8b_finetune_magicoder/ \
        OrthoMerge/models/Llama-3.1-8B_OFT_adapters/llama3-1_8b_finetune_numinamath/ \
        OrthoMerge/models/Llama-3.1-8B_OFT_adapters/llama3-1_8b_finetune_commonsense/ \
        OrthoMerge/models/Llama-3.1-8B_OFT_adapters/llama3-1_8b_finetune_socialiqa/ \
        OrthoMerge/models/Llama-3.1-8B_OFT_adapters/llama3-1_8b_finetune_scienceqa/ \
    --fisher_paths \
        data/empirical_diagonal_fishers/llama3-1_8b_finetune_magicoder.safetensors \
        data/empirical_diagonal_fishers/llama3-1_8b_finetune_numinamath.safetensors \
        data/empirical_diagonal_fishers/llama3-1_8b_finetune_commonsense.safetensors \
        data/empirical_diagonal_fishers/llama3-1_8b_finetune_socialiqa.safetensors \
        data/empirical_diagonal_fishers/llama3-1_8b_finetune_scienceqa.safetensors \
    --output_dir outputs/OrthoMerge_Llama-3.1-8B-fisher \
    --merge_mode "diagonal_fisher" \
    --save_merged_model \
    --gpu 0
