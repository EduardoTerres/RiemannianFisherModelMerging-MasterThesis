#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=merging
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=01:00:00
#SBATCH --output=merging_%A.out

set -e 

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate OrthoMerge

# python src/scripts/perform_merging.py \
#     --language_model_name "data/models/Llama-3.1-8B/" \
#     --adapter_paths \
#         data/models/Llama-3.1-8B_OFT_adapters/llama3-1_8b_finetune_magicoder/ \
#         data/models/Llama-3.1-8B_OFT_adapters/llama3-1_8b_finetune_numinamath/ \
#         data/models/Llama-3.1-8B_OFT_adapters/llama3-1_8b_finetune_commonsense/ \
#         data/models/Llama-3.1-8B_OFT_adapters/llama3-1_8b_finetune_socialiqa/ \
#         data/models/Llama-3.1-8B_OFT_adapters/llama3-1_8b_finetune_scienceqa/ \
#     --fisher_paths \
#         data/empirical_fishers/llama3-1_8b_finetune_magicoder.safetensors \
#         data/empirical_fishers/llama3-1_8b_finetune_numinamath.safetensors \
#         data/empirical_fishers/llama3-1_8b_finetune_commonsense.safetensors \
#         data/empirical_fishers/llama3-1_8b_finetune_socialiqa.safetensors \
#         data/empirical_fishers/llama3-1_8b_finetune_scienceqa.safetensors \
#     --output_dir data/output_models/Llama-3.1-8B-merged-fisher \
#     --merge_mode "standard" \
#     --save_merged_model \
#     --gpu 0

# bash scripts/eval.sh


# Qwen
python src/scripts/perform_merging.py \
    --language_model_name "data/models/Qwen-2.5-3B/" \
    --adapter_paths \
        data/models/Qwen-2.5-3B_OFT_adapters/qwen2.5_3b_finetune_magicoder/ \
        data/models/Qwen-2.5-3B_OFT_adapters/qwen2.5_3b_finetune_numinamath/ \
        data/models/Qwen-2.5-3B_OFT_adapters/qwen2.5_3b_finetune_commonsense/ \
        data/models/Qwen-2.5-3B_OFT_adapters/qwen2.5_3b_finetune_socialiqa/ \
        data/models/Qwen-2.5-3B_OFT_adapters/qwen2.5_3b_finetune_scienceqa/ \
    --output_dir data/output_models/Qwen-2.5-3B-merged \
    --merge_mode "standard" \
    --save_merged_model \
    --gpu 0

bash scripts/eval.sh
