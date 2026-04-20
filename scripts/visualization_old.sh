#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=visualization
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=01:00:00
#SBATCH --output=visualization_%A.out

set -e

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate OrthoMerge

python -m src.analysis.visualization \
    --adapter_paths \
        OrthoMerge/models/Llama-3.1-8B_OFT_adapters/llama3-1_8b_finetune_magicoder/ \
        OrthoMerge/models/Llama-3.1-8B_OFT_adapters/llama3-1_8b_finetune_numinamath/ \
        OrthoMerge/models/Llama-3.1-8B_OFT_adapters/llama3-1_8b_finetune_commonsense/ \
        OrthoMerge/models/Llama-3.1-8B_OFT_adapters/llama3-1_8b_finetune_socialiqa/ \
        OrthoMerge/models/Llama-3.1-8B_OFT_adapters/llama3-1_8b_finetune_scienceqa/ \
    --fisher_paths \
        data/empirical_fishers/llama3-1_8b_finetune_magicoder.safetensors \
        data/empirical_fishers/llama3-1_8b_finetune_numinamath.safetensors \
        data/empirical_fishers/llama3-1_8b_finetune_commonsense.safetensors \
        data/empirical_fishers/llama3-1_8b_finetune_socialiqa.safetensors \
        data/empirical_fishers/llama3-1_8b_finetune_scienceqa.safetensors \
    --task_names magicoder numinamath commonsense socialiqa scienceqa \
    --lam 1.0 \
    --output outputs/task_vector_analysis.png
