#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --job-name=oft_info
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=outputs/diffusion/slurms/oft_info_%A.out

set -e

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate orthofuse_env

QWEN_ADAPTER="/path/to/models/Qwen-2.5-3B_OFT_dataset3_adapters/qwen2.5_3b_finetune_mbpp"
LLAMA_ADAPTER="/path/to/models/Llama-3.1-8B_OFT_dataset3_adapters/llama3-1_8b_finetune_wikitext"
DIFFUSION_ADAPTER="/path/to/sdxl_data/concepts/adapters/pytorch_lora_weights_cat.safetensors"

python -m src.scripts.oft_info \
  --qwen-adapter="${QWEN_ADAPTER}" \
  --llama-adapter="${LLAMA_ADAPTER}" \
  --diffusion-adapter="${DIFFUSION_ADAPTER}" \
  --output=outputs/oft_info/oft_info.txt
