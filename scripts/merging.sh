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

python src/scripts/perform_merging.py \
    --model_family llama3.1 \
    --merge_method "gradients" \
    --merge_mode "diagonal_fisher" \
    --output_dir outputs/models/Llama-3.1-8B-merged \
    --save_merged_model \
    --lam 0.0 \
    --device gpu

bash scripts/eval.sh


# Qwen
# python src/scripts/perform_merging.py \
#     --model_family qwen2.5 \
#     --merge_mode "standard" \
#     --output_dir outputs/models/Qwen-2.5-3B-merged-std \
#     --save_merged_model \
#     --device gpu

# bash scripts/eval.sh
