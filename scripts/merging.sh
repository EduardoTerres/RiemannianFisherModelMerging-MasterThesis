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
    --language_model_name "data/models/Llama-3.1-8B/" \
    --merge_mode "standard" \
    --save_merged_model \
    --output_dir outputs/models/Llama-3.1-8B-merged-fisher

bash scripts/eval.sh


# Qwen
python src/scripts/perform_merging.py \
    --model_family qwen2.5 \
    --language_model_name "data/models/Qwen-2.5-3B/" \
    --merge_mode "standard" \
    --save_merged_model \
    --output_dir outputs/models/Qwen-2.5-3B-merged

bash scripts/eval.sh
