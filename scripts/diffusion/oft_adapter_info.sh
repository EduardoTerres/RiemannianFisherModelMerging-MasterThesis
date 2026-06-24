#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --job-name=oft_adapter_info
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=outputs/diffusion/slurms/oft_adapter_info_%A.out

set -e

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate orthofuse_env

cd /home/eterres/MasterThesis/OrthoFuse

if [ "$#" -gt 0 ]; then
  python ../src/diffusion/oft_adapter_info.py "$@"
else
  python ../src/diffusion/oft_adapter_info.py \
    /scratch-shared/eterres/SDXL/concepts/pytorch_lora_weights_cat.safetensors \
    /scratch-shared/eterres/SDXL/adapters/pytorch_lora_weights_01_07.safetensors
fi
