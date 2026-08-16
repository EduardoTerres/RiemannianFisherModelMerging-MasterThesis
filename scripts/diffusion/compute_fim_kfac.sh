#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=fim_kfac
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=05:00:00
#SBATCH --array=0-17%5
#SBATCH --output=outputs/diffusion/slurms/fim_kfac_%A_%a.out

set -e

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate orthofuse_env

export HF_HOME=/path/to/hf_cache
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TASKS=(
  "concept cat2"
  "concept cat"
  "concept dog"
  "concept dog2"
  "concept dog3"
  "concept dog6"
  "style 01_01"
  "style 01_02"
  "style 01_03"
  "style 01_07"
  "style 01_08"
  "style 02_03"
  "style 03_04"
  "style dolina"
  "style etsy"
  "style gondoliers"
  "style image_scan"
  "style pots"
)

read -r ENTRY_TYPE DATASET_NAME <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"
echo "Running KFAC for ${ENTRY_TYPE}:${DATASET_NAME}"

REPEATS=500
if [[ "${ENTRY_TYPE}" == "style" ]]; then
  REPEATS=2500
fi
echo "Using repeats=${REPEATS}"

python src/diffusion/compute_fim_kfac.py \
  --config_path=src/diffusion/config/config.yaml \
  --output_dir=/path/to/fishers/sdxl \
  --entry_type="${ENTRY_TYPE}" \
  --dataset_name="${DATASET_NAME}" \
  --batch_size=1 \
  --repeats="${REPEATS}" \
  --seed=42 \
  --device=cuda \
  --weight_dtype=float32
