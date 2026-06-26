#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=pipe_all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=12:00:00
#SBATCH --output=outputs/diffusion/slurms/pipe_all_%A.out

set -e

REPO_ROOT="/gpfs/home6/eterres/MasterThesis"
OUTPUT_DIR="${REPO_ROOT}/outputs/diffusion"
CONFIG_PATH="${REPO_ROOT}/src/diffusion/config/config.yaml"

METHODS=(
  "diagonal_fisher_rescaled"
  "diagonal_fisher"
  "standard_rescaled"
  "orthofuse"
)

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate orthofuse_env

export HF_HOME=/scratch-shared/eterres/huggingface-cache
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export DIFFUSERS_OFFLINE="${DIFFUSERS_OFFLINE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${REPO_ROOT}/OrthoFuse:${REPO_ROOT}:${PYTHONPATH:-}"

mapfile -t PAIRS < <(
  python - <<'PY'
from src.diffusion.dataset_1 import DIFFUSION_MERGE_PAIRS

for pair in DIFFUSION_MERGE_PAIRS:
    print(f"{pair['concept']['name']}:{pair['style']['name']}")
PY
)

echo "[pipe] running ${#PAIRS[@]} dataset pairs"

for pair in "${PAIRS[@]}"; do
  concept="${pair%%:*}"
  style="${pair#*:}"

  for method in "${METHODS[@]}"; do
    echo "[pipe] ${method} ${pair}"

    if [[ "${method}" == "orthofuse" ]]; then
      python "${REPO_ROOT}/src/diffusion/pipe_orthofuse.py" \
        --config_path="${CONFIG_PATH}" \
        --output_dir="${OUTPUT_DIR}" \
        --concept_name="${concept}" \
        --style_name="${style}" \
        --t=0.6 \
        --num_images_per_medium_prompt=10 \
        --replace_inference_output
    elif [[ "${method}" == diagonal_fisher* ]]; then
      python "${REPO_ROOT}/src/diffusion/pipe_gradients.py" \
        --config_path="${CONFIG_PATH}" \
        --output_dir="${OUTPUT_DIR}" \
        --samples "${pair}" \
        --merge_mode="${method}" \
        --alphas 1.0 1.0 \
        --num_images_per_medium_prompt=10 \
        --replace_inference_output \
        --fisher_min=1e-14 \
        --fisher_rescale=1e10
    else
      python "${REPO_ROOT}/src/diffusion/pipe_gradients.py" \
        --config_path="${CONFIG_PATH}" \
        --output_dir="${OUTPUT_DIR}" \
        --samples "${pair}" \
        --merge_mode="${method}" \
        --alphas 1.0 1.0 \
        --num_images_per_medium_prompt=10 \
        --replace_inference_output
    fi
  done
done

bash "${REPO_ROOT}/scripts/diffusion/eval.sh"
