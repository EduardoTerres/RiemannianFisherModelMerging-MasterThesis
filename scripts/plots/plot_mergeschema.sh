#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=mergeschema
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/home6/eterres/MasterThesis/outputs/mergeschema/slurm/mergeschema_%A_%a.out

set -e

REPO_ROOT="/home/eterres/MasterThesis"
OUTPUT_DIR="${REPO_ROOT}/outputs/mergeschema"
export MPLCONFIGDIR="${REPO_ROOT}/.matplotlib"
export XDG_CACHE_HOME="${REPO_ROOT}/.cache"

mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}" "${OUTPUT_DIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate merge
python "${REPO_ROOT}/src/plots/plot_mergeschema.py" \
    --repo-root "${REPO_ROOT}" \
    --cache "${OUTPUT_DIR}/mergeschema_drop_triviaqa_larger_losses.npz" \
    --output "${OUTPUT_DIR}/merge_schema.png" \
    --geometry-output "${OUTPUT_DIR}/merge_schema_geometry.png" \
    --manifold-output "${OUTPUT_DIR}/merge_schema_3d.png" \
    --from-saved
