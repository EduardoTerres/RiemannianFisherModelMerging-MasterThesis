#!/bin/bash

set -e

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate OrthoMerge

python src/analysis/interpolation.py \
    --num-points 4 \
    --num-samples 64 \
    --batch-size 64 \
    --max-length 256 \
    --save-path outputs/interpolation