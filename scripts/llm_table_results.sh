#!/bin/bash

set -e

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate merge

# Entries should be METHOD or METHOD=DISPLAY_LABEL, where METHOD is the output folder name.
METHODS=(
  "finetunes=Finetuned"
  "standard_rescaled=OrthoMerge"
  "diagonal_fisher=diagonal fisher"
  "orthomerge_c_ties=TIES"
  "orthomerge_c_tsvm=TSVM"
  "standard=Lie avg. (alpha=1)"
  "wudi=Wudi"
)

python -m src.analysis.performance_table \
  --decimals 2 \
  --methods "${METHODS[@]}" \
  "$@"
