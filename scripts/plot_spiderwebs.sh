#!/bin/bash

set -e

EVALUATION_TYPE="${1:-pretrained}"
REPO_ROOT="/home/eterres/MasterThesis"

python "${REPO_ROOT}/src/plots/plot_spiderwebs.py" "${EVALUATION_TYPE}" --repo-root "${REPO_ROOT}"
