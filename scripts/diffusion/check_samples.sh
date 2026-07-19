#!/bin/bash
set -e

REPO_ROOT="/gpfs/home6/eterres/MasterThesis"
SUMMARY="${1:-${REPO_ROOT}/outputs/diffusion/tables/eval_summary.json}"

cd "${REPO_ROOT}"
python -m src.diffusion.check_samples "${SUMMARY}" "${@:2}"
