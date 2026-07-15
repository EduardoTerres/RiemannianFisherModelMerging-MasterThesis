#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

python "${SCRIPT_DIR}/interpolation_geodesic_results.py" \
  --config_path="${REPO_ROOT}/src/diffusion/config/config.yaml" \
  "$@"
