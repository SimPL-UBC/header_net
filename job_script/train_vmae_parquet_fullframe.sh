#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export SPATIAL_MODE="${SPATIAL_MODE:-full_frame}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/output/vmae_parquet_ratio10_new_fullframe}"
export RUN_NAME="${RUN_NAME:-vmae_parquet_ratio10_new_fullframe}"

exec "${SCRIPT_DIR}/train_vmae_parquet.sh"
