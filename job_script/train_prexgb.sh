#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"

# Env overrides:
# METADATA_DIR: directory with *_meta.json files (default: output/dataset_generation/train)
# OUTPUT_DIR: output directory for models/results (default: output/pre_xgb/train)
# NEG_RATIO: negative:positive sampling ratio (default: 3.0)
# N_FOLDS: number of CV folds (default: 5)
# THRESHOLD: proposal threshold (ball_det_dict mode only; default: 0.3)
# MAX_PROPOSALS_PER_MIN: proposal cap (ball_det_dict mode only; default: 5)
# DISABLE_PLAYER_FEATURES: set to 1 to pass --no_player_features (default: 0)

CONDA_SH="${CONDA_SH:-${HOME}/anaconda3/etc/profile.d/conda.sh}"
if [[ ! -f "${CONDA_SH}" ]]; then
  echo "[ERROR] Conda init not found at ${CONDA_SH}; aborting." >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${CONDA_SH}"
if ! conda activate deep_impact_env; then
  echo "[ERROR] Failed to activate conda env deep_impact_env; aborting." >&2
  exit 1
fi

PYTHON_BIN="$(command -v python)"
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "[ERROR] Python executable not found after conda activate; aborting." >&2
  exit 1
fi

METADATA_DIR="${METADATA_DIR:-${REPO_ROOT}/output/dataset_generation/train}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/pre_xgb/train}"
NEG_RATIO="${NEG_RATIO:-3.0}"
N_FOLDS="${N_FOLDS:-5}"
THRESHOLD="${THRESHOLD:-0.3}"
MAX_PROPOSALS_PER_MIN="${MAX_PROPOSALS_PER_MIN:-5}"
DISABLE_PLAYER_FEATURES="${DISABLE_PLAYER_FEATURES:-0}"

mkdir -p "${OUTPUT_DIR}"

PREXGB_ARGS=(
  --metadata_dir "${METADATA_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --neg_ratio "${NEG_RATIO}"
  --n_folds "${N_FOLDS}"
  --threshold "${THRESHOLD}"
  --max_proposals_per_min "${MAX_PROPOSALS_PER_MIN}"
)

if [[ "${DISABLE_PLAYER_FEATURES}" == "1" ]]; then
  PREXGB_ARGS+=(--no_player_features)
fi

"${PYTHON_BIN}" "${REPO_ROOT}/tree/pre_xgb.py" \
  "${PREXGB_ARGS[@]}"
  --metadata_dir "${METADATA_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --neg_ratio "${NEG_RATIO}" \
  --n_folds "${N_FOLDS}"
