#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"

# Env overrides:
# INPUT_CSV: source CSV (default: output/dataset_generation/train/train_cache_header.csv)
# OUTPUT_DIR: destination directory for augmented dataset
# OUTPUT_NAME: output CSV filename
# SEED: random seed (default: 42)
# FLIP_P: horizontal flip prob (default: 0.5)
# CROP_P: random resized crop prob (default: 0.5)
# CROP_SCALE_MIN / CROP_SCALE_MAX: crop scale range (default: 0.8-1.0)
# ROTATION_P: rotation prob (default: 0.2)
# ROTATION_DEG: rotation range (+/- degrees, default: 10)
# BLUR_P: blur prob (default: 0.2)
# BLUR_RADIUS: Gaussian blur radius (default: 1.0)
# OVERWRITE: set to 1 to overwrite existing outputs

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

INPUT_CSV="${INPUT_CSV:-${REPO_ROOT}/output/dataset_generation/train/train_cache_header.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/dataset_generation/train_aug}"
OUTPUT_NAME="${OUTPUT_NAME:-train_with_aug.csv}"
SEED="${SEED:-42}"
FLIP_P="${FLIP_P:-0.5}"
CROP_P="${CROP_P:-0.5}"
CROP_SCALE_MIN="${CROP_SCALE_MIN:-0.8}"
CROP_SCALE_MAX="${CROP_SCALE_MAX:-1.0}"
ROTATION_P="${ROTATION_P:-0.2}"
ROTATION_DEG="${ROTATION_DEG:-10}"
BLUR_P="${BLUR_P:-0.2}"
BLUR_RADIUS="${BLUR_RADIUS:-1.0}"
OVERWRITE="${OVERWRITE:-0}"

EXTRA_ARGS=()
if [[ "${OVERWRITE}" == "1" ]]; then
  EXTRA_ARGS+=(--overwrite)
fi

"${PYTHON_BIN}" "${REPO_ROOT}/dataset_generation/augment_cache_dataset.py" \
  --input_csv "${INPUT_CSV}" \
  --output_dir "${OUTPUT_DIR}" \
  --output_name "${OUTPUT_NAME}" \
  --seed "${SEED}" \
  --flip_p "${FLIP_P}" \
  --crop_p "${CROP_P}" \
  --crop_scale_min "${CROP_SCALE_MIN}" \
  --crop_scale_max "${CROP_SCALE_MAX}" \
  --rotation_p "${ROTATION_P}" \
  --rotation_deg "${ROTATION_DEG}" \
  --blur_p "${BLUR_P}" \
  --blur_radius "${BLUR_RADIUS}" \
  "${EXTRA_ARGS[@]}"
