#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"

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

if ! "${PYTHON_BIN}" - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    print("[ERROR] torch.cuda.is_available() is False; aborting.")
    sys.exit(1)
print(f"[INFO] Using torch {torch.__version__} CUDA {torch.version.cuda}")
print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
PY
then
  exit 1
fi

# Configurable options.
WEIGHTS_PATH="${REPO_ROOT}/RFDETR-Soccernet/weights/checkpoint_best_regular.pth"
CONF_THRESHOLD="0.25"
BATCH_SIZE="4"
TOPK="15"
NEGATIVE_RATIO="3.0"
GUARD_FRAMES="10"
CROP_SCALE_FACTOR="4.5"
WINDOW_OFFSETS=(-24 -18 -12 -6 -3 0 3 6 12 18 24)

# Optional toggles.
DEVICE="${DEVICE:-}"
OPTIMIZE="${OPTIMIZE:-0}"
OPTIMIZE_BATCH_SIZE="${OPTIMIZE_BATCH_SIZE:-1}"
OPTIMIZE_COMPILE="${OPTIMIZE_COMPILE:-0}"

OUTPUT_BASE="${REPO_ROOT}/output/dataset_generation"
mkdir -p "${OUTPUT_BASE}/train" "${OUTPUT_BASE}/val"

COMMON_ARGS=(
  --weights "${WEIGHTS_PATH}"
  --confidence-threshold "${CONF_THRESHOLD}"
  --batch-size "${BATCH_SIZE}"
  --topk "${TOPK}"
  --negative-ratio "${NEGATIVE_RATIO}"
  --guard-frames "${GUARD_FRAMES}"
  --seed "42"
  --window "${WINDOW_OFFSETS[@]}"
  --crop-scale-factor "${CROP_SCALE_FACTOR}"
)

if [[ -n "${DEVICE}" ]]; then
  COMMON_ARGS+=(--device "${DEVICE}")
fi

if [[ "${OPTIMIZE}" == "1" ]]; then
  COMMON_ARGS+=(--optimize --optimize-batch-size "${OPTIMIZE_BATCH_SIZE}")
  if [[ "${OPTIMIZE_COMPILE}" == "1" ]]; then
    COMMON_ARGS+=(--optimize-compile)
  fi
fi

"${PYTHON_BIN}" "${REPO_ROOT}/dataset_generation/generate_positive_negative_dataset.py" \
  --dataset-path "${REPO_ROOT}/SoccerNet/train" \
  --labels-dir "${REPO_ROOT}/SoccerNet/train/labelled_header" \
  --output-dir "${OUTPUT_BASE}/train" \
  --output-name "train_cache_header.csv" \
  "${COMMON_ARGS[@]}"

"${PYTHON_BIN}" "${REPO_ROOT}/dataset_generation/generate_positive_negative_dataset.py" \
  --dataset-path "${REPO_ROOT}/SoccerNet/val" \
  --labels-dir "${REPO_ROOT}/SoccerNet/val/labelled_header" \
  --output-dir "${OUTPUT_BASE}/val" \
  --output-name "val_cache_header.csv" \
  "${COMMON_ARGS[@]}"
