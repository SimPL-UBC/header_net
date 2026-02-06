#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"

# Env overrides:
# LABEL_MODE: "one_frame" or "continuous" (default: one_frame)
# WEIGHTS_PATH: path to SoccerNet RF-DETR weights
# CONF_THRESHOLD: detection confidence threshold (default: 0.25)
# BATCH_SIZE: inference batch size (default: 8)
# TOPK: max detections per frame (default: 15)
# DEVICE: torch device override (cpu/cuda/mps; auto if empty)
# OPTIMIZE: 1 to enable RF-DETR optimization (default: 1)
# OPTIMIZE_BATCH_SIZE: batch size during optimization (default: 1)
# OPTIMIZE_COMPILE: 1 to enable torch compile (default: 0)
# OUTPUT_BASE: base output directory (default: ${REPO_ROOT}/output/dense_dataset)
# MATCH_FILTER: space-separated match names to process (default: all)

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
LABEL_MODE="${LABEL_MODE:-one_frame}"
WEIGHTS_PATH="${WEIGHTS_PATH:-${REPO_ROOT}/RFDETR-Soccernet/weights/checkpoint_best_regular.pth}"
CONF_THRESHOLD="${CONF_THRESHOLD:-0.25}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TOPK="${TOPK:-15}"
OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/output/dense_dataset}"

# Optional toggles.
DEVICE="${DEVICE:-}"
OPTIMIZE="${OPTIMIZE:-1}"
OPTIMIZE_BATCH_SIZE="${OPTIMIZE_BATCH_SIZE:-1}"
OPTIMIZE_COMPILE="${OPTIMIZE_COMPILE:-0}"
MATCH_FILTER="${MATCH_FILTER:-}"

mkdir -p "${OUTPUT_BASE}"

# Resolve label mode flag.
if [[ "${LABEL_MODE}" == "continuous" ]]; then
  LABEL_FLAG="--continuous-frame-header"
elif [[ "${LABEL_MODE}" == "one_frame" ]]; then
  LABEL_FLAG="--one-frame-header"
else
  echo "[ERROR] LABEL_MODE must be 'one_frame' or 'continuous', got '${LABEL_MODE}'" >&2
  exit 1
fi

COMMON_ARGS=(
  --weights "${WEIGHTS_PATH}"
  --confidence-threshold "${CONF_THRESHOLD}"
  --batch-size "${BATCH_SIZE}"
  --topk "${TOPK}"
  "${LABEL_FLAG}"
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

if [[ -n "${MATCH_FILTER}" ]]; then
  # shellcheck disable=SC2206
  COMMON_ARGS+=(--match-filter ${MATCH_FILTER})
fi

echo "============================================================"
echo "Dense dataset generation — ${LABEL_MODE} mode"
echo "============================================================"

# ── Train split ──
echo ""
echo "[STEP 1/2] Generating train split..."
"${PYTHON_BIN}" "${REPO_ROOT}/dataset_generation/generate_dense_dataset.py" \
  --dataset-path "${REPO_ROOT}/SoccerNet/train" \
  --labels-dir "${REPO_ROOT}/SoccerNet/train/labelled_header" \
  --output-path "${OUTPUT_BASE}/dense_train.parquet" \
  "${COMMON_ARGS[@]}"

# ── Val split ──
echo ""
echo "[STEP 2/2] Generating val split..."
"${PYTHON_BIN}" "${REPO_ROOT}/dataset_generation/generate_dense_dataset.py" \
  --dataset-path "${REPO_ROOT}/SoccerNet/val" \
  --labels-dir "${REPO_ROOT}/SoccerNet/val/labelled_header" \
  --output-path "${OUTPUT_BASE}/dense_val.parquet" \
  "${COMMON_ARGS[@]}"

echo ""
echo "============================================================"
echo "Done. Outputs:"
echo "  Train: ${OUTPUT_BASE}/dense_train.parquet"
echo "  Val:   ${OUTPUT_BASE}/dense_val.parquet"
echo "============================================================"
