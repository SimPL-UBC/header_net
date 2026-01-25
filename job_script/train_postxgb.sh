#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"

# Env overrides:
# DATASET_ROOT: Parent directory containing SoccerNet/ folder with videos
#               Expected structure: DATASET_ROOT/SoccerNet/league/season/match/*.mp4
#               For validation only, point to the val split parent (default: ../DeepImpact)
# LABELS_DIR: Ground truth labels directory (default: ../SoccerNet/val/labelled_header)
# PRE_XGB_MODEL: Pre-XGB model path (default: output/pre_xgb/train/pre_xgb_final.pkl)
# VMAE_RUN_DIR: VMAE trained run directory (default: output/vmae/vmae_full_base)
# BACKBONE_CKPT: VideoMAE pretrained weights (default: checkpoints/VideoMAEv2-Base)
# RF_DETR_WEIGHTS: RF-DETR weights (optional)
# OUTPUT_DIR: Output directory (default: output/post_xgb)
# MODE: Frame selection mode - ball_only or every_n (default: ball_only)
# WINDOW_STRIDE: Frame stride for every_n mode (default: 5)
# BATCH_SIZE: Inference batch size (default: 4)
# N_FOLDS: Number of CV folds (default: 5)
# PRE_XGB_THRESHOLD: Pre-XGB threshold for naming (default: 0.2)
# DEVICE: GPU device for inference, e.g., cuda:0 or cuda:1 (default: cuda:0)

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

# Default paths
# DATASET_ROOT should be the parent directory containing SoccerNet/
# The function looks for: DATASET_ROOT/SoccerNet/{train,val,test}/league/season/match/*.mp4
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/..}"
LABELS_DIR="${LABELS_DIR:-${REPO_ROOT}/../SoccerNet/val/labelled_header}"
PRE_XGB_MODEL="${PRE_XGB_MODEL:-${REPO_ROOT}/output/pre_xgb/train/pre_xgb_final.pkl}"
VMAE_RUN_DIR="${VMAE_RUN_DIR:-${REPO_ROOT}/output/vmae/vmae_full_base}"
BACKBONE_CKPT="${BACKBONE_CKPT:-${REPO_ROOT}/checkpoints/VideoMAEv2-Base}"
RF_DETR_WEIGHTS="${RF_DETR_WEIGHTS:-${REPO_ROOT}/rf-detr-medium.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/post_xgb}"
MODE="${MODE:-ball_only}"
WINDOW_STRIDE="${WINDOW_STRIDE:-5}"
BATCH_SIZE="${BATCH_SIZE:-16}"
N_FOLDS="${N_FOLDS:-5}"
PRE_XGB_THRESHOLD="${PRE_XGB_THRESHOLD:-0.2}"
DEVICE="${DEVICE:-cuda:0}"

# Validate required files
if [[ ! -f "${PRE_XGB_MODEL}" ]]; then
  echo "[ERROR] Pre-XGB model not found: ${PRE_XGB_MODEL}" >&2
  exit 1
fi

if [[ ! -d "${VMAE_RUN_DIR}" ]]; then
  echo "[ERROR] VMAE run directory not found: ${VMAE_RUN_DIR}" >&2
  exit 1
fi

if [[ ! -f "${VMAE_RUN_DIR}/best_metrics.json" ]]; then
  echo "[ERROR] best_metrics.json not found in ${VMAE_RUN_DIR}" >&2
  exit 1
fi

# Read VMAE checkpoint from best_metrics.json dynamically
echo "Reading VMAE checkpoint from ${VMAE_RUN_DIR}/best_metrics.json..."
VMAE_CHECKPOINT_REL=$("${PYTHON_BIN}" -c "
import json
from pathlib import Path
with open('${VMAE_RUN_DIR}/best_metrics.json') as f:
    print(json.load(f)['checkpoint'])
")
VMAE_CHECKPOINT="${VMAE_RUN_DIR}/${VMAE_CHECKPOINT_REL}"

if [[ ! -f "${VMAE_CHECKPOINT}" ]]; then
  echo "[ERROR] VMAE checkpoint not found: ${VMAE_CHECKPOINT}" >&2
  exit 1
fi

echo "Using VMAE checkpoint: ${VMAE_CHECKPOINT}"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Build output filename based on mode
if [[ "${MODE}" == "ball_only" ]]; then
  PROBS_CSV="${OUTPUT_DIR}/val_probs_ball_only.csv"
else
  PROBS_CSV="${OUTPUT_DIR}/val_probs_every_${WINDOW_STRIDE}.csv"
fi

echo ""
echo "============================================================"
echo "Post-XGB Training Pipeline"
echo "============================================================"
echo "Dataset root:      ${DATASET_ROOT}"
echo "Labels directory:  ${LABELS_DIR}"
echo "Pre-XGB model:     ${PRE_XGB_MODEL}"
echo "VMAE run dir:      ${VMAE_RUN_DIR}"
echo "VMAE checkpoint:   ${VMAE_CHECKPOINT}"
echo "Backbone ckpt:     ${BACKBONE_CKPT}"
echo "RF-DETR weights:   ${RF_DETR_WEIGHTS}"
echo "Output directory:  ${OUTPUT_DIR}"
echo "Mode:              ${MODE}"
echo "Window stride:     ${WINDOW_STRIDE}"
echo "Batch size:        ${BATCH_SIZE}"
echo "CV folds:          ${N_FOLDS}"
echo "Pre-XGB threshold: ${PRE_XGB_THRESHOLD}"
echo "Device:            ${DEVICE}"
echo "============================================================"
echo ""

# Step 1: Export probabilities from raw video
echo "Step 1: Exporting CNN and Pre-XGB probabilities..."
echo "Output: ${PROBS_CSV}"
echo ""

EXPORT_ARGS=(
  --dataset_root "${DATASET_ROOT}"
  --split val
  --backbone vmae
  --checkpoint "${VMAE_CHECKPOINT}"
  --backbone_ckpt "${BACKBONE_CKPT}"
  --pre_xgb_model "${PRE_XGB_MODEL}"
  --pre_xgb_threshold "${PRE_XGB_THRESHOLD}"
  --mode "${MODE}"
  --batch_size "${BATCH_SIZE}"
  --device "${DEVICE}"
  --output "${PROBS_CSV}"
)

# Add optional RF-DETR weights
if [[ -f "${RF_DETR_WEIGHTS}" ]]; then
  EXPORT_ARGS+=(--rf_detr_weights "${RF_DETR_WEIGHTS}")
fi

# Add stride for every_n mode
if [[ "${MODE}" == "every_n" ]]; then
  EXPORT_ARGS+=(--window_stride "${WINDOW_STRIDE}")
fi

"${PYTHON_BIN}" "${REPO_ROOT}/export_probs_raw_video.py" \
  "${EXPORT_ARGS[@]}"

echo ""
echo "Step 1 complete: Probabilities exported to ${PROBS_CSV}"
echo ""

# Step 2: Train post-XGB model
echo "Step 2: Training post-XGB model..."
echo ""

POST_XGB_ARGS=(
  --probs_csv "${PROBS_CSV}"
  --labels_csv "${LABELS_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --window_size 15
  --n_folds "${N_FOLDS}"
  --backbone vmae
  --export_mode "${MODE}"
  --export_threshold "${PRE_XGB_THRESHOLD}"
)

# Add stride for every_n mode
if [[ "${MODE}" == "every_n" ]]; then
  POST_XGB_ARGS+=(--export_stride "${WINDOW_STRIDE}")
fi

"${PYTHON_BIN}" "${REPO_ROOT}/tree/post_xgb.py" \
  "${POST_XGB_ARGS[@]}"

echo ""
echo "============================================================"
echo "Post-XGB Training Complete!"
echo "============================================================"
echo "Probabilities:  ${PROBS_CSV}"
echo "Model outputs:  ${OUTPUT_DIR}/post_xgb_vmae_*/"
echo "============================================================"
