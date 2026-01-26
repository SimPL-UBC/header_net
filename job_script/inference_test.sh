#!/usr/bin/env bash
# =============================================================================
# Test Inference Pipeline Job Script
# =============================================================================
# Runs the full header detection pipeline (pre-XGB → VMAE → post-XGB)
# on the test dataset.
#
# Environment variables (with defaults):
#   DATASET_ROOT      - Path to SoccerNet videos (default: ../SoccerNet/test/)
#   OUTPUT_DIR        - Output directory for predictions
#   DEVICE            - GPU device (default: cuda:1)
#   PRE_XGB_THRESHOLD - Pre-XGB filtering threshold (default: 0.05)
#   BATCH_SIZE        - Inference batch size (default: 16)
#
# Usage:
#   # With defaults (uses ../SoccerNet/test/)
#   bash job_script/inference_test.sh
#
#   # With custom dataset path
#   DATASET_ROOT=/data/SoccerNet/test bash job_script/inference_test.sh
# =============================================================================

set -euo pipefail

# Get script and repo directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "============================================================"
echo "Test Inference Pipeline"
echo "============================================================"
echo "Repository root: ${REPO_ROOT}"
echo ""

# =============================================================================
# Conda Environment Setup
# =============================================================================
CONDA_SH="${CONDA_SH:-${HOME}/anaconda3/etc/profile.d/conda.sh}"
if [[ ! -f "${CONDA_SH}" ]]; then
	# Try miniconda path
	CONDA_SH="${HOME}/miniconda3/etc/profile.d/conda.sh"
fi
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
echo "[OK] Using Python: ${PYTHON_BIN}"

# =============================================================================
# Configuration with Defaults
# =============================================================================

# Dataset root (default: SoccerNet/test/)
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/SoccerNet/test/}"

# Output directory
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/test_inference}"

# Device configuration
DEVICE="${DEVICE:-cuda:1}"

# Pre-XGB filtering threshold (center frame only)
PRE_XGB_THRESHOLD="${PRE_XGB_THRESHOLD:-0.05}"

# Processing parameters
BATCH_SIZE="${BATCH_SIZE:-16}"

# Model paths (verified locations)
PRE_XGB_MODEL="${PRE_XGB_MODEL:-${REPO_ROOT}/output/pre_xgb/train/pre_xgb_final.pkl}"
VMAE_CHECKPOINT="${VMAE_CHECKPOINT:-${REPO_ROOT}/output/vmae/vmae_full_base/checkpoints/best_epoch_26.pt}"
BACKBONE_CKPT="${BACKBONE_CKPT:-${REPO_ROOT}/checkpoints/VideoMAEv2-Base}"
POST_XGB_MODEL="${POST_XGB_MODEL:-${REPO_ROOT}/output/post_xgb/post_xgb_vmae_ball_only_thr0.2/post_xgb_final.pkl}"
RF_DETR_WEIGHTS="${RF_DETR_WEIGHTS:-${REPO_ROOT}/RFDETR-Soccernet/weights/checkpoint_best_regular.pth}"

# Ball detection threshold
BALL_CONF_THRESHOLD="${BALL_CONF_THRESHOLD:-0.3}"

# =============================================================================
# Validate Model Paths
# =============================================================================
echo ""
echo "Validating model paths..."

validate_path() {
	local name="$1"
	local path="$2"
	if [[ -e "${path}" ]]; then
		echo "  [OK] ${name}: ${path}"
	else
		echo "  [ERROR] ${name} not found: ${path}" >&2
		return 1
	fi
}

validate_path "Pre-XGB model" "${PRE_XGB_MODEL}" || exit 1
validate_path "VMAE checkpoint" "${VMAE_CHECKPOINT}" || exit 1
validate_path "VideoMAE backbone" "${BACKBONE_CKPT}" || exit 1
validate_path "Post-XGB model" "${POST_XGB_MODEL}" || exit 1
validate_path "RF-DETR weights" "${RF_DETR_WEIGHTS}" || exit 1
validate_path "Dataset root" "${DATASET_ROOT}" || exit 1

# =============================================================================
# Print Configuration Summary
# =============================================================================
echo ""
echo "============================================================"
echo "Configuration Summary"
echo "============================================================"
echo "  Dataset root:       ${DATASET_ROOT}"
echo "  Output directory:   ${OUTPUT_DIR}"
echo "  Device:             ${DEVICE}"
echo "  Pre-XGB threshold:  ${PRE_XGB_THRESHOLD}"
echo "  Batch size:         ${BATCH_SIZE}"
echo "  Ball conf threshold:${BALL_CONF_THRESHOLD}"
echo "============================================================"
echo ""

# =============================================================================
# Create Output Directory
# =============================================================================
mkdir -p "${OUTPUT_DIR}"

# =============================================================================
# Run Inference
# =============================================================================
echo "Starting inference pipeline..."
echo ""

"${PYTHON_BIN}" "${REPO_ROOT}/inference_test.py" \
	--dataset_root "${DATASET_ROOT}" \
	--output_dir "${OUTPUT_DIR}" \
	--device "${DEVICE}" \
	--pre_xgb_model "${PRE_XGB_MODEL}" \
	--vmae_checkpoint "${VMAE_CHECKPOINT}" \
	--backbone_ckpt "${BACKBONE_CKPT}" \
	--post_xgb_model "${POST_XGB_MODEL}" \
	--rf_detr_weights "${RF_DETR_WEIGHTS}" \
	--rf_detr_label_mode soccernet \
	--ball_conf_threshold "${BALL_CONF_THRESHOLD}" \
	--pre_xgb_threshold "${PRE_XGB_THRESHOLD}" \
	--batch_size "${BATCH_SIZE}"

# =============================================================================
# Completion
# =============================================================================
echo ""
echo "============================================================"
echo "Job completed!"
echo "Output saved to: ${OUTPUT_DIR}/test_predictions.csv"
echo "============================================================"
