#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_SH="${CONDA_SH:-${HOME}/anaconda3/etc/profile.d/conda.sh}"
if [[ ! -f "${CONDA_SH}" ]]; then
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

PREDICTIONS_CSV="${PREDICTIONS_CSV:-${REPO_ROOT}/output/vmae_parquet_ratio10/test_inference/test_predictions_raw.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/vmae_parquet_ratio10/test_inference/analysis}"
DECISION_THRESHOLD="${DECISION_THRESHOLD:-}"
F1_THRESHOLD_STEP="${F1_THRESHOLD_STEP:-0.01}"
RENDER_VIDEO_ID="${RENDER_VIDEO_ID:-}"
RENDER_HALF="${RENDER_HALF:-}"
RENDER_FRAME_STRIDE="${RENDER_FRAME_STRIDE:-5}"
RENDER_OUTPUT="${RENDER_OUTPUT:-}"

if [[ ! -f "${PREDICTIONS_CSV}" ]]; then
	echo "[ERROR] Predictions CSV not found: ${PREDICTIONS_CSV}" >&2
	exit 1
fi

ARGS=(
	"${REPO_ROOT}/analyze_test_inference.py"
	--predictions-csv "${PREDICTIONS_CSV}"
	--output-dir "${OUTPUT_DIR}"
	--f1-threshold-step "${F1_THRESHOLD_STEP}"
	--render-frame-stride "${RENDER_FRAME_STRIDE}"
)

if [[ -n "${DECISION_THRESHOLD}" ]]; then
	ARGS+=(--decision-threshold "${DECISION_THRESHOLD}")
fi
if [[ -n "${RENDER_VIDEO_ID}" ]]; then
	ARGS+=(--render-video-id "${RENDER_VIDEO_ID}")
fi
if [[ -n "${RENDER_HALF}" ]]; then
	ARGS+=(--render-half "${RENDER_HALF}")
fi
if [[ -n "${RENDER_OUTPUT}" ]]; then
	ARGS+=(--render-output "${RENDER_OUTPUT}")
fi

"${PYTHON_BIN}" "${ARGS[@]}"

echo ""
echo "Analysis outputs:"
echo "  ${OUTPUT_DIR}/test_metrics.json"
echo "  ${OUTPUT_DIR}/test_predictions_scored.csv"
