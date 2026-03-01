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

DATASET_PATH="${DATASET_PATH:-${REPO_ROOT}/SoccerNet/test}"
LABELS_DIR="${LABELS_DIR:-${REPO_ROOT}/SoccerNet/test/labelled_header}"
OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/output/dense_dataset}"
WEIGHTS_PATH="${WEIGHTS_PATH:-${REPO_ROOT}/RFDETR-Soccernet/weights/checkpoint_best_regular.pth}"
CONF_THRESHOLD="${CONF_THRESHOLD:-0.25}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TOPK="${TOPK:-15}"
MODEL_NUM_FRAMES="${MODEL_NUM_FRAMES:-16}"
MIN_DECODE_RATIO="${MIN_DECODE_RATIO:-0.5}"
STRICT_DECODE_ERRORS="${STRICT_DECODE_ERRORS:-1}"
FFMPEG_BIN="${FFMPEG_BIN:-ffmpeg}"
DEVICE="${DEVICE:-}"
MATCH_FILTER="${MATCH_FILTER:-}"
OPTIMIZE="${OPTIMIZE:-1}"
OPTIMIZE_BATCH_SIZE="${OPTIMIZE_BATCH_SIZE:-32}"
OPTIMIZE_COMPILE="${OPTIMIZE_COMPILE:-0}"
LABEL_MODE="${LABEL_MODE:-continuous}"

OUTPUT_PATH="${OUTPUT_BASE}/dense_test.parquet"
FAILED_LOG_PATH="${OUTPUT_BASE}/dense_test_failed_videos.csv"
FAILED_FRAME_LOG_PATH="${OUTPUT_BASE}/dense_test_failed_frames.csv"

if [[ ! -d "${DATASET_PATH}" ]]; then
	echo "[ERROR] Dataset path not found: ${DATASET_PATH}" >&2
	exit 1
fi
if [[ ! -d "${LABELS_DIR}" ]]; then
	echo "[ERROR] Labels directory not found: ${LABELS_DIR}" >&2
	exit 1
fi
if [[ ! -f "${WEIGHTS_PATH}" ]]; then
	echo "[ERROR] RF-DETR weights not found: ${WEIGHTS_PATH}" >&2
	exit 1
fi

if [[ -z "${DEVICE}" || "${DEVICE}" == cuda* ]]; then
	if ! "${PYTHON_BIN}" - <<'PY'; then
import sys
import torch

if not torch.cuda.is_available():
    print("[ERROR] torch.cuda.is_available() is False; aborting.")
    sys.exit(1)
print(f"[INFO] Using torch {torch.__version__} CUDA {torch.version.cuda}")
print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
PY
		exit 1
	fi
fi

mkdir -p "${OUTPUT_BASE}"

ARGS=(
	"${REPO_ROOT}/dataset_generation/generate_dense_test_dataset.py"
	--dataset-path "${DATASET_PATH}"
	--labels-dir "${LABELS_DIR}"
	--output-path "${OUTPUT_PATH}"
	--failed-log-path "${FAILED_LOG_PATH}"
	--failed-frame-log-path "${FAILED_FRAME_LOG_PATH}"
	--weights "${WEIGHTS_PATH}"
	--confidence-threshold "${CONF_THRESHOLD}"
	--batch-size "${BATCH_SIZE}"
	--topk "${TOPK}"
	--model-num-frames "${MODEL_NUM_FRAMES}"
	--min-decode-ratio "${MIN_DECODE_RATIO}"
)

if [[ "${LABEL_MODE}" == "continuous" ]]; then
	ARGS+=(--continuous-frame-header)
elif [[ "${LABEL_MODE}" == "one_frame" ]]; then
	ARGS+=(--one-frame-header)
else
	echo "[ERROR] LABEL_MODE must be 'continuous' or 'one_frame', got '${LABEL_MODE}'" >&2
	exit 1
fi

if [[ "${STRICT_DECODE_ERRORS}" == "1" ]]; then
	ARGS+=(--drop-on-decode-error --ffmpeg-bin "${FFMPEG_BIN}")
fi

if [[ -n "${DEVICE}" ]]; then
	ARGS+=(--device "${DEVICE}")
fi

if [[ "${OPTIMIZE}" == "1" ]]; then
	ARGS+=(--optimize --optimize-batch-size "${OPTIMIZE_BATCH_SIZE}")
	if [[ "${OPTIMIZE_COMPILE}" == "1" ]]; then
		ARGS+=(--optimize-compile)
	fi
fi

if [[ -n "${MATCH_FILTER}" ]]; then
	# shellcheck disable=SC2206
	MATCH_FILTER_ARR=(${MATCH_FILTER})
	ARGS+=(--match-filter "${MATCH_FILTER_ARR[@]}")
fi

"${PYTHON_BIN}" "${ARGS[@]}"

echo ""
echo "Dense test parquet created:"
echo "  ${OUTPUT_PATH}"
