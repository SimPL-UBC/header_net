#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"

# Env overrides:
# LABEL_MODE: "one_frame" or "continuous" (default: continuous)
# CONTINUOUS_OFFSETS_25FPS: space-separated offsets for continuous labels at 25fps
#   default: "-1 0 1 2 3"
# CONTINUOUS_OFFSETS_50FPS: space-separated offsets for continuous labels at 50fps/high-fps videos
#   default: "-2 -1 0 1 2 3 4 5 6"
# MODEL_NUM_FRAMES: model clip length used for window-valid row filtering (default: 16)
# WEIGHTS_PATH: path to SoccerNet RF-DETR weights
# CONF_THRESHOLD: detection confidence threshold (default: 0.25)
# BATCH_SIZE: inference batch size (default: 8)
# DECODE_CHUNK_SIZE: decord decode chunk size (default: 256)
# TOPK: max detections per frame (default: 15)
# DEVICE: torch device override (cpu/cuda/mps; auto if empty)
# OPTIMIZE: 1 to enable RF-DETR optimization (default: 1)
# OPTIMIZE_BATCH_SIZE: batch size during optimization (default: 1)
# OPTIMIZE_COMPILE: 1 to enable torch compile (default: 0)
# OUTPUT_BASE: base output directory (default: ${REPO_ROOT}/output/dense_dataset)
# MATCH_FILTER: space-separated match names to process (default: all)
# MIN_DECODE_RATIO: minimum decoded/expected frame ratio to keep a video (default: 0.5)
# REMOVE_HEAVILY_CORRUPTED: set 1 to delete videos flagged with decode_rate_below_threshold (default: 0)
# STRICT_DECODE_ERRORS: set 1 to skip videos with any ffmpeg decode error (default: 1)
# FFMPEG_BIN: ffmpeg executable for strict decode scans (default: ffmpeg)

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
BATCH_SIZE="${BATCH_SIZE:-96}"
DECODE_CHUNK_SIZE="${DECODE_CHUNK_SIZE:-256}"
TOPK="${TOPK:-15}"
MODEL_NUM_FRAMES="${MODEL_NUM_FRAMES:-16}"
CONTINUOUS_OFFSETS_25FPS="${CONTINUOUS_OFFSETS_25FPS:--1 0 1 2 3}"
CONTINUOUS_OFFSETS_50FPS="${CONTINUOUS_OFFSETS_50FPS:--2 -1 0 1 2 3 4 5 6}"
MIN_DECODE_RATIO="${MIN_DECODE_RATIO:-0.9}"
REMOVE_HEAVILY_CORRUPTED="${REMOVE_HEAVILY_CORRUPTED:-0}"
STRICT_DECODE_ERRORS="${STRICT_DECODE_ERRORS:-0}"
FFMPEG_BIN="${FFMPEG_BIN:-ffmpeg}"
DEVICE="${DEVICE:-}"
MATCH_FILTER="${MATCH_FILTER:-}"
OPTIMIZE="${OPTIMIZE:-1}"
OPTIMIZE_BATCH_SIZE="${OPTIMIZE_BATCH_SIZE:-1}"
OPTIMIZE_COMPILE="${OPTIMIZE_COMPILE:-0}"
LABEL_MODE="${LABEL_MODE:-continuous}"

OUTPUT_PATH="${OUTPUT_BASE}/dense_test"
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

if ! "${PYTHON_BIN}" - "${MIN_DECODE_RATIO}" <<'PY'; then
import sys

value = float(sys.argv[1])
if not 0.0 <= value <= 1.0:
    raise ValueError("MIN_DECODE_RATIO must be in [0.0, 1.0]")
PY
	echo "[ERROR] Invalid MIN_DECODE_RATIO: ${MIN_DECODE_RATIO}" >&2
	exit 1
fi

cleanup_heavily_corrupted_videos() {
	local failed_csv="$1"
	local split_name="$2"

	if [[ "${REMOVE_HEAVILY_CORRUPTED}" != "1" ]]; then
		return 0
	fi

	if [[ ! -f "${failed_csv}" ]]; then
		echo "[INFO] ${split_name}: no failed video log at ${failed_csv}; skipping cleanup."
		return 0
	fi

	if ! "${PYTHON_BIN}" - "${failed_csv}" "${split_name}" <<'PY'; then
import csv
import sys
from pathlib import Path

failed_csv = Path(sys.argv[1])
split_name = sys.argv[2]
reasons_to_remove = {"decode_rate_below_threshold", "ffmpeg_decode_error"}

paths = []
with failed_csv.open("r", encoding="utf-8", newline="") as fh:
    reader = csv.DictReader(fh)
    for row in reader:
        reason = (row.get("reason") or "").strip()
        if reason not in reasons_to_remove:
            continue
        path_text = (row.get("path") or "").strip()
        if path_text:
            paths.append(path_text)

unique_paths = []
seen = set()
for value in paths:
    if value in seen:
        continue
    seen.add(value)
    unique_paths.append(Path(value))

removed = 0
missing = 0
delete_errors = 0
for path in unique_paths:
    try:
        if path.exists():
            path.unlink()
            removed += 1
        else:
            missing += 1
    except Exception as exc:
        delete_errors += 1
        print(f"[WARN] {split_name}: unable to delete {path}: {exc}")

print(
    f"[INFO] {split_name}: corrupted cleanup candidates={len(unique_paths)} "
    f"removed={removed} missing={missing} delete_errors={delete_errors}"
)
PY
		echo "[WARN] ${split_name}: cleanup step failed for ${failed_csv}" >&2
	fi
}

echo "============================================================"
echo "Dense test dataset generation — ${LABEL_MODE} mode"
echo "============================================================"
echo "Model num frames: ${MODEL_NUM_FRAMES}"
if [[ "${LABEL_MODE}" == "continuous" ]]; then
	echo "Continuous offsets @25fps: ${CONTINUOUS_OFFSETS_25FPS}"
	echo "Continuous offsets @50fps: ${CONTINUOUS_OFFSETS_50FPS}"
fi
echo "Min decode ratio: ${MIN_DECODE_RATIO}"
echo "Strict decode errors: ${STRICT_DECODE_ERRORS} (ffmpeg=${FFMPEG_BIN})"
echo "Delete heavily corrupted videos: ${REMOVE_HEAVILY_CORRUPTED}"
echo "Dataset path: ${DATASET_PATH}"
echo "Output path:  ${OUTPUT_PATH}"

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
	--decode-chunk-size "${DECODE_CHUNK_SIZE}"
	--topk "${TOPK}"
	--model-num-frames "${MODEL_NUM_FRAMES}"
	--min-decode-ratio "${MIN_DECODE_RATIO}"
)

if [[ "${LABEL_MODE}" == "continuous" ]]; then
	# shellcheck disable=SC2206
	CONTINUOUS_OFFSETS_25_ARR=(${CONTINUOUS_OFFSETS_25FPS})
	# shellcheck disable=SC2206
	CONTINUOUS_OFFSETS_50_ARR=(${CONTINUOUS_OFFSETS_50FPS})
	if [[ "${#CONTINUOUS_OFFSETS_25_ARR[@]}" -eq 0 ]]; then
		echo "[ERROR] CONTINUOUS_OFFSETS_25FPS cannot be empty in continuous mode." >&2
		exit 1
	fi
	if [[ "${#CONTINUOUS_OFFSETS_50_ARR[@]}" -eq 0 ]]; then
		echo "[ERROR] CONTINUOUS_OFFSETS_50FPS cannot be empty in continuous mode." >&2
		exit 1
	fi
	ARGS+=(--continuous-frame-header)
	ARGS+=(--continuous-offsets-25fps "${CONTINUOUS_OFFSETS_25_ARR[@]}")
	ARGS+=(--continuous-offsets-50fps "${CONTINUOUS_OFFSETS_50_ARR[@]}")
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
cleanup_heavily_corrupted_videos "${FAILED_LOG_PATH}" "test"

echo ""
echo "Dense test parquet created:"
echo "  ${OUTPUT_PATH}"
