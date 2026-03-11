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

PARQUET="${PARQUET:-${REPO_ROOT}/output/dense_dataset/dense_test.parquet}"
CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/output/vmae_parquet_ratio10/checkpoints/last.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/vmae_parquet_ratio10/test_inference}"
OUTPUT_CSV="${OUTPUT_CSV:-}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/SoccerNet}"
VIDEO_ID="${VIDEO_ID:-}"
HALF="${HALF:-}"
BACKBONE_CKPT="${BACKBONE_CKPT:-}"
NUM_FRAMES="${NUM_FRAMES:-}"
INPUT_SIZE="${INPUT_SIZE:-}"
BATCH_SIZE="${BATCH_SIZE:-128}"
# decord is required: pip install decord>=0.6.0
NUM_WORKERS="${NUM_WORKERS:-6}"
MAX_OPEN_VIDEOS="${MAX_OPEN_VIDEOS:-1}"
PIN_MEMORY="${PIN_MEMORY:-off}"
DEBUG_MEMORY="${DEBUG_MEMORY:-0}"
GPUS="${GPUS:-0 1}"
DEVICE="${DEVICE:-}"
SEED="${SEED:-}"

if [[ ! -f "${PARQUET}" ]]; then
	echo "[ERROR] Parquet not found: ${PARQUET}" >&2
	exit 1
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
	echo "[ERROR] Checkpoint not found: ${CHECKPOINT}" >&2
	exit 1
fi
if [[ ! -d "${DATASET_ROOT}" ]]; then
	echo "[ERROR] Dataset root not found: ${DATASET_ROOT}" >&2
	exit 1
fi

ARGS=(
	"${REPO_ROOT}/inference_parquet_test.py"
	--parquet "${PARQUET}"
	--checkpoint "${CHECKPOINT}"
	--output-dir "${OUTPUT_DIR}"
	--dataset-root "${DATASET_ROOT}"
	--batch-size "${BATCH_SIZE}"
	--num-workers "${NUM_WORKERS}"
	--max-open-videos "${MAX_OPEN_VIDEOS}"
	--pin-memory "${PIN_MEMORY}"
)

if [[ -n "${OUTPUT_CSV}" ]]; then
	ARGS+=(--output-csv "${OUTPUT_CSV}")
fi
if [[ -n "${VIDEO_ID}" ]]; then
	ARGS+=(--video-id "${VIDEO_ID}")
fi
if [[ -n "${HALF}" ]]; then
	ARGS+=(--half "${HALF}")
fi
if [[ -n "${BACKBONE_CKPT}" ]]; then
	ARGS+=(--backbone-ckpt "${BACKBONE_CKPT}")
fi
if [[ -n "${NUM_FRAMES}" ]]; then
	ARGS+=(--num-frames "${NUM_FRAMES}")
fi
if [[ -n "${INPUT_SIZE}" ]]; then
	ARGS+=(--input-size "${INPUT_SIZE}")
fi
if [[ -n "${GPUS}" ]]; then
	# shellcheck disable=SC2206
	GPU_ARR=(${GPUS})
	ARGS+=(--gpus "${GPU_ARR[@]}")
elif [[ -n "${DEVICE}" ]]; then
	ARGS+=(--device "${DEVICE}")
fi
if [[ -n "${SEED}" ]]; then
	ARGS+=(--seed "${SEED}")
fi
case "${DEBUG_MEMORY}" in
	1|true|TRUE|on|ON|yes|YES)
		ARGS+=(--debug-memory)
		;;
esac

"${PYTHON_BIN}" "${ARGS[@]}"

echo ""
echo "Inference output:"
if [[ -n "${OUTPUT_CSV}" ]]; then
	echo "  ${OUTPUT_CSV}"
else
	echo "  ${OUTPUT_DIR}/test_predictions_raw.csv"
fi
