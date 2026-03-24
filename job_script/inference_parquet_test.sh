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

PARQUET="${PARQUET:-${REPO_ROOT}/output/dense_dataset/dense_test}"
CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/output/vmae_parquet_ratio10/checkpoints/last.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/vmae_parquet_ratio10/test_inference}"
OUTPUT_CSV="${OUTPUT_CSV:-}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/SoccerNet}"
VIDEO_ID="${VIDEO_ID:-}"
VIDEO_IDS="${VIDEO_IDS:-}"
HALF="${HALF:-}"
BACKBONE_CKPT="${BACKBONE_CKPT:-}"
NUM_FRAMES="${NUM_FRAMES:-}"
INPUT_SIZE="${INPUT_SIZE:-}"
BATCH_SIZE="${BATCH_SIZE:-128}"
# decord is required: pip install decord>=0.6.0
NUM_WORKERS="${NUM_WORKERS:-6}"
MAX_OPEN_VIDEOS="${MAX_OPEN_VIDEOS:-1}"
FRAME_CACHE_SIZE="${FRAME_CACHE_SIZE:-128}"
LOADER_START_METHOD="${LOADER_START_METHOD:-spawn}"
PIN_MEMORY="${PIN_MEMORY:-off}"
DEBUG_MEMORY="${DEBUG_MEMORY:-0}"
GPUS="${GPUS:-0 1}"
DEVICE="${DEVICE:-}"
SEED="${SEED:-}"
PROCESS_BY_MATCH="${PROCESS_BY_MATCH:-0}"

if [[ ! -e "${PARQUET}" ]]; then
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

is_enabled() {
	case "$1" in
	1|true|TRUE|on|ON|yes|YES)
		return 0
		;;
	*)
		return 1
		;;
	esac
}

resolve_parquet_video_ids() {
	"${PYTHON_BIN}" - "${PARQUET}" <<'PY'
import sys

import pandas as pd

parquet_path = sys.argv[1]
df = pd.read_parquet(parquet_path, columns=["video_id"])
seen = set()
for value in df["video_id"].astype(str).tolist():
    if not value or value in seen:
        continue
    seen.add(value)
    print(value)
PY
}

COMMON_ARGS=(
	"${REPO_ROOT}/inference_parquet_test.py"
	--parquet "${PARQUET}"
	--checkpoint "${CHECKPOINT}"
	--output-dir "${OUTPUT_DIR}"
	--dataset-root "${DATASET_ROOT}"
	--batch-size "${BATCH_SIZE}"
	--num-workers "${NUM_WORKERS}"
	--max-open-videos "${MAX_OPEN_VIDEOS}"
	--frame-cache-size "${FRAME_CACHE_SIZE}"
	--loader-start-method "${LOADER_START_METHOD}"
	--pin-memory "${PIN_MEMORY}"
)

if [[ -n "${HALF}" ]]; then
	COMMON_ARGS+=(--half "${HALF}")
fi
if [[ -n "${BACKBONE_CKPT}" ]]; then
	COMMON_ARGS+=(--backbone-ckpt "${BACKBONE_CKPT}")
fi
if [[ -n "${NUM_FRAMES}" ]]; then
	COMMON_ARGS+=(--num-frames "${NUM_FRAMES}")
fi
if [[ -n "${INPUT_SIZE}" ]]; then
	COMMON_ARGS+=(--input-size "${INPUT_SIZE}")
fi
if [[ -n "${GPUS}" ]]; then
	# shellcheck disable=SC2206
	GPU_ARR=(${GPUS})
	COMMON_ARGS+=(--gpus "${GPU_ARR[@]}")
elif [[ -n "${DEVICE}" ]]; then
	COMMON_ARGS+=(--device "${DEVICE}")
fi
if [[ -n "${SEED}" ]]; then
	COMMON_ARGS+=(--seed "${SEED}")
fi
if is_enabled "${DEBUG_MEMORY}"; then
	COMMON_ARGS+=(--debug-memory)
fi

if is_enabled "${PROCESS_BY_MATCH}"; then
	if [[ -n "${OUTPUT_CSV}" ]]; then
		echo "[ERROR] OUTPUT_CSV is not supported when PROCESS_BY_MATCH=1." >&2
		exit 1
	fi

	video_ids_output="$(resolve_parquet_video_ids)"
	if [[ -z "${video_ids_output}" ]]; then
		echo "[ERROR] No video_id values found in parquet: ${PARQUET}" >&2
		exit 1
	fi

	mapfile -t available_matches <<< "${video_ids_output}"
	declare -A available_lookup=()
	for match_id in "${available_matches[@]}"; do
		available_lookup["${match_id}"]=1
	done

	requested_matches=()
	if [[ -n "${VIDEO_IDS}" ]]; then
		# shellcheck disable=SC2206
		requested_matches=(${VIDEO_IDS})
	elif [[ -n "${VIDEO_ID}" ]]; then
		requested_matches=("${VIDEO_ID}")
	else
		requested_matches=("${available_matches[@]}")
	fi

	if [[ "${#requested_matches[@]}" -eq 0 ]]; then
		echo "[ERROR] PROCESS_BY_MATCH=1 but no matches were selected." >&2
		exit 1
	fi

	declare -A seen_matches=()
	matches_to_run=()
	failed_matches=()
	for match_id in "${requested_matches[@]}"; do
		if [[ -z "${match_id}" || -n "${seen_matches[${match_id}]:-}" ]]; then
			continue
		fi
		seen_matches["${match_id}"]=1
		if [[ -z "${available_lookup[${match_id}]:-}" ]]; then
			echo "[WARN] Match not found in parquet: ${match_id}" >&2
			failed_matches+=("${match_id}")
			continue
		fi
		matches_to_run+=("${match_id}")
	done

	if [[ "${#matches_to_run[@]}" -eq 0 ]]; then
		echo "[ERROR] No valid matches selected for per-match inference." >&2
		exit 1
	fi

	per_match_dir="${OUTPUT_DIR}/per_match"
	mkdir -p "${per_match_dir}"
	succeeded_matches=()

	echo "[INFO] PROCESS_BY_MATCH=1"
	echo "[INFO] Per-match output dir: ${per_match_dir}"
	echo "[INFO] Matches queued: ${#matches_to_run[@]}"

	for match_id in "${matches_to_run[@]}"; do
		match_output="${per_match_dir}/${match_id}"
		if [[ -n "${HALF}" ]]; then
			match_output+="_half${HALF}"
		fi
		match_output+=".csv"

		args=("${COMMON_ARGS[@]}" --video-id "${match_id}" --output-csv "${match_output}")

		echo ""
		echo "[MATCH] ${match_id}"
		echo "[MATCH] Output: ${match_output}"
		if "${PYTHON_BIN}" "${args[@]}"; then
			succeeded_matches+=("${match_id}")
		else
			failed_matches+=("${match_id}")
			echo "[WARN] Match failed: ${match_id}" >&2
		fi
	done

	echo ""
	echo "Per-match inference summary:"
	echo "  Output dir: ${per_match_dir}"
	echo "  Succeeded:  ${#succeeded_matches[@]}"
	for match_id in "${succeeded_matches[@]}"; do
		echo "    ${match_id}"
	done
	echo "  Failed:     ${#failed_matches[@]}"
	for match_id in "${failed_matches[@]}"; do
		echo "    ${match_id}"
	done

	if [[ "${#failed_matches[@]}" -ne 0 ]]; then
		exit 1
	fi
	exit 0
fi

ARGS=("${COMMON_ARGS[@]}")
if [[ -n "${OUTPUT_CSV}" ]]; then
	ARGS+=(--output-csv "${OUTPUT_CSV}")
fi
if [[ -n "${VIDEO_ID}" ]]; then
	ARGS+=(--video-id "${VIDEO_ID}")
fi

"${PYTHON_BIN}" "${ARGS[@]}"

echo ""
echo "Inference output:"
if [[ -n "${OUTPUT_CSV}" ]]; then
	echo "  ${OUTPUT_CSV}"
else
	echo "  ${OUTPUT_DIR}/test_predictions_raw.csv"
fi
