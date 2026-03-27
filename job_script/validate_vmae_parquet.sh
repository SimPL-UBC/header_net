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

# Env overrides:
# CHECKPOINT_PATH: checkpoint .pt to evaluate
# VAL_PARQUET: validation parquet file or partitioned parquet dataset directory
# DATASET_ROOT: SoccerNet root for strict path validation
# OUTPUT_DIR: optional validation output directory; empty uses checkpoint run dir
# BATCH_SIZE, NUM_FRAMES
# VAL_NUM_WORKERS, VAL_PIN_MEMORY, VAL_PROGRESS_EVERY
# MAX_OPEN_VIDEOS, FRAME_CACHE_SIZE, LOADER_START_METHOD
# VAL_NEG_POS_RATIO: all|positive integer
# VIDEO_ID, HALF: optional validation subset filters
# BACKBONE, FINETUNE_MODE, UNFREEZE_BLOCKS, BACKBONE_CKPT
# BASE_LR, LAYER_LR_DECAY
# LOSS, FOCAL_GAMMA, FOCAL_ALPHA
# SEED, GPUS
# F1_THRESHOLD_STEP
# SAVE_PREDICTIONS: true|false
# SKIP_EXISTING: true|false, skip when metrics output already exists
# REUSE_PREDICTIONS: true|false, rebuild metrics from an existing predictions CSV when possible
# PREDICTIONS_PATH, METRICS_PATH

CHECKPOINT_PATH="${CHECKPOINT_PATH:-${REPO_ROOT}/output/vmae_parquet_ratio10/checkpoints/last.pt}"
VAL_PARQUET="${VAL_PARQUET:-${REPO_ROOT}/output/dense_dataset/dense_val}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/SoccerNet}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

BATCH_SIZE="${BATCH_SIZE:-}"
NUM_FRAMES="${NUM_FRAMES:-}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-8}"
VAL_PIN_MEMORY="${VAL_PIN_MEMORY:-true}"
VAL_PROGRESS_EVERY="${VAL_PROGRESS_EVERY:-1000}"
MAX_OPEN_VIDEOS="${MAX_OPEN_VIDEOS:-8}"
FRAME_CACHE_SIZE="${FRAME_CACHE_SIZE:-128}"
LOADER_START_METHOD="${LOADER_START_METHOD:-spawn}"
VAL_NEG_POS_RATIO="${VAL_NEG_POS_RATIO:-all}"
VIDEO_ID="${VIDEO_ID:-}"
HALF="${HALF:-}"

BACKBONE="${BACKBONE:-}"
FINETUNE_MODE="${FINETUNE_MODE:-}"
UNFREEZE_BLOCKS="${UNFREEZE_BLOCKS:-}"
BACKBONE_CKPT="${BACKBONE_CKPT:-}"
BASE_LR="${BASE_LR:-}"
LAYER_LR_DECAY="${LAYER_LR_DECAY:-}"
LOSS="${LOSS:-}"
FOCAL_GAMMA="${FOCAL_GAMMA:-}"
FOCAL_ALPHA="${FOCAL_ALPHA:-}"
SEED="${SEED:-}"
GPUS="${GPUS:-0}"
F1_THRESHOLD_STEP="${F1_THRESHOLD_STEP:-0.01}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-true}"
SKIP_EXISTING="${SKIP_EXISTING:-false}"
REUSE_PREDICTIONS="${REUSE_PREDICTIONS:-true}"
PREDICTIONS_PATH="${PREDICTIONS_PATH:-}"
METRICS_PATH="${METRICS_PATH:-}"

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
	echo "[ERROR] Checkpoint not found: ${CHECKPOINT_PATH}" >&2
	exit 1
fi
if [[ ! -e "${VAL_PARQUET}" ]]; then
	echo "[ERROR] Validation parquet not found: ${VAL_PARQUET}" >&2
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

ARGS=(
	-m training.cli_train_header_parquet_eval
	--checkpoint_path "${CHECKPOINT_PATH}"
	--val_parquet "${VAL_PARQUET}"
	--dataset_root "${DATASET_ROOT}"
	--val_num_workers "${VAL_NUM_WORKERS}"
	--val_progress_every "${VAL_PROGRESS_EVERY}"
	--max_open_videos "${MAX_OPEN_VIDEOS}"
	--frame_cache_size "${FRAME_CACHE_SIZE}"
	--loader_start_method "${LOADER_START_METHOD}"
	--val_neg_pos_ratio "${VAL_NEG_POS_RATIO}"
	--f1_threshold_step "${F1_THRESHOLD_STEP}"
)

if [[ -n "${OUTPUT_DIR}" ]]; then
	ARGS+=(--output_dir "${OUTPUT_DIR}")
fi
if [[ -n "${BATCH_SIZE}" ]]; then
	ARGS+=(--batch_size "${BATCH_SIZE}")
fi
if [[ -n "${NUM_FRAMES}" ]]; then
	ARGS+=(--num_frames "${NUM_FRAMES}")
fi
if [[ -n "${VIDEO_ID}" ]]; then
	ARGS+=(--video_id "${VIDEO_ID}")
fi
if [[ -n "${HALF}" ]]; then
	ARGS+=(--half "${HALF}")
fi
if [[ -n "${BACKBONE}" ]]; then
	ARGS+=(--backbone "${BACKBONE}")
fi
if [[ -n "${FINETUNE_MODE}" ]]; then
	ARGS+=(--finetune_mode "${FINETUNE_MODE}")
fi
if [[ -n "${UNFREEZE_BLOCKS}" ]]; then
	ARGS+=(--unfreeze_blocks "${UNFREEZE_BLOCKS}")
fi
if [[ -n "${BACKBONE_CKPT}" ]]; then
	ARGS+=(--backbone_ckpt "${BACKBONE_CKPT}")
fi
if [[ -n "${BASE_LR}" ]]; then
	ARGS+=(--base_lr "${BASE_LR}")
fi
if [[ -n "${LAYER_LR_DECAY}" ]]; then
	ARGS+=(--layer_lr_decay "${LAYER_LR_DECAY}")
fi
if [[ -n "${LOSS}" ]]; then
	ARGS+=(--loss "${LOSS}")
fi
if [[ -n "${FOCAL_GAMMA}" ]]; then
	ARGS+=(--focal_gamma "${FOCAL_GAMMA}")
fi
if [[ -n "${FOCAL_ALPHA}" ]]; then
	ARGS+=(--focal_alpha "${FOCAL_ALPHA}")
fi
if [[ -n "${SEED}" ]]; then
	ARGS+=(--seed "${SEED}")
fi
if [[ -n "${GPUS}" ]]; then
	# shellcheck disable=SC2206
	GPU_ARR=(${GPUS})
	ARGS+=(--gpus "${GPU_ARR[@]}")
fi
if [[ -n "${PREDICTIONS_PATH}" ]]; then
	ARGS+=(--predictions_path "${PREDICTIONS_PATH}")
fi
if [[ -n "${METRICS_PATH}" ]]; then
	ARGS+=(--metrics_path "${METRICS_PATH}")
fi

if is_enabled "${VAL_PIN_MEMORY}"; then
	ARGS+=(--val_pin_memory)
else
	ARGS+=(--no-val_pin_memory)
fi

if is_enabled "${SAVE_PREDICTIONS}"; then
	ARGS+=(--save_predictions)
else
	ARGS+=(--no-save_predictions)
fi
if is_enabled "${SKIP_EXISTING}"; then
	ARGS+=(--skip_existing)
else
	ARGS+=(--no-skip_existing)
fi
if is_enabled "${REUSE_PREDICTIONS}"; then
	ARGS+=(--reuse_predictions)
else
	ARGS+=(--no-reuse_predictions)
fi

echo "============================================================"
echo "Parquet validation"
echo "============================================================"
echo "Checkpoint:       ${CHECKPOINT_PATH}"
echo "Val parquet:      ${VAL_PARQUET}"
echo "Dataset root:     ${DATASET_ROOT}"
echo "Workers:          ${VAL_NUM_WORKERS}"
echo "Pin memory:       ${VAL_PIN_MEMORY}"
echo "Max open videos:  ${MAX_OPEN_VIDEOS}"
echo "Frame cache:      ${FRAME_CACHE_SIZE}"
echo "Start method:     ${LOADER_START_METHOD}"
echo "Val neg:pos:      ${VAL_NEG_POS_RATIO}"
echo "Save predictions: ${SAVE_PREDICTIONS}"
echo "Skip existing:    ${SKIP_EXISTING}"
echo "Reuse preds:      ${REUSE_PREDICTIONS}"
if [[ -n "${VIDEO_ID}" ]]; then
	echo "Video filter:     ${VIDEO_ID}"
fi
if [[ -n "${HALF}" ]]; then
	echo "Half filter:      ${HALF}"
fi
if [[ -n "${OUTPUT_DIR}" ]]; then
	echo "Output dir:       ${OUTPUT_DIR}"
fi
echo "============================================================"

"${PYTHON_BIN}" "${ARGS[@]}"
