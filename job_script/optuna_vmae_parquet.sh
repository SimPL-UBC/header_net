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

# Common user overrides:
# BACKBONES: "base", "giant", or "base giant" (default: base giant)
# N_TRIALS: Optuna trials per backbone (default: 20)
# OBJECTIVE_METRIC: metric to maximize from validation JSON (default: val_f1)
# TRAIN_PARQUET, VAL_PARQUET, DATASET_ROOT, OUTPUT_ROOT, OPTUNA_STORAGE
# ALPHA_MIN, ALPHA_MAX, GAMMA_MIN, GAMMA_MAX, NEG_POS_RATIOS
# EPOCHS, BATCH_SIZE, BASE_BATCH_SIZE, GIANT_BATCH_SIZE
# GRADIENT_ACCUMULATION_STEPS, BASE_GRADIENT_ACCUMULATION_STEPS, GIANT_GRADIENT_ACCUMULATION_STEPS
# BASE_LR, WEIGHT_DECAY, LAYER_LR_DECAY, GPUS, DDP_MODE
# RUN_BEST_VAL_INFERENCE, SAVE_VAL_PREDICTIONS, DRY_RUN

BACKBONES="${BACKBONES:-base giant}"
N_TRIALS="${N_TRIALS:-20}"
OBJECTIVE_METRIC="${OBJECTIVE_METRIC:-val_f1}"
STUDY_NAME_PREFIX="${STUDY_NAME_PREFIX:-optuna_vmae_parquet}"
STUDY_NAME_SUFFIX="${STUDY_NAME_SUFFIX:-auto}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/output/optuna_vmae_parquet}"
OPTUNA_STORAGE="${OPTUNA_STORAGE:-}"
SAMPLER_SEED="${SAMPLER_SEED:-42}"

TRAIN_PARQUET="${TRAIN_PARQUET:-${REPO_ROOT}/output/dense_dataset/dense_train}"
VAL_PARQUET="${VAL_PARQUET:-${REPO_ROOT}/output/dense_dataset/dense_val}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/SoccerNet}"
SPATIAL_MODE="${SPATIAL_MODE:-ball_crop}"
BASE_BACKBONE_CKPT="${BASE_BACKBONE_CKPT:-${REPO_ROOT}/checkpoints/VideoMAEv2-Base}"
GIANT_BACKBONE_CKPT="${GIANT_BACKBONE_CKPT:-${REPO_ROOT}/checkpoints/VideoMAEv2-giant}"

ALPHA_MIN="${ALPHA_MIN:-0.55}"
ALPHA_MAX="${ALPHA_MAX:-0.95}"
GAMMA_MIN="${GAMMA_MIN:-0.5}"
GAMMA_MAX="${GAMMA_MAX:-5.0}"
NEG_POS_RATIOS="${NEG_POS_RATIOS:-3 5 8 10 15 20}"

EPOCHS="${EPOCHS:-30}"
NUM_FRAMES="${NUM_FRAMES:-16}"
BATCH_SIZE="${BATCH_SIZE:-}"
BASE_BATCH_SIZE="${BASE_BATCH_SIZE:-16}"
GIANT_BATCH_SIZE="${GIANT_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-}"
BASE_GRADIENT_ACCUMULATION_STEPS="${BASE_GRADIENT_ACCUMULATION_STEPS:-1}"
GIANT_GRADIENT_ACCUMULATION_STEPS="${GIANT_GRADIENT_ACCUMULATION_STEPS:-2}"
NUM_WORKERS="${NUM_WORKERS:-}"
BASE_NUM_WORKERS="${BASE_NUM_WORKERS:-4}"
GIANT_NUM_WORKERS="${GIANT_NUM_WORKERS:-1}"
MAX_OPEN_VIDEOS="${MAX_OPEN_VIDEOS:-}"
BASE_MAX_OPEN_VIDEOS="${BASE_MAX_OPEN_VIDEOS:-4}"
GIANT_MAX_OPEN_VIDEOS="${GIANT_MAX_OPEN_VIDEOS:-1}"
FRAME_CACHE_SIZE="${FRAME_CACHE_SIZE:-128}"
LOADER_START_METHOD="${LOADER_START_METHOD:-spawn}"
OPTIMIZER="${OPTIMIZER:-adamw}"
BASE_LR="${BASE_LR:-1e-3}"
LAYER_LR_DECAY="${LAYER_LR_DECAY:-0.75}"
BETAS="${BETAS:-0.9 0.999}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
LOSS="${LOSS:-focal}"
FINETUNE_MODE="${FINETUNE_MODE:-full}"
UNFREEZE_BLOCKS="${UNFREEZE_BLOCKS:-4}"
SEED="${SEED:-42}"
GPUS="${GPUS:-0 1}"
DDP_MODE="${DDP_MODE:-auto}"
AMP="${AMP:-true}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
SAVE_EPOCH_INDICES="${SAVE_EPOCH_INDICES:-false}"
SAVE_EVERY_N_EPOCHS="${SAVE_EVERY_N_EPOCHS:-1}"
SAVE_EVERY_N_STEPS="${SAVE_EVERY_N_STEPS:-0}"
KEEP_LAST_N_STEP_CHECKPOINTS="${KEEP_LAST_N_STEP_CHECKPOINTS:-2}"
TRAIN_AUGMENTATION_MODE="${TRAIN_AUGMENTATION_MODE:-clip_consistent}"
RESAMPLE_ON_DECODE_FAILURE="${RESAMPLE_ON_DECODE_FAILURE:-false}"
VALIDATE_VIDEO_LOAD="${VALIDATE_VIDEO_LOAD:-false}"
VALIDATE_VIDEO_LOAD_MAX_ERRORS="${VALIDATE_VIDEO_LOAD_MAX_ERRORS:-20}"

VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-8}"
VAL_PIN_MEMORY="${VAL_PIN_MEMORY:-true}"
VAL_PROGRESS_EVERY="${VAL_PROGRESS_EVERY:-1000}"
VAL_NEG_POS_RATIO="${VAL_NEG_POS_RATIO:-all}"
F1_THRESHOLD_STEP="${F1_THRESHOLD_STEP:-0.01}"
SAVE_VAL_PREDICTIONS="${SAVE_VAL_PREDICTIONS:-false}"
REUSE_VAL_PREDICTIONS="${REUSE_VAL_PREDICTIONS:-false}"
RUN_BEST_VAL_INFERENCE="${RUN_BEST_VAL_INFERENCE:-false}"
BEST_INFERENCE_BATCH_SIZE="${BEST_INFERENCE_BATCH_SIZE:-128}"
BEST_INFERENCE_NUM_WORKERS="${BEST_INFERENCE_NUM_WORKERS:-6}"
BEST_INFERENCE_MAX_OPEN_VIDEOS="${BEST_INFERENCE_MAX_OPEN_VIDEOS:-1}"
DRY_RUN="${DRY_RUN:-0}"

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

normalize_words() {
	local text="$1"
	text="${text//,/ }"
	printf '%s\n' "${text}"
}

read -r -a BACKBONE_ARR <<<"$(normalize_words "${BACKBONES}")"
read -r -a NEG_POS_RATIO_ARR <<<"$(normalize_words "${NEG_POS_RATIOS}")"
read -r -a GPU_ARR <<<"$(normalize_words "${GPUS}")"
read -r -a BETA_ARR <<<"$(normalize_words "${BETAS}")"

ARGS=(
	"${REPO_ROOT}/tools/optuna_vmae_parquet.py"
	--backbones "${BACKBONE_ARR[@]}"
	--n-trials "${N_TRIALS}"
	--objective-metric "${OBJECTIVE_METRIC}"
	--study-name-prefix "${STUDY_NAME_PREFIX}"
	--study-name-suffix "${STUDY_NAME_SUFFIX}"
	--output-root "${OUTPUT_ROOT}"
	--sampler-seed "${SAMPLER_SEED}"
	--train-parquet "${TRAIN_PARQUET}"
	--val-parquet "${VAL_PARQUET}"
	--dataset-root "${DATASET_ROOT}"
	--spatial-mode "${SPATIAL_MODE}"
	--base-backbone-ckpt "${BASE_BACKBONE_CKPT}"
	--giant-backbone-ckpt "${GIANT_BACKBONE_CKPT}"
	--alpha-min "${ALPHA_MIN}"
	--alpha-max "${ALPHA_MAX}"
	--gamma-min "${GAMMA_MIN}"
	--gamma-max "${GAMMA_MAX}"
	--neg-pos-ratios "${NEG_POS_RATIO_ARR[@]}"
	--epochs "${EPOCHS}"
	--num-frames "${NUM_FRAMES}"
	--base-batch-size "${BASE_BATCH_SIZE}"
	--giant-batch-size "${GIANT_BATCH_SIZE}"
	--base-gradient-accumulation-steps "${BASE_GRADIENT_ACCUMULATION_STEPS}"
	--giant-gradient-accumulation-steps "${GIANT_GRADIENT_ACCUMULATION_STEPS}"
	--base-num-workers "${BASE_NUM_WORKERS}"
	--giant-num-workers "${GIANT_NUM_WORKERS}"
	--base-max-open-videos "${BASE_MAX_OPEN_VIDEOS}"
	--giant-max-open-videos "${GIANT_MAX_OPEN_VIDEOS}"
	--frame-cache-size "${FRAME_CACHE_SIZE}"
	--loader-start-method "${LOADER_START_METHOD}"
	--optimizer "${OPTIMIZER}"
	--base-lr "${BASE_LR}"
	--layer-lr-decay "${LAYER_LR_DECAY}"
	--betas "${BETA_ARR[@]}"
	--weight-decay "${WEIGHT_DECAY}"
	--loss "${LOSS}"
	--finetune-mode "${FINETUNE_MODE}"
	--unfreeze-blocks "${UNFREEZE_BLOCKS}"
	--seed "${SEED}"
	--ddp-mode "${DDP_MODE}"
	--save-every-n-epochs "${SAVE_EVERY_N_EPOCHS}"
	--save-every-n-steps "${SAVE_EVERY_N_STEPS}"
	--keep-last-n-step-checkpoints "${KEEP_LAST_N_STEP_CHECKPOINTS}"
	--train-augmentation-mode "${TRAIN_AUGMENTATION_MODE}"
	--validate-video-load-max-errors "${VALIDATE_VIDEO_LOAD_MAX_ERRORS}"
	--val-num-workers "${VAL_NUM_WORKERS}"
	--val-progress-every "${VAL_PROGRESS_EVERY}"
	--val-neg-pos-ratio "${VAL_NEG_POS_RATIO}"
	--f1-threshold-step "${F1_THRESHOLD_STEP}"
	--best-inference-batch-size "${BEST_INFERENCE_BATCH_SIZE}"
	--best-inference-num-workers "${BEST_INFERENCE_NUM_WORKERS}"
	--best-inference-max-open-videos "${BEST_INFERENCE_MAX_OPEN_VIDEOS}"
)

if [[ -n "${OPTUNA_STORAGE}" ]]; then
	ARGS+=(--storage "${OPTUNA_STORAGE}")
fi
if [[ -n "${BATCH_SIZE}" ]]; then
	ARGS+=(--batch-size "${BATCH_SIZE}")
fi
if [[ -n "${GRADIENT_ACCUMULATION_STEPS}" ]]; then
	ARGS+=(--gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}")
fi
if [[ -n "${NUM_WORKERS}" ]]; then
	ARGS+=(--num-workers "${NUM_WORKERS}")
fi
if [[ -n "${MAX_OPEN_VIDEOS}" ]]; then
	ARGS+=(--max-open-videos "${MAX_OPEN_VIDEOS}")
fi
if [[ -n "${GPUS}" ]]; then
	ARGS+=(--gpus "${GPU_ARR[@]}")
fi
if [[ -n "${VAL_BATCH_SIZE}" ]]; then
	ARGS+=(--val-batch-size "${VAL_BATCH_SIZE}")
fi

if is_enabled "${AMP}"; then
	ARGS+=(--amp)
else
	ARGS+=(--no-amp)
fi
if is_enabled "${GRADIENT_CHECKPOINTING}"; then
	ARGS+=(--gradient-checkpointing)
else
	ARGS+=(--no-gradient-checkpointing)
fi
if is_enabled "${SAVE_EPOCH_INDICES}"; then
	ARGS+=(--save-epoch-indices)
else
	ARGS+=(--no-save-epoch-indices)
fi
if is_enabled "${RESAMPLE_ON_DECODE_FAILURE}"; then
	ARGS+=(--resample-on-decode-failure)
else
	ARGS+=(--no-resample-on-decode-failure)
fi
if is_enabled "${VALIDATE_VIDEO_LOAD}"; then
	ARGS+=(--validate-video-load)
else
	ARGS+=(--no-validate-video-load)
fi
if is_enabled "${VAL_PIN_MEMORY}"; then
	ARGS+=(--val-pin-memory)
else
	ARGS+=(--no-val-pin-memory)
fi
if is_enabled "${SAVE_VAL_PREDICTIONS}"; then
	ARGS+=(--save-val-predictions)
else
	ARGS+=(--no-save-val-predictions)
fi
if is_enabled "${REUSE_VAL_PREDICTIONS}"; then
	ARGS+=(--reuse-val-predictions)
else
	ARGS+=(--no-reuse-val-predictions)
fi
if is_enabled "${RUN_BEST_VAL_INFERENCE}"; then
	ARGS+=(--run-best-val-inference)
else
	ARGS+=(--no-run-best-val-inference)
fi
if is_enabled "${DRY_RUN}"; then
	ARGS+=(--dry-run)
fi

echo "============================================================"
echo "Optuna VMAE2 parquet tuning"
echo "============================================================"
echo "Backbones:       ${BACKBONE_ARR[*]}"
echo "Trials/backbone: ${N_TRIALS}"
echo "Objective:       ${OBJECTIVE_METRIC}"
echo "Study suffix:    ${STUDY_NAME_SUFFIX}"
echo "Train parquet:   ${TRAIN_PARQUET}"
echo "Val parquet:     ${VAL_PARQUET}"
echo "Output root:     ${OUTPUT_ROOT}"
echo "Search alpha:    ${ALPHA_MIN}..${ALPHA_MAX}"
echo "Search gamma:    ${GAMMA_MIN}..${GAMMA_MAX}"
echo "Search ratios:   ${NEG_POS_RATIO_ARR[*]}"
echo "GPUs:            ${GPU_ARR[*]:-auto}"
echo "DDP mode:        ${DDP_MODE}"
echo "Dry run:         ${DRY_RUN}"
echo "============================================================"

"${PYTHON_BIN}" "${ARGS[@]}"
