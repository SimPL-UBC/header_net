#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"

# Env overrides:
# TRAIN_PARQUET, VAL_PARQUET
# DATASET_ROOT: path to SoccerNet root
# NEG_POS_RATIO: 10|20|30|all
# BACKBONE_CKPT: VideoMAE checkpoint directory
# OUTPUT_ROOT, RUN_NAME
# FINETUNE_MODE: full|frozen|partial
# UNFREEZE_BLOCKS
# EPOCHS, BATCH_SIZE, NUM_FRAMES, NUM_WORKERS
# OPTIMIZER, BASE_LR, LAYER_LR_DECAY, BETAS, WEIGHT_DECAY
# LOSS, FOCAL_GAMMA, FOCAL_ALPHA
# F1_THRESHOLD_STEP
# SEED, GPUS
# SAVE_EPOCH_INDICES: true|false

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

TRAIN_PARQUET="${TRAIN_PARQUET:-${REPO_ROOT}/output/dense_dataset/dense_train.parquet}"
VAL_PARQUET="${VAL_PARQUET:-${REPO_ROOT}/output/dense_dataset/dense_val.parquet}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/SoccerNet}"
NEG_POS_RATIO="${NEG_POS_RATIO:-10}"
BACKBONE_CKPT="${BACKBONE_CKPT:-${REPO_ROOT}/checkpoints/VideoMAEv2-Base}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/output/vmae}"
RUN_NAME="${RUN_NAME:-vmae_parquet_ratio10}"
FINETUNE_MODE="${FINETUNE_MODE:-full}"
UNFREEZE_BLOCKS="${UNFREEZE_BLOCKS:-4}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_FRAMES="${NUM_FRAMES:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
OPTIMIZER="${OPTIMIZER:-adamw}"
BASE_LR="${BASE_LR:-1e-3}"
LAYER_LR_DECAY="${LAYER_LR_DECAY:-0.75}"
BETAS="${BETAS:-0.9 0.999}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
LOSS="${LOSS:-focal}"
FOCAL_GAMMA="${FOCAL_GAMMA:-2.0}"
FOCAL_ALPHA="${FOCAL_ALPHA:-0.75}"
F1_THRESHOLD_STEP="${F1_THRESHOLD_STEP:-0.01}"
SEED="${SEED:-42}"
GPUS="${GPUS:-0 1}"
SAVE_EPOCH_INDICES="${SAVE_EPOCH_INDICES:-true}"

ARGS=(
  -m training.cli_train_header_parquet
  --train_parquet "${TRAIN_PARQUET}"
  --val_parquet "${VAL_PARQUET}"
  --dataset_root "${DATASET_ROOT}"
  --neg_pos_ratio "${NEG_POS_RATIO}"
  --backbone vmae
  --finetune_mode "${FINETUNE_MODE}"
  --unfreeze_blocks "${UNFREEZE_BLOCKS}"
  --backbone_ckpt "${BACKBONE_CKPT}"
  --run_name "${RUN_NAME}"
  --output_root "${OUTPUT_ROOT}"
  --epochs "${EPOCHS}"
  --num_frames "${NUM_FRAMES}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --optimizer "${OPTIMIZER}"
  --base_lr "${BASE_LR}"
  --layer_lr_decay "${LAYER_LR_DECAY}"
  --betas ${BETAS}
  --weight_decay "${WEIGHT_DECAY}"
  --loss "${LOSS}"
  --focal_gamma "${FOCAL_GAMMA}"
  --focal_alpha "${FOCAL_ALPHA}"
  --f1_threshold_step "${F1_THRESHOLD_STEP}"
  --seed "${SEED}"
  --gpus ${GPUS}
)

if [[ "${SAVE_EPOCH_INDICES}" == "true" ]]; then
  ARGS+=(--save_epoch_indices)
else
  ARGS+=(--no-save_epoch_indices)
fi

"${PYTHON_BIN}" "${ARGS[@]}"
