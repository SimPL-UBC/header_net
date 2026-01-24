#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"

# Env overrides:
# TRAIN_CSV: path to training CSV
# VAL_CSV: path to validation CSV
# BACKBONE_CKPT: VideoMAE checkpoint directory
# OUTPUT_ROOT: base output directory for runs
# RUN_NAME: experiment name
# FINETUNE_MODE: full|frozen|partial (default: full)
# UNFREEZE_BLOCKS: last N blocks to unfreeze (partial mode only)
# EPOCHS, BATCH_SIZE, NUM_FRAMES, NUM_WORKERS
# OPTIMIZER: adamw|sgd
# BASE_LR: base learning rate (scaled by batch_size/256)
# LAYER_LR_DECAY: layer-wise LR decay for VMAE (default: 0.75)
# BETAS: AdamW betas (default: "0.9 0.999")
# WEIGHT_DECAY: AdamW weight decay (default: 0.05)
# LOSS: focal|ce
# FOCAL_GAMMA, FOCAL_ALPHA
# SEED
# GPUS: space-separated GPU IDs (default: "0 1")

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

TRAIN_CSV="${TRAIN_CSV:-${REPO_ROOT}/output/dataset_generation/train_aug/train_with_aug.csv}"
VAL_CSV="${VAL_CSV:-${REPO_ROOT}/output/dataset_generation/val/val_cache_header.csv}"
BACKBONE_CKPT="${BACKBONE_CKPT:-${REPO_ROOT}/checkpoints/VideoMAEv2-Base}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/output/vmae}"
RUN_NAME="${RUN_NAME:-vmae_full_base}"
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
GPUS="${GPUS:-0 1}"
SEED="${SEED:-42}"

"${PYTHON_BIN}" -m training.cli_train_header \
  --train_csv "${TRAIN_CSV}" \
  --val_csv "${VAL_CSV}" \
  --backbone vmae \
  --finetune_mode "${FINETUNE_MODE}" \
  --unfreeze_blocks "${UNFREEZE_BLOCKS}" \
  --backbone_ckpt "${BACKBONE_CKPT}" \
  --run_name "${RUN_NAME}" \
  --output_root "${OUTPUT_ROOT}" \
  --epochs "${EPOCHS}" \
  --num_frames "${NUM_FRAMES}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --optimizer "${OPTIMIZER}" \
  --base_lr "${BASE_LR}" \
  --layer_lr_decay "${LAYER_LR_DECAY}" \
  --betas ${BETAS} \
  --weight_decay "${WEIGHT_DECAY}" \
  --loss "${LOSS}" \
  --focal_gamma "${FOCAL_GAMMA}" \
  --focal_alpha "${FOCAL_ALPHA}" \
  --seed "${SEED}" \
  --gpus ${GPUS}
