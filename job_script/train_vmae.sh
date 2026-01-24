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
# LR_BACKBONE, LR_HEAD, WEIGHT_DECAY, SEED
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

TRAIN_CSV="${TRAIN_CSV:-${REPO_ROOT}/output/dataset_generation/train/train_cache_header.csv}"
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
LR_BACKBONE="${LR_BACKBONE:-1e-5}"
LR_HEAD="${LR_HEAD:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
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
  --lr_backbone "${LR_BACKBONE}" \
  --lr_head "${LR_HEAD}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --seed "${SEED}" \
  --gpus ${GPUS}
