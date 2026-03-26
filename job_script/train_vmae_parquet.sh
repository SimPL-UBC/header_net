#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"

# Env overrides:
# TRAIN_PARQUET
# DATASET_ROOT: path to SoccerNet root
# NEG_POS_RATIO: all|positive integer (e.g., 6)
# BACKBONE_CKPT: VideoMAE checkpoint directory
# OUTPUT_ROOT, RUN_NAME
# RESUME_CHECKPOINT: optional periodic checkpoint path (e.g. output/vmae/<run>/checkpoints/epoch_014.pt)
# FINETUNE_MODE: full|frozen|partial
# UNFREEZE_BLOCKS
# EPOCHS, BATCH_SIZE, NUM_FRAMES, NUM_WORKERS
# MAX_OPEN_VIDEOS, FRAME_CACHE_SIZE, LOADER_START_METHOD
# OPTIMIZER, BASE_LR, LAYER_LR_DECAY, BETAS, WEIGHT_DECAY
# LOSS, FOCAL_GAMMA, FOCAL_ALPHA
# SEED, GPUS
# SAVE_EPOCH_INDICES: true|false
# SAVE_EVERY_N_EPOCHS: save periodic checkpoints every N epochs (default: 1)
# VALIDATE_VIDEO_LOAD: 1 to verify train parquet video paths can be opened/decoded before training (default: 1)
# VALIDATE_VIDEO_LOAD_MAX_ERRORS: max unreadable paths to print (default: 20)
# FILTER_BAD_WINDOWS: optional legacy pre-filter pass before training (default: 0)
#   Dense parquet generator now drops rows with incomplete model windows.
# FILTER_BAD_WINDOWS_FORCE_REBUILD: 1 to rebuild filtered parquet even if cached output exists (default: 0)
# FILTER_BAD_WINDOWS_OUTPUT_DIR: directory for filtered parquet cache/reports (default: ${REPO_ROOT}/output/dense_dataset/filtered_windows)
# FILTER_BAD_WINDOWS_CHUNK_SIZE: row chunk size for vectorized filtering (default: 200000)

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

TRAIN_PARQUET="${TRAIN_PARQUET:-${REPO_ROOT}/output/dense_dataset/dense_train}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/SoccerNet}"
NEG_POS_RATIO="${NEG_POS_RATIO:-10}"
BACKBONE_CKPT="${BACKBONE_CKPT:-${REPO_ROOT}/checkpoints/VideoMAEv2-Base}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/output/vmae_parquet_ratio10_new}"
RUN_NAME="${RUN_NAME:-vmae_parquet_ratio10_new}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
FINETUNE_MODE="${FINETUNE_MODE:-full}"
UNFREEZE_BLOCKS="${UNFREEZE_BLOCKS:-4}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_FRAMES="${NUM_FRAMES:-16}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MAX_OPEN_VIDEOS="${MAX_OPEN_VIDEOS:-2}"
FRAME_CACHE_SIZE="${FRAME_CACHE_SIZE:-128}"
LOADER_START_METHOD="${LOADER_START_METHOD:-spawn}"
OPTIMIZER="${OPTIMIZER:-adamw}"
BASE_LR="${BASE_LR:-1e-3}"
LAYER_LR_DECAY="${LAYER_LR_DECAY:-0.75}"
BETAS="${BETAS:-0.9 0.999}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
LOSS="${LOSS:-focal}"
FOCAL_GAMMA="${FOCAL_GAMMA:-2.0}"
FOCAL_ALPHA="${FOCAL_ALPHA:-0.75}"
SEED="${SEED:-42}"
GPUS="${GPUS:-0 1}"
SAVE_EPOCH_INDICES="${SAVE_EPOCH_INDICES:-true}"
SAVE_EVERY_N_EPOCHS="${SAVE_EVERY_N_EPOCHS:-1}"
VALIDATE_VIDEO_LOAD="${VALIDATE_VIDEO_LOAD:-1}"
VALIDATE_VIDEO_LOAD_MAX_ERRORS="${VALIDATE_VIDEO_LOAD_MAX_ERRORS:-20}"
FILTER_BAD_WINDOWS="${FILTER_BAD_WINDOWS:-0}"
FILTER_BAD_WINDOWS_FORCE_REBUILD="${FILTER_BAD_WINDOWS_FORCE_REBUILD:-0}"
FILTER_BAD_WINDOWS_OUTPUT_DIR="${FILTER_BAD_WINDOWS_OUTPUT_DIR:-${REPO_ROOT}/output/dense_dataset/filtered_windows}"
FILTER_BAD_WINDOWS_CHUNK_SIZE="${FILTER_BAD_WINDOWS_CHUNK_SIZE:-200000}"

if [[ "${FILTER_BAD_WINDOWS}" == "1" ]]; then
  mkdir -p "${FILTER_BAD_WINDOWS_OUTPUT_DIR}"

  TRAIN_STEM="$(basename "${TRAIN_PARQUET%.parquet}")"
  TRAIN_PARQUET_FILTERED="${FILTER_BAD_WINDOWS_OUTPUT_DIR}/${TRAIN_STEM}.valid_window_nf${NUM_FRAMES}.parquet"
  TRAIN_FILTER_REPORT="${FILTER_BAD_WINDOWS_OUTPUT_DIR}/${TRAIN_STEM}.valid_window_nf${NUM_FRAMES}.report.csv"

  filter_split() {
    local split_name="$1"
    local input_parquet="$2"
    local output_parquet="$3"
    local report_csv="$4"

    if [[ "${FILTER_BAD_WINDOWS_FORCE_REBUILD}" != "1" && -f "${output_parquet}" ]]; then
      echo "[INFO] Reusing filtered ${split_name} parquet: ${output_parquet}"
      return
    fi

    echo "[INFO] Filtering ${split_name} parquet windows..."
    "${PYTHON_BIN}" "${REPO_ROOT}/tools/filter_parquet_decode_windows.py" \
      --input-parquet "${input_parquet}" \
      --output-parquet "${output_parquet}" \
      --num-frames "${NUM_FRAMES}" \
      --chunk-size "${FILTER_BAD_WINDOWS_CHUNK_SIZE}" \
      --report-csv "${report_csv}"
  }

  filter_split "train" "${TRAIN_PARQUET}" "${TRAIN_PARQUET_FILTERED}" "${TRAIN_FILTER_REPORT}"

  TRAIN_PARQUET="${TRAIN_PARQUET_FILTERED}"
fi

if [[ "${VALIDATE_VIDEO_LOAD}" == "1" ]]; then
  echo "[INFO] Validating video readability from train parquet..."
  "${PYTHON_BIN}" - "${TRAIN_PARQUET}" "${VALIDATE_VIDEO_LOAD_MAX_ERRORS}" <<'PY'
import sys
from pathlib import Path

import decord
import pandas as pd

decord.bridge.set_bridge("native")

train_parquet = Path(sys.argv[1])
max_errors = int(sys.argv[2])

bad = []
if not train_parquet.exists():
    bad.append(("train", str(train_parquet), "parquet_missing"))
else:
    df = pd.read_parquet(train_parquet, columns=["video_path"])
    for video_path in pd.unique(df["video_path"].astype(str)):
        try:
            reader = decord.VideoReader(video_path, ctx=decord.cpu())
        except Exception:
            bad.append(("train", video_path, "open_failed"))
            continue

        frame_count = len(reader)
        if frame_count <= 0:
            bad.append(("train", video_path, f"invalid_frame_count={frame_count}"))
            continue
        try:
            reader.get_batch([0]).asnumpy()
        except Exception:
            bad.append(("train", video_path, "first_frame_decode_failed"))

if bad:
    print(
        f"[ERROR] Found {len(bad)} unreadable video path(s) referenced by parquet files.",
        file=sys.stderr,
    )
    for split_name, path, reason in bad[:max_errors]:
        print(f"  - {split_name}: {path} ({reason})", file=sys.stderr)
    if len(bad) > max_errors:
        print(f"  ... and {len(bad) - max_errors} more", file=sys.stderr)
    print(
        "[ERROR] Regenerate parquet with corrupted videos removed, "
        "or set VALIDATE_VIDEO_LOAD=0 to bypass this check.",
        file=sys.stderr,
    )
    sys.exit(1)

print("[INFO] Video readability check passed.")
PY
fi

ARGS=(
  -m training.cli_train_header_parquet_train
  --train_parquet "${TRAIN_PARQUET}"
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
  --max_open_videos "${MAX_OPEN_VIDEOS}"
  --frame_cache_size "${FRAME_CACHE_SIZE}"
  --loader_start_method "${LOADER_START_METHOD}"
  --optimizer "${OPTIMIZER}"
  --base_lr "${BASE_LR}"
  --layer_lr_decay "${LAYER_LR_DECAY}"
  --betas ${BETAS}
  --weight_decay "${WEIGHT_DECAY}"
  --loss "${LOSS}"
  --focal_gamma "${FOCAL_GAMMA}"
  --focal_alpha "${FOCAL_ALPHA}"
  --save_every_n_epochs "${SAVE_EVERY_N_EPOCHS}"
  --seed "${SEED}"
  --gpus ${GPUS}
)

if [[ "${SAVE_EPOCH_INDICES}" == "true" ]]; then
  ARGS+=(--save_epoch_indices)
else
  ARGS+=(--no-save_epoch_indices)
fi

if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  ARGS+=(--resume_checkpoint "${RESUME_CHECKPOINT}")
fi

"${PYTHON_BIN}" "${ARGS[@]}"
