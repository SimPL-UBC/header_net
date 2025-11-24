# VideoMAE v2 Header Training (Phase 2)

Minimal instructions to train the frozen VideoMAE v2 + head classifier on the header cache.

## 0) Environment
- Activate env: `conda activate deep_impact_env`
- Ensure deps installed (add if missing): `pip install -r requirements.txt`
- Local HF cache is set inside the repo (`.cache/hf_local`) by the code; no network needed if checkpoints are present.

## 1) Prepare splits
The cache CSV lives at `scratch_output/generate_dataset_test/16_frames_ver/dataset_generation/train_cache_header.csv`.

Split by `video_id` with stratification:
```bash
python tools/split_cache_csv.py \
  --input_csv scratch_output/generate_dataset_test/16_frames_ver/dataset_generation/train_cache_header.csv \
  --train_csv scratch_output/generate_dataset_test/16_frames_ver/dataset_generation/train_split.csv \
  --val_csv scratch_output/generate_dataset_test/16_frames_ver/dataset_generation/val_split.csv \
  --val_frac 0.2 \
  --seed 42 \
  --override_root scratch_output/generate_dataset_test/16_frames_ver/dataset_generation
```

If paths in the CSV already point to accessible locations on your cluster, omit `--override_root`.

## 2) Single-GPU / CPU run
```bash
python -m training.cli_train_header \
  --train_csv scratch_output/generate_dataset_test/16_frames_ver/dataset_generation/train_split.csv \
  --val_csv scratch_output/generate_dataset_test/16_frames_ver/dataset_generation/val_split.csv \
  --backbone_ckpt checkpoints/VideoMAEv2-Base \
  --run_name vmae_frozen_phase2_run1 \
  --epochs 50 \
  --batch_size 16 \
  --num_workers 4 \
  --optimizer_type adamw \
  --lr_head 1e-3 \
  --seed 0
```

Outputs land in `report/header_experiments/<run_name>/`:
- `config.yaml`, `metrics_epoch.csv`, `best_metrics.json`, `val_predictions.csv`
- `checkpoints/` with `best_epoch_*.pt`

## 3) Multi-GPU (DataParallel)
`cli_train_header` accepts `--gpus` to wrap the model with `torch.nn.DataParallel`. Example on 2 GPUs (IDs 0 and 1):
```bash
python -m training.cli_train_header \
  --train_csv scratch_output/generate_dataset_test/16_frames_ver/dataset_generation/train_split.csv \
  --val_csv scratch_output/generate_dataset_test/16_frames_ver/dataset_generation/val_split.csv \
  --backbone_ckpt checkpoints/VideoMAEv2-Base \
  --run_name vmae_frozen_phase2_run_dp \
  --epochs 50 \
  --batch_size 16 \
  --num_workers 8 \
  --optimizer_type adamw \
  --lr_head 1e-3 \
  --seed 0 \
  --gpus 0 1
```
Notes:
- DataParallel uses the first ID for the main device. Adjust batch size and workers per cluster resources.
- Only the head is trainable; backbone stays frozen.

## 4) Checkpoints
Available checkpoints under `checkpoints/`:
- `VideoMAEv2-Base` (default), `VideoMAEv2-Large`, `VideoMAEv2-giant`
Use `--backbone_ckpt` to point to the desired folder.

## 5) Troubleshooting
- Permission errors in `~/.cache/huggingface`: already redirected to `.cache/hf_local`, but ensure the repo is writable.
- Multiprocessing lock errors: set `--num_workers 0` as a fallback.
- Ensure `config.num_frames` (defaults to 16) matches the checkpoint’s `num_frames`; a mismatch raises early.
