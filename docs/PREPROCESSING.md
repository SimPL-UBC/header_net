# Header Net Preprocessing Pipeline

This checklist covers everything required to regenerate detections, caches, and the XGBoost weak pre-filter now that RF-DETR is wired in.

## 0. Environment & Data
- Activate the project environment: `conda activate deep_impact_env`
- Change into the project directory: `cd header_net` (all commands below assume this working directory)
- Verify the dataset layout (already wired into `configs/header_default.py`):
  - Videos and JSONs live under `DeepImpact/SoccerNet/<league>/<season>/<match>/`
  - Header annotations are the Excel workbooks in `DeepImpact/header_dataset/SoccerNetV2/`
  - RF-DETR checkpoint `header_net/checkpoints/rf-detr-medium.pth` is the default weight file

If you store assets elsewhere, override the config or pass explicit CLI arguments to the scripts below.

## 1. Build the Ball Detection Dictionary (RF-DETR)
From the project root run `cd header_net`, then execute:
```
python cache/build_ball_det_dict.py \
    --dataset-path ../DeepImpact \
    --detector rf-detr \
    --detector-weights checkpoints/rf-detr-medium.pth \
    --rf-target-classes "sports ball" \
    --output cache/ball_det_dict.npy \
    --rf-device cuda:0
```
Adjust `--rf-device` to `cpu` if no GPU is available, and tune `--rf-batch-size`/`--rf-frame-stride` if you need to trade accuracy for speed. Set `--rf-target-classes all` if you want to keep every COCO class from RF-DETR instead of filtering to the ball. The script reports how many matches received RF-DETR detections and falls back to synthesized centre crops only when the detector yields nothing.

## 2. Create the Training Cache
```
python cache/create_cache_header.py \
    --dataset-path ../DeepImpact \
    --ball-det-dict cache/ball_det_dict.npy \
    --output-dir cache/cache_header
```
Expect a summary of positives/negatives and the total number of cropped tensors written under `header_net/cache/cache_header/`.

## 3. Run the Weak Prefilter (XGBoost)
OpenMP shared memory must be available (the Codex sandbox blocks it). Execute on a normal workstation if needed:
```
python tree/pre_xgb.py \
    --dataset_path ../DeepImpact \
    --ball_det_dict_path cache/ball_det_dict.npy \
    --output_dir cache/pre_xgb \
    --n_folds 5
```
This produces fold models, `pre_xgb_final.pkl`, `feature_importance.csv`, `proposals.csv`, and `training_metadata.csv` inside `header_net/cache/pre_xgb/`.

## 4. Optional Spot Checks
- Inspect a few crops: `python - <<'PY'` block loading `np.load('header_net/cache/cache_header/<sample>_s.npy')`
- Examine `header_net/cache/pre_xgb/feature_importance.csv` and `proposals.csv` for sanity before continuing to full training

Repeat the sequence whenever the detector weights or raw data change.

  CUDA_VISIBLE_DEVICES=1 python cache/build_ball_det_dict.py \
    --dataset-path /data/diskb/gyan \
    --header-dataset /data/diskb/gyan/header_dataset \
    --detector rf-detr \
    --detector-weights checkpoints/rf-detr-medium.pth \
    --output cache/ball_det_dict.npy
