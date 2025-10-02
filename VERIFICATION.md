# VERIFICATION GUIDE

Use this checklist to confirm the preprocessing stack is healthy after any change.

## 0. Environment
- Activate the project environment: `conda activate deep_impact_env`.

## 1. Detection Dictionary
```
python cache/build_ball_det_dict.py \
    --dataset-path ../DeepImpact \
    --output cache/ball_det_dict.npy
```
- Expects: summary line like `Saved detections for <N> videos`. Any `Skipped … without detections` entries mean the YOLO outputs were missing; either supply them under `YOLO_DETECTIONS_PATH` or rely on the fallback (centre crops around labelled frames).
- Config knobs used: `DATASET_PATH`, `YOLO_DETECTIONS_PATH`, `FALLBACK_CONTEXT_FRAMES` (set in `configs/header_default.py`).

## 2. Cache Generation
```
python cache/create_cache_header.py \
    --dataset-path ../DeepImpact \
    --ball-det-dict cache/ball_det_dict.npy \
    --output-dir cache/cache_header
```
- Expects: label counts, warnings for missing video sources, and a final line `Created <M> cache samples`.
- Outputs live in `cache/cache_header/` with metadata in `train_cache_header.csv`.
- Config knobs: `DATASET_PATH`, `HEADER_DATASET_PATH`, `OUTPUT_SIZE`, `LOW_RES_OUTPUT_SIZE`, `LOW_RES_MAX_DIM`, `WINDOW_SIZE`, `CROP_SCALE_FACTOR`.

## 3. Weak Prefilter (XGBoost)
```
python tree/pre_xgb.py \
    --dataset_path ../DeepImpact \
    --ball_det_dict_path cache/ball_det_dict.npy \
    --output_dir cache/pre_xgb \
    --n_folds 5
```
- Produces fold models, a final model, `proposals.csv`, and `feature_importance.csv`.
- **Sandbox note:** the local CLI blocks OpenMP shared memory (`OMP Error #179`). Run this on a machine without that restriction to avoid the crash.
- Config knobs: same path settings plus `NUM_FOLDS` (default 5) in `configs/header_default.py` if you prefer editing there instead of the CLI.

## 4. Spot Checks
- Inspect a few cache tiles: `python - <<'PY'` block loading `numpy.load('cache/cache_header/<sample>_s.npy')` to verify crops and masks look reasonable.
- Optional: re-run the inline feature extraction test to ensure the design matrix forms correctly (`python - <<'PY'` script from the main summary).

## Configurable Paths & Their Effects (`configs/header_default.py`)
- `DATASET_PATH`: root folder containing `SoccerNet/` (raw videos) and `header_dataset/`. All commands use this to look up media and annotations.
- `HEADER_DATASET_PATH`: defaults to `<DATASET_PATH>/header_dataset`; override only if the annotations live elsewhere.
- `YOLO_DETECTIONS_PATH`: where `build_ball_det_dict.py` searches for YOLO outputs. Mirror the league/season/match layout or keep all files flat—the script checks both patterns.
- `CACHE_PATH`: target directory for generated detection dictionaries, cache tensors, and intermediate CSVs.
- `OUTPUT_PATH`: default destination for model outputs (used by training/inference scripts).
- `FALLBACK_CONTEXT_FRAMES`: when YOLO results are missing, detections are synthesized for labelled frames ± this many frames; raise/lower based on how much context you need.

Adjust these values before running the verification commands to point at a different dataset, annotation set, or output directory.
