# HANDSOFF – Header Net Follow-up Tasks

State as of this hand-off:

- `header_net/` is now self-contained. We inlined the Kalman filter, label readers, and detection utilities (`utils/`) and rewired `cache/build_ball_det_dict.py`, `cache/create_cache_header.py`, and `tree/pre_xgb.py` to rely solely on them.
- `configs/header_default.py` discovers the project roots automatically and exposes `DATASET_PATH`, `HEADER_DATASET_PATH`, `YOLO_DETECTIONS_PATH`, and a `FALLBACK_CONTEXT_FRAMES` knob. Changing those values controls where every script looks for SoccerNet video, header annotations, and ball detections.
- Building detections currently succeeds for **25** matches. Because the legacy YOLOv5 outputs are missing, the builder synthesized centre crops around each labelled frame (±30 frame context). These fallback boxes allow cache creation to continue, but you should regenerate true detections once a modern detector is available.
- Cache generation now indexes only the matches referenced in `header_dataset/` and loads only the frames needed per window. With the fallback detections, `cache/create_cache_header.py` produced **4 837** crops (829 positives / 4 008 negatives). Warnings in the log list header events whose raw videos are absent in the low-res SoccerNet dump or the `SoccerDB` spreadsheets—those cannot be cached until footage is present.
- The weak-supervision stage (`tree/pre_xgb.py`) parses the new structures correctly and can construct the training matrix, but **XGBoost crashes in this sandbox** (`OMP Error #179: Can't open SHM2`). That is an environment restriction (POSIX shared memory is blocked). Expect the script to finish normally on any workstation without those limits.

## Immediate Checks for the Next Session

1. Activate the environment: `conda activate deep_impact_env`.
2. (Optional) Inspect `configs/header_default.py` if the dataset lives elsewhere. Updating `DATASET_PATH` cascades to every script (detections, cache creation, preprocessing).
3. Re-run the cache steps to confirm they still succeed:
   - `python cache/build_ball_det_dict.py --dataset-path ../DeepImpact --output cache/ball_det_dict.npy`
   - `python cache/create_cache_header.py --dataset-path ../DeepImpact --ball-det-dict cache/ball_det_dict.npy --output-dir cache/cache_header`
   The second command currently takes ~3 minutes on this machine.
4. Try `python tree/pre_xgb.py ...` on a workstation with unrestricted OpenMP shared memory (see “Weak supervision” below).

## High-Priority TODOs

- **Upgrade ball detection.** The pipeline still assumes YOLOv5 outputs under `DeepImpact/yolo_detections/`. Swap in a stronger detector (YOLOv8, RT-DETR, etc.), ensuring that the detector works, and rerun `build_ball_det_dict.py`. Once real detections exist you can re-enable Kalman smoothing and drop the centre fallback.
- **Rebuild caches with real detections.** After you have true boxes, rerun `cache/build_ball_det_dict.py` and `cache/create_cache_header.py` so all crops reflect accurate ball motion and mask overlays. Spot-check a few `_s.npy` files to make sure the motion and cropping look correct.
- **Finish the weak-pre filter.** Run `tree/pre_xgb.py --dataset_path ...` in an environment with functioning OpenMP shared memory. This will save cross-val models, feature importances, proposals, and metadata in `cache/pre_xgb/`.

## Notes on Weak Supervision (`tree/pre_xgb.py`)

- Feature extraction and dataset assembly work (see the quick inline test in `VERIFICATION.md`).
- Training aborts here due to `OMP: Error #179`. This is unrelated to our code — it’s the sandbox blocking `/dev/shm`. When you move to a different machine the same command should finish and produce `pre_xgb_fold_*.pkl`, `pre_xgb_final.pkl`, `proposals.csv`, and `training_metadata.csv`.

## Next Steps for the Project

1. Regenerate detections, caches, and XGBoost outputs on a full-access machine once a new detector is ready.
2. Prepare train/val lists from `cache/cache_header/train_cache_header.csv` (respecting folds produced by the XGB proposals, if used). Wire them into `train_header.py` and run sanity training.
3. Refresh evaluation scripts in `eval/` to consume the new cache layout, apply temporal NMS, and report precision/recall per match.
4. Document a reproducible pipeline (commands + config overrides) so the heavy training run can happen on the larger compute node.

Keep this file handy for the next Codex CLI session.
