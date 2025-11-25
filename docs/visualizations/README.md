# Visualization guide

This folder documents how to generate the header-classifier visualizations and where outputs are written.

## Prerequisites
- Use the existing conda env: `conda activate deep_impact_env` (or prefix commands with `conda run -n deep_impact_env`).
- Default run data is read from `scratch_output/csn_16frames_test/` (`metrics_epoch.csv`, `val_predictions.csv`).

## Quick start (all quantitative plots)
From the repo root:
```
conda run -n deep_impact_env python -m visualizations.run_all \
  --run-dir scratch_output/csn_16frames_test \
  --save-dir visualizations/output \
  --no-show
```
Outputs (PNGs) land in `visualizations/output/`:
- `loss.png`, `f1.png`
- `roc.png`, `precision_recall.png`
- `metrics_vs_threshold.png`, `probability_distributions.png`
- `confusion_matrix.png`
- (if enabled) `gradcam_correct_*.png`, `correct_gallery.png` (error gallery only if misclassifications exist)
- (if enabled) `embedding.png`

## Optional extras (Grad-CAM, embeddings, galleries)
`visualizations/user_context.py` is pre-populated to:
- load the best checkpoint from `scratch_output/csn_16frames_test/checkpoints/`
- build a small validation loader from `scratch_output/generate_dataset_test/16_frames_ver/dataset_generation/val_split_debug.csv` (falls back to `val_split.csv`)
- prepare a balanced subset for Grad-CAM/galleries and point `gradcam_target_layer` to `model.layer4`

You can edit it if you want a different checkpoint, dataset, or target layer.

Then rerun with flags, for example:
```
conda run -n deep_impact_env python -m visualizations.run_all \
  --run-dir scratch_output/csn_16frames_test \
  --save-dir visualizations/output \
  --enable-gradcam --enable-embedding --enable-galleries \
  --feature-layer-name backbone.layer4
```

## Individual plots
Each plot can be run standalone, e.g.:
```
conda run -n deep_impact_env python -m visualizations.plot_roc_curve \
  --  # no args; uses default run dir
```
Most scripts read from `scratch_output/csn_16frames_test` by default; override with `--run-dir` in `visualizations.run_all`.
