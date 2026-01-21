# Post-XGB Post-Processor

## Overview
The post-XGB model is a temporal smoothing stage that refines CNN/VMAE
probabilities using a fixed window around each frame. It learns temporal
patterns that distinguish true headers from spurious detections.

**Note**: This version uses only CNN and Pre-XGB probabilities.
Ensemble features have been removed.

## Inputs
Post-XGB consumes a CSV of per-frame probabilities. Required columns:
- `video_id`
- `frame_id`
- `cnn_prob`
- `pre_xgb_prob` (required, no default)

Use `export_probs_raw_video.py` to generate this CSV from raw videos.

## Feature Set (82 features)
Features are constructed from CNN and pre-XGB probabilities only:

**Center frame (2 features):**
- `center_cnn_prob`, `center_pre_xgb_prob`

**Per-frame window features (62 features):**
For each offset in `[-15, +15]`:
- `cnn_prob_{offset}` (31 features)
- `pre_xgb_prob_{offset}` (31 features)

**Aggregates over the window (10 features):**
- `cnn_prob_mean/std/max/min/median` (5 features)
- `pre_xgb_prob_mean/std/max/min/median` (5 features)

**Trend and local maxima (4 features):**
- `cnn_prob_slope`, `pre_xgb_prob_slope`
- `is_local_max_cnn`, `is_local_max_pre_xgb`

## Generating Probabilities from Raw Video

### Ball-only mode
Processes only frames where a ball is detected:
```bash
python export_probs_raw_video.py \
    --dataset_root /path/to/SoccerNet \
    --backbone vmae \
    --checkpoint checkpoints/vmae_best.pt \
    --backbone_ckpt checkpoints/vit_b_k710_dl_from_giant.pth \
    --pre_xgb_model tree/pre_xgb/pre_xgb_final.pkl \
    --pre_xgb_threshold 0.3 \
    --mode ball_only \
    --output probs_ball_only.csv
```

### Every-N mode
Processes every N frames, but drops frames without ball detection:
```bash
python export_probs_raw_video.py \
    --dataset_root /path/to/SoccerNet \
    --backbone vmae \
    --checkpoint checkpoints/vmae_best.pt \
    --pre_xgb_model tree/pre_xgb/pre_xgb_final.pkl \
    --pre_xgb_threshold 0.3 \
    --mode every_n \
    --window_stride 5 \
    --output probs_every_5.csv
```

### Export CLI Options
| Option | Description | Default |
|--------|-------------|---------|
| `--input_csv` | CSV with video_id, half, frame (optional) | None |
| `--videos` | Explicit list of video keys (optional) | None |
| `--dataset_root` | SoccerNet root directory | Required |
| `--backbone` | Model backbone (vmae, csn) | vmae |
| `--checkpoint` | Path to trained CNN/VMAE checkpoint | Required |
| `--backbone_ckpt` | VideoMAE pretrained weights | None |
| `--pre_xgb_model` | Path to pre-XGB model | Required |
| `--pre_xgb_threshold` | Threshold for naming (metadata only) | Required |
| `--mode` | Frame selection (ball_only, every_n) | ball_only |
| `--window_stride` | Stride for every_n mode | 5 |
| `--rf_detr_variant` | RF-DETR variant | medium |
| `--ball_conf_threshold` | Ball detection threshold | 0.3 |
| `--batch_size` | Processing batch size | 4 |
| `--output` | Output CSV path | Required |

## Training

```bash
python tree/post_xgb.py \
    --probs_csv probs_ball_only.csv \
    --labels_csv cache/val.csv \
    --output_dir tree/post_xgb_output \
    --backbone vmae \
    --export_mode ball_only \
    --export_threshold 0.3 \
    --window_size 15
```

### Training CLI Options
| Option | Description | Default |
|--------|-------------|---------|
| `--probs_csv` | Path to probabilities CSV | Required |
| `--labels_csv` | Path(s) to ground truth CSVs | None |
| `--dataset_path` | Legacy annotation path | ../../DeepImpact |
| `--output_dir` | Output directory | post_xgb |
| `--backbone` | Backbone used in export | vmae |
| `--export_mode` | Mode used in export | None |
| `--export_stride` | Stride used in every_n export | None |
| `--export_threshold` | Pre-XGB threshold in export | None |
| `--window_size` | Temporal window size (must be 15) | 15 |
| `--n_folds` | Cross-validation folds | 5 |
| `--inference_only` | Skip training, run inference only | False |
| `--model_path` | Pre-trained model for inference | None |

### Output Naming Convention
Models are automatically named based on configuration:
- `post_xgb_{backbone}_ball_only_thr{T}` for ball_only mode
- `post_xgb_{backbone}_every_n_stride{N}_thr{T}` for every_n mode

### Outputs
- `post_xgb_final.pkl` - Final trained model
- `post_xgb_fold_{0-4}.pkl` - Per-fold models
- `feature_names.pkl` - Feature schema for inference
- `feature_importance.csv` - Feature importance ranking
- `final_predictions.csv` - Training predictions

## Key Constraints

1. **window_size must be 15** - This is a hard constraint. The model will
   fail if a different window size is specified.

2. **Pre-XGB is mandatory** - Unlike the old version, the `pre_xgb_prob`
   column is required. Use `export_probs_raw_video.py` to generate
   probabilities with pre-XGB.

3. **Frame dropping behavior**:
   - `ball_only` mode: Only frames with ball detection are included
   - `every_n` mode: Every N frames, but frames WITHOUT ball detection
     are DROPPED (not included with pre_xgb_prob=0)

4. **Missing kinematics** - Frames where kinematic features cannot be
   computed (e.g., insufficient temporal context) get `pre_xgb_prob=0`
   but are not dropped.

## Inference Integration

Configure the inference pipeline with post-XGB:
```python
from inference.config import InferenceConfig
from inference.pipeline import InferencePipeline

config = InferenceConfig(
    video_path="match.mp4",
    model_checkpoint="checkpoints/vmae_best.pt",
    backbone="vmae",
    pre_xgb_model="tree/pre_xgb/pre_xgb_final.pkl",
    post_xgb_model="tree/post_xgb_output/post_xgb_final.pkl",
    window_mode="ball_only",
)

pipeline = InferencePipeline(config)
results = pipeline.run()
```

## Common Pitfalls

- **Sparse inputs**: Proposal-based CSVs are not continuous. Zero padding
  is expected when true frame offsets are missing.
- **Video ID mismatch**: If your CSV has `video_id` without `_half`, ensure
  the labels CSV includes a `half` column so keys align.
- **Feature schema mismatch**: Always use the `feature_names.pkl` produced
  during training for inference.
- **Old ensemble models**: Models trained with ensemble features (123 features)
  are incompatible with this version. Retrain post-XGB with the new 82-feature
  schema.

## Migration from Ensemble Version

If you have existing models trained with ensemble features:

1. **Regenerate probabilities** using `export_probs_raw_video.py`
2. **Retrain post-XGB** using the new `tree/post_xgb.py`
3. **Update inference configs** to use the new post-XGB model

The old ensemble version used 123 features; the new version uses 82 features.
Models are not interchangeable.
