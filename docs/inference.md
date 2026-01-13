# Inference Pipeline Guide

This guide explains how to use the inference pipeline to detect soccer headers in video files.

## Overview

The inference pipeline processes raw soccer match videos and outputs per-frame predictions indicating whether a header event is occurring. It uses a four-stage cascade architecture:

1. **Ball Detection**: RF-DETR locates the ball with Kalman smoothing
2. **Pre-XGB Filter** (optional): Filters easy negatives using ball kinematics
3. **CNN/VMAE Inference**: Classifies 16-frame temporal windows
4. **Post-XGB Filter** (optional): Temporal smoothing to suppress spurious detections

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Inference Pipeline Architecture                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  [Input Video] ──► [Ball Detection] ──► [Window Mode Selection]     │
│                     (RF-DETR + Kalman)   (dense/every_n/ball_only)  │
│                                                │                     │
│                                                ▼                     │
│                          ┌────────────────────────────────┐         │
│                          │  Pre-XGB Filter (optional)     │         │
│                          │  • 36 kinematic/spatial feats  │         │
│                          │  • Prunes ~90% of frames       │         │
│                          └────────────────────────────────┘         │
│                                                │                     │
│                                                ▼                     │
│                          ┌────────────────────────────────┐         │
│                          │     CNN/VMAE Inference         │         │
│                          │  • 16-frame temporal window    │         │
│                          │  • Ball-centered 224×224 crop  │         │
│                          │  • CSN or VideoMAE backbone    │         │
│                          └────────────────────────────────┘         │
│                                                │                     │
│                                                ▼                     │
│                          ┌────────────────────────────────┐         │
│                          │  Post-XGB Filter (optional)    │         │
│                          │  • 31-frame probability window │         │
│                          │  • Temporal smoothing          │         │
│                          └────────────────────────────────┘         │
│                                                │                     │
│                                                ▼                     │
│                                    [Output CSV]                     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Basic Usage

```bash
conda activate deep_impact_env

# Run inference on a video
python -m inference.cli \
    --video /path/to/match.mp4 \
    --checkpoint /path/to/model.pt \
    --backbone csn \
    --output predictions.csv
```

### Minimal Example

```bash
# Using the CSN model with ball-only mode (fastest)
python -m inference.cli \
    --video match.mp4 \
    --checkpoint scratch_output/csn_16frames_test/checkpoints/best_epoch_48.pt \
    --backbone csn \
    --window-mode ball_only \
    --output predictions.csv
```

---

## CLI Reference

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--video`, `-v` | Path to input video file (mp4/mkv) |
| `--checkpoint`, `-c` | Path to trained model checkpoint (.pt) |

### Model Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--backbone` | `vmae` | Model backbone: `vmae` or `csn` |
| `--backbone-ckpt` | None | VideoMAE pretrained weights directory (for vmae only) |
| `--num-frames` | `16` | Temporal window size in frames |
| `--input-size` | `224` | CNN input resolution in pixels |

### Sliding Window Mode

| Argument | Default | Description |
|----------|---------|-------------|
| `--window-mode` | `dense` | Window mode: `dense`, `every_n`, or `ball_only` |
| `--window-stride` | `5` | Frame stride for `every_n` mode |

### XGBoost Filters (Optional)

| Argument | Default | Description |
|----------|---------|-------------|
| `--pre-xgb` | None | Pre-XGB model path for filtering |
| `--post-xgb` | None | Post-XGB model path for temporal smoothing |
| `--pre-xgb-threshold` | `0.3` | Pre-XGB filter probability threshold |

### Ball Detection

| Argument | Default | Description |
|----------|---------|-------------|
| `--rf-detr-weights` | None | RF-DETR weights path (uses default if not specified) |
| `--rf-detr-variant` | `medium` | RF-DETR variant: `nano`, `small`, `medium`, `base`, `large` |
| `--ball-threshold` | `0.3` | Ball detection confidence threshold |
| `--no-kalman` | False | Disable Kalman smoothing for ball detection |

### Output Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--output`, `-o` | `predictions.csv` | Output CSV path |
| `--confidence-threshold` | `0.5` | Final prediction confidence threshold |

### Processing

| Argument | Default | Description |
|----------|---------|-------------|
| `--batch-size` | `4` | Inference batch size |
| `--device` | Auto | Device: `cuda`, `cuda:0`, `cpu` (auto-detects if not specified) |

---

## Window Modes

The pipeline supports three window modes that control which frames are processed:

### Dense Mode (Default)

Processes every frame in the video. Most thorough but slowest.

```bash
python -m inference.cli \
    --video match.mp4 \
    --checkpoint model.pt \
    --window-mode dense
```

**Use when:** You need complete frame-by-frame analysis.

### Every-N Mode

Processes every Nth frame based on `--window-stride`.

```bash
python -m inference.cli \
    --video match.mp4 \
    --checkpoint model.pt \
    --window-mode every_n \
    --window-stride 5
```

**Use when:** You want faster processing with acceptable coverage.

### Ball-Only Mode

Only processes frames where the ball was detected. Most efficient.

```bash
python -m inference.cli \
    --video match.mp4 \
    --checkpoint model.pt \
    --window-mode ball_only
```

**Use when:** You have good ball detection and want maximum speed.

---

## Output Format

The pipeline outputs a CSV file with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `frame` | int | Frame index (0-indexed) |
| `prediction` | int | Binary prediction: 0 (non-header) or 1 (header) |
| `confidence` | float | Confidence score [0.0, 1.0] |
| `ball_detected` | bool | Whether ball was detected in this frame |
| `ball_x` | float | Ball center x-coordinate (pixels) |
| `ball_y` | float | Ball center y-coordinate (pixels) |

### Example Output

```csv
frame,prediction,confidence,ball_detected,ball_x,ball_y
0,0,0.12,True,450.5,320.2
1,0,0.15,True,455.2,318.8
2,0,0.18,True,460.1,315.5
...
1250,1,0.87,True,512.0,180.5
1251,1,0.92,True,510.5,175.2
1252,0,0.45,True,508.0,172.0
```

---

## Examples

### Basic CSN Inference

```bash
python -m inference.cli \
    --video /data/matches/game1.mp4 \
    --checkpoint scratch_output/csn_16frames_test/checkpoints/best_epoch_48.pt \
    --backbone csn \
    --output game1_predictions.csv
```

### VideoMAE with Pre-XGB Filter

```bash
python -m inference.cli \
    --video /data/matches/game1.mp4 \
    --checkpoint checkpoints/vmae_header.pt \
    --backbone vmae \
    --backbone-ckpt checkpoints/videomae_base \
    --pre-xgb scratch_output/cache_pre_xgb/pre_xgb_final.pkl \
    --window-mode ball_only \
    --output game1_predictions.csv
```

### High-Precision with Both XGB Filters

```bash
python -m inference.cli \
    --video /data/matches/game1.mp4 \
    --checkpoint checkpoints/vmae_header.pt \
    --backbone vmae \
    --pre-xgb cache/pre_xgb/pre_xgb_final.pkl \
    --post-xgb cache/post_xgb/post_xgb_final.pkl \
    --window-mode ball_only \
    --confidence-threshold 0.6 \
    --output game1_predictions.csv
```

### Fast Processing (Every 10 Frames)

```bash
python -m inference.cli \
    --video /data/matches/game1.mp4 \
    --checkpoint model.pt \
    --backbone csn \
    --window-mode every_n \
    --window-stride 10 \
    --batch-size 8 \
    --output game1_predictions.csv
```

### CPU-Only Inference

```bash
python -m inference.cli \
    --video /data/matches/game1.mp4 \
    --checkpoint model.pt \
    --backbone csn \
    --device cpu \
    --batch-size 2 \
    --output game1_predictions.csv
```

### Specific GPU Selection

```bash
python -m inference.cli \
    --video /data/matches/game1.mp4 \
    --checkpoint model.pt \
    --backbone csn \
    --device cuda:1 \
    --output game1_predictions.csv
```

---

## Python API

For programmatic usage, you can use the `HeaderDetectionPipeline` class directly:

### Basic Usage

```python
from pathlib import Path
from inference import InferenceConfig, HeaderDetectionPipeline

# Create configuration
config = InferenceConfig(
    video_path=Path("match.mp4"),
    model_checkpoint=Path("model.pt"),
    output_csv=Path("predictions.csv"),
    backbone="csn",
    window_mode="ball_only",
    batch_size=4,
)

# Run pipeline
pipeline = HeaderDetectionPipeline(config)
results_df = pipeline.run()

# Access results
print(f"Total frames: {len(results_df)}")
print(f"Headers detected: {results_df['prediction'].sum()}")
```

### With XGB Filters

```python
from pathlib import Path
from inference import InferenceConfig, HeaderDetectionPipeline

config = InferenceConfig(
    video_path=Path("match.mp4"),
    model_checkpoint=Path("model.pt"),
    output_csv=Path("predictions.csv"),
    backbone="vmae",
    backbone_ckpt=Path("checkpoints/videomae_base"),
    pre_xgb_model=Path("cache/pre_xgb/pre_xgb_final.pkl"),
    post_xgb_model=Path("cache/post_xgb/post_xgb_final.pkl"),
    window_mode="ball_only",
    pre_xgb_threshold=0.3,
    confidence_threshold=0.5,
)

pipeline = HeaderDetectionPipeline(config)
results_df = pipeline.run()
```

### Convenience Function

```python
from inference.pipeline import run_inference

# Simple one-liner
results_df = run_inference(
    video_path="match.mp4",
    checkpoint_path="model.pt",
    output_path="predictions.csv",
    backbone="csn",
    window_mode="every_n",
    window_stride=5,
)
```

### Accessing Individual Stages

```python
from inference.preprocessing import VideoReader, FrameCropper
from inference.stages import BallDetector, CNNInference

# Read video
with VideoReader("match.mp4") as reader:
    frame = reader.get_frame(100)
    print(f"Frame shape: {frame.shape}")
    print(f"Total frames: {reader.frame_count}")
    print(f"FPS: {reader.fps}")
```

---

## Configuration Reference

### InferenceConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `video_path` | Path | required | Input video path |
| `output_csv` | Path | `predictions.csv` | Output CSV path |
| `model_checkpoint` | Path | required | Model checkpoint path |
| `backbone` | str | `vmae` | Backbone type: `vmae` or `csn` |
| `backbone_ckpt` | Path | None | VideoMAE pretrained weights |
| `window_mode` | str | `dense` | Window mode |
| `window_stride` | int | `5` | Stride for `every_n` mode |
| `num_frames` | int | `16` | Temporal window size |
| `input_size` | int | `224` | CNN input resolution |
| `batch_size` | int | `4` | Inference batch size |
| `pre_xgb_model` | Path | None | Pre-XGB model path |
| `post_xgb_model` | Path | None | Post-XGB model path |
| `pre_xgb_threshold` | float | `0.3` | Pre-XGB filter threshold |
| `rf_detr_weights` | Path | None | RF-DETR weights |
| `rf_detr_variant` | str | `medium` | RF-DETR variant |
| `ball_conf_threshold` | float | `0.3` | Ball detection threshold |
| `use_kalman` | bool | `True` | Enable Kalman smoothing |
| `confidence_threshold` | float | `0.5` | Final prediction threshold |
| `device` | str | None | Device (auto-detect if None) |

---

## Troubleshooting

### No Ball Detected

If ball detection fails (0 detections), try:
- Lower `--ball-threshold` (e.g., `0.2`)
- Use a different RF-DETR variant (e.g., `--rf-detr-variant large`)
- Use `--window-mode dense` or `--window-mode every_n` to process frames regardless

### Out of Memory

If you encounter GPU memory issues:
- Reduce `--batch-size` (e.g., `2` or `1`)
- Use `--device cpu` for CPU inference
- Use `--window-mode every_n --window-stride 10` to process fewer frames

### Slow Inference

To speed up inference:
- Use `--window-mode ball_only` (most efficient)
- Increase `--batch-size` if GPU memory allows
- Use `--pre-xgb` to filter frames before CNN inference
- Use `--rf-detr-variant nano` for faster ball detection

---

## Module Structure

```
inference/
├── __init__.py              # Package exports
├── __main__.py              # Enables `python -m inference`
├── config.py                # InferenceConfig dataclass
├── pipeline.py              # HeaderDetectionPipeline main class
├── cli.py                   # CLI entry point
│
├── stages/
│   ├── __init__.py
│   ├── ball_detection.py    # BallDetector (RF-DETR + Kalman)
│   ├── pre_filter.py        # PreXGBFilter
│   ├── model_inference.py   # CNNInference (VMAE/CSN)
│   └── post_filter.py       # PostXGBFilter
│
├── preprocessing/
│   ├── __init__.py
│   ├── video_reader.py      # VideoReader (OpenCV wrapper)
│   ├── frame_cropper.py     # Ball-centered cropping
│   └── transforms.py        # Inference transforms
│
└── utils/
    ├── __init__.py
    └── device.py            # Device auto-detection
```

---

## See Also

- [XGBoost Filters Guide](xgboost_filters.md) - Details on Pre-XGB and Post-XGB models
- [Dataset Generation](dataset_generation.md) - Creating training data
- [Preprocessing Explanation](preprocessing_explanation.md) - Data preprocessing pipeline
