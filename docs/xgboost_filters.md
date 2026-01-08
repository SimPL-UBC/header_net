# XGBoost Pre-filter and Post-filter Guide

This guide explains the XGBoost models used as pre-filter and post-filter in the header detection pipeline.

## Overview

The header detection pipeline uses two XGBoost models to improve detection accuracy:

1. **Pre-filter XGBoost**: A weak classifier that filters out easy negatives before CNN inference, reducing computational cost
2. **Post-filter XGBoost**: A temporal refinement model that smooths CNN outputs and suppresses spurious detections

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Full Header Detection Pipeline                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  [Video Frames] ──► [Ball Detection] ──► [Cache Generation]         │
│                                              │                       │
│                                              ▼                       │
│                          ┌────────────────────────────────┐         │
│                          │  Pre-filter XGBoost (36 feat)  │         │
│                          │  • Ball kinematics             │         │
│                          │  • Player spatial context      │         │
│                          └────────────────────────────────┘         │
│                                              │                       │
│                                              ▼                       │
│                          ┌────────────────────────────────┐         │
│                          │     3D CNN (CSN/VideoMAE)      │         │
│                          │  • Visual feature extraction   │         │
│                          │  • Per-clip classification     │         │
│                          └────────────────────────────────┘         │
│                                              │                       │
│                                              ▼                       │
│                          ┌────────────────────────────────┐         │
│                          │  Post-filter XGBoost (31-frame)│         │
│                          │  • Temporal smoothing          │         │
│                          │  • Local maximum detection     │         │
│                          └────────────────────────────────┘         │
│                                              │                       │
│                                              ▼                       │
│                                    [Final Predictions]              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 1. Pre-filter XGBoost

### Purpose

The pre-filter model uses ball kinematics and player spatial context to identify frames that are likely candidates for a header event. This helps prune easy negative samples before running the expensive CNN.

### Features (36 total)

#### Kinematic Features (24)

| Feature | Description |
|---------|-------------|
| `x`, `y` | Ball position (center of bounding box) |
| `vx`, `vy` | Ball velocity (pixels/second) |
| `speed` | Ball speed magnitude |
| `ax`, `ay` | Ball acceleration (pixels/second²) |
| `accel_mag` | Acceleration magnitude |
| `angle_change` | Direction change between consecutive frames |
| `speed_change` | Absolute speed difference |
| `speed_drop_ratio` | Relative speed decrease (indicator of contact) |
| `curvature` | Trajectory curvature approximation |
| `jerk` | Change in acceleration |
| `confidence` | Ball detection confidence |
| `ball_size` | Ball bounding box area |
| `speed_mean_w`, `speed_std_w`, `speed_max_w` | Speed statistics over 5-frame window |
| `accel_mean_w`, `accel_std_w`, `accel_max_w` | Acceleration statistics over 5-frame window |
| `angle_mean_w`, `angle_std_w`, `angle_max_w` | Angle change statistics over 5-frame window |

#### Player Spatial Features (12)

| Feature | Description |
|---------|-------------|
| `dist_to_nearest_player` | Distance from ball to nearest player center |
| `dist_to_nearest_head` | Distance from ball to nearest player head (top of bbox) |
| `num_players_50px` | Number of players within 50 pixels of ball |
| `num_players_100px` | Number of players within 100 pixels of ball |
| `num_players_200px` | Number of players within 200 pixels of ball |
| `nearest_player_rel_vx` | Relative x-velocity of nearest player to ball |
| `nearest_player_rel_vy` | Relative y-velocity of nearest player to ball |
| `nearest_head_y_offset` | Vertical offset from ball to nearest head |
| `ball_above_nearest_head` | Binary: is ball above nearest player head? |
| `avg_player_density` | Average distance to all visible players |
| `player_count` | Total players detected in frame |
| `goalkeeper_nearby` | Binary: is there a goalkeeper within 200px? |

### Training

#### Mode 1: With Player Features (Recommended)

Requires metadata JSON files with player detections:

```bash
conda activate deep_impact_env
python tree/pre_xgb.py \
    --metadata_dir scratch_output/generate_dataset_test/16_frames_ver/dataset_generation \
    --output_dir cache/pre_xgb_full \
    --neg_ratio 3.0 \
    --n_folds 5
```

#### Mode 2: Kinematic Features Only

Uses ball detection dictionary (no player features):

```bash
conda activate deep_impact_env
python tree/pre_xgb.py \
    --ball_det_dict_path cache/ball_det_dict.npy \
    --header_dataset ../DeepImpact/header_dataset \
    --output_dir cache/pre_xgb_kinematic \
    --neg_ratio 3.0 \
    --n_folds 5
```

### CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--metadata_dir` | Directory with `*_meta.json` files (enables player features) | None |
| `--ball_det_dict_path` | Path to ball detection dictionary | None |
| `--header_dataset` | Path to header annotation dataset | `../DeepImpact/header_dataset` |
| `--output_dir` | Output directory for models | `cache/pre_xgb` |
| `--neg_ratio` | Negative to positive sampling ratio | 3.0 |
| `--threshold` | Probability threshold for proposals | 0.3 |
| `--n_folds` | Number of cross-validation folds | 5 |
| `--use_player_features` | Include player features (with `--metadata_dir`) | True |
| `--no_player_features` | Disable player features | False |

### Output Artifacts

Saved to `<output_dir>/`:

- `pre_xgb_final.pkl` - Trained XGBoost model
- `pre_xgb_fold_*.pkl` - Cross-validation fold models
- `feature_names.pkl` - Feature names in order
- `feature_importance.csv` - Feature importance rankings
- `training_config.json` - Training configuration
- `proposals.csv` - Generated proposals (ball_det_dict mode only)

---

## 2. Post-filter XGBoost

### Purpose

The post-filter model refines CNN predictions by learning temporal patterns characteristic of true header events. It uses a 31-frame window (±15 frames) to smooth outputs and suppress spurious detections.

### Features

#### Per-Frame Probabilities (31 features)
- `cnn_prob_{-15}` to `cnn_prob_{15}`: Raw CNN probability for each frame in window

#### Statistical Aggregates (13 features)
- `cnn_prob_mean`, `cnn_prob_std`, `cnn_prob_max`, `cnn_prob_min`, `cnn_prob_median`
- `ensemble_prob_mean`, `ensemble_prob_std`, `ensemble_prob_max`, `ensemble_prob_min`, `ensemble_prob_median`
- `pre_xgb_prob_mean`, `pre_xgb_prob_std`, `pre_xgb_prob_max`

#### Trend Features (2 features)
- `cnn_prob_slope`: Linear fit slope over the window
- `ensemble_prob_slope`: Linear fit slope for ensemble probabilities

#### Local Maximum Indicators (2 features)
- `is_local_max_cnn`: Binary indicator if center frame is local maximum
- `is_local_max_ensemble`: Binary indicator for ensemble probabilities

### Training

Requires CNN probability output from `export_probs.py`:

```bash
conda activate deep_impact_env
python tree/post_xgb.py \
    --probs_csv cache/cnn_probabilities.csv \
    --dataset_path ../DeepImpact \
    --output_dir cache/post_xgb \
    --window_size 15 \
    --n_folds 5
```

### Inference Only

Apply a trained model to new predictions:

```bash
python tree/post_xgb.py \
    --probs_csv new_predictions.csv \
    --inference_only \
    --model_path cache/post_xgb/post_xgb_final.pkl \
    --output_dir cache/post_xgb_inference
```

### CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--probs_csv` | Path to CNN probabilities CSV | Required |
| `--dataset_path` | Path to dataset for ground truth | `../DeepImpact` |
| `--output_dir` | Output directory | `post_xgb` |
| `--window_size` | Temporal window size (total = 2×window+1) | 15 |
| `--n_folds` | Number of cross-validation folds | 5 |
| `--inference_only` | Only run inference with pre-trained model | False |
| `--model_path` | Path to pre-trained model (for inference) | None |

### Output Artifacts

Saved to `<output_dir>/`:

- `post_xgb_final.pkl` - Trained XGBoost model
- `post_xgb_fold_*.pkl` - Cross-validation fold models
- `feature_names.pkl` - Feature names in order
- `feature_importance.csv` - Feature importance rankings
- `final_predictions.csv` - Refined predictions with final probabilities

---

## 3. Expected Feature Importance

Based on domain knowledge, these features typically have high importance:

### Pre-filter
| Feature | Expected Importance | Rationale |
|---------|---------------------|-----------|
| `dist_to_nearest_head` | Very High | Headers require ball near player head |
| `ball_above_nearest_head` | High | Ball trajectory before head contact |
| `speed_drop_ratio` | High | Ball decelerates on contact |
| `angle_change` | High | Direction change indicates contact |
| `num_players_100px` | Medium | Contested headers involve multiple players |

### Post-filter
| Feature | Expected Importance | Rationale |
|---------|---------------------|-----------|
| `cnn_prob_0` | Very High | Center frame probability |
| `cnn_prob_max` | High | Peak probability in window |
| `is_local_max_cnn` | High | True headers are local maxima |
| `cnn_prob_slope` | Medium | Probability trend direction |

---

## 4. Hyperparameters

### Pre-filter XGBoost
```python
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_estimators': 100
}
```

### Post-filter XGBoost
```python
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 4,        # Shallower to prevent overfitting
    'learning_rate': 0.05,  # Lower for stability
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_estimators': 200     # More trees for refinement
}
```

---

## 5. Data Format Requirements

### Pre-filter Metadata JSON

Each `*_meta.json` file should have this structure:

```json
{
  "video_id": "match_name",
  "half": 1,
  "frame": 1000,
  "label": 0,
  "frames": [
    {
      "offset": -7,
      "frame": 993,
      "ball": {
        "box": [x, y, w, h],
        "confidence": 0.9,
        "velocity": [vx, vy]
      },
      "players": [
        {
          "box": [x, y, w, h],
          "confidence": 0.87,
          "class_id": 1
        }
      ]
    }
  ]
}
```

### Post-filter Probability CSV

Required columns:
- `video_id`: Video identifier
- `frame_id`: Frame number
- `cnn_prob`: CNN output probability (P(header))
- `ensemble_prob`: Combined probability
- `pre_xgb_prob`: Pre-filter probability (optional)
