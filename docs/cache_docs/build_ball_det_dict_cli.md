# Ball Detection Cache Scripts

Two scripts in `header_net/cache/` generate the ball-detection dictionary consumed by later preprocessing steps. Both expose the same command-line interface with a few default differences called out below.

## `build_ball_det_dict.py`

This variant scans every SoccerNet video under `--dataset-path`, optionally smooths detections with a Kalman filter, and writes the combined dictionary to disk.

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset-path` | `../DeepImpact` | Root containing the SoccerNet tree (`SoccerNet/<league>/<season>/<match>/`). Change this when your dataset lives elsewhere. |
| `--yolo-dir` | `../DeepImpact/yolo_detections` | Location of precomputed YOLO detections. The script searches this directory for files matching each video. Ignored when `--detector rf-detr`. |
| `--detector` | `yolo` | Detector backend: `yolo` loads files from `--yolo-dir`; `rf-detr` runs the transformer model on the fly. |
| `--detector-weights` | `header_net/checkpoints/rf-detr-medium.pth` | Path (or hosted filename) for RF-DETR weights; only used when `--detector rf-detr`. |
| `--rf-batch-size` | `4` | Frame batch size while running RF-DETR inference. Larger batches improve throughput but require more GPU memory. |
| `--rf-score-threshold` | `0.3` | Minimum confidence for RF-DETR predictions to be kept. Reduce to recover more low-scoring detections. |
| `--rf-frame-stride` | `1` | Process every Nth frame with RF-DETR. Raising this skips frames, trading runtime for coverage. Must remain ≥ 1. |
| `--rf-device` | `None` (falls back to PyTorch default) | Device for RF-DETR (`cuda:0`, `cpu`, etc.). |
| `--rf-variant` | `medium` | RF-DETR checkpoint variant to instantiate. |
| `--rf-target-classes` | `["sports ball"]` | Filter RF-DETR outputs to specific COCO class names. Use `all` to keep every class. |
| `--rf-optimize` / `--no-rf-optimize` | enabled by default | Toggle torch.jit tracing of the RF-DETR model for faster inference. |
| `--rf-optimize-batch-size` | `1` | Batch size used while tracing the optimised RF-DETR model. |
| `--rf-optimize-compile` | disabled by default | Enable experimental torch.jit compilation during optimisation. |
| `--rf-topk` | `5` | Maximum RF-DETR detections stored per frame before selecting the top entry. |
| `--header-dataset` | `../DeepImpact/header_dataset` | Path housing header annotations (`SoccerNetV2` spreadsheets or `SoccerDB` files). Drives the lookup used for missing-frame reporting. |
| `--output` | `cache/ball_det_dict.npy` | Destination `.npy` file for the detection dictionary (video → frame → detection). |
| `--no-kalman` | disabled | Skip Kalman smoothing and keep raw per-frame detections. |
| `--missing-report` | `cache/missing_detections.csv` | CSV log listing labelled frames that lacked detections after processing. |

### Behavioural Notes

- The script prints warnings whenever no detections are found for a labelled frame, helping you identify gaps that may require retraining or re-exporting detections.
- When `--detector yolo` is selected, the loader understands `.json`, `.npy`, and plain-text YOLO formats, automatically picking the highest-confidence detection per frame (or applying Kalman smoothing if enabled).

Run:

```bash
python cache/build_ball_det_dict.py --detector rf-detr --detector-weights checkpoints/rf-detr-medium.pth --rf-device cuda:0
```

## `build_labelled-only_ball_det_dict.py`

This script reuses the same arguments but filters the SoccerNet corpus to matches (and halves) that have header annotations. That keeps runtime focused on labelled data.

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset-path` | `../DeepImpact` | Same meaning as above: root containing the SoccerNet videos. |
| `--header-dataset` | `../DeepImpact/header_dataset` | Required; the script exits if no labels are found. Match directory names are normalised (punctuation stripped) so they align with SoccerNet folder names. You can point directly to the root or to its `SoccerNetV2` child—the tool resolves the proper root automatically and reports it. |
| `--output` | `cache/ball_det_dict_labelled_only.npy` | Separate default output so the labelled-only cache does not overwrite the all-video cache. Override if you prefer a different filename. |
| `--missing-report` | `cache/missing_detections_labelled_only.csv` | Missing-frame CSV for the filtered run. |
| _All other flags_ | _Same as `build_ball_det_dict.py`_ | Detector, RF-DETR, and Kalman settings behave identically. |

### Behavioural Notes

- The script prints a warning for each labelled match half that cannot be paired with a SoccerNet video (e.g., missing second halves). Use this to verify your dataset layout.
- `collect_labelled_videos` produces a detection list limited to labelled halves while still checking for missing detections against the exact labelled frames.
- Labels located outside the SoccerNetV2 hierarchy (for example, SoccerDB spreadsheets or experimental files) are ignored automatically after logging a short notice, so they no longer inflate the missing-match warnings.

Typical invocation:

```bash
python cache/build_labelled-only_ball_det_dict.py \
    --dataset-path ../DeepImpact \
    --header-dataset ../DeepImpact/header_dataset \
    --detector rf-detr \
    --detector-weights checkpoints/rf-detr-medium.pth \
    --rf-target-classes "sports ball" \
    --rf-device cuda:0 \
    --output cache/ball_det_dict_labelled_only.npy \
    --missing-report cache/missing_detections_labelled_only.csv
```

Run either script with `--help` to view live defaults after changing `configs/header_default.py`.
