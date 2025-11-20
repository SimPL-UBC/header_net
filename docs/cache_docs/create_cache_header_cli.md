# `create_cache_header.py` Command Reference

This script transforms the ball detection dictionary into cropped training samples. Each invocation reads the detections (`cache/ball_det_dict.npy`), combines them with header annotations, generates negative samples, and writes the stack of NumPy patches plus metadata under `cache/cache_header/`.

The options below can be combined; unspecified flags fall back to the defaults defined in `configs/header_default.py`.

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset-path` | `../DeepImpact` | Location of the dataset root. The script expects match videos under `SoccerNet/<league>/<season>/<match>/`. Adjust this when your directory layout differs. |
| `--header-dataset` | `../DeepImpact/header_dataset` | Path containing the header annotations (Excel/CSV files). Only frames listed here become positive samples; if you point this elsewhere, make sure the structure matches what `utils.labels.load_header_labels` expects. |
| `--ball-det-dict` | `cache/ball_det_dict.npy` | The detection dictionary produced by `build_ball_det_dict.py`. Each frame listed here supplies the bounding boxes that drive cropping. If the file is missing or stale, rerun Step 1 before generating the cache. |
| `--output-dir` | `cache/cache_header` | Destination directory for cached samples and metadata. The script writes:<br/>• `<video_key>_<frame>_<label>_s.npy` tensors<br/>• `train_cache_header.csv` manifest<br/>• `skipped_samples.csv` listing frames discarded because no detections were available across the temporal window. |
| `--negative-ratio` | `3.0` | Controls how many background frames are sampled per positive. For example, `--negative-ratio 1.5` gathers roughly 1.5 negatives for every labelled header. Larger values grow the cache and may increase training imbalance. |
| `--guard-frames` | `10` | Frames on either side of each positive that are excluded from negative sampling. With the default temporal window (±24 frames), you may want to increase this if negatives still sit too close to true headers. |
| `--window` | `[ -24 -18 -12 -6 -3 0 3 6 12 18 24 ]` | Explicit list of temporal offsets to include around each labelled frame. Supplying a new window changes how many frames are pulled from the video and how wide each cached sample is. |

### Behavioural Notes

- **Missing detections:** When the detector fails to find the ball across the entire temporal window, the script now skips that sample entirely and logs it in `skipped_samples.csv`. No centre-cropped placeholders are emitted.
- **Skipped sample log:** `skipped_samples.csv` includes `video_id`, `half`, `frame`, `label`, and a `reason` string. Use this to audit positives that were dropped because the detector never fired in that neighbourhood.
- **Manifest (`train_cache_header.csv`):** Only samples with at least one detection in the window make it into the manifest. Downstream scripts (e.g., `run_pipeline.py`) rely on this CSV when creating train/validation splits.
- **Negative sampling pool:** If detections are available for a match, negatives are drawn from frames where the detector fired; otherwise the script falls back to all frame indices in that video. This keeps negatives focused around plausible ball locations when possible.

Run `python cache/create_cache_header.py --help` to see real-time defaults after any configuration changes. The arguments above cover the typical adjustments needed when regenerating the cache.
