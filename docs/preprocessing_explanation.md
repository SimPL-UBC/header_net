# Header Net Preprocessing Reference

This note inventories every script and helper module that runs **before** model training. Each section explains the code in plain language that a senior undergraduate should follow.

Below you will find short code excerpts from each preprocessing file followed by explanations. The snippets focus on blocks that commonly confuse students.

## configs/header_default.py
```python
HEADER_NET_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = HEADER_NET_ROOT.parent

DATASET_PATH = (REPO_ROOT / "DeepImpact").resolve()
CACHE_PATH = (HEADER_NET_ROOT / "cache").resolve()
HEADER_DATASET_PATH = (DATASET_PATH / "header_dataset").resolve()
YOLO_DETECTIONS_PATH = (DATASET_PATH / "yolo_detections").resolve()
```
*Explanation* — The first two lines locate the project folder on your computer regardless of where the repo is cloned. Everything else combines that root with folder names (like `DeepImpact` or `cache`) to produce absolute paths. Downstream scripts import these constants instead of hard-coding file locations.

```python
WINDOW_SIZE = [-24, -18, -12, -6, -3, 0, 3, 6, 12, 18, 24]
CROP_SCALE_FACTOR = 4.5
OUTPUT_SIZE = 256
LOW_RES_OUTPUT_SIZE = 64
```
*Explanation* — `WINDOW_SIZE` defines which neighbour frames to collect around each labelled frame (e.g., 24 frames before, 24 after). `CROP_SCALE_FACTOR` determines how large a region around the ball we cut out. `OUTPUT_SIZE` and `LOW_RES_OUTPUT_SIZE` say how big the cropped patches should be when saved.

## utils/labels.py
```python
def load_header_labels(header_dataset_root: Path) -> pd.DataFrame:
    entries: List[dict] = []

    soccernet_root = header_dataset_root / "SoccerNetV2"
    if soccernet_root.exists():
        for match_dir in soccernet_root.iterdir():
            ...
            frames = _extract_numeric_frames(df)
            half = _infer_half_from_name(file_path.stem)
            for frame in frames:
                entries.append({"video_id": video_id,
                                "half": half,
                                "frame": int(frame),
                                "label": 1})
```
*Explanation* — This function walks through every spreadsheet in `SoccerNetV2`, reads frame numbers, guesses whether a file is for half 1 or half 2 (based on the filename), and records each occurrence as a row with `label = 1` (meaning “header present”). The same logic repeats for the `SoccerDB` folder. In the end, it returns a tidy pandas table that other scripts can join on.

```python
def build_half_frame_lookup(df: pd.DataFrame) -> dict:
    lookup: dict[str, set[int]] = {}
    for _, row in df.iterrows():
        key = f"{row['video_id']}_half{int(row.get('half', 1))}"
        lookup.setdefault(key, set()).add(int(row['frame']))
    return lookup
```
*Explanation* — The look-up dictionary makes it quick to ask “is frame 123 of match X half 2 a header?”. Each key combines `video_id` and `half`; the value is a set of frame numbers. Using a set gives O(1) lookups instead of scanning the entire dataframe repeatedly.

## utils/detections.py
```python
BallDetections = Dict[str, Dict[int, Dict[int, Dict[str, float]]]]

def make_video_key(match_name: str, half: int) -> str:
    return f"{match_name}_half{int(half)}"
```
*Explanation* — `BallDetections` is a type hint that describes the nested dictionary structure used everywhere: top level is video key, second level is frame number, third level is detection index within that frame. `make_video_key` standardises the text key so files agree on naming.

```python
def load_ball_det_dict(path: Path) -> BallDetections:
    data = np.load(path, allow_pickle=True).item()
    result: BallDetections = {}
    for video_id, frame_dict in data.items():
        frame_entries: Dict[int, Dict[int, Dict[str, float]]] = {}
        for frame_key, detections in frame_dict.items():
            frame_id = int(frame_key)
            if isinstance(detections, dict):
                det_map = detections
            else:
                det_map = {idx: det for idx, det in enumerate(detections)}
            frame_entries[frame_id] = det_map
        result[str(video_id)] = frame_entries
    return result
```
*Explanation* — The detection dictionary is saved as a NumPy `.npy` file. When we load it, everything comes back as generic Python objects. This code rebuilds the nested dictionary so that each frame maps to a dictionary of detection entries (with consistent integer keys). Students often get confused because `.npy` files can store arbitrary Python objects — this block handles that conversion safely.

## utils/kalman.py
```python
@dataclass
class KalmanFilter4D:
    dt: float = 1.0
    process_var: float = 5.0
    meas_var: float = 25.0

    def predict(self) -> None:
        if self._state is None:
            return
        self._state = self._F @ self._state
        self._covariance = self._F @ self._covariance @ self._F.T + self._Q

    def update(self, x: float, y: float) -> None:
        if self._state is None:
            self.init_state(x, y)
            return
        ...
```
*Explanation* — The Kalman filter keeps track of position `(x, y)` and velocity `(vx, vy)` using a state vector. `predict` computes where the ball should be in the next frame, assuming constant velocity. `update` corrects that prediction when a detection exists. If there is no initial state, `update` bootstraps the filter with the first measurement. This smoothing avoids jittery bounding boxes when detections flicker.

## detectors/rf_detr/model.py
```python
_VARIANT_MAP = {
    "nano": RFDETRNano,
    "small": RFDETRSmall,
    "medium": RFDETRMedium,
    "base": RFDETRBase,
    "large": RFDETRLarge,
}

def build_rf_detr(config: Optional[RFDetrConfig] = None):
    if config is None:
        config = RFDetrConfig()
    variant_key = _normalise_variant(config.variant)
    cls = _VARIANT_MAP[variant_key]
    kwargs = {"device": config.rfdetr_device()}
    if config.weights_path:
        kwargs["pretrain_weights"] = str(config.weights_path)
    model = _instantiate_quietly(cls, **kwargs)
    return model
```
*Explanation* — RF-DETR comes in different sizes. `_VARIANT_MAP` picks the right class based on the user’s choice (`nano`, `medium`, etc.). `build_rf_detr` takes a config object, swaps in the requested weights file if provided, and builds the detector model on the correct device (GPU/CPU). `_instantiate_quietly` simply hides the verbose logs printed by the third-party package unless something goes wrong.

```python
class RFDetrInference:
    def __call__(self, frames: Sequence[np.ndarray], score_threshold: float, topk: int):
        outputs = self.model(frames)
        detections = []
        for det in outputs:
            mask = det.confidence >= score_threshold
            det = det[mask]
            det = det.topk(topk, by_confidence=True)
            detections.append([...])
        return detections
```
*Explanation* — Once the model is built, this inference helper turns a batch of RGB frames into filtered detections. It keeps only boxes above the score threshold and the top `k` results per frame. The return value matches the format expected by the caching code.

## cache/build_ball_det_dict.py (Step 1)
```python
def discover_videos(dataset_root: Path) -> List[VideoInfo]:
    soccer_root = dataset_root / "SoccerNet"
    for match_path in soccer_root.glob("*/*/*"):
        for video_file in match_path.glob("*.*"):
            if video_file.suffix.lower() not in {".mp4", ".mkv"}:
                continue
            stem = video_file.stem
            half = 1
            if stem.startswith("2") or "_2" in stem:
                half = 2
            canonical_name = canonical_match_name(match_path.name)
            video_id = make_video_key(canonical_name, half)
            videos.append(VideoInfo(video_id=video_id, path=video_file, half=half, rel_dir=match_path.relative_to(soccer_root)))
    return videos
```
*Explanation* — This bit walks the dataset folder and yields structured information about each video (path, which half, etc.). It relies on file naming conventions (filenames containing “2” imply second half), so if matches are organised differently you’d update this logic.

```python
if detector == "rf-detr":
    raw_detections = run_rf_detr_on_video(...)
else:
    det_file = find_detection_file(video, det_dir)
    if det_file is not None:
        raw_detections = load_yolo_detections(det_file)

if not raw_detections:
    print("[WARN] RF-DETR produced no detections ...")

if raw_detections:
    video_detections = apply_kalman_smoothing(raw_detections) if use_kalman else best_detection_per_frame(raw_detections)
```
*Explanation* — The core decision tree: either run RF-DETR live or read stored YOLO detections. When the detector produces no boxes, the script emits a warning and logs the miss; it no longer synthesises centre crops, so downstream stages operate strictly on real detections.

```python
np.save(output_path, detections)
print(f"Saved detections for {len(detections)} videos to {output_path}")
```
*Explanation* — Saves the nested dictionary to disk so future runs can load it without recomputing detections.

## cache/create_cache_header.py (Step 2)
```python
def gather_boxes(frame_id, window, dets, source):
    boxes = np.full((len(window), 4), np.nan, dtype=np.float32)
    has_detection = [False] * len(window)
    for idx, offset in enumerate(window):
        frame_key = frame_id + offset
        det_entry = dets.get(frame_key)
        if det_entry:
            det = det_entry.get(0) or next(iter(det_entry.values()))
            box = maybe_denormalise_box(det.get("box", [0, 0, 0, 0]), source)
            boxes[idx] = box
            has_detection[idx] = True
    if np.isnan(boxes).all():
        return boxes, has_detection
    df = pd.DataFrame(boxes, columns=["x", "y", "w", "h"])
    df.interpolate(inplace=True, limit_direction="both")
    return df.to_numpy(dtype=np.float32), has_detection
```
*Explanation* — For the target frame plus its temporal neighbours, this function collects the ball boxes. If some frames are missing detections, it leaves `NaN` values and later uses pandas interpolation to fill gaps smoothly. Students often get confused by `.interpolate`: here it just estimates the box position in missing frames by using surrounding detections.

```python
prev_frame = frames.get(target_frame - 2, base_frame)
next_frame = frames.get(target_frame + 2, base_frame)
stack = np.stack([prev_frame, base_frame, next_frame], axis=-1)
```
*Explanation* — This builds a 3-channel image where each channel is a different time step (`t-2`, `t`, `t+2`). Because the original frames are grayscale, stacking them provides temporal context (motion cues) without needing RGB.

```python
mask = np.zeros((source.height, source.width), dtype=np.uint8)
if has_detection[idx] and not np.isnan(boxes[idx]).any():
    radius_mask = int(max(boxes[idx][2], boxes[idx][3]) * 0.35)
    cv2.circle(mask, (int(center_x), int(center_y)), max(radius_mask, 2), 255, thickness=-1)
crop_img, crop_mask = crop_patch(stack, mask, center_x, center_y, radius, output_size)
crop_img[crop_mask > 100] = 255
```
*Explanation* — A mask is drawn around the ball location (filled circle). After cropping and resizing, pixels where the mask is strong (`>100`) are forced to 255 (bright white). This emphasises the ball region for the model. `crop_patch` handles boundary cases by padding when the crop would fall outside the frame.

```python
if not has_valid_detection:
    skipped.append({
        "video_id": match_name,
        "half": int(half),
        "frame": frame_id,
        "label": label,
        "reason": "no_ball_detections_in_window",
    })
    continue
```
*Explanation* — When every offset in the temporal window lacks detector output, the sample is skipped and logged instead of resorting to a centre crop. Check `skipped_samples.csv` to see which positives/negatives were dropped.

```python
cache_name = f"{key}_{frame_id:06d}_{label}"
cache_path = output_dir / cache_name
np.save(str(cache_path) + "_s.npy", np.array(images, dtype=np.uint8))
records.append({"path": str(cache_path), "label": label, ...})
```
*Explanation* — Each sample is saved under a unique name encoding the video key, frame number (zero-padded), and label (0 or 1). The `_s.npy` suffix is convention for stacked frames. Metadata is also stored in a dataframe so training can later iterate over samples without rescanning the filesystem.

## tree/pre_xgb.py (Step 3)
```python
def compute_kinematics_features(ball_positions, fps=25):
    ball_positions = sorted(ball_positions, key=lambda x: x[0])
    dt = 1.0 / fps
    for i in range(1, len(ball_positions)-1):
        x_prev, y_prev = ball_positions[i-1][1], ball_positions[i-1][2]
        x_curr, y_curr = ball_positions[i][1], ball_positions[i][2]
        x_next, y_next = ball_positions[i+1][1], ball_positions[i+1][2]
        vx = (x_next - x_prev) / (2 * dt)
        vy = (y_next - y_prev) / (2 * dt)
        ax = (x_next - 2*x_curr + x_prev) / (dt**2)
        ay = (y_next - 2*y_curr + y_prev) / (dt**2)
        ...
```
*Explanation* — This loop transforms raw (frame, x, y, width, height, confidence) detections into physics-style features. Central differences approximate velocity (`vx`, `vy`) using positions before and after the current frame. Second differences approximate acceleration. The function also computes angle changes and curvature to capture sudden direction shifts — useful cues for headers.

```python
def create_training_data(ball_det_dict, labels_df, neg_sampling_ratio=3.0):
    label_lookup = build_half_frame_lookup(labels_df)
    for video_id in ball_det_dict.keys():
        features = extract_features_from_video(video_id, ball_det_dict)
        header_frames = set(label_lookup.get(video_id, []))
        ...
        n_negatives = min(len(negative_frames), int(len(positive_samples) * neg_sampling_ratio))
        sampled_negatives = rng.choice(negative_frames, n_negatives, replace=False)
```
*Explanation* — After features are computed per video, this function matches them with the labels. Every positive frame becomes a training sample. It then randomly picks a limited number of negative frames (`neg_sampling_ratio` times as many as positives) so the dataset stays balanced. `rng.choice(..., replace=False)` avoids selecting the same negative frame twice.

```python
model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train)
y_pred_proba = model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_pred_proba)
```
*Explanation* — Typical scikit-learn style training loop: instantiate XGBoost with chosen hyperparameters, fit on the training fold, then compute validation metrics (AUC, precision, recall, etc.). Students often wonder about `predict_proba(...)[:, 1]`: the classifier outputs probabilities for both classes; `[:, 1]` selects the probability of “header”.

```python
proposals = generate_proposals(...)
proposals_df = pd.DataFrame(proposals_csv_data)
proposals_df.to_csv(output_dir / 'proposals.csv', index=False)
```
*Explanation* — After training, every frame is scored. High-probability frames become “proposals,” which can be reviewed or used to reduce the number of negatives the deep model must consider.

## tools/cache_inspect.py
```python
overlays, patches, _, frame_indices, offsets_used = prepare_detection_sequence(...)
display_sequence(
    title=f"{name} sample (frame {sample.frame}, label {sample.label})",
    overlays=overlays,
    patches=patches,
    frame_indices=frame_indices,
    offsets=offsets_used,
    auto=args.auto,
    delay=args.delay,
)
```
*Explanation* — This interactive helper lets you browse cached samples alongside the original video frames. It overlays detector boxes (and confidences) on the raw footage while displaying the corresponding preprocessed crop, making it easy to sanity-check both positive and negative samples without writing new files.

## run_pipeline.py (Orchestration helper)
```python
def run_command(cmd, description="", check=True):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        sys.exit(1)
    return result
```
*Explanation* — This wrapper prints a banner, runs a shell command, shows stdout/stderr, and exits the program if the command fails. It lets you run the individual preprocessing scripts from one master command.

```python
if args.step in ['all', 'ball_det']:
    cmd = f"cd header_net/cache && python build_ball_det_dict.py"
    run_command(cmd, "Building ball detection dictionary from YOLO results")
```
*Explanation* — Depending on `--step`, the script constructs plain shell commands that call the actual preprocessing programs. Notice that it changes into the subdirectory before running each script so relative paths inside those scripts continue to work.

## PREPROCESSING.md
This markdown file does not contain executable code but is worth noting: it explains the same three-step process in checklist form so you can rerun preprocessing quickly. Read it alongside the explanations above.

## Generated Artifacts
- `cache/ball_det_dict.npy` — Output of Step 1; the master lookup of ball detections per frame.
- `cache/cache_header/` — Output of Step 2; contains cropped samples, `train_cache_header.csv`, and `skipped_samples.csv` (frames dropped because no detections were available).
- `cache/pre_xgb/` — Output of Step 3 (if used); contains trained XGBoost models, feature importances, proposals, and metadata.

Follow the order: configure paths → build detections → create cache → (optionally) train the XGBoost prefilter → move on to the deep model (`train_header.py`). Whenever raw videos, detector weights, or labels change, rerun the necessary steps so the cache stays in sync.
