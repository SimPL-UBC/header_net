#!/usr/bin/env python3
"""Build a detection dictionary using the fine-tuned SoccerNet RF-DETR model with negative sampling."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

HEADER_NET_ROOT = Path(__file__).resolve().parents[1]
if str(HEADER_NET_ROOT) not in sys.path:
    sys.path.append(str(HEADER_NET_ROOT))

from configs import header_default as cfg  # noqa: E402
from utils.detections import make_video_key  # noqa: E402
from utils.kalman import KalmanFilter4D  # noqa: E402
from utils.labels import load_header_labels  # noqa: E402

from detectors.rf_detr.model import RFDetrConfig, RFDetrInference, build_rf_detr  # noqa: E402

# Import from create_cache_header for merged functionality
from cache.create_cache_header import (
    create_header_cache,
    discover_video_sources,
    generate_negative_samples,
    VideoSource,
)

FrameDetections = Dict[int, Dict[int, Dict[str, float]]]
VideoDetections = Dict[str, FrameDetections]


CLASS_ID_TO_NAME = {
    0: "ball",
    1: "player",
    2: "referee",
    3: "goalkeeper",
}


@dataclass
class SoccerNetInference:
    """Thin wrapper around the fine-tuned SoccerNet RF-DETR model."""

    inference: RFDetrInference
    class_names: Sequence[str]
    device: str

    @classmethod
    def build(
        cls,
        weights_path: Path,
        device: Optional[str] = None,
        optimize: bool = False,
        optimize_batch_size: int = 1,
        optimize_compile: bool = False,
    ) -> "SoccerNetInference":
        weights_path = weights_path.expanduser()
        if not weights_path.exists():
            raise FileNotFoundError(f"SoccerNet RF-DETR weights not found: {weights_path}")

        config = RFDetrConfig(
            variant="large",
            weights_path=str(weights_path),
            device=device,
            target_class_ids=(0, 1, 2, 3),
            optimize=optimize,
            optimize_batch_size=optimize_batch_size,
            optimize_compile=optimize_compile,
        )
        model = build_rf_detr(config)
        inference = RFDetrInference(model, config)
        class_names = tuple(CLASS_ID_TO_NAME[idx] for idx in sorted(CLASS_ID_TO_NAME))
        return cls(inference=inference, class_names=class_names, device=inference.device)

    def __call__(
        self,
        frames: Sequence[np.ndarray],
        score_threshold: float,
        topk: int,
    ) -> List[List[Dict[str, float]]]:
        return self.inference(frames, score_threshold=score_threshold, topk=topk)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ball/player detection dictionary using fine-tuned SoccerNet RF-DETR with negative sampling"
    )
    parser.add_argument(
        "--dataset-path",
        default=str(cfg.DATASET_PATH),
        help="Path to dataset root containing SoccerNet videos",
    )
    parser.add_argument(
        "--header-dataset",
        default=str(cfg.HEADER_DATASET_PATH),
        help="Path to header dataset annotations",
    )
    parser.add_argument(
        "--weights",
        default=str(cfg.SOCCERNET_RFDETR_WEIGHTS),
        help="Path to fine-tuned RF-DETR SoccerNet weights",
    )
    parser.add_argument(
        "--output-dir",
        default=str(cfg.CACHE_PATH / "cache_header"),
        help="Directory to store generated cache",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.25,
        help="Score threshold for detections",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for model inference",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=15,
        help="Maximum detections to keep per frame",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Torch device override (cpu/cuda/mps)",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Enable RF-DETR optimization pass",
    )
    parser.add_argument(
        "--optimize-batch-size",
        type=int,
        default=1,
        help="Batch size to use when optimising the model",
    )
    parser.add_argument(
        "--optimize-compile",
        action="store_true",
        help="Enable torch compile during optimization",
    )
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=3.0,
        help="Number of negatives per positive",
    )
    parser.add_argument(
        "--guard-frames",
        type=int,
        default=10,
        help="Frames to exclude around each positive sample",
    )
    parser.add_argument(
        "--window",
        nargs="*",
        type=int,
        default=cfg.WINDOW_SIZE,
        help="Temporal window offsets",
    )
    parser.add_argument(
        "--crop-scale-factor",
        type=float,
        default=cfg.CROP_SCALE_FACTOR,
        help="Scale factor for crop radius relative to ball size",
    )
    return parser.parse_args()


def run_soccernet_on_video_sparse(
    video: VideoSource,
    model: SoccerNetInference,
    target_frames: Set[int],
    batch_size: int,
    score_threshold: float,
    topk: int,
) -> Dict[int, List[Dict[str, float]]]:
    cap = cv2.VideoCapture(str(video.path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video {video.path}")

    sorted_frames = sorted(target_frames)
    raw: Dict[int, List[Dict[str, float]]] = {}
    
    batch_frames: List[np.ndarray] = []
    batch_indices: List[int] = []

    # Optimize seeking: if frames are sequential, read sequentially.
    # Otherwise seek.
    
    current_pos = -1
    
    for frame_id in sorted_frames:
        if frame_id >= video.frame_count:
            continue
            
        if current_pos != frame_id:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            current_pos = frame_id
        
        ret, frame = cap.read()
        if not ret:
            break
        current_pos += 1 # Advance position after read

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        batch_frames.append(rgb)
        batch_indices.append(frame_id)

        if len(batch_frames) >= batch_size:
            preds = model(batch_frames, score_threshold=score_threshold, topk=topk)
            for fid, dets in zip(batch_indices, preds):
                raw[fid] = dets
            batch_frames.clear()
            batch_indices.clear()

    if batch_frames:
        preds = model(batch_frames, score_threshold=score_threshold, topk=topk)
        for fid, dets in zip(batch_indices, preds):
            raw[fid] = dets

    cap.release()
    return raw


def smooth_ball_detections(raw: Dict[int, List[Dict[str, float]]]) -> Dict[int, Dict[str, float]]:
    # Note: Kalman smoothing on sparse frames is tricky. 
    # If the window is contiguous, it works locally.
    # If frames are far apart, predict() will project far into the future/past which might be bad.
    # However, for the purpose of this cache generation, we are mostly interested in the window around the event.
    # We will reset the Kalman filter if the gap is too large.
    
    kalman = KalmanFilter4D()
    smoothed: Dict[int, Dict[str, float]] = {}
    last_size: Optional[Tuple[float, float]] = None
    last_frame_id: Optional[int] = None
    MAX_GAP = 5 # Reset if gap is larger than this

    for frame_id in sorted(raw.keys()):
        if last_frame_id is not None and (frame_id - last_frame_id) > MAX_GAP:
             kalman = KalmanFilter4D() # Reset
             last_size = None

        detections = raw.get(frame_id, [])
        ball_candidates = [det for det in detections if int(det.get("class_id", -1)) == 0]

        if ball_candidates:
            best = max(ball_candidates, key=lambda det: det.get("confidence", 0.0))
            x, y, w, h = best["box"]
            cx = x + w / 2.0
            cy = y + h / 2.0
            if smoothed and kalman._state is not None:
                kalman.predict()
                kalman.update(cx, cy)
            else:
                kalman.init_state(cx, cy)
            state_x, state_y, vx, vy = kalman.get_state()
            entry = best.copy()
            entry.update(
                {
                    "box": [state_x - w / 2.0, state_y - h / 2.0, w, h],
                    "class_id": 0,
                    "class_name": "ball",
                    "velocity": [vx, vy],
                    "source": "model",
                }
            )
            smoothed[frame_id] = entry
            last_size = (w, h)
            last_frame_id = frame_id
            continue

        if kalman._state is None or last_size is None:
            last_frame_id = frame_id
            continue

        kalman.predict()
        state_x, state_y, vx, vy = kalman.get_state()
        w, h = last_size
        smoothed[frame_id] = {
            "box": [state_x - w / 2.0, state_y - h / 2.0, w, h],
            "confidence": 0.0,
            "class_id": 0,
            "class_name": "ball",
            "velocity": [vx, vy],
            "source": "kalman",
        }
        last_frame_id = frame_id

    return smoothed


def assemble_frame_detections(
    raw: Dict[int, List[Dict[str, float]]],
    ball: Dict[int, Dict[str, float]],
) -> FrameDetections:
    frames: FrameDetections = {}
    for frame_id in sorted(set(raw.keys()) | set(ball.keys())):
        entries: Dict[int, Dict[str, float]] = {}
        ball_entry = ball.get(frame_id)
        if ball_entry:
            entries[0] = ball_entry

        others = [det for det in raw.get(frame_id, []) if int(det.get("class_id", -1)) != 0]
        next_idx = 1
        for det in others:
            class_id = int(det.get("class_id", -1))
            det_copy = det.copy()
            det_copy.setdefault("class_name", CLASS_ID_TO_NAME.get(class_id, str(class_id)))
            entries[next_idx] = det_copy
            next_idx += 1

        if entries:
            frames[int(frame_id)] = entries

    return frames


def main() -> None:
    args = parse_args()

    if args.batch_size < 1:
        raise ValueError("batch-size must be >= 1")

    dataset_path = Path(args.dataset_path)
    header_dataset = Path(args.header_dataset)
    weights_path = Path(args.weights)
    output_dir = Path(args.output_dir)
    window = args.window if args.window else cfg.WINDOW_SIZE

    # 1. Load Labels
    labels_df = load_header_labels(header_dataset)
    if labels_df.empty:
        raise SystemExit("No header labels found. Verify header dataset path.")
    print(f"[INFO] Loaded {len(labels_df)} header labels")

    # 2. Discover Videos
    match_names = set(labels_df["video_id"].astype(str))
    sources = discover_video_sources(dataset_path, matches=match_names)
    print(f"[INFO] Indexed {len(sources)} video sources")

    # 3. Generate Negative Samples
    # We need an empty detections dict for generate_negative_samples because we haven't run detection yet.
    # The original logic used detections to avoid sampling frames that already had detections?
    # Checking create_cache_header.py:
    # "det_frames = set(detections.get(key, {}).keys()) ... if det_frames: available_frames = det_frames else: available_frames = set(range(source.frame_count))"
    # Since we want to sample from the whole video (minus positives), we pass empty detections so it falls back to range(frame_count).
    
    print("[INFO] Generating negative samples...")
    negatives_df = generate_negative_samples(
        labels_df,
        {}, # No pre-existing detections
        sources,
        args.negative_ratio,
        args.guard_frames,
        window,
    )
    print(f"[INFO] Generated {len(negatives_df)} negative samples")

    all_labels = pd.concat([labels_df, negatives_df], ignore_index=True)
    print(f"[INFO] Total samples to process: {len(all_labels)}")

    # 4. Initialize Model
    inference = SoccerNetInference.build(
        weights_path=weights_path,
        device=args.device,
        optimize=args.optimize,
        optimize_batch_size=args.optimize_batch_size,
        optimize_compile=args.optimize_compile,
    )
    print(f"[INFO] SoccerNet RF-DETR running on {inference.device}")

    # 5. Process Videos (Sparse Detection)
    detections: VideoDetections = {}
    
    # Group by video to minimize video opening/closing
    for (match_name, half), group in tqdm(all_labels.groupby(["video_id", "half"], sort=False), desc="Processing Videos"):
        key = make_video_key(match_name, int(half))
        source = sources.get(key)
        if source is None:
            continue

        # Identify all frames needed for this video
        target_frames: Set[int] = set()
        for frame_val in group['frame'].astype(int):
            for offset in window:
                target_frames.add(frame_val + offset)
        
        # Run detection on these frames
        raw = run_soccernet_on_video_sparse(
            source,
            inference,
            target_frames,
            batch_size=args.batch_size,
            score_threshold=args.confidence_threshold,
            topk=args.topk,
        )
        
        ball = smooth_ball_detections(raw)
        frames = assemble_frame_detections(raw, ball)
        if frames:
            detections[key] = frames

    # 6. Create Cache
    print("[INFO] Creating cache files...")
    cache_df = create_header_cache(
        detections,
        all_labels,
        sources,
        output_dir,
        window,
        args.crop_scale_factor,
        cfg.OUTPUT_SIZE,
        cfg.LOW_RES_OUTPUT_SIZE,
        cfg.LOW_RES_MAX_DIM,
    )

    print(f"[INFO] Created {len(cache_df)} cache samples in {output_dir}")


if __name__ == "__main__":
    main()
