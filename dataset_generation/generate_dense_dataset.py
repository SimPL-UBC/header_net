#!/usr/bin/env python3
"""Generate a dense, per-frame detection Parquet file for the header classification pipeline.

Unlike the sparse pipeline (``generate_positive_negative_dataset.py``), this script
processes **every** frame of every video, runs RF-DETR detection, applies Kalman
smoothing to ball trajectories, and writes a single compact Parquet file.  No ``.npy``
clips are cached — the training dataloader reads video on-the-fly using this metadata.

Two labeling modes are supported (mutually exclusive):
  --one-frame-header        Only annotated frames get label=1
  --continuous-frame-header  Annotated + FPS-adaptive surrounding frames get label=1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import pyarrow  # noqa: F401 — needed by pandas Parquet engine
except ImportError:
    print(
        "[ERROR] pyarrow is required for Parquet output. "
        "Install with: pip install pyarrow"
    )
    sys.exit(1)

HEADER_NET_ROOT = Path(__file__).resolve().parents[1]
if str(HEADER_NET_ROOT) not in sys.path:
    sys.path.append(str(HEADER_NET_ROOT))

from configs import header_default as cfg  # noqa: E402
from utils.detections import make_video_key  # noqa: E402
from utils.kalman import KalmanFilter4D  # noqa: E402
from detectors.rf_detr.model import RFDetrConfig, RFDetrInference, build_rf_detr  # noqa: E402
from dataset_generation.dataset_utils import (  # noqa: E402
    discover_video_sources,
    load_labels_dataframe,
    VideoSource,
)


# ---------------------------------------------------------------------------
# SoccerNet RF-DETR class map (copied to avoid coupling with sparse script)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate dense per-frame detection Parquet for header classification. "
            "Runs RF-DETR on every frame, applies Kalman smoothing, and writes "
            "a single .parquet file with per-frame metadata and labels."
        ),
    )

    parser.add_argument(
        "--dataset-path",
        default=str(cfg.DATASET_PATH),
        help="Path to dataset root containing SoccerNet videos",
    )
    parser.add_argument(
        "--labels-dir",
        required=True,
        help="Path to labelled_header directory",
    )
    parser.add_argument(
        "--weights",
        default=str(cfg.SOCCERNET_RFDETR_WEIGHTS),
        help="Path to SoccerNet RF-DETR weights",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Full path for output .parquet file",
    )

    # Mutually exclusive labeling mode
    label_group = parser.add_mutually_exclusive_group(required=True)
    label_group.add_argument(
        "--one-frame-header",
        action="store_true",
        help="Only annotated frames get label=1",
    )
    label_group.add_argument(
        "--continuous-frame-header",
        action="store_true",
        help="Annotated + FPS-adaptive surrounding frames get label=1",
    )

    # Detection parameters
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

    # Optional filters
    parser.add_argument(
        "--match-filter",
        nargs="*",
        type=str,
        help="Process only these match names (space-separated)",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_frame_stride(fps: float) -> int:
    """Return 2 for high-fps (>=40) videos, else 1.

    At 50fps the effective temporal coverage with stride=2 matches 25fps stride=1.
    """
    return 2 if fps >= 40 else 1


def build_label_lookup(
    labels_df: pd.DataFrame,
    stride_map: Dict[str, int],
    continuous: bool,
) -> Dict[str, Set[int]]:
    """Build ``{video_key: set_of_positive_frames}`` from annotation data.

    In *continuous* mode the annotated frame is expanded by FPS-adaptive offsets
    so that the positive window covers ~0.64 s regardless of FPS.

    Parameters
    ----------
    stride_map : dict
        Pre-computed ``{video_key: stride}`` (avoids re-opening videos).
    """
    lookup: Dict[str, Set[int]] = {}

    for _, row in labels_df.iterrows():
        video_id = str(row["video_id"])
        half = int(row["half"])
        frame = int(row["frame"])
        key = make_video_key(video_id, half)

        if key not in lookup:
            lookup[key] = set()

        if not continuous:
            lookup[key].add(frame)
        else:
            stride = stride_map.get(key, 1)
            # ~0.64s coverage: 16 frames at 25fps (stride=1), 32 frames at 50fps (stride=2)
            half_window = 8 * stride
            for offset in range(-half_window, half_window):
                f = frame + offset
                if f >= 0:
                    lookup[key].add(f)

    return lookup


def smooth_ball_detections(
    raw: Dict[int, List[Dict[str, float]]],
) -> Dict[int, Dict[str, float]]:
    """Apply Kalman smoothing to ball detections across dense sequential frames.

    Since frames are dense (no gaps), the filter runs continuously without resets.
    """
    kalman = KalmanFilter4D()
    smoothed: Dict[int, Dict[str, float]] = {}
    last_size: Optional[Tuple[float, float]] = None

    for frame_id in sorted(raw.keys()):
        detections = raw.get(frame_id, [])
        ball_candidates = [
            det for det in detections if int(det.get("class_id", -1)) == 0
        ]

        if ball_candidates:
            best = max(ball_candidates, key=lambda d: d.get("confidence", 0.0))
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
            continue

        # No ball detected — predict only if filter is initialised
        if kalman._state is None or last_size is None:
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

    return smoothed


def run_dense_detection(
    source: VideoSource,
    model: SoccerNetInference,
    batch_size: int,
    score_threshold: float,
    topk: int,
) -> Tuple[Dict[int, List[Dict[str, float]]], int, int]:
    """Process ALL frames of a video with batched RF-DETR inference.

    Returns ``(raw_detections, total_frames, decoded_frames)``.
    """
    cap = cv2.VideoCapture(str(source.path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video {source.path}")

    total_frames = source.frame_count
    raw: Dict[int, List[Dict[str, float]]] = {}
    decoded = 0

    batch_frames: List[np.ndarray] = []
    batch_indices: List[int] = []
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        decoded += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        batch_frames.append(rgb)
        batch_indices.append(frame_id)

        if len(batch_frames) >= batch_size:
            preds = model(batch_frames, score_threshold=score_threshold, topk=topk)
            for fid, dets in zip(batch_indices, preds):
                raw[fid] = dets
            batch_frames.clear()
            batch_indices.clear()

        frame_id += 1

    # Flush remaining batch
    if batch_frames:
        preds = model(batch_frames, score_threshold=score_threshold, topk=topk)
        for fid, dets in zip(batch_indices, preds):
            raw[fid] = dets

    cap.release()
    return raw, total_frames, decoded


def build_frame_records(
    raw_dets: Dict[int, List[Dict[str, float]]],
    smoothed_ball: Dict[int, Dict[str, float]],
    header_frames: Set[int],
    source: VideoSource,
    video_path: str,
    fps: float,
    stride: int,
) -> List[Dict]:
    """Convert per-frame detections into flat record dicts matching the Parquet schema."""
    records: List[Dict] = []
    all_frames = sorted(set(raw_dets.keys()) | set(smoothed_ball.keys()))

    for frame_id in all_frames:
        label = 1 if frame_id in header_frames else 0

        # Ball columns (flat)
        ball = smoothed_ball.get(frame_id)
        if ball is not None:
            bx, by, bw, bh = ball["box"]
            vel = ball.get("velocity", [0.0, 0.0])
            ball_record = {
                "ball_x": float(bx),
                "ball_y": float(by),
                "ball_w": float(bw),
                "ball_h": float(bh),
                "ball_confidence": float(ball.get("confidence", 0.0)),
                "ball_vx": float(vel[0]) if len(vel) >= 1 else 0.0,
                "ball_vy": float(vel[1]) if len(vel) >= 2 else 0.0,
                "ball_source": ball.get("source", ""),
            }
        else:
            ball_record = {
                "ball_x": None,
                "ball_y": None,
                "ball_w": None,
                "ball_h": None,
                "ball_confidence": None,
                "ball_vx": None,
                "ball_vy": None,
                "ball_source": "",
            }

        # Other detections (non-ball) — count by class, serialise to JSON
        frame_dets = raw_dets.get(frame_id, [])
        num_players = 0
        num_referees = 0
        num_goalkeepers = 0
        other_list: List[Dict] = []
        for det in frame_dets:
            class_id = int(det.get("class_id", -1))
            if class_id == 0:
                continue  # ball handled above
            class_name = CLASS_ID_TO_NAME.get(class_id, str(class_id))
            if class_id == 1:
                num_players += 1
            elif class_id == 2:
                num_referees += 1
            elif class_id == 3:
                num_goalkeepers += 1
            other_list.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(float(det.get("confidence", 0.0)), 4),
                    "box": [round(float(v), 2) for v in det.get("box", [0, 0, 0, 0])],
                }
            )

        record = {
            "video_id": source.match_name,
            "half": source.half,
            "frame": frame_id,
            "label": label,
            "video_path": video_path,
            "fps": fps,
            "frame_stride": stride,
            "frame_width": source.width,
            "frame_height": source.height,
            **ball_record,
            "num_players": num_players,
            "num_referees": num_referees,
            "num_goalkeepers": num_goalkeepers,
            "other_detections": json.dumps(other_list) if other_list else "[]",
        }
        records.append(record)

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.batch_size < 1:
        raise ValueError("batch-size must be >= 1")

    output_path = Path(args.output_path)
    if not str(output_path).lower().endswith(".parquet"):
        raise ValueError("--output-path must end with .parquet")

    dataset_path = Path(args.dataset_path)
    labels_dir = Path(args.labels_dir)
    weights_path = Path(args.weights)
    label_mode = "continuous" if args.continuous_frame_header else "one_frame"

    # ── Step 1: Load header labels ──
    labels_df, resolved_root = load_labels_dataframe(labels_dir)
    if labels_df.empty:
        raise SystemExit(f"No header labels found. Verify labels dir: {labels_dir}")
    print(f"[INFO] Loaded {len(labels_df)} header labels from {resolved_root}")

    # ── Step 2: Discover videos ──
    match_names = set(labels_df["video_id"].astype(str))
    if args.match_filter:
        match_names = match_names & set(args.match_filter)
        if not match_names:
            raise SystemExit(
                f"No matches remaining after --match-filter. "
                f"Available: {sorted(set(labels_df['video_id'].astype(str)))}"
            )
        print(f"[INFO] Filtered to {len(match_names)} match(es): {sorted(match_names)}")

    sources = discover_video_sources(dataset_path, matches=match_names)
    if not sources:
        raise SystemExit(f"No video sources found in {dataset_path} for matches: {sorted(match_names)}")
    print(f"[INFO] Indexed {len(sources)} video source(s)")

    # ── Step 3: Compute FPS and stride per video ──
    fps_map: Dict[str, float] = {}
    stride_map: Dict[str, int] = {}
    for key, source in sources.items():
        cap = cv2.VideoCapture(str(source.path))
        fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 25.0
        cap.release()
        fps_map[key] = fps
        stride_map[key] = compute_frame_stride(fps)
        print(f"  {key}: {fps:.1f} fps → stride {stride_map[key]}")

    # ── Step 4: Build header frame lookup ──
    label_lookup = build_label_lookup(
        labels_df, stride_map, continuous=args.continuous_frame_header,
    )
    total_positive_frames = sum(len(v) for v in label_lookup.values())
    print(
        f"[INFO] Label mode: {label_mode} | "
        f"Positive frames: {total_positive_frames} across {len(label_lookup)} video(s)"
    )

    # ── Step 5: Initialize model ──
    inference = SoccerNetInference.build(
        weights_path=weights_path,
        device=args.device,
        optimize=args.optimize,
        optimize_batch_size=args.optimize_batch_size,
        optimize_compile=args.optimize_compile,
    )
    print(f"[INFO] SoccerNet RF-DETR running on {inference.device}")

    # ── Step 6: Process each video half (dense detection) ──
    all_chunks: List[pd.DataFrame] = []
    failed_records: List[Dict] = []
    total_ball_detected = 0
    total_frames_processed = 0
    t_start = time.perf_counter()

    for key, source in tqdm(sources.items(), desc="Processing videos"):
        fps = fps_map[key]
        stride = stride_map[key]
        header_frames = label_lookup.get(key, set())

        t_video = time.perf_counter()
        try:
            raw_dets, total_frames, decoded = run_dense_detection(
                source,
                inference,
                batch_size=args.batch_size,
                score_threshold=args.confidence_threshold,
                topk=args.topk,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - t_video
            failed_records.append(
                {
                    "video_id": source.match_name,
                    "half": source.half,
                    "key": key,
                    "path": str(source.path),
                    "reason": f"exception: {exc}",
                    "elapsed_seconds": round(elapsed, 2),
                }
            )
            print(f"\n[WARN] Failed to process {key}: {exc}")
            continue

        if decoded == 0:
            failed_records.append(
                {
                    "video_id": source.match_name,
                    "half": source.half,
                    "key": key,
                    "path": str(source.path),
                    "reason": "no_frames_decoded",
                    "elapsed_seconds": round(time.perf_counter() - t_video, 2),
                }
            )
            print(f"\n[WARN] No frames decoded for {key}")
            continue

        # Kalman smooth ball trajectory
        smoothed_ball = smooth_ball_detections(raw_dets)
        total_ball_detected += len(smoothed_ball)
        total_frames_processed += decoded

        # Build per-frame records
        records = build_frame_records(
            raw_dets,
            smoothed_ball,
            header_frames,
            source,
            str(source.path),
            fps,
            stride,
        )

        if records:
            chunk_df = pd.DataFrame(records)
            all_chunks.append(chunk_df)

        elapsed = time.perf_counter() - t_video
        ball_rate = len(smoothed_ball) / decoded * 100 if decoded else 0
        pos_count = sum(1 for r in records if r["label"] == 1)
        print(
            f"\n[INFO] {key}: {decoded}/{total_frames} frames decoded, "
            f"ball={ball_rate:.1f}%, positives={pos_count}, "
            f"time={elapsed:.1f}s"
        )

    # ── Step 7: Write Parquet ──
    if not all_chunks:
        raise SystemExit("[ERROR] No data to write — all videos failed.")

    df = pd.concat(all_chunks, ignore_index=True)

    # Cast to efficient types
    df["half"] = df["half"].astype("int8")
    df["frame"] = df["frame"].astype("int32")
    df["label"] = df["label"].astype("int8")
    df["fps"] = df["fps"].astype("float32")
    df["frame_stride"] = df["frame_stride"].astype("int8")
    df["frame_width"] = df["frame_width"].astype("int16")
    df["frame_height"] = df["frame_height"].astype("int16")
    df["num_players"] = df["num_players"].astype("int16")
    df["num_referees"] = df["num_referees"].astype("int16")
    df["num_goalkeepers"] = df["num_goalkeepers"].astype("int16")

    for col in ["ball_x", "ball_y", "ball_w", "ball_h", "ball_confidence", "ball_vx", "ball_vy"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Attach file-level metadata via pyarrow
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(df)
    metadata = table.schema.metadata or {}
    metadata.update(
        {
            b"generator": b"generate_dense_dataset.py",
            b"label_mode": label_mode.encode(),
            b"confidence_threshold": str(args.confidence_threshold).encode(),
            b"creation_date": datetime.now().isoformat().encode(),
        }
    )
    table = table.replace_schema_metadata(metadata)
    pq.write_table(table, str(output_path), compression="snappy")

    total_time = time.perf_counter() - t_start

    # ── Step 8: Save failures ──
    if failed_records:
        failed_df = pd.DataFrame(failed_records)
        failed_path = output_path.parent / "failed_videos.csv"
        failed_df.to_csv(failed_path, index=False)
        print(f"[WARN] Logged {len(failed_df)} failed video(s) to {failed_path}")

    # ── Step 9: Summary ──
    ball_rate = total_ball_detected / total_frames_processed * 100 if total_frames_processed else 0
    label_dist = df["label"].value_counts().to_dict()

    print("\n" + "=" * 60)
    print("DENSE DATASET GENERATION SUMMARY")
    print("=" * 60)
    print(f"  Output:           {output_path}")
    print(f"  Label mode:       {label_mode}")
    print(f"  Videos processed: {len(sources) - len(failed_records)}")
    print(f"  Videos failed:    {len(failed_records)}")
    print(f"  Total frames:     {len(df):,}")
    print(f"  Ball detected:    {total_ball_detected:,} / {total_frames_processed:,} ({ball_rate:.1f}%)")
    print(f"  Label=0:          {label_dist.get(0, 0):,}")
    print(f"  Label=1:          {label_dist.get(1, 0):,}")
    print(f"  FPS values:       {sorted(df['fps'].unique())}")
    print(f"  Stride values:    {sorted(df['frame_stride'].unique())}")
    print(f"  Parquet size:     {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  Total time:       {total_time:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
