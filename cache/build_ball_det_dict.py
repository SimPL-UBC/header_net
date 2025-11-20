#!/usr/bin/env python3
"""Build a dictionary containing per-frame ball detections."""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

try:  # Torch might not be present during static analysis.
    from torch.jit import TracerWarning
except Exception:  # pragma: no cover - defensive guard for lints.
    TracerWarning = None
else:
    warnings.filterwarnings("ignore", category=TracerWarning)

warnings.filterwarnings(
    "ignore",
    message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.*",
)
warnings.filterwarnings("ignore", message="`loss_type=None` was set in the config but it is unrecognized.*")
warnings.filterwarnings(
    "ignore",
    message="Using a different number of positional encodings than DINOv2.*",
)
warnings.filterwarnings(
    "ignore",
    message="Using patch size 16 instead of 14.*",
)
warnings.filterwarnings(
    "ignore",
    message="CUDA initialization: Unexpected error from cudaGetDeviceCount\\(\\).*",
)
warnings.filterwarnings(
    "ignore",
    message="\\[Errno 13] Permission denied\\.  joblib will operate in serial mode",
)

if TracerWarning is None:
    warnings.filterwarnings("ignore", message="TracerWarning:.*")

try:  # Transformers logs underpin RF-DETR configs; keep their noise suppressed.
    import transformers
except Exception:  # pragma: no cover - optional dependency.
    transformers = None
else:
    try:
        transformers.logging.set_verbosity_error()  # type: ignore[attr-defined]
    except AttributeError:  # Older transformers versions.
        pass

HEADER_NET_ROOT = Path(__file__).resolve().parents[1]
if str(HEADER_NET_ROOT) not in sys.path:
    sys.path.append(str(HEADER_NET_ROOT))

from configs import header_default as cfg
from utils.kalman import KalmanFilter4D
from utils.labels import load_header_labels, build_half_frame_lookup, canonical_match_name
from utils.detections import make_video_key
from detectors import RFDetrConfig, RFDetrInference, build_rf_detr
from utils.videos import infer_half_from_stem

DetectionDict = Dict[int, Dict[int, Dict[str, float]]]
BallDetections = Dict[str, DetectionDict]


@dataclass
class VideoInfo:
    video_id: str
    path: Path
    half: int
    rel_dir: Path


def load_yolo_detections(det_file: Path) -> Dict[int, List[Dict[str, float]]]:
    """Load YOLO detections for a single video."""
    if not det_file.exists():
        return {}

    if det_file.suffix == ".json":
        with det_file.open("r") as fp:
            raw = json.load(fp)
        detections: Dict[int, List[Dict[str, float]]] = {}
        for frame_str, entries in raw.items():
            try:
                frame_id = int(frame_str)
            except ValueError:
                continue
            detections[frame_id] = [
                {
                    "box": entry.get("box", entry.get("bbox", [0, 0, 0, 0])),
                    "confidence": float(entry.get("confidence", entry.get("score", 0.0))),
                    "class_id": int(entry.get("class_id", entry.get("label", 0))),
                }
                for entry in entries
            ]
        return detections

    if det_file.suffix == ".npy":
        raw = np.load(det_file, allow_pickle=True).item()
        return {
            int(frame_id): [
                {
                    "box": list(entry.get("box", entry.get("bbox", [0, 0, 0, 0]))),
                    "confidence": float(entry.get("confidence", entry.get("score", 0.0))),
                    "class_id": int(entry.get("class_id", entry.get("label", 0))),
                }
                for entry in entries
            ]
            for frame_id, entries in raw.items()
        }

    detections: Dict[int, List[Dict[str, float]]] = {}
    with det_file.open("r") as fh:
        for line in fh:
            parts = line.strip().split()
            if not parts:
                continue

            try:
                frame_id = int(parts[0])
                floats = [float(x) for x in parts[1:]]
            except ValueError:
                if len(parts) > 1:
                    try:
                        frame_id = int(parts[1])
                        floats = [float(x) for x in parts[2:]]
                    except ValueError:
                        continue
                else:
                    continue

            if len(floats) < 4:
                continue

            if len(floats) == 5:
                x, y, w, h, conf = floats
                class_id = 0
            else:
                x, y, w, h, conf = floats[:5]
                class_id = int(round(floats[5])) if len(floats) >= 6 else 0

            detections.setdefault(frame_id, []).append(
                {
                    "box": [x, y, w, h],
                    "confidence": conf,
                    "class_id": class_id,
                }
            )
    return detections


def apply_kalman_smoothing(detections: Dict[int, List[Dict[str, float]]]) -> DetectionDict:
    if not detections:
        return {}

    kalman = KalmanFilter4D()
    smoothed: DetectionDict = {}

    for frame_id in sorted(detections.keys()):
        frame_dets = detections[frame_id]
        if not frame_dets:
            kalman.predict()
            state_x, state_y, *_ = kalman.get_state()
            if not smoothed:
                continue
            last_frame = max(smoothed)
            last_box = smoothed[last_frame][0]["box"]
            last_w, last_h = last_box[2], last_box[3]
            smoothed[frame_id] = {
                0: {
                    "box": [state_x - last_w / 2, state_y - last_h / 2, last_w, last_h],
                    "confidence": 0.0,
                    "class_id": 0,
                    "predicted": True,
                }
            }
            continue

        best_det = max(frame_dets, key=lambda x: x.get("confidence", 0.0))
        x, y, w, h = best_det["box"]
        cx = x + w / 2
        cy = y + h / 2

        if smoothed:
            kalman.predict()
            kalman.update(cx, cy)
        else:
            kalman.init_state(cx, cy)

        state_x, state_y, *_ = kalman.get_state()
        smoothed[frame_id] = {
            0: {
                "box": [state_x - w / 2, state_y - h / 2, w, h],
                "confidence": best_det.get("confidence", 0.0),
                "class_id": best_det.get("class_id", 0),
            }
        }

    return smoothed


def discover_videos(dataset_root: Path) -> List[VideoInfo]:
    dataset_root = dataset_root.expanduser()
    soccer_root = dataset_root / "SoccerNet"
    videos: List[VideoInfo] = []

    if not soccer_root.exists():
        return videos

    for match_path in soccer_root.glob("*/*/*"):
        if not match_path.is_dir():
            continue
        for video_file in match_path.glob("*.*"):
            if video_file.suffix.lower() not in {".mp4", ".mkv"}:
                continue
            stem = video_file.stem
            half = infer_half_from_stem(stem)
            canonical_name = canonical_match_name(match_path.name)
            video_id = make_video_key(canonical_name, half)
            rel_dir = match_path.relative_to(soccer_root)
            videos.append(VideoInfo(video_id=video_id, path=video_file, half=half, rel_dir=rel_dir))
    return videos


def find_detection_file(video: VideoInfo, det_root: Path) -> Optional[Path]:
    det_root = det_root.expanduser()
    candidates: List[Path] = []
    rel_dir = det_root / video.rel_dir
    base_names: Sequence[str] = (
        video.path.stem,
        f"{video.rel_dir.as_posix().replace('/', '_')}_{video.path.stem}",
        video.video_id,
    )

    for base in base_names:
        for ext in (".json", ".npy", ".txt"):
            candidates.append(rel_dir / f"{base}{ext}")
            candidates.append(det_root / f"{base}{ext}")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def best_detection_per_frame(raw: Dict[int, List[Dict[str, float]]]) -> DetectionDict:
    """Select the highest-confidence detection per frame."""
    best: DetectionDict = {}
    for frame_id, detections in raw.items():
        if not detections:
            continue
        top = max(detections, key=lambda det: det.get("confidence", 0.0))
        best[frame_id] = {0: top}
    return best


def run_rf_detr_on_video(
    video: VideoInfo,
    inference: RFDetrInference,
    batch_size: int = 4,
    score_threshold: float = 0.3,
    frame_stride: int = 1,
    topk: int = 5,
) -> Dict[int, List[Dict[str, float]]]:
    cap = cv2.VideoCapture(str(video.path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video for RF-DETR detections: {video.path}")

    frames: List[np.ndarray] = []
    frame_indices: List[int] = []
    raw: Dict[int, List[Dict[str, float]]] = {}

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_stride > 1 and frame_id % frame_stride != 0:
            frame_id += 1
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(rgb)
        frame_indices.append(frame_id)

        if len(frames) >= batch_size:
            detections = inference(frames, score_threshold=score_threshold, topk=topk)
            for fid, dets in zip(frame_indices, detections):
                if dets:
                    raw[fid] = dets
            frames.clear()
            frame_indices.clear()
        frame_id += 1

    if frames:
        detections = inference(frames, score_threshold=score_threshold, topk=topk)
        for fid, dets in zip(frame_indices, detections):
            if dets:
                raw[fid] = dets

    cap.release()
    return raw


def build_ball_detection_dict(
    videos: Iterable[VideoInfo],
    det_dir: Path,
    label_lookup: Dict[str, Sequence[int]],
    use_kalman: bool,
    detector: str = "yolo",
    rf_inference: Optional[RFDetrInference] = None,
    rf_batch_size: int = cfg.RF_DETR_BATCH_SIZE,
    rf_score_threshold: float = cfg.RF_DETR_SCORE_THRESHOLD,
    rf_frame_stride: int = cfg.RF_DETR_FRAME_STRIDE,
    rf_topk: int = cfg.RF_DETR_TOPK,
) -> tuple[BallDetections, List[Dict[str, Any]]]:
    det_dir = det_dir.expanduser()
    all_detections: BallDetections = {}
    missing: List[str] = []
    missing_records: List[Dict[str, Any]] = []

    video_list = list(videos)
    for video in tqdm(video_list, desc="Videos"):
        video_detections: DetectionDict = {}
        raw_detections: Dict[int, List[Dict[str, float]]] = {}
        det_file: Optional[Path] = None

        if detector == "rf-detr":
            if rf_inference is None:
                raise ValueError("RF-DETR inference object is required when detector is 'rf-detr'")
            raw_detections = run_rf_detr_on_video(
                video,
                rf_inference,
                batch_size=rf_batch_size,
                score_threshold=rf_score_threshold,
                frame_stride=rf_frame_stride,
                topk=rf_topk,
            )
        else:
            det_file = find_detection_file(video, det_dir)
            if det_file is not None:
                raw_detections = load_yolo_detections(det_file)

        if not raw_detections:
            if detector == "rf-detr":
                print(f"[WARN] RF-DETR produced no detections for {video.video_id} ({video.path})")
            elif det_file is None:
                print(f"[WARN] No detection file found for {video.video_id} ({video.path})")

        if raw_detections:
            video_detections = apply_kalman_smoothing(raw_detections) if use_kalman else best_detection_per_frame(raw_detections)

        if video_detections:
            all_detections[video.video_id] = video_detections
        else:
            missing.append(video.video_id)

        if label_lookup:
            label_frames = label_lookup.get(video.video_id, ())
            if label_frames:
                missing_frames = [
                    int(frame)
                    for frame in label_frames
                    if int(frame) not in video_detections
                ]
                if missing_frames:
                    preview = ", ".join(str(f) for f in missing_frames[:10])
                    suffix = " ..." if len(missing_frames) > 10 else ""
                    print(
                        f"[WARN] Missing ball detections for {video.video_id} ({video.path}): "
                        f"frames {preview}{suffix}"
                    )
                    for frame in missing_frames:
                        missing_records.append(
                            {
                                "video_id": video.video_id,
                                "frame": int(frame),
                                "video_path": str(video.path),
                            }
                        )

    if missing:
        print(
            f"Skipped {len(missing)} videos without detections. "
            f"Example: {missing[:3]}"
        )

    return all_detections, missing_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ball detection dictionary")
    parser.add_argument(
        "--dataset-path",
        default=str(cfg.DATASET_PATH),
        help="Path to dataset root containing SoccerNet folders",
    )
    parser.add_argument(
        "--yolo-dir",
        default=str(cfg.YOLO_DETECTIONS_PATH),
        help="Directory containing YOLO detection files",
    )
    parser.add_argument(
        "--detector",
        default="yolo",
        choices=["yolo", "rf-detr"],
        help="Detector backend to use for ball localisation",
    )
    parser.add_argument(
        "--detector-weights",
        type=str,
        default=str(cfg.RF_DETR_WEIGHTS),
        help="Checkpoint file for RF-DETR weights (detector=rf-detr)",
    )
    parser.add_argument(
        "--rf-batch-size",
        type=int,
        default=cfg.RF_DETR_BATCH_SIZE,
        help="Batch size for RF-DETR inference",
    )
    parser.add_argument(
        "--rf-score-threshold",
        type=float,
        default=cfg.RF_DETR_SCORE_THRESHOLD,
        help="Score threshold for RF-DETR detections",
    )
    parser.add_argument(
        "--rf-frame-stride",
        type=int,
        default=cfg.RF_DETR_FRAME_STRIDE,
        help="Process every Nth frame with RF-DETR (>=1)",
    )
    parser.add_argument(
        "--rf-device",
        type=str,
        default=str(cfg.RF_DETR_DEVICE) if cfg.RF_DETR_DEVICE is not None else None,
        help="Torch device for RF-DETR inference (e.g., cuda:0)",
    )
    parser.add_argument(
        "--rf-variant",
        default="medium",
        choices=["nano", "small", "medium", "base", "large"],
        help="RF-DETR variant to instantiate",
    )
    parser.add_argument(
        "--rf-target-classes",
        nargs="+",
        default=["sports ball"],
        help="Class names to keep from RF-DETR predictions (COCO names)",
    )
    parser.add_argument(
        "--rf-optimize",
        dest="rf_optimize",
        action="store_true",
        help="Prepare the RF-DETR model for inference via torch.jit tracing (enabled by default)",
    )
    parser.add_argument(
        "--no-rf-optimize",
        dest="rf_optimize",
        action="store_false",
        help="Skip RF-DETR inference optimisation",
    )
    parser.add_argument(
        "--rf-optimize-batch-size",
        type=int,
        default=1,
        help="Batch size to use when tracing the optimized RF-DETR model",
    )
    parser.add_argument(
        "--rf-optimize-compile",
        action="store_true",
        help="Enable torch.jit compilation during RF-DETR optimization",
    )
    parser.add_argument(
        "--rf-topk",
        type=int,
        default=cfg.RF_DETR_TOPK,
        help="Maximum number of RF-DETR detections to keep per frame",
    )
    parser.add_argument(
        "--header-dataset",
        default=str(cfg.HEADER_DATASET_PATH),
        help="Path to header dataset annotations",
    )
    parser.add_argument(
        "--output",
        default=str(cfg.BALL_DET_DICT_PATH),
        help="Output path for the ball detection dictionary",
    )
    parser.add_argument(
        "--no-kalman",
        action="store_true",
        help="Disable Kalman smoothing",
    )
    parser.add_argument(
        "--missing-report",
        type=str,
        default="cache/missing_detections.csv",
        help="CSV file to log frames without detections",
    )
    parser.set_defaults(rf_optimize=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rf_optimize = True if args.rf_optimize is None else args.rf_optimize

    dataset_path = Path(args.dataset_path)
    det_dir = Path(args.yolo_dir)
    header_dataset = Path(args.header_dataset)
    output_path = Path(args.output)

    videos = discover_videos(dataset_path)
    if not videos:
        raise SystemExit("No videos found under SoccerNet. Check dataset path.")

    labels_df = load_header_labels(header_dataset)
    label_lookup = build_half_frame_lookup(labels_df) if not labels_df.empty else {}

    rf_inference = None
    if args.detector == "rf-detr":
        if args.rf_frame_stride < 1:
            raise ValueError("rf-frame-stride must be >= 1")
        print("Initialising RF-DETR detector")
        requested_weight = Path(args.detector_weights).expanduser() if args.detector_weights else None
        resolved_weight: Optional[str] = None
        if requested_weight:
            if requested_weight.exists():
                resolved_weight = str(requested_weight)
            else:
                alt_suffix = '.pth' if requested_weight.suffix == '.pt' else '.pt'
                alt_candidate = requested_weight.with_suffix(alt_suffix)
                if alt_candidate.exists():
                    print(f"RF-DETR weights not found at {requested_weight}; using {alt_candidate} instead")
                    resolved_weight = str(alt_candidate)
                else:
                    hosted_name = requested_weight.name
                    print(
                        f"RF-DETR weights not found at {requested_weight}; "
                        f"falling back to hosted name '{hosted_name}' if available"
                    )
                    resolved_weight = hosted_name
        target_names = tuple(args.rf_target_classes) if args.rf_target_classes else ()
        if any(name.lower() == "all" for name in target_names):
            target_names = ()
        rf_config = RFDetrConfig(
            variant=args.rf_variant,
            weights_path=resolved_weight,
            device=args.rf_device,
            target_class_names=target_names,
            optimize=rf_optimize,
            optimize_batch_size=args.rf_optimize_batch_size,
            optimize_compile=args.rf_optimize_compile,
        )
        model = build_rf_detr(rf_config)
        rf_inference = RFDetrInference(model, rf_config)
        print(f"RF-DETR running on {rf_inference.device}")

    detections, missing_records = build_ball_detection_dict(
        videos=videos,
        det_dir=det_dir,
        label_lookup=label_lookup,
        use_kalman=not args.no_kalman,
        detector=args.detector,
        rf_inference=rf_inference,
        rf_batch_size=args.rf_batch_size,
        rf_score_threshold=args.rf_score_threshold,
        rf_frame_stride=args.rf_frame_stride,
        rf_topk=args.rf_topk,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, detections)
    print(f"Saved detections for {len(detections)} videos to {output_path}")

    if missing_records:
        missing_path = Path(args.missing_report)
        missing_path.parent.mkdir(parents=True, exist_ok=True)
        missing_df = pd.DataFrame(missing_records)
        missing_df.sort_values(["video_id", "frame"], inplace=True)
        missing_df.to_csv(missing_path, index=False)
        print(f"Logged {len(missing_records)} missing detections to {missing_path}")
    else:
        print("All labelled frames had corresponding detections.")


if __name__ == "__main__":
    main()
