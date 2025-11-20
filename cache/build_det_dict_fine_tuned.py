#!/usr/bin/env python3
"""Build a detection dictionary using the fine-tuned SoccerNet RF-DETR model."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

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

from cache.build_ball_det_dict import VideoInfo, discover_videos  # noqa: E402

def _load_label_helpers():
    module_name = "cache.build_labelled_only_ball_det_dict"
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        module_path = HEADER_NET_ROOT / "cache" / "build_labelled-only_ball_det_dict.py"
        if not module_path.exists():
            raise
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:  # pragma: no cover - defensive guard
            raise ImportError(f"Unable to load helpers from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[module_name] = module
        return module


_label_module = _load_label_helpers()
collect_labelled_videos = _label_module.collect_labelled_videos
load_labels_dataframe = _label_module.load_labels_dataframe
build_label_lookup = _label_module.build_label_lookup


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
        description="Build ball/player detection dictionary using fine-tuned SoccerNet RF-DETR"
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
        "--output",
        default=str(cfg.BALL_PLAYER_DET_DICT_PATH),
        help="Output path for the detection dictionary",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.25,
        help="Score threshold for detections",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Process every Nth frame (>=1)",
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
        "--missing-report",
        default=str(cfg.CACHE_PATH / "missing_soccernet_detections.csv"),
        help="Path to CSV capturing labelled frames without ball detections",
    )
    return parser.parse_args()


def run_soccernet_on_video(
    video: VideoInfo,
    model: SoccerNetInference,
    batch_size: int,
    frame_stride: int,
    score_threshold: float,
    topk: int,
) -> Dict[int, List[Dict[str, float]]]:
    cap = cv2.VideoCapture(str(video.path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video {video.path}")

    frame_id = 0
    frames: List[np.ndarray] = []
    indices: List[int] = []
    raw: Dict[int, List[Dict[str, float]]] = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_stride > 1 and frame_id % frame_stride != 0:
            frame_id += 1
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(rgb)
        indices.append(frame_id)

        if len(frames) >= batch_size:
            preds = model(frames, score_threshold=score_threshold, topk=topk)
            for fid, dets in zip(indices, preds):
                raw[fid] = dets
            frames.clear()
            indices.clear()

        frame_id += 1

    if frames:
        preds = model(frames, score_threshold=score_threshold, topk=topk)
        for fid, dets in zip(indices, preds):
            raw[fid] = dets

    cap.release()
    return raw


def smooth_ball_detections(raw: Dict[int, List[Dict[str, float]]]) -> Dict[int, Dict[str, float]]:
    kalman = KalmanFilter4D()
    smoothed: Dict[int, Dict[str, float]] = {}
    last_size: Optional[Tuple[float, float]] = None

    for frame_id in sorted(raw.keys()):
        detections = raw.get(frame_id, [])
        ball_candidates = [det for det in detections if int(det.get("class_id", -1)) == 0]

        if ball_candidates:
            best = max(ball_candidates, key=lambda det: det.get("confidence", 0.0))
            x, y, w, h = best["box"]
            cx = x + w / 2.0
            cy = y + h / 2.0
            if smoothed:
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

        if kalman._state is None or last_size is None:  # noqa: SLF001
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


def build_detection_dict(
    videos: Iterable[VideoInfo],
    model: SoccerNetInference,
    batch_size: int,
    frame_stride: int,
    score_threshold: float,
    topk: int,
) -> VideoDetections:
    detections: VideoDetections = {}

    for video in tqdm(list(videos), desc="Videos"):
        raw = run_soccernet_on_video(
            video,
            model,
            batch_size=batch_size,
            frame_stride=frame_stride,
            score_threshold=score_threshold,
            topk=topk,
        )
        ball = smooth_ball_detections(raw)
        frames = assemble_frame_detections(raw, ball)
        if frames:
            detections[video.video_id] = frames

    return detections


def report_missing_ball_frames(
    detections: VideoDetections,
    label_lookup: Dict[str, Sequence[int]],
) -> List[Dict[str, object]]:
    missing: List[Dict[str, object]] = []
    for video_id, frames in label_lookup.items():
        frame_map = detections.get(video_id, {})
        for frame in frames:
            entry = frame_map.get(int(frame), {})
            ball_entry = entry.get(0)
            if not ball_entry:
                missing.append({"video_id": video_id, "frame": int(frame)})
    return missing


def main() -> None:
    args = parse_args()

    if args.frame_stride < 1:
        raise ValueError("frame-stride must be >= 1")
    if args.batch_size < 1:
        raise ValueError("batch-size must be >= 1")

    dataset_path = Path(args.dataset_path)
    header_dataset = Path(args.header_dataset)
    weights_path = Path(args.weights)
    output_path = Path(args.output)
    missing_report = Path(args.missing_report)

    videos = discover_videos(dataset_path)
    if not videos:
        raise SystemExit("No SoccerNet videos discovered. Check dataset path.")

    labels_df, resolved_header_root = load_labels_dataframe(header_dataset)
    if labels_df.empty:
        raise SystemExit("No header labels found. Verify header dataset path.")
    print(f"[INFO] Loaded {len(labels_df)} header labels")

    filtered_videos, missing_halves, name_map, skipped = collect_labelled_videos(videos, labels_df)
    if not filtered_videos:
        raise SystemExit("No labelled videos matched dataset contents. Check naming conventions.")

    print(f"[INFO] Using {len(filtered_videos)} labelled videos for detection")
    if missing_halves:
        print("[WARN] Missing video files for labelled matches:")
        for match, halves in sorted(missing_halves.items()):
            half_list = ", ".join(str(h) for h in sorted(halves))
            print(f"  {match}: half(s) {half_list}")
    if skipped:
        preview = ", ".join(sorted(skipped)[:10])
        suffix = " ..." if len(skipped) > 10 else ""
        print(f"[INFO] Ignoring labels without videos: {preview}{suffix}")

    inference = SoccerNetInference.build(
        weights_path=weights_path,
        device=args.device,
        optimize=args.optimize,
        optimize_batch_size=args.optimize_batch_size,
        optimize_compile=args.optimize_compile,
    )
    print(f"[INFO] SoccerNet RF-DETR running on {inference.device}")

    detections = build_detection_dict(
        filtered_videos,
        model=inference,
        batch_size=args.batch_size,
        frame_stride=args.frame_stride,
        score_threshold=args.confidence_threshold,
        topk=args.topk,
    )

    label_lookup = build_label_lookup(labels_df)
    missing_frames = report_missing_ball_frames(detections, label_lookup)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, detections)
    print(f"Saved detections for {len(detections)} videos to {output_path}")

    if missing_frames:
        missing_report.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(missing_frames)
        df.sort_values(["video_id", "frame"], inplace=True)
        df.to_csv(missing_report, index=False)
        print(f"[WARN] Logged {len(missing_frames)} labelled frames without ball detections to {missing_report}")
    else:
        if missing_report.exists():
            missing_report.unlink()
        print("All labelled frames have a corresponding ball detection.")


if __name__ == "__main__":
    main()
