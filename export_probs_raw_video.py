#!/usr/bin/env python3
"""
Export CNN/VMAE and Pre-XGB probabilities from raw video files.

This script processes raw SoccerNet videos directly, computing both
CNN/VMAE probabilities and pre-XGB kinematic probabilities for each frame.
Output is suitable for training post-XGB models.

Supports two frame selection modes:
- ball_only: Only frames where ball is detected
- every_n: Every N frames (dropping frames without ball detection)

Example usage:
    python export_probs_raw_video.py \
        --dataset_root /path/to/SoccerNet \
        --backbone vmae \
        --checkpoint checkpoints/vmae_best.pt \
        --pre_xgb_model tree/pre_xgb/pre_xgb_final.pkl \
        --pre_xgb_threshold 0.3 \
        --mode ball_only \
        --output probs_ball_only.csv
"""

import argparse
import sys
import pickle
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple
from dataclasses import dataclass, field

import pandas as pd
import torch
from tqdm import tqdm

# Add paths
HEADER_NET_ROOT = Path(__file__).resolve().parent
if str(HEADER_NET_ROOT) not in sys.path:
    sys.path.insert(0, str(HEADER_NET_ROOT))

from dataset_generation.dataset_utils import discover_video_sources, VideoSource
from inference.preprocessing.video_reader import VideoReader
from inference.preprocessing.frame_cropper import FrameCropper
from inference.stages.model_inference import CNNInference
from inference.config import InferenceConfig
from detectors.rf_detr.model import RFDetrConfig, RFDetrInference, build_rf_detr
from tree.pre_xgb import (
    compute_kinematics_features,
    add_temporal_features,
    KINEMATIC_FEATURE_NAMES,
)
from utils.player_features import compute_player_features, PLAYER_FEATURE_NAMES

# Full feature set for pre-XGB
FULL_FEATURE_NAMES = KINEMATIC_FEATURE_NAMES + PLAYER_FEATURE_NAMES


@dataclass
class ExportConfig:
    """Configuration for probability export."""

    # Input sources
    input_csv: Optional[Path] = None
    videos: Optional[List[str]] = None
    dataset_root: Path = field(default_factory=Path)
    split: Optional[str] = None  # train, val, or test

    # CNN/VMAE model
    backbone: str = "vmae"
    checkpoint: Path = field(default_factory=Path)
    backbone_ckpt: Optional[Path] = None

    # Pre-XGB model
    pre_xgb_model: Path = field(default_factory=Path)
    pre_xgb_threshold: float = 0.2

    # Frame selection
    mode: str = "ball_only"  # ball_only or every_n
    window_stride: int = 5

    # RF-DETR config
    # TODO change it to fine-tuned RF-DETR
    rf_detr_weights: Optional[Path] = None
    rf_detr_variant: str = "medium"
    rf_detr_label_mode: str = "coco"
    rf_detr_optimize: bool = False
    rf_detr_optimize_batch_size: int = 1
    rf_detr_optimize_compile: bool = False
    ball_conf_threshold: float = 0.3

    # Processing
    device: str = "cuda"
    batch_size: int = 4
    num_frames: int = 16
    input_size: int = 224
    min_decode_ratio: float = 0.5


def load_pre_xgb_model(model_path: Path) -> Tuple:
    """Load pre-XGB model and feature names.

    Args:
        model_path: Path to pre-XGB model pickle file

    Returns:
        Tuple of (model, feature_names)
    """
    print(f"Loading Pre-XGB model from {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    feature_path = model_path.parent / "feature_names.pkl"
    if feature_path.exists():
        with open(feature_path, "rb") as f:
            feature_names = pickle.load(f)
    else:
        # Fallback to kinematic features only
        feature_names = KINEMATIC_FEATURE_NAMES

    print(f"Pre-XGB model loaded with {len(feature_names)} features")
    return model, feature_names


def run_ball_detection(
    video_reader: VideoReader,
    rf_detr: RFDetrInference,
    ball_class_ids: set[int],
    player_class_ids: set[int],
    ball_conf_threshold: float,
    batch_size: int = 4,
) -> Tuple[Dict[int, Dict], Dict[int, List[Dict]], int, int]:
    """Run RF-DETR ball and player detection on all frames.

    Args:
        video_reader: VideoReader instance
        rf_detr: RF-DETR inference wrapper
        ball_conf_threshold: Minimum confidence for ball detection
        batch_size: Processing batch size

    Returns:
        Tuple of (ball_detections, player_detections) where:
        - ball_detections: Dict[frame_idx -> {box, confidence}]
        - player_detections: Dict[frame_idx -> List[{box, confidence, class_id}]]
    """
    ball_detections = {}
    player_detections = {}

    frame_count = video_reader.frame_count

    attempted = 0
    decoded = 0

    for batch_start in tqdm(
        range(0, frame_count, batch_size),
        desc="Ball detection",
        unit="batch",
        leave=False,
    ):
        batch_end = min(batch_start + batch_size, frame_count)
        frame_indices = list(range(batch_start, batch_end))
        attempted += len(frame_indices)

        # Load frames
        frames = [video_reader.get_frame(idx) for idx in frame_indices]
        valid_frames = [
            (idx, f) for idx, f in zip(frame_indices, frames) if f is not None
        ]
        decoded += len(valid_frames)

        if not valid_frames:
            continue

        indices, images = zip(*valid_frames)

        # Run detection
        results = rf_detr(images, score_threshold=ball_conf_threshold)

        for idx, dets in zip(indices, results):
            # Separate ball and player detections
            ball_det = None
            players = []

            for det in dets:
                class_id = det.get("class_id", -1)
                if class_id in ball_class_ids:
                    if ball_det is None or det["confidence"] > ball_det["confidence"]:
                        ball_det = det
                elif class_id in player_class_ids:
                    players.append(det)

            if ball_det is not None:
                ball_detections[idx] = ball_det
            if players:
                player_detections[idx] = players

    print(f"  Detected ball in {len(ball_detections)}/{frame_count} frames")
    return ball_detections, player_detections, attempted, decoded


def get_candidate_frames(
    ball_detections: Dict[int, Dict],
    total_frames: int,
    mode: str,
    stride: int,
) -> List[int]:
    """Get candidate frames based on mode.

    Args:
        ball_detections: Dict mapping frame_idx to detection
        total_frames: Total number of frames in video
        mode: Selection mode ("ball_only" or "every_n")
        stride: Frame stride for every_n mode

    Returns:
        Sorted list of candidate frame indices
    """
    if mode == "ball_only":
        return sorted(ball_detections.keys())

    elif mode == "every_n":
        # Every N frames, but DROP frames without ball detection
        candidates = []
        for frame_idx in range(0, total_frames, stride):
            if frame_idx in ball_detections:
                candidates.append(frame_idx)
        return candidates

    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'ball_only' or 'every_n'")


def compute_pre_xgb_probabilities(
    ball_detections: Dict[int, Dict],
    player_detections: Dict[int, List[Dict]],
    model,
    feature_names: List[str],
    fps: int = 25,
) -> Dict[int, float]:
    """Compute pre-XGB probabilities for all frames with ball detections.

    Args:
        ball_detections: Dict mapping frame_idx to ball detection
        player_detections: Dict mapping frame_idx to player detections
        model: Pre-XGB model
        feature_names: List of feature names model expects
        fps: Video frame rate

    Returns:
        Dict mapping frame_idx to pre-XGB probability
    """
    if len(ball_detections) < 3:
        # Need at least 3 frames for kinematic features
        return {}

    # Convert to list format for feature extraction
    ball_positions = []
    for frame_id in sorted(ball_detections.keys()):
        det = ball_detections[frame_id]
        box = det.get("box", [0, 0, 0, 0])
        conf = det.get("confidence", 0.0)
        ball_positions.append(
            (
                frame_id,
                box[0],  # x
                box[1],  # y
                box[2],  # w
                box[3],  # h
                conf,
            )
        )

    # Compute kinematic features
    features = compute_kinematics_features(ball_positions, fps)
    features = add_temporal_features(features, window=5)

    # Check if model expects player features
    use_player_features = any(fn in PLAYER_FEATURE_NAMES for fn in feature_names)

    if use_player_features:
        frame_ids = sorted(features.keys())
        for i, frame_id in enumerate(frame_ids):
            ball_det = ball_detections.get(frame_id, {})
            players = player_detections.get(frame_id, [])

            prev_players = None
            if i > 0:
                prev_frame_id = frame_ids[i - 1]
                prev_players = player_detections.get(prev_frame_id, [])

            player_feats = compute_player_features(
                ball_det, players, prev_players=prev_players
            )
            features[frame_id].update(player_feats)

    # Compute probabilities
    probabilities = {}
    for frame_id, feat_dict in features.items():
        feat_vector = [feat_dict.get(fn, 0.0) for fn in feature_names]
        try:
            prob = model.predict_proba([feat_vector])[0, 1]
            probabilities[frame_id] = prob
        except Exception:
            # If prediction fails, set probability to 0
            probabilities[frame_id] = 0.0

    return probabilities


def run_cnn_inference(
    candidate_frames: List[int],
    ball_detections: Dict[int, Dict],
    video_reader: VideoReader,
    cnn_inference: CNNInference,
    frame_cropper: FrameCropper,
    batch_size: int,
    num_frames: int,
) -> Dict[int, float]:
    """Run CNN inference on candidate frames.

    Args:
        candidate_frames: List of center frame indices
        ball_detections: Dict mapping frame_idx to ball detection
        video_reader: VideoReader instance
        cnn_inference: CNNInference instance
        frame_cropper: FrameCropper instance
        batch_size: Processing batch size
        num_frames: Temporal window size

    Returns:
        Dict mapping frame_idx to CNN probability
    """
    cnn_probs = {}

    half_window = num_frames // 2
    window_offsets = list(range(-half_window, half_window))

    for batch_start in tqdm(
        range(0, len(candidate_frames), batch_size),
        desc="CNN inference",
        unit="batch",
        leave=False,
    ):
        batch_frames = candidate_frames[batch_start : batch_start + batch_size]
        batch_windows = []

        for center_frame in batch_frames:
            required_frames = [center_frame + offset for offset in window_offsets]
            frames = video_reader.get_frames(required_frames)

            window = frame_cropper.crop_temporal_window(
                frames,
                ball_detections,
                center_frame,
                window_offsets,
            )
            batch_windows.append(window)

        probs = cnn_inference.predict_batch(batch_windows)

        for frame_idx, prob in zip(batch_frames, probs):
            cnn_probs[frame_idx] = prob

    return cnn_probs


def process_video(
    video_source: VideoSource,
    config: ExportConfig,
    rf_detr: RFDetrInference,
    pre_xgb_model,
    pre_xgb_feature_names: List[str],
    cnn_inference: CNNInference,
    frame_cropper: FrameCropper,
) -> Tuple[List[Dict], Dict[str, object]]:
    """Process a single video and return probability records.

    Args:
        video_source: VideoSource object with video metadata
        config: Export configuration
        rf_detr: RF-DETR inference wrapper
        pre_xgb_model: Loaded pre-XGB model
        pre_xgb_feature_names: Feature names for pre-XGB
        cnn_inference: CNN inference wrapper
        frame_cropper: Frame cropper instance

    Returns:
        List of dicts with video_id, frame_id, cnn_prob, pre_xgb_prob
    """
    records: List[Dict] = []
    failure: Dict[str, object] = {}

    with VideoReader(video_source.path) as video_reader:
        # Step 1: Ball and player detection
        if config.rf_detr_label_mode == "soccernet":
            ball_ids = {0}
            player_ids = {1, 2, 3}
        else:
            ball_ids = {32}
            player_ids = {0}

        ball_detections, player_detections, attempted, decoded = run_ball_detection(
            video_reader,
            rf_detr,
            ball_ids,
            player_ids,
            config.ball_conf_threshold,
            config.batch_size,
        )

        read_ratio = (decoded / attempted) if attempted else 0.0
        if attempted == 0 or read_ratio < config.min_decode_ratio:
            failure = {
                "video_id": video_source.key,
                "path": str(video_source.path),
                "attempted_frames": attempted,
                "decoded_frames": decoded,
                "decoded_ratio": round(read_ratio, 4),
                "reason": "decode_rate_below_threshold",
            }
            return [], failure

        if not ball_detections:
            print(f"  No ball detections in {video_source.key}")
            return [], failure

        # Step 2: Get candidate frames
        candidate_frames = get_candidate_frames(
            ball_detections,
            video_reader.frame_count,
            config.mode,
            config.window_stride,
        )

        if not candidate_frames:
            print(f"  No candidate frames in {video_source.key}")
            return [], failure

        print(f"  {len(candidate_frames)} candidate frames")

        # Step 3: Compute pre-XGB probabilities
        pre_xgb_probs = compute_pre_xgb_probabilities(
            ball_detections,
            player_detections,
            pre_xgb_model,
            pre_xgb_feature_names,
            fps=int(video_reader.fps),
        )

        # Step 4: Run CNN inference
        cnn_probs = run_cnn_inference(
            candidate_frames,
            ball_detections,
            video_reader,
            cnn_inference,
            frame_cropper,
            config.batch_size,
            config.num_frames,
        )

        # Step 5: Build output records
        for frame_idx in candidate_frames:
            cnn_prob = cnn_probs.get(frame_idx, 0.0)
            # User decision: frames without kinematic features get pre_xgb_prob=0
            pre_xgb_prob = pre_xgb_probs.get(frame_idx, 0.0)

            records.append(
                {
                    "video_id": video_source.key,
                    "frame_id": frame_idx,
                    "cnn_prob": cnn_prob,
                    "pre_xgb_prob": pre_xgb_prob,
                }
            )

    return records, failure


def main():
    parser = argparse.ArgumentParser(
        description="Export CNN/VMAE and Pre-XGB probabilities from raw video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export using ball_only mode
  python export_probs_raw_video.py \\
      --dataset_root ~/SoccerNet \\
      --backbone vmae \\
      --checkpoint checkpoints/vmae_best.pt \\
      --pre_xgb_model tree/pre_xgb/pre_xgb_final.pkl \\
      --pre_xgb_threshold 0.3 \\
      --mode ball_only \\
      --output probs_ball_only.csv

  # Export using every_n mode
  python export_probs_raw_video.py \\
      --dataset_root ~/SoccerNet \\
      --backbone vmae \\
      --checkpoint checkpoints/vmae_best.pt \\
      --pre_xgb_model tree/pre_xgb/pre_xgb_final.pkl \\
      --pre_xgb_threshold 0.3 \\
      --mode every_n \\
      --window_stride 5 \\
      --output probs_every_5.csv
        """,
    )

    # Input sources (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--input_csv",
        type=str,
        default=None,
        help="CSV with video_id, half, frame columns",
    )
    input_group.add_argument(
        "--videos",
        type=str,
        nargs="*",
        default=None,
        help="Explicit list of video_id_half keys (e.g., 'match_name_half1')",
    )

    # Dataset root (required)
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Raw video root for SoccerNet structure",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "val", "test"],
        help="Filter to specific split (train/val/test). If None, process all.",
    )

    # CNN/VMAE model
    parser.add_argument(
        "--backbone",
        type=str,
        default="vmae",
        choices=["vmae", "csn"],
        help="CNN backbone type (default: vmae)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained CNN/VMAE checkpoint",
    )
    parser.add_argument(
        "--backbone_ckpt",
        type=str,
        default=None,
        help="Path to VideoMAE pretrained weights (for vmae backbone)",
    )

    # Pre-XGB model (required)
    parser.add_argument(
        "--pre_xgb_model",
        type=str,
        required=True,
        help="Path to pre-XGB model (required)",
    )
    parser.add_argument(
        "--pre_xgb_threshold",
        type=float,
        required=True,
        help="Pre-XGB threshold used during export (for naming only)",
    )

    # Frame selection mode
    parser.add_argument(
        "--mode",
        type=str,
        default="ball_only",
        choices=["ball_only", "every_n"],
        help="Frame selection mode (default: ball_only)",
    )
    parser.add_argument(
        "--window_stride",
        type=int,
        default=5,
        help="Frame stride for every_n mode (default: 5)",
    )

    # RF-DETR config
    parser.add_argument(
        "--rf_detr_weights",
        type=str,
        default=None,
        help="Path to RF-DETR weights (uses default medium if None)",
    )
    parser.add_argument(
        "--rf_detr_label_mode",
        type=str,
        default="coco",
        choices=["coco", "soccernet"],
        help="RF-DETR label mode (default: coco)",
    )
    parser.add_argument(
        "--rf_detr_variant",
        type=str,
        default="medium",
        choices=["nano", "small", "medium", "base", "large"],
        help="RF-DETR model variant (default: medium)",
    )
    parser.add_argument(
        "--rf_detr_optimize",
        action="store_true",
        help="Enable RF-DETR optimize_for_inference",
    )
    parser.add_argument(
        "--rf_detr_optimize_batch_size",
        type=int,
        default=1,
        help="Batch size for RF-DETR optimization",
    )
    parser.add_argument(
        "--rf_detr_optimize_compile",
        action="store_true",
        help="Enable torch.compile during RF-DETR optimization",
    )
    parser.add_argument(
        "--ball_conf_threshold",
        type=float,
        default=0.3,
        help="Ball detection confidence threshold (default: 0.3)",
    )

    # Processing
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu, auto-detect if None)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for inference (default: 4)",
    )
    parser.add_argument(
        "--min_decode_ratio",
        type=float,
        default=0.5,
        help="Minimum decoded frame ratio required to process a video",
    )
    parser.add_argument("--output", type=str, required=True, help="Output CSV path")
    parser.add_argument(
        "--failed_log",
        type=str,
        default=None,
        help="Optional CSV path to log failed videos",
    )

    args = parser.parse_args()

    # Validate mode-specific args
    if args.mode == "every_n" and args.window_stride <= 0:
        raise ValueError("window_stride must be positive for every_n mode")

    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build config
    config = ExportConfig(
        input_csv=Path(args.input_csv) if args.input_csv else None,
        videos=args.videos,
        dataset_root=Path(args.dataset_root),
        split=args.split,
        backbone=args.backbone,
        checkpoint=Path(args.checkpoint),
        backbone_ckpt=Path(args.backbone_ckpt) if args.backbone_ckpt else None,
        pre_xgb_model=Path(args.pre_xgb_model),
        pre_xgb_threshold=args.pre_xgb_threshold,
        mode=args.mode,
        window_stride=args.window_stride,
        rf_detr_weights=Path(args.rf_detr_weights) if args.rf_detr_weights else None,
        rf_detr_variant=args.rf_detr_variant,
        rf_detr_label_mode=args.rf_detr_label_mode,
        rf_detr_optimize=args.rf_detr_optimize,
        rf_detr_optimize_batch_size=args.rf_detr_optimize_batch_size,
        rf_detr_optimize_compile=args.rf_detr_optimize_compile,
        ball_conf_threshold=args.ball_conf_threshold,
        device=str(device),
        batch_size=args.batch_size,
        min_decode_ratio=args.min_decode_ratio,
    )

    # Discover video sources
    print(f"Discovering videos in {config.dataset_root}...")

    match_names = None
    if config.input_csv:
        # Load video list from CSV
        df = pd.read_csv(config.input_csv)
        match_names = set(df["video_id"].astype(str))
    elif config.videos:
        # Extract match names from video keys
        match_names = set()
        for key in config.videos:
            if "_half" in key:
                match_names.add(key.rsplit("_half", 1)[0])
            else:
                match_names.add(key)

    sources = discover_video_sources(config.dataset_root, matches=match_names)

    if config.videos:
        # Filter to requested videos
        sources = {k: v for k, v in sources.items() if k in config.videos}

    if config.split:
        # Filter to specific split (train/val/test)
        # Video paths are like: SoccerNet/val/league/season/match/1.mp4
        split_filter = f"/{config.split}/"
        sources = {
            k: v for k, v in sources.items()
            if split_filter in str(v.path)
        }
        print(f"Filtered to {config.split} split: {len(sources)} videos")

    print(f"Found {len(sources)} video sources")

    if not sources:
        print("No videos found!")
        raise SystemExit(1)

    # Initialize models
    print("\nInitializing models...")

    # RF-DETR for ball + player detection
    target_class_names = ("sports ball", "person")
    target_class_ids = None
    if config.rf_detr_label_mode == "soccernet":
        target_class_names = ()
        target_class_ids = (0, 1, 2, 3)

    rf_detr_config = RFDetrConfig(
        variant=config.rf_detr_variant,
        weights_path=str(config.rf_detr_weights) if config.rf_detr_weights else None,
        device=config.device,
        target_class_names=target_class_names,
        target_class_ids=target_class_ids,
        optimize=config.rf_detr_optimize,
        optimize_batch_size=config.rf_detr_optimize_batch_size,
        optimize_compile=config.rf_detr_optimize_compile,
    )
    rf_detr_model = build_rf_detr(rf_detr_config)
    rf_detr = RFDetrInference(rf_detr_model, rf_detr_config)

    # Pre-XGB model
    pre_xgb_model, pre_xgb_feature_names = load_pre_xgb_model(config.pre_xgb_model)

    # CNN/VMAE model
    backbone_type: Literal["vmae", "csn"] = (
        "vmae" if config.backbone == "vmae" else "csn"
    )
    inference_config = InferenceConfig(
        model_checkpoint=config.checkpoint,
        backbone=backbone_type,
        backbone_ckpt=config.backbone_ckpt,
        num_frames=config.num_frames,
        input_size=config.input_size,
    )
    cnn_inference = CNNInference(inference_config, device)

    # Frame cropper
    frame_cropper = FrameCropper()

    # Process videos
    print(f"\nProcessing {len(sources)} videos...")
    all_records: List[Dict] = []
    failed_records: List[Dict[str, object]] = []

    for key, source in tqdm(sources.items(), desc="Videos"):
        print(f"\nProcessing {key}...")
        records, failure = process_video(
            source,
            config,
            rf_detr,
            pre_xgb_model,
            pre_xgb_feature_names,
            cnn_inference,
            frame_cropper,
        )
        if failure:
            failed_records.append(failure)
            print(
                f"[WARN] Skipping {source.key}: decoded "
                f"{failure.get('decoded_frames')}/{failure.get('attempted_frames')} "
                f"frames ({failure.get('decoded_ratio'):.2%})"
            )
            continue
        all_records.extend(records)

    # Save output
    output_df = pd.DataFrame(all_records)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    if failed_records:
        failed_path = (
            Path(args.failed_log)
            if args.failed_log
            else output_path.parent / "failed_videos.csv"
        )
        pd.DataFrame(failed_records).to_csv(failed_path, index=False)
        print(f"[WARN] Logged {len(failed_records)} failed videos to {failed_path}")

    print(f"\n{'=' * 60}")
    print(f"Exported {len(output_df)} probability records to {output_path}")
    print(f"Videos processed: {len(sources)}")
    if len(output_df) > 0:
        print(
            f"CNN prob: mean={output_df['cnn_prob'].mean():.4f}, "
            f"std={output_df['cnn_prob'].std():.4f}"
        )
        print(
            f"Pre-XGB prob: mean={output_df['pre_xgb_prob'].mean():.4f}, "
            f"std={output_df['pre_xgb_prob'].std():.4f}"
        )
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
