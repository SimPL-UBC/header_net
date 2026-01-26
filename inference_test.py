#!/usr/bin/env python3
"""
Test Inference Pipeline for Header Detection.

Runs the full three-stage pipeline (pre-XGB → VMAE → post-XGB) on test videos.
Outputs comprehensive CSV with all probability columns.

Pipeline flow:
1. RF-DETR ball detection on all frames
2. Pre-XGB filtering (skip VMAE if center frame prob < threshold)
3. VMAE inference on filtered frames only
4. Post-XGB temporal smoothing for final predictions

Example usage:
    python inference_test.py \
        --dataset_root /path/to/SoccerNet \
        --output_dir output/test_inference \
        --device cuda:1 \
        --pre_xgb_threshold 0.05
"""

import argparse
import sys
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
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
from inference.stages.post_filter import PostXGBFilter
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
class TestInferenceConfig:
    """Configuration for test inference pipeline."""

    # Input/Output
    dataset_root: Path = field(default_factory=Path)
    output_dir: Path = field(default_factory=lambda: Path("output/test_inference"))

    # Pre-XGB model and filtering
    pre_xgb_model: Path = field(default_factory=Path)
    pre_xgb_threshold: float = 0.05  # Threshold for filtering

    # VMAE model
    vmae_checkpoint: Path = field(default_factory=Path)
    backbone_ckpt: Optional[Path] = None

    # Post-XGB model
    post_xgb_model: Path = field(default_factory=Path)

    # RF-DETR config
    rf_detr_weights: Optional[Path] = None
    rf_detr_variant: str = "large"
    rf_detr_label_mode: str = "soccernet"
    ball_conf_threshold: float = 0.3

    # Processing
    device: str = "cuda:1"
    batch_size: int = 4
    num_frames: int = 16
    input_size: int = 224
    min_decode_ratio: float = 0.5

    # Frame selection
    mode: str = "ball_only"


def load_pre_xgb_model(model_path: Path) -> Tuple:
    """Load pre-XGB model and feature names."""
    print(f"Loading Pre-XGB model from {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    feature_path = model_path.parent / "feature_names.pkl"
    if feature_path.exists():
        with open(feature_path, "rb") as f:
            feature_names = pickle.load(f)
    else:
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
    """Run RF-DETR ball and player detection on all frames."""
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

        frames = [video_reader.get_frame(idx) for idx in frame_indices]
        valid_frames = [
            (idx, f) for idx, f in zip(frame_indices, frames) if f is not None
        ]
        decoded += len(valid_frames)

        if not valid_frames:
            continue

        indices, images = zip(*valid_frames)
        results = rf_detr(images, score_threshold=ball_conf_threshold)

        for idx, dets in zip(indices, results):
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


def compute_pre_xgb_probabilities(
    ball_detections: Dict[int, Dict],
    player_detections: Dict[int, List[Dict]],
    model,
    feature_names: List[str],
    fps: int = 25,
) -> Dict[int, float]:
    """Compute pre-XGB probabilities for all frames with ball detections."""
    if len(ball_detections) < 3:
        return {}

    # Convert to list format for feature extraction
    ball_positions = []
    for frame_id in sorted(ball_detections.keys()):
        det = ball_detections[frame_id]
        box = det.get("box", [0, 0, 0, 0])
        conf = det.get("confidence", 0.0)
        ball_positions.append(
            (frame_id, box[0], box[1], box[2], box[3], conf)
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
    """Run CNN inference on candidate frames."""
    cnn_probs = {}

    half_window = num_frames // 2
    window_offsets = list(range(-half_window, half_window))

    for batch_start in tqdm(
        range(0, len(candidate_frames), batch_size),
        desc="VMAE inference",
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
    config: TestInferenceConfig,
    rf_detr: RFDetrInference,
    pre_xgb_model,
    pre_xgb_feature_names: List[str],
    cnn_inference: CNNInference,
    frame_cropper: FrameCropper,
    post_xgb_filter: PostXGBFilter,
) -> Tuple[List[Dict], Dict[str, object]]:
    """
    Process a single video through the full pipeline.

    Returns:
        Tuple of (records, failure_info)
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

        # Step 2: Get all candidate frames (ball_only mode)
        candidate_frames = sorted(ball_detections.keys())

        if not candidate_frames:
            print(f"  No candidate frames in {video_source.key}")
            return [], failure

        print(f"  {len(candidate_frames)} candidate frames (ball detected)")

        # Step 3: Compute pre-XGB probabilities for ALL frames
        pre_xgb_probs = compute_pre_xgb_probabilities(
            ball_detections,
            player_detections,
            pre_xgb_model,
            pre_xgb_feature_names,
            fps=int(video_reader.fps),
        )

        # Step 4: Filter frames by pre-XGB threshold for VMAE inference
        filtered_frames = [
            frame_idx for frame_idx in candidate_frames
            if pre_xgb_probs.get(frame_idx, 0.0) >= config.pre_xgb_threshold
        ]

        print(
            f"  {len(filtered_frames)}/{len(candidate_frames)} frames pass "
            f"pre-XGB threshold ({config.pre_xgb_threshold})"
        )

        # Step 5: Run VMAE inference ONLY on filtered frames
        vmae_probs = {}
        if filtered_frames:
            vmae_probs = run_cnn_inference(
                filtered_frames,
                ball_detections,
                video_reader,
                cnn_inference,
                frame_cropper,
                config.batch_size,
                config.num_frames,
            )

        # Frames that didn't pass pre-XGB get vmae_prob = 0.0
        for frame_idx in candidate_frames:
            if frame_idx not in vmae_probs:
                vmae_probs[frame_idx] = 0.0

        # Step 6: Post-XGB temporal smoothing
        # Post-XGB uses both vmae_probs and pre_xgb_probs
        post_xgb_probs = post_xgb_filter.refine_with_cnn_probs(
            frame_probs=vmae_probs,
            pre_xgb_probs=pre_xgb_probs,
        )

        # Step 7: Build output records
        for frame_idx in candidate_frames:
            ball_det = ball_detections.get(frame_idx, {})
            box = ball_det.get("box", [0, 0, 0, 0])
            ball_conf = ball_det.get("confidence", 0.0)

            pre_prob = pre_xgb_probs.get(frame_idx, 0.0)
            vmae_prob = vmae_probs.get(frame_idx, 0.0)
            post_prob = post_xgb_probs.get(frame_idx, 0.0)

            # Final prediction based on post-XGB probability
            final_pred = int(post_prob >= 0.5)

            records.append(
                {
                    "video_id": video_source.key,
                    "half": video_source.half,
                    "frame_id": frame_idx,
                    "ball_x": box[0] + box[2] / 2,  # center x
                    "ball_y": box[1] + box[3] / 2,  # center y
                    "ball_conf": ball_conf,
                    "pre_xgb_prob": pre_prob,
                    "vmae_prob": vmae_prob,
                    "post_xgb_prob": post_prob,
                    "final_prediction": final_pred,
                }
            )

    return records, failure


def main():
    parser = argparse.ArgumentParser(
        description="Test Inference Pipeline for Header Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input/Output
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Path to SoccerNet dataset root",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/test_inference",
        help="Output directory for predictions",
    )

    # Model paths
    parser.add_argument(
        "--pre_xgb_model",
        type=str,
        default="output/pre_xgb/train/pre_xgb_final.pkl",
        help="Path to pre-XGB model",
    )
    parser.add_argument(
        "--vmae_checkpoint",
        type=str,
        default="output/vmae/vmae_full_base/checkpoints/best_epoch_26.pt",
        help="Path to VMAE checkpoint",
    )
    parser.add_argument(
        "--backbone_ckpt",
        type=str,
        default="checkpoints/VideoMAEv2-Base",
        help="Path to VideoMAE pretrained weights",
    )
    parser.add_argument(
        "--post_xgb_model",
        type=str,
        default="output/post_xgb/post_xgb_vmae_ball_only_thr0.2/post_xgb_final.pkl",
        help="Path to post-XGB model",
    )

    # RF-DETR config
    parser.add_argument(
        "--rf_detr_weights",
        type=str,
        default="RFDETR-Soccernet/weights/checkpoint_best_regular.pth",
        help="Path to RF-DETR weights",
    )
    parser.add_argument(
        "--rf_detr_label_mode",
        type=str,
        default="soccernet",
        choices=["coco", "soccernet"],
        help="RF-DETR label mode",
    )
    parser.add_argument(
        "--ball_conf_threshold",
        type=float,
        default=0.3,
        help="Ball detection confidence threshold",
    )

    # Pre-XGB filtering
    parser.add_argument(
        "--pre_xgb_threshold",
        type=float,
        default=0.05,
        help="Pre-XGB probability threshold for filtering (default: 0.05)",
    )

    # Processing
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:1",
        help="Device for inference (default: cuda:1)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for inference",
    )

    args = parser.parse_args()

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build config
    config = TestInferenceConfig(
        dataset_root=Path(args.dataset_root),
        output_dir=Path(args.output_dir),
        pre_xgb_model=Path(args.pre_xgb_model),
        pre_xgb_threshold=args.pre_xgb_threshold,
        vmae_checkpoint=Path(args.vmae_checkpoint),
        backbone_ckpt=Path(args.backbone_ckpt),
        post_xgb_model=Path(args.post_xgb_model),
        rf_detr_weights=Path(args.rf_detr_weights) if args.rf_detr_weights else None,
        rf_detr_label_mode=args.rf_detr_label_mode,
        ball_conf_threshold=args.ball_conf_threshold,
        device=str(device),
        batch_size=args.batch_size,
    )

    # Validate paths
    print("\n" + "=" * 60)
    print("Validating model paths...")
    required_paths = [
        ("Pre-XGB model", config.pre_xgb_model),
        ("VMAE checkpoint", config.vmae_checkpoint),
        ("Post-XGB model", config.post_xgb_model),
    ]
    if config.rf_detr_weights:
        required_paths.append(("RF-DETR weights", config.rf_detr_weights))
    if config.backbone_ckpt:
        required_paths.append(("VideoMAE backbone", config.backbone_ckpt))

    for name, path in required_paths:
        if not path.exists():
            print(f"[ERROR] {name} not found: {path}")
            sys.exit(1)
        print(f"  [OK] {name}: {path}")

    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Discover test videos
    print(f"\n{'=' * 60}")
    print(f"Discovering videos in {config.dataset_root}...")

    sources = discover_video_sources(config.dataset_root)

    # If dataset_root already points to a split folder (train/val/test),
    # use all discovered videos. Otherwise, try to filter for test split.
    dataset_root_str = str(config.dataset_root)
    if any(split in dataset_root_str for split in ["/test", "/val", "/train"]):
        # Already pointing to a specific split
        test_sources = sources
        print(f"Dataset root points to split folder, using all discovered videos")
    else:
        # Try to filter to test split
        test_filter = "/test/"
        test_sources = {
            k: v for k, v in sources.items()
            if test_filter in str(v.path)
        }
        if not test_sources:
            print(f"[WARN] No videos found in test split, using all videos")
            test_sources = sources

    print(f"Found {len(test_sources)} videos")

    if not test_sources:
        print("[ERROR] No videos found!")
        sys.exit(1)

    # Initialize models
    print(f"\n{'=' * 60}")
    print("Initializing models...")

    # RF-DETR for ball + player detection
    if config.rf_detr_label_mode == "soccernet":
        target_class_names = ()
        target_class_ids = (0, 1, 2, 3)
    else:
        target_class_names = ("sports ball", "person")
        target_class_ids = None

    rf_detr_config = RFDetrConfig(
        variant="large",
        weights_path=str(config.rf_detr_weights) if config.rf_detr_weights else None,
        device=config.device,
        target_class_names=target_class_names,
        target_class_ids=target_class_ids,
        optimize=True,
        optimize_batch_size=8,
        optimize_compile=False,
    )
    rf_detr_model = build_rf_detr(rf_detr_config)
    rf_detr = RFDetrInference(rf_detr_model, rf_detr_config)
    print("  [OK] RF-DETR loaded")

    # Pre-XGB model
    pre_xgb_model, pre_xgb_feature_names = load_pre_xgb_model(config.pre_xgb_model)
    print("  [OK] Pre-XGB loaded")

    # VMAE model
    inference_config = InferenceConfig(
        model_checkpoint=config.vmae_checkpoint,
        backbone="vmae",
        backbone_ckpt=config.backbone_ckpt,
        num_frames=config.num_frames,
        input_size=config.input_size,
    )
    cnn_inference = CNNInference(inference_config, device)
    print("  [OK] VMAE loaded")

    # Post-XGB filter
    post_config = InferenceConfig(
        post_xgb_model=config.post_xgb_model,
    )
    post_xgb_filter = PostXGBFilter(post_config)
    print("  [OK] Post-XGB loaded")

    # Frame cropper
    frame_cropper = FrameCropper()

    # Process videos
    print(f"\n{'=' * 60}")
    print(f"Processing {len(test_sources)} test videos...")
    print(f"Pre-XGB threshold: {config.pre_xgb_threshold}")
    print("=" * 60)

    all_records: List[Dict] = []
    failed_records: List[Dict[str, object]] = []

    for key, source in tqdm(test_sources.items(), desc="Videos"):
        print(f"\nProcessing {key}...")
        records, failure = process_video(
            source,
            config,
            rf_detr,
            pre_xgb_model,
            pre_xgb_feature_names,
            cnn_inference,
            frame_cropper,
            post_xgb_filter,
        )
        if failure:
            failed_records.append(failure)
            print(
                f"[WARN] Skipping {source.key}: decoded "
                f"{failure.get('decoded_frames')}/{failure.get('attempted_frames')} "
                f"frames ({failure.get('decoded_ratio', 0):.2%})"
            )
            continue
        all_records.extend(records)

    # Save output
    output_path = config.output_dir / "test_predictions.csv"
    output_df = pd.DataFrame(all_records)
    output_df.to_csv(output_path, index=False)

    if failed_records:
        failed_path = config.output_dir / "failed_videos.csv"
        pd.DataFrame(failed_records).to_csv(failed_path, index=False)
        print(f"\n[WARN] Logged {len(failed_records)} failed videos to {failed_path}")

    # Summary
    print(f"\n{'=' * 60}")
    print("INFERENCE COMPLETE")
    print(f"{'=' * 60}")
    print(f"Output: {output_path}")
    print(f"Total frames: {len(output_df)}")
    print(f"Videos processed: {len(test_sources) - len(failed_records)}")
    print(f"Videos failed: {len(failed_records)}")

    if len(output_df) > 0:
        print(f"\nProbability statistics:")
        print(
            f"  Pre-XGB:  mean={output_df['pre_xgb_prob'].mean():.4f}, "
            f"std={output_df['pre_xgb_prob'].std():.4f}"
        )
        print(
            f"  VMAE:     mean={output_df['vmae_prob'].mean():.4f}, "
            f"std={output_df['vmae_prob'].std():.4f}"
        )
        print(
            f"  Post-XGB: mean={output_df['post_xgb_prob'].mean():.4f}, "
            f"std={output_df['post_xgb_prob'].std():.4f}"
        )

        # Count frames passing pre-XGB filter
        vmae_processed = (output_df['vmae_prob'] > 0).sum()
        print(
            f"\nFrames with VMAE inference: {vmae_processed}/{len(output_df)} "
            f"({100*vmae_processed/len(output_df):.1f}%)"
        )

        # Final predictions
        positive_preds = output_df['final_prediction'].sum()
        print(
            f"Final positive predictions: {positive_preds}/{len(output_df)} "
            f"({100*positive_preds/len(output_df):.1f}%)"
        )

    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
