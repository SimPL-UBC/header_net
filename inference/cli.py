#!/usr/bin/env python3
"""
CLI entry point for header detection inference.

Usage:
    python -m inference.cli \\
        --video /path/to/match.mp4 \\
        --checkpoint /path/to/model.pt \\
        --output predictions.csv \\
        --backbone vmae \\
        --window-mode dense \\
        --batch-size 4

Examples:
    # Basic VMAE inference (dense mode)
    python -m inference.cli \\
        --video match.mp4 \\
        --checkpoint model.pt \\
        --backbone vmae

    # CSN with stride-based processing
    python -m inference.cli \\
        --video match.mp4 \\
        --checkpoint scratch_output/csn_16frames_test/checkpoints/best_epoch_48.pt \\
        --backbone csn \\
        --window-mode every_n \\
        --window-stride 5

    # Ball-only mode (most efficient)
    python -m inference.cli \\
        --video match.mp4 \\
        --checkpoint model.pt \\
        --window-mode ball_only

    # With Pre-XGB filtering
    python -m inference.cli \\
        --video match.mp4 \\
        --checkpoint model.pt \\
        --pre-xgb scratch_output/cache_pre_xgb/pre_xgb_final.pkl
"""

import argparse
from pathlib import Path
import sys

from .config import InferenceConfig
from .pipeline import HeaderDetectionPipeline


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Soccer header detection inference pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument(
        "--video", "-v",
        type=Path,
        required=True,
        help="Path to input video file (mp4/mkv)",
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=Path,
        required=True,
        help="Path to trained model checkpoint (.pt)",
    )

    # Output
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("predictions.csv"),
        help="Output CSV path for predictions (default: predictions.csv)",
    )

    # Model configuration
    parser.add_argument(
        "--backbone",
        choices=["vmae", "csn"],
        default="vmae",
        help="Model backbone type (default: vmae)",
    )
    parser.add_argument(
        "--backbone-ckpt",
        type=Path,
        default=None,
        help="VideoMAE pretrained weights directory (for vmae backbone)",
    )

    # Sliding window mode
    parser.add_argument(
        "--window-mode",
        choices=["dense", "every_n", "ball_only"],
        default="dense",
        help="Sliding window mode (default: dense)",
    )
    parser.add_argument(
        "--window-stride",
        type=int,
        default=5,
        help="Frame stride for 'every_n' mode (default: 5)",
    )

    # Processing parameters
    parser.add_argument(
        "--num-frames",
        type=int,
        default=16,
        help="Temporal window size in frames (default: 16)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Inference batch size (default: 4)",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=224,
        help="CNN input resolution (default: 224)",
    )

    # XGB filters (optional)
    parser.add_argument(
        "--pre-xgb",
        type=Path,
        default=None,
        help="Pre-XGB model path for filtering (optional)",
    )
    parser.add_argument(
        "--post-xgb",
        type=Path,
        default=None,
        help="Post-XGB model path for temporal smoothing (optional)",
    )
    parser.add_argument(
        "--pre-xgb-threshold",
        type=float,
        default=0.3,
        help="Pre-XGB filter threshold (default: 0.3)",
    )

    # Ball detection
    parser.add_argument(
        "--rf-detr-weights",
        type=Path,
        default=None,
        help="RF-DETR weights path (uses default if not specified)",
    )
    parser.add_argument(
        "--rf-detr-variant",
        type=str,
        default="medium",
        choices=["nano", "small", "medium", "base", "large"],
        help="RF-DETR model variant (default: medium)",
    )
    parser.add_argument(
        "--no-kalman",
        action="store_true",
        help="Disable Kalman smoothing for ball detection",
    )
    parser.add_argument(
        "--ball-threshold",
        type=float,
        default=0.3,
        help="Ball detection confidence threshold (default: 0.3)",
    )

    # Output thresholds
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Final prediction confidence threshold (default: 0.5)",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Auto-detect if not specified.",
    )

    # Debug
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Save intermediate results for debugging",
    )
    parser.add_argument(
        "--intermediate-dir",
        type=Path,
        default=None,
        help="Directory for intermediate outputs",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Validate inputs
    if not args.video.exists():
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    if not args.checkpoint.exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    if args.pre_xgb and not args.pre_xgb.exists():
        print(f"Error: Pre-XGB model not found: {args.pre_xgb}")
        sys.exit(1)

    if args.post_xgb and not args.post_xgb.exists():
        print(f"Error: Post-XGB model not found: {args.post_xgb}")
        sys.exit(1)

    # Build configuration
    config = InferenceConfig(
        video_path=args.video,
        output_csv=args.output,
        model_checkpoint=args.checkpoint,
        backbone=args.backbone,
        backbone_ckpt=args.backbone_ckpt,
        window_mode=args.window_mode,
        window_stride=args.window_stride,
        num_frames=args.num_frames,
        batch_size=args.batch_size,
        input_size=args.input_size,
        pre_xgb_model=args.pre_xgb,
        post_xgb_model=args.post_xgb,
        pre_xgb_threshold=args.pre_xgb_threshold,
        rf_detr_weights=args.rf_detr_weights,
        rf_detr_variant=args.rf_detr_variant,
        use_kalman=not args.no_kalman,
        ball_conf_threshold=args.ball_threshold,
        confidence_threshold=args.confidence_threshold,
        device=args.device,
        save_intermediate=args.save_intermediate,
        intermediate_dir=args.intermediate_dir,
    )

    # Run pipeline
    try:
        pipeline = HeaderDetectionPipeline(config)
        result_df = pipeline.run()

        # Print summary
        print(f"\nPredictions summary:")
        print(f"  Total frames: {len(result_df)}")
        print(f"  Headers detected: {result_df['prediction'].sum()}")
        if len(result_df) > 0:
            print(f"  Detection rate: {result_df['prediction'].mean() * 100:.2f}%")

    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
