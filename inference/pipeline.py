"""Main inference pipeline for soccer header detection."""

from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

from .config import InferenceConfig
from .utils.device import get_device, get_device_info
from .preprocessing.video_reader import VideoReader
from .preprocessing.frame_cropper import FrameCropper
from .stages.ball_detection import BallDetector
from .stages.model_inference import CNNInference
from .stages.pre_filter import PreXGBFilter
from .stages.post_filter import PostXGBFilter


@dataclass
class FramePrediction:
    """Prediction result for a single frame.

    Attributes:
        frame_idx: Frame index in the video.
        prediction: Binary prediction (0 = non-header, 1 = header).
        confidence: Final confidence score [0, 1].
        pre_xgb_prob: Pre-XGB probability (if used).
        cnn_prob: CNN/VMAE probability.
        post_xgb_prob: Post-XGB probability (if used).
        ball_detected: Whether ball was detected in this frame.
        ball_x: Ball center x coordinate.
        ball_y: Ball center y coordinate.
    """

    frame_idx: int
    prediction: int
    confidence: float
    pre_xgb_prob: Optional[float] = None
    cnn_prob: Optional[float] = None
    post_xgb_prob: Optional[float] = None
    ball_detected: bool = False
    ball_x: Optional[float] = None
    ball_y: Optional[float] = None


class HeaderDetectionPipeline:
    """
    Three-stage cascade pipeline for soccer header detection.

    Pipeline stages:
    1. Ball Detection (RF-DETR + Kalman smoothing)
    2. Pre-XGB Filter (kinematic features) - optional
    3. CNN/VMAE Inference (16-frame temporal windows)
    4. Post-XGB Filter (temporal smoothing) - optional

    Example:
        >>> config = InferenceConfig(
        ...     video_path="match.mp4",
        ...     model_checkpoint="model.pt",
        ...     backbone="csn",
        ... )
        >>> pipeline = HeaderDetectionPipeline(config)
        >>> results = pipeline.run()
        >>> results.to_csv("predictions.csv")
    """

    def __init__(self, config: InferenceConfig):
        """
        Initialize the inference pipeline.

        Args:
            config: Inference configuration.
        """
        self.config = config
        self.device = get_device(config.device)

        print(f"Initializing pipeline on device: {get_device_info(self.device)}")

        # Initialize stages
        self.ball_detector = BallDetector(config, str(self.device))

        self.pre_filter = None
        if config.pre_xgb_model:
            self.pre_filter = PreXGBFilter(config)

        self.cnn_inference = CNNInference(config, self.device)

        self.post_filter = None
        if config.post_xgb_model:
            self.post_filter = PostXGBFilter(config)

        # Preprocessing
        self.frame_cropper = FrameCropper()

    def run(self) -> pd.DataFrame:
        """
        Execute the full inference pipeline.

        Returns:
            DataFrame with columns:
            - frame: Frame index
            - prediction: Binary prediction (0 or 1)
            - confidence: Confidence score
            - ball_detected: Whether ball was detected
            - ball_x, ball_y: Ball center coordinates
        """
        print(f"\n{'='*60}")
        print(f"Running header detection on: {self.config.video_path}")
        print(f"Backbone: {self.config.backbone}")
        print(f"Window mode: {self.config.window_mode}")
        print(f"{'='*60}\n")

        # Open video
        video_reader = VideoReader(self.config.video_path)
        print(f"Video: {video_reader}")

        try:
            # Stage 1: Ball detection
            print("\n[Stage 1] Ball Detection...")
            ball_detections = self.ball_detector.detect_all(video_reader)
            print(f"Detected ball in {len(ball_detections)}/{video_reader.frame_count} frames")

            # Determine candidate frames based on window mode
            candidate_frames = self._get_candidate_frames(
                ball_detections,
                video_reader.frame_count,
            )
            print(f"Candidate frames ({self.config.window_mode}): {len(candidate_frames)}")

            # Stage 2: Pre-XGB filtering (optional)
            if self.pre_filter:
                print("\n[Stage 2] Pre-XGB Filtering...")
                candidate_frames = self.pre_filter.filter(
                    candidate_frames,
                    ball_detections,
                    threshold=self.config.pre_xgb_threshold,
                    fps=video_reader.fps,
                )

            # Stage 3: CNN/VMAE inference
            print(f"\n[Stage 3] CNN Inference on {len(candidate_frames)} candidates...")
            predictions = self._run_cnn_inference(
                candidate_frames,
                ball_detections,
                video_reader,
            )

            # Stage 4: Post-XGB filtering (optional)
            if self.post_filter and predictions:
                print("\n[Stage 4] Post-XGB Temporal Smoothing...")
                predictions = self.post_filter.refine(predictions)

            # Apply confidence threshold
            final_predictions = self._apply_threshold(predictions)

            # Convert to DataFrame
            result_df = self._to_dataframe(final_predictions)

            # Save output
            self.config.output_csv.parent.mkdir(parents=True, exist_ok=True)
            result_df.to_csv(self.config.output_csv, index=False)

            # Summary
            print(f"\n{'='*60}")
            print(f"Results saved to: {self.config.output_csv}")
            print(f"Total frames processed: {len(result_df)}")
            headers_detected = result_df["prediction"].sum()
            print(f"Headers detected: {headers_detected}")
            print(f"{'='*60}\n")

            return result_df

        finally:
            video_reader.release()

    def _get_candidate_frames(
        self,
        ball_detections: Dict[int, Dict],
        total_frames: int,
    ) -> List[int]:
        """Get candidate frames based on window mode."""
        if self.config.window_mode == "ball_only":
            # Only frames with ball detection
            return sorted(ball_detections.keys())

        elif self.config.window_mode == "every_n":
            # Every N frames
            stride = self.config.window_stride
            return list(range(0, total_frames, stride))

        else:  # dense
            # All frames
            return list(range(total_frames))

    def _run_cnn_inference(
        self,
        candidate_frames: List[int],
        ball_detections: Dict[int, Dict],
        video_reader: VideoReader,
    ) -> List[FramePrediction]:
        """Run batched CNN inference on candidate frames."""
        predictions = []
        batch_size = self.config.batch_size
        num_frames = self.config.num_frames

        # Compute window offsets (evenly spaced around center)
        half_window = num_frames // 2
        window_offsets = list(range(-half_window, half_window))

        # Process in batches
        for batch_start in tqdm(
            range(0, len(candidate_frames), batch_size),
            desc="CNN inference",
            unit="batch",
        ):
            batch_frames = candidate_frames[batch_start : batch_start + batch_size]

            # Extract temporal windows
            batch_windows = []
            batch_frame_info = []

            for center_frame in batch_frames:
                # Get required frame indices
                required_frames = [
                    center_frame + offset for offset in window_offsets
                ]

                # Load frames
                frames = video_reader.get_frames(required_frames)

                # Get ball detection for center frame
                center_det = ball_detections.get(center_frame)

                # Crop temporal window
                window = self.frame_cropper.crop_temporal_window(
                    frames,
                    ball_detections,
                    center_frame,
                    window_offsets,
                )

                batch_windows.append(window)
                batch_frame_info.append((center_frame, center_det))

            # Run inference
            probs = self.cnn_inference.predict_batch(batch_windows)

            # Create predictions
            for i, (center_frame, center_det) in enumerate(batch_frame_info):
                prob = probs[i]

                ball_x, ball_y = None, None
                ball_detected = center_det is not None
                if ball_detected and "box" in center_det:
                    box = center_det["box"]
                    ball_x = box[0] + box[2] / 2
                    ball_y = box[1] + box[3] / 2

                predictions.append(
                    FramePrediction(
                        frame_idx=center_frame,
                        prediction=int(prob >= 0.5),
                        confidence=prob,
                        cnn_prob=prob,
                        ball_detected=ball_detected,
                        ball_x=ball_x,
                        ball_y=ball_y,
                    )
                )

        return predictions

    def _apply_threshold(
        self,
        predictions: List[FramePrediction],
    ) -> List[FramePrediction]:
        """Apply confidence threshold to predictions."""
        threshold = self.config.confidence_threshold

        for pred in predictions:
            pred.prediction = int(pred.confidence >= threshold)

        return predictions

    def _to_dataframe(
        self,
        predictions: List[FramePrediction],
    ) -> pd.DataFrame:
        """Convert predictions to DataFrame."""
        if not predictions:
            return pd.DataFrame(columns=[
                "frame", "prediction", "confidence",
                "ball_detected", "ball_x", "ball_y",
            ])

        data = []
        for pred in sorted(predictions, key=lambda p: p.frame_idx):
            data.append({
                "frame": pred.frame_idx,
                "prediction": pred.prediction,
                "confidence": pred.confidence,
                "ball_detected": pred.ball_detected,
                "ball_x": pred.ball_x,
                "ball_y": pred.ball_y,
            })

        return pd.DataFrame(data)


def run_inference(
    video_path: str,
    checkpoint_path: str,
    output_path: str = "predictions.csv",
    backbone: str = "vmae",
    window_mode: str = "dense",
    **kwargs,
) -> pd.DataFrame:
    """
    Convenience function to run inference.

    Args:
        video_path: Path to input video.
        checkpoint_path: Path to model checkpoint.
        output_path: Path for output CSV.
        backbone: Model backbone ("vmae" or "csn").
        window_mode: Sliding window mode.
        **kwargs: Additional config options.

    Returns:
        DataFrame with predictions.
    """
    config = InferenceConfig(
        video_path=Path(video_path),
        model_checkpoint=Path(checkpoint_path),
        output_csv=Path(output_path),
        backbone=backbone,
        window_mode=window_mode,
        **kwargs,
    )

    pipeline = HeaderDetectionPipeline(config)
    return pipeline.run()
