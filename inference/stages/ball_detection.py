"""Ball detection stage using RF-DETR with Kalman smoothing."""

from typing import Dict, List, Optional
import numpy as np
from tqdm import tqdm

import sys
from pathlib import Path

HEADER_NET_ROOT = Path(__file__).resolve().parents[2]
if str(HEADER_NET_ROOT) not in sys.path:
    sys.path.insert(0, str(HEADER_NET_ROOT))

from detectors.rf_detr import RFDetrConfig, RFDetrInference, build_rf_detr
from utils.kalman import KalmanFilter4D

from ..config import InferenceConfig
from ..preprocessing.video_reader import VideoReader


class BallDetector:
    """
    Ball detection stage using RF-DETR with optional Kalman smoothing.

    This stage detects the soccer ball in video frames using the RF-DETR
    object detector, then optionally applies Kalman filtering to smooth
    the ball trajectory and fill gaps in detections.

    Attributes:
        config: Inference configuration.
        detector: RF-DETR inference wrapper.
        use_kalman: Whether to apply Kalman smoothing.
    """

    def __init__(self, config: InferenceConfig, device: str):
        """
        Initialize the ball detector.

        Args:
            config: Inference configuration.
            device: Device string (e.g., "cuda", "cpu").
        """
        self.config = config
        self.device = device
        self.use_kalman = config.use_kalman
        self.conf_threshold = config.ball_conf_threshold
        self.batch_size = config.batch_size

        # Initialize RF-DETR
        rf_config = RFDetrConfig(
            variant=config.rf_detr_variant,
            weights_path=str(config.rf_detr_weights) if config.rf_detr_weights else None,
            device=device,
            target_class_names=("sports ball",),
        )
        model = build_rf_detr(rf_config)
        self.detector = RFDetrInference(model, rf_config)

    def detect_all(
        self,
        video_reader: VideoReader,
        show_progress: bool = True,
    ) -> Dict[int, Dict]:
        """
        Detect ball in all video frames.

        Args:
            video_reader: VideoReader instance for the input video.
            show_progress: Whether to show a progress bar.

        Returns:
            Dictionary mapping frame_idx -> detection dict with:
            - 'box': [x, y, w, h] in pixels
            - 'confidence': detection confidence score
        """
        raw_detections = self._run_detection(video_reader, show_progress)

        if self.use_kalman:
            return self._apply_kalman_smoothing(raw_detections)

        return self._select_best_per_frame(raw_detections)

    def _run_detection(
        self,
        video_reader: VideoReader,
        show_progress: bool = True,
    ) -> Dict[int, List[Dict]]:
        """Run RF-DETR on all frames in batches."""
        detections = {}
        total_frames = video_reader.frame_count

        frames_batch = []
        indices_batch = []

        iterator = range(total_frames)
        if show_progress:
            iterator = tqdm(iterator, desc="Ball detection", unit="frame")

        for frame_idx in iterator:
            frame = video_reader.get_frame(frame_idx)
            if frame is None:
                continue

            frames_batch.append(frame)
            indices_batch.append(frame_idx)

            if len(frames_batch) >= self.batch_size:
                self._process_batch(frames_batch, indices_batch, detections)
                frames_batch = []
                indices_batch = []

        # Process remaining frames
        if frames_batch:
            self._process_batch(frames_batch, indices_batch, detections)

        return detections

    def _process_batch(
        self,
        frames: List[np.ndarray],
        indices: List[int],
        detections: Dict[int, List[Dict]],
    ):
        """Process a batch of frames through the detector."""
        batch_results = self.detector(
            frames,
            score_threshold=self.conf_threshold,
            topk=5,
        )

        for idx, dets in zip(indices, batch_results):
            if dets:
                detections[idx] = dets

    def _apply_kalman_smoothing(
        self,
        raw_detections: Dict[int, List[Dict]],
    ) -> Dict[int, Dict]:
        """
        Apply Kalman filter smoothing to detections.

        This smooths the ball trajectory and can interpolate
        positions for frames with missing detections.
        """
        if not raw_detections:
            return {}

        smoothed = {}
        kalman = KalmanFilter4D()
        initialized = False

        # Process frames in order
        sorted_frames = sorted(raw_detections.keys())
        prev_frame_id = None

        for frame_id in sorted_frames:
            frame_dets = raw_detections[frame_id]
            if not frame_dets:
                continue

            # Select highest confidence detection
            best_det = max(frame_dets, key=lambda x: x.get("confidence", 0.0))
            box = best_det["box"]
            cx = box[0] + box[2] / 2
            cy = box[1] + box[3] / 2

            # Handle gaps in detection sequence
            if initialized and prev_frame_id is not None:
                gap = frame_id - prev_frame_id
                # Predict through gap
                for _ in range(gap - 1):
                    kalman.predict()

            if not initialized:
                kalman.init_state(cx, cy)
                initialized = True
            else:
                kalman.predict()
                kalman.update(cx, cy)

            state_x, state_y, vx, vy = kalman.get_state()

            smoothed[frame_id] = {
                "box": [
                    state_x - box[2] / 2,
                    state_y - box[3] / 2,
                    box[2],
                    box[3],
                ],
                "confidence": best_det.get("confidence", 0.0),
                "velocity": [vx, vy],
            }

            prev_frame_id = frame_id

        return smoothed

    def _select_best_per_frame(
        self,
        raw_detections: Dict[int, List[Dict]],
    ) -> Dict[int, Dict]:
        """Select highest confidence detection per frame."""
        return {
            frame_id: max(dets, key=lambda x: x.get("confidence", 0.0))
            for frame_id, dets in raw_detections.items()
            if dets
        }

    def detect_sparse(
        self,
        video_reader: VideoReader,
        target_frames: List[int],
        show_progress: bool = True,
    ) -> Dict[int, Dict]:
        """
        Detect ball only on specified frames (sparse detection).

        More efficient when only a subset of frames need detection.

        Args:
            video_reader: VideoReader instance.
            target_frames: List of frame indices to process.
            show_progress: Whether to show progress bar.

        Returns:
            Dictionary mapping frame_idx -> detection dict.
        """
        raw_detections = {}
        sorted_frames = sorted(set(target_frames))

        iterator = sorted_frames
        if show_progress:
            iterator = tqdm(iterator, desc="Sparse ball detection", unit="frame")

        frames_batch = []
        indices_batch = []

        for frame_idx in iterator:
            frame = video_reader.get_frame(frame_idx)
            if frame is None:
                continue

            frames_batch.append(frame)
            indices_batch.append(frame_idx)

            if len(frames_batch) >= self.batch_size:
                self._process_batch(frames_batch, indices_batch, raw_detections)
                frames_batch = []
                indices_batch = []

        if frames_batch:
            self._process_batch(frames_batch, indices_batch, raw_detections)

        if self.use_kalman:
            return self._apply_kalman_smoothing(raw_detections)

        return self._select_best_per_frame(raw_detections)
