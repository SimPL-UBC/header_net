"""Pre-XGB filter stage for filtering easy negatives."""

from typing import Dict, List, Optional
import pickle
import numpy as np

import sys
from pathlib import Path

HEADER_NET_ROOT = Path(__file__).resolve().parents[2]
if str(HEADER_NET_ROOT) not in sys.path:
    sys.path.insert(0, str(HEADER_NET_ROOT))

from tree.pre_xgb import (
    compute_kinematics_features,
    add_temporal_features,
    KINEMATIC_FEATURE_NAMES,
)

from ..config import InferenceConfig


class PreXGBFilter:
    """
    Pre-filter stage using XGBoost on kinematic features.

    This stage computes ball kinematics features (velocity, acceleration,
    curvature, etc.) and uses a trained XGBoost model to filter out
    frames that are unlikely to contain headers.

    Features (24 kinematic):
    - Position: x, y
    - Velocity: vx, vy, speed
    - Acceleration: ax, ay, accel_mag
    - Trajectory: angle_change, speed_change, speed_drop_ratio, curvature, jerk
    - Quality: confidence, ball_size
    - Temporal: speed_mean_w, speed_std_w, speed_max_w, etc.

    Attributes:
        config: Inference configuration.
        model: Trained XGBoost classifier.
        feature_names: List of feature names in order.
    """

    def __init__(self, config: InferenceConfig):
        """
        Initialize the Pre-XGB filter.

        Args:
            config: Inference configuration with pre_xgb_model path.
        """
        self.config = config

        if config.pre_xgb_model is None:
            raise ValueError("pre_xgb_model path is required for PreXGBFilter")

        # Load XGBoost model
        model_path = Path(config.pre_xgb_model)
        print(f"Loading Pre-XGB model from {model_path}")

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        # Load feature names
        feature_path = model_path.parent / "feature_names.pkl"
        if feature_path.exists():
            with open(feature_path, "rb") as f:
                self.feature_names = pickle.load(f)
        else:
            # Fall back to kinematic features only
            self.feature_names = KINEMATIC_FEATURE_NAMES

        print(f"Pre-XGB model loaded with {len(self.feature_names)} features")

    def compute_features(
        self,
        ball_detections: Dict[int, Dict],
        fps: float = 25.0,
    ) -> Dict[int, Dict]:
        """
        Compute kinematic features from ball detections.

        Args:
            ball_detections: Dictionary mapping frame_idx -> detection dict.
            fps: Video frame rate.

        Returns:
            Dictionary mapping frame_idx -> feature dict.
        """
        if len(ball_detections) < 3:
            return {}

        # Convert to list format for feature extraction
        ball_positions = []
        for frame_id in sorted(ball_detections.keys()):
            det = ball_detections[frame_id]
            box = det.get("box", [0, 0, 0, 0])
            conf = det.get("confidence", 0.0)
            ball_positions.append((
                frame_id,
                box[0],  # x
                box[1],  # y
                box[2],  # w
                box[3],  # h
                conf,
            ))

        # Compute kinematic features
        features = compute_kinematics_features(ball_positions, fps)

        # Add temporal features
        features = add_temporal_features(features, window=5)

        return features

    def filter(
        self,
        candidate_frames: List[int],
        ball_detections: Dict[int, Dict],
        threshold: float = None,
        fps: float = 25.0,
    ) -> List[int]:
        """
        Filter candidate frames using Pre-XGB model.

        Args:
            candidate_frames: List of frame indices to consider.
            ball_detections: Dictionary of ball detections per frame.
            threshold: Probability threshold for keeping frames.
                      Uses config value if None.
            fps: Video frame rate.

        Returns:
            Filtered list of frame indices that pass the threshold.
        """
        if threshold is None:
            threshold = self.config.pre_xgb_threshold

        # Compute features
        features = self.compute_features(ball_detections, fps)

        if not features:
            # Not enough data for kinematics - return all candidates
            return candidate_frames

        # Filter candidates
        passed_frames = []
        probabilities = {}

        for frame_idx in candidate_frames:
            if frame_idx not in features:
                # No features for this frame - skip
                continue

            feat_dict = features[frame_idx]
            feat_vector = [feat_dict.get(fn, 0.0) for fn in self.feature_names]

            # Get probability from XGBoost
            prob = self.model.predict_proba([feat_vector])[0, 1]
            probabilities[frame_idx] = prob

            if prob >= threshold:
                passed_frames.append(frame_idx)

        print(
            f"Pre-XGB filter: {len(passed_frames)}/{len(candidate_frames)} frames "
            f"passed (threshold={threshold:.2f})"
        )

        return passed_frames

    def get_probabilities(
        self,
        ball_detections: Dict[int, Dict],
        fps: float = 25.0,
    ) -> Dict[int, float]:
        """
        Get Pre-XGB probabilities for all frames with detections.

        Args:
            ball_detections: Dictionary of ball detections per frame.
            fps: Video frame rate.

        Returns:
            Dictionary mapping frame_idx -> probability.
        """
        features = self.compute_features(ball_detections, fps)

        probabilities = {}
        for frame_idx, feat_dict in features.items():
            feat_vector = [feat_dict.get(fn, 0.0) for fn in self.feature_names]
            prob = self.model.predict_proba([feat_vector])[0, 1]
            probabilities[frame_idx] = prob

        return probabilities
