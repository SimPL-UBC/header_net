"""Post-XGB filter stage for temporal smoothing."""

from typing import Dict, List, Optional, TYPE_CHECKING
import pickle
import numpy as np

from pathlib import Path

from ..config import InferenceConfig

if TYPE_CHECKING:
    from ..pipeline import FramePrediction


class PostXGBFilter:
    """
    Post-filter stage using XGBoost for temporal smoothing.

    This stage uses CNN predictions from a temporal window (±15 frames)
    to refine predictions and suppress spurious detections.

    Features (48 total):
    - 31 per-frame CNN probabilities (from -15 to +15)
    - Statistical aggregates: mean, std, max, min, median
    - Trend features: slope
    - Local maxima indicators

    Attributes:
        config: Inference configuration.
        model: Trained XGBoost classifier.
        feature_names: List of feature names in order.
        window_size: Temporal window size (frames on each side).
    """

    def __init__(self, config: InferenceConfig):
        """
        Initialize the Post-XGB filter.

        Args:
            config: Inference configuration with post_xgb_model path.
        """
        self.config = config
        self.window_size = 15  # ±15 frames

        if config.post_xgb_model is None:
            raise ValueError("post_xgb_model path is required for PostXGBFilter")

        # Load XGBoost model
        model_path = Path(config.post_xgb_model)
        print(f"Loading Post-XGB model from {model_path}")

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        # Load feature names
        feature_path = model_path.parent / "feature_names.pkl"
        if feature_path.exists():
            with open(feature_path, "rb") as f:
                self.feature_names = pickle.load(f)
        else:
            # Build default feature names
            self.feature_names = self._build_default_feature_names()

        print(f"Post-XGB model loaded with {len(self.feature_names)} features")

    def _build_default_feature_names(self) -> List[str]:
        """Build default feature names for post-filter."""
        names = []

        # Per-frame probabilities
        for offset in range(-self.window_size, self.window_size + 1):
            names.append(f"cnn_prob_{offset}")

        # Statistical aggregates
        names.extend([
            "cnn_prob_mean",
            "cnn_prob_std",
            "cnn_prob_max",
            "cnn_prob_min",
            "cnn_prob_median",
            "cnn_prob_slope",
            "is_local_max_cnn",
        ])

        return names

    def _extract_temporal_features(
        self,
        predictions: List["FramePrediction"],
        center_idx: int,
    ) -> Optional[Dict[str, float]]:
        """
        Extract temporal features around center prediction.

        Args:
            predictions: Sorted list of FramePrediction objects.
            center_idx: Index of center prediction in the list.

        Returns:
            Feature dictionary or None if window is incomplete.
        """
        window = self.window_size
        cnn_probs = []

        # Collect probabilities in window
        for offset in range(-window, window + 1):
            target_idx = center_idx + offset
            if 0 <= target_idx < len(predictions):
                prob = predictions[target_idx].cnn_prob or 0.0
                cnn_probs.append(prob)
            else:
                cnn_probs.append(0.0)

        # Build feature dict
        features = {}

        # Per-frame probabilities
        for i, prob in enumerate(cnn_probs):
            offset = i - window
            features[f"cnn_prob_{offset}"] = prob

        # Statistical aggregates
        features["cnn_prob_mean"] = np.mean(cnn_probs)
        features["cnn_prob_std"] = np.std(cnn_probs)
        features["cnn_prob_max"] = np.max(cnn_probs)
        features["cnn_prob_min"] = np.min(cnn_probs)
        features["cnn_prob_median"] = np.median(cnn_probs)

        # Trend (slope)
        if len(cnn_probs) >= 3:
            x = np.arange(len(cnn_probs))
            features["cnn_prob_slope"] = np.polyfit(x, cnn_probs, 1)[0]
        else:
            features["cnn_prob_slope"] = 0.0

        # Local maxima indicator
        center = window
        is_local_max = (
            center > 0
            and center < len(cnn_probs) - 1
            and cnn_probs[center] >= cnn_probs[center - 1]
            and cnn_probs[center] >= cnn_probs[center + 1]
        )
        features["is_local_max_cnn"] = float(is_local_max)

        return features

    def refine(
        self,
        predictions: List["FramePrediction"],
    ) -> List["FramePrediction"]:
        """
        Refine predictions using temporal context.

        Args:
            predictions: List of FramePrediction from CNN stage.

        Returns:
            List of FramePrediction with refined confidences.
        """
        if not predictions:
            return predictions

        # Sort by frame index
        predictions = sorted(predictions, key=lambda p: p.frame_idx)

        # Refine each prediction
        for i, pred in enumerate(predictions):
            features = self._extract_temporal_features(predictions, i)

            if features is not None:
                # Build feature vector
                feat_vector = [
                    features.get(fn, 0.0) for fn in self.feature_names
                ]

                # Get refined probability
                try:
                    refined_prob = self.model.predict_proba([feat_vector])[0, 1]

                    # Update prediction
                    pred.post_xgb_prob = refined_prob
                    pred.confidence = refined_prob
                    pred.prediction = int(refined_prob >= 0.5)
                except Exception as e:
                    # Keep original prediction if refinement fails
                    print(f"Warning: Post-XGB refinement failed for frame {pred.frame_idx}: {e}")

        return predictions

    def refine_with_cnn_probs(
        self,
        frame_probs: Dict[int, float],
    ) -> Dict[int, float]:
        """
        Refine CNN probabilities using temporal context.

        Alternative interface using dictionary of probabilities.

        Args:
            frame_probs: Dictionary mapping frame_idx -> CNN probability.

        Returns:
            Dictionary mapping frame_idx -> refined probability.
        """
        if not frame_probs:
            return {}

        sorted_frames = sorted(frame_probs.keys())
        probs_list = [frame_probs[f] for f in sorted_frames]

        refined = {}

        for i, frame_idx in enumerate(sorted_frames):
            # Build window
            cnn_probs = []
            for offset in range(-self.window_size, self.window_size + 1):
                target_idx = i + offset
                if 0 <= target_idx < len(probs_list):
                    cnn_probs.append(probs_list[target_idx])
                else:
                    cnn_probs.append(0.0)

            # Build features
            features = {}
            for j, prob in enumerate(cnn_probs):
                offset = j - self.window_size
                features[f"cnn_prob_{offset}"] = prob

            features["cnn_prob_mean"] = np.mean(cnn_probs)
            features["cnn_prob_std"] = np.std(cnn_probs)
            features["cnn_prob_max"] = np.max(cnn_probs)
            features["cnn_prob_min"] = np.min(cnn_probs)
            features["cnn_prob_median"] = np.median(cnn_probs)

            x = np.arange(len(cnn_probs))
            features["cnn_prob_slope"] = np.polyfit(x, cnn_probs, 1)[0]

            center = self.window_size
            is_local_max = (
                cnn_probs[center] >= cnn_probs[center - 1]
                and cnn_probs[center] >= cnn_probs[center + 1]
            )
            features["is_local_max_cnn"] = float(is_local_max)

            # Predict
            feat_vector = [features.get(fn, 0.0) for fn in self.feature_names]
            refined[frame_idx] = self.model.predict_proba([feat_vector])[0, 1]

        return refined
