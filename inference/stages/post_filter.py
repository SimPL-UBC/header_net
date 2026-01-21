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

    This stage uses CNN and pre-XGB predictions from a temporal window
    (±15 frames) to refine predictions and suppress spurious detections.

    Features (82 total):
    - Center frame probs: CNN, pre-XGB (2)
    - 31 per-frame CNN probabilities (from -15 to +15)
    - 31 per-frame pre-XGB probabilities (from -15 to +15)
    - Statistical aggregates for CNN: mean, std, max, min, median (5)
    - Statistical aggregates for pre-XGB: mean, std, max, min, median (5)
    - Trend features: cnn_prob_slope, pre_xgb_prob_slope (2)
    - Local maxima indicators: is_local_max_cnn, is_local_max_pre_xgb (2)

    Note: Ensemble features have been removed from this version.

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
        """Build default feature names for post-filter (82 features, no ensemble)."""
        names = []

        # Center frame probabilities (2)
        names.extend([
            "center_cnn_prob",
            "center_pre_xgb_prob",
        ])

        # Per-frame probabilities (62: 31 cnn + 31 pre_xgb)
        for offset in range(-self.window_size, self.window_size + 1):
            names.append(f"cnn_prob_{offset}")
        for offset in range(-self.window_size, self.window_size + 1):
            names.append(f"pre_xgb_prob_{offset}")

        # Statistical aggregates (10: 5 cnn + 5 pre_xgb)
        names.extend([
            "cnn_prob_mean",
            "cnn_prob_std",
            "cnn_prob_max",
            "cnn_prob_min",
            "cnn_prob_median",
            "pre_xgb_prob_mean",
            "pre_xgb_prob_std",
            "pre_xgb_prob_max",
            "pre_xgb_prob_min",
            "pre_xgb_prob_median",
        ])

        # Slopes (2)
        names.extend([
            "cnn_prob_slope",
            "pre_xgb_prob_slope",
        ])

        # Local maxima indicators (2)
        names.extend([
            "is_local_max_cnn",
            "is_local_max_pre_xgb",
        ])

        return names

    def _extract_temporal_features(
        self,
        predictions: List["FramePrediction"],
        center_idx: int,
    ) -> Optional[Dict[str, float]]:
        """
        Extract temporal features around center prediction (no ensemble).

        Args:
            predictions: Sorted list of FramePrediction objects.
            center_idx: Index of center prediction in the list.

        Returns:
            Feature dictionary with 82 features.
        """
        window = self.window_size
        cnn_probs = []
        pre_xgb_probs = []

        frame_to_pred = {p.frame_idx: p for p in predictions}
        center_frame = predictions[center_idx].frame_idx

        # Collect probabilities in window based on true frame offsets
        for offset in range(-window, window + 1):
            target_frame = center_frame + offset
            target_pred = frame_to_pred.get(target_frame)
            if target_pred is not None:
                cnn_prob = target_pred.cnn_prob or 0.0
                pre_xgb_prob = target_pred.pre_xgb_prob or 0.0
            else:
                cnn_prob = 0.0
                pre_xgb_prob = 0.0

            cnn_probs.append(cnn_prob)
            pre_xgb_probs.append(pre_xgb_prob)

        # Build feature dict
        features = {}

        # Center frame probabilities
        features["center_cnn_prob"] = cnn_probs[window]
        features["center_pre_xgb_prob"] = pre_xgb_probs[window]

        # Per-frame probabilities
        for i, prob in enumerate(cnn_probs):
            offset = i - window
            features[f"cnn_prob_{offset}"] = prob
        for i, prob in enumerate(pre_xgb_probs):
            offset = i - window
            features[f"pre_xgb_prob_{offset}"] = prob

        # Statistical aggregates
        features["cnn_prob_mean"] = np.mean(cnn_probs)
        features["cnn_prob_std"] = np.std(cnn_probs)
        features["cnn_prob_max"] = np.max(cnn_probs)
        features["cnn_prob_min"] = np.min(cnn_probs)
        features["cnn_prob_median"] = np.median(cnn_probs)
        features["pre_xgb_prob_mean"] = np.mean(pre_xgb_probs)
        features["pre_xgb_prob_std"] = np.std(pre_xgb_probs)
        features["pre_xgb_prob_max"] = np.max(pre_xgb_probs)
        features["pre_xgb_prob_min"] = np.min(pre_xgb_probs)
        features["pre_xgb_prob_median"] = np.median(pre_xgb_probs)

        # Trend (slope)
        if len(cnn_probs) >= 3:
            x = np.arange(len(cnn_probs))
            features["cnn_prob_slope"] = np.polyfit(x, cnn_probs, 1)[0]
            features["pre_xgb_prob_slope"] = np.polyfit(x, pre_xgb_probs, 1)[0]
        else:
            features["cnn_prob_slope"] = 0.0
            features["pre_xgb_prob_slope"] = 0.0

        # Local maxima indicator
        center = window
        is_local_max_cnn = (
            center > 0
            and center < len(cnn_probs) - 1
            and cnn_probs[center] >= cnn_probs[center - 1]
            and cnn_probs[center] >= cnn_probs[center + 1]
        )
        features["is_local_max_cnn"] = float(is_local_max_cnn)
        is_local_max_pre_xgb = (
            center > 0
            and center < len(pre_xgb_probs) - 1
            and pre_xgb_probs[center] >= pre_xgb_probs[center - 1]
            and pre_xgb_probs[center] >= pre_xgb_probs[center + 1]
        )
        features["is_local_max_pre_xgb"] = float(is_local_max_pre_xgb)

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
        pre_xgb_probs: Optional[Dict[int, float]] = None,
    ) -> Dict[int, float]:
        """
        Refine CNN probabilities using temporal context (no ensemble).

        Alternative interface using dictionary of probabilities.

        Args:
            frame_probs: Dictionary mapping frame_idx -> CNN probability.
            pre_xgb_probs: Optional dictionary mapping frame_idx -> pre-XGB probability.

        Returns:
            Dictionary mapping frame_idx -> refined probability.
        """
        if not frame_probs:
            return {}

        refined = {}

        sorted_frames = sorted(frame_probs.keys())

        for frame_idx in sorted_frames:
            # Build window based on true frame offsets
            cnn_probs = []
            pre_probs = []

            for offset in range(-self.window_size, self.window_size + 1):
                target_frame = frame_idx + offset
                cnn_prob = frame_probs.get(target_frame, 0.0)
                pre_prob = 0.0
                if pre_xgb_probs is not None:
                    pre_prob = pre_xgb_probs.get(target_frame, 0.0)

                cnn_probs.append(cnn_prob)
                pre_probs.append(pre_prob)

            # Build features (no ensemble)
            features = {
                "center_cnn_prob": cnn_probs[self.window_size],
                "center_pre_xgb_prob": pre_probs[self.window_size],
            }

            for j, prob in enumerate(cnn_probs):
                offset = j - self.window_size
                features[f"cnn_prob_{offset}"] = prob
            for j, prob in enumerate(pre_probs):
                offset = j - self.window_size
                features[f"pre_xgb_prob_{offset}"] = prob

            features["cnn_prob_mean"] = np.mean(cnn_probs)
            features["cnn_prob_std"] = np.std(cnn_probs)
            features["cnn_prob_max"] = np.max(cnn_probs)
            features["cnn_prob_min"] = np.min(cnn_probs)
            features["cnn_prob_median"] = np.median(cnn_probs)
            features["pre_xgb_prob_mean"] = np.mean(pre_probs)
            features["pre_xgb_prob_std"] = np.std(pre_probs)
            features["pre_xgb_prob_max"] = np.max(pre_probs)
            features["pre_xgb_prob_min"] = np.min(pre_probs)
            features["pre_xgb_prob_median"] = np.median(pre_probs)

            x = np.arange(len(cnn_probs))
            features["cnn_prob_slope"] = np.polyfit(x, cnn_probs, 1)[0]
            features["pre_xgb_prob_slope"] = np.polyfit(x, pre_probs, 1)[0]

            center = self.window_size
            is_local_max_cnn = (
                cnn_probs[center] >= cnn_probs[center - 1]
                and cnn_probs[center] >= cnn_probs[center + 1]
            )
            features["is_local_max_cnn"] = float(is_local_max_cnn)
            is_local_max_pre = (
                pre_probs[center] >= pre_probs[center - 1]
                and pre_probs[center] >= pre_probs[center + 1]
            )
            features["is_local_max_pre_xgb"] = float(is_local_max_pre)

            # Predict
            feat_vector = [features.get(fn, 0.0) for fn in self.feature_names]
            refined[frame_idx] = self.model.predict_proba([feat_vector])[0, 1]

        return refined
