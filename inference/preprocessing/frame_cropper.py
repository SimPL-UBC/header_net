"""Ball-centered frame cropping for inference pipeline."""

from typing import Dict, Optional, Tuple, List
import numpy as np
import cv2

# Import configuration constants
import sys
from pathlib import Path

HEADER_NET_ROOT = Path(__file__).resolve().parents[2]
if str(HEADER_NET_ROOT) not in sys.path:
    sys.path.insert(0, str(HEADER_NET_ROOT))

from configs import header_default as cfg


class FrameCropper:
    """
    Ball-centered frame cropping for inference.

    Extracts square patches centered on ball positions, handling
    boundary conditions and resizing to a consistent output size.

    Attributes:
        crop_scale_factor: Multiplier for ball size to determine crop radius.
        output_size: Size of the output crop (pixels).
        default_radius: Default crop radius when ball size is unknown.
    """

    def __init__(
        self,
        crop_scale_factor: float = None,
        output_size: int = None,
        default_radius: int = 100,
    ):
        """
        Initialize the frame cropper.

        Args:
            crop_scale_factor: Multiplier for ball size. Defaults to config value.
            output_size: Output crop size in pixels. Defaults to config value.
            default_radius: Default radius when ball size unavailable.
        """
        self.crop_scale_factor = crop_scale_factor or cfg.CROP_SCALE_FACTOR
        self.output_size = output_size or cfg.OUTPUT_SIZE
        self.default_radius = default_radius

    def compute_radius(
        self,
        ball_detection: Optional[Dict],
        frame_shape: Tuple[int, int],
    ) -> int:
        """
        Compute crop radius based on ball size.

        Args:
            ball_detection: Ball detection dict with 'box' key [x, y, w, h].
            frame_shape: Frame dimensions (height, width).

        Returns:
            Crop radius in pixels.
        """
        if ball_detection is None or "box" not in ball_detection:
            return self.default_radius

        box = ball_detection["box"]
        ball_w, ball_h = box[2], box[3]
        mean_size = max(ball_w, ball_h)

        # Scale radius based on ball size
        radius = int(max(self.default_radius, self.crop_scale_factor * mean_size / 2))

        # Clamp to reasonable bounds (max 75% of frame dimension)
        max_dim = max(frame_shape[0], frame_shape[1])
        radius = int(min(radius, 0.75 * max_dim))

        return radius

    def get_ball_center(
        self,
        ball_detection: Optional[Dict],
        frame_shape: Tuple[int, int],
    ) -> Tuple[float, float]:
        """
        Get ball center coordinates.

        Args:
            ball_detection: Ball detection dict with 'box' key.
            frame_shape: Frame dimensions (height, width).

        Returns:
            Tuple of (center_x, center_y). Falls back to frame center if no detection.
        """
        if ball_detection is None or "box" not in ball_detection:
            # Fall back to frame center
            return frame_shape[1] / 2, frame_shape[0] / 2

        box = ball_detection["box"]
        center_x = box[0] + box[2] / 2
        center_y = box[1] + box[3] / 2
        return center_x, center_y

    def crop(
        self,
        frame: np.ndarray,
        ball_detection: Optional[Dict],
        radius: Optional[int] = None,
    ) -> np.ndarray:
        """
        Crop a square region around the ball position.

        Args:
            frame: Input frame (H, W, C) in RGB.
            ball_detection: Ball detection dict with 'box' key.
            radius: Optional override for crop radius.

        Returns:
            Cropped and resized frame (output_size, output_size, C).
        """
        h, w = frame.shape[:2]

        # Get ball center
        center_x, center_y = self.get_ball_center(ball_detection, (h, w))

        # Compute radius if not provided
        if radius is None:
            radius = self.compute_radius(ball_detection, (h, w))

        # Compute crop bounds
        x1 = int(center_x - radius)
        y1 = int(center_y - radius)
        x2 = int(center_x + radius)
        y2 = int(center_y + radius)

        # Handle boundary conditions with padding
        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x2 - w)
        pad_bottom = max(0, y2 - h)

        # Clamp to valid range
        x1_clamped = max(0, x1)
        y1_clamped = max(0, y1)
        x2_clamped = min(w, x2)
        y2_clamped = min(h, y2)

        # Extract region
        crop = frame[y1_clamped:y2_clamped, x1_clamped:x2_clamped]

        # Apply padding if needed
        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            crop = cv2.copyMakeBorder(
                crop,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )

        # Resize to output size
        if crop.shape[0] != self.output_size or crop.shape[1] != self.output_size:
            crop = cv2.resize(
                crop,
                (self.output_size, self.output_size),
                interpolation=cv2.INTER_LINEAR,
            )

        return crop

    def crop_temporal_window(
        self,
        frames: Dict[int, np.ndarray],
        ball_detections: Dict[int, Dict],
        center_frame: int,
        window_offsets: List[int],
    ) -> Optional[np.ndarray]:
        """
        Crop a temporal window of frames around a center frame.

        Args:
            frames: Dictionary mapping frame_idx -> frame array.
            ball_detections: Dictionary mapping frame_idx -> ball detection.
            center_frame: The center frame index.
            window_offsets: List of frame offsets from center (e.g., [-8, -4, 0, 4, 8]).

        Returns:
            Stacked array (T, H, W, C) or None if frames unavailable.
        """
        crops = []

        # Compute consistent radius based on center frame
        center_det = ball_detections.get(center_frame)
        if center_frame in frames:
            radius = self.compute_radius(center_det, frames[center_frame].shape[:2])
        else:
            radius = self.default_radius

        for offset in window_offsets:
            frame_idx = center_frame + offset
            frame = frames.get(frame_idx)

            if frame is None:
                return None

            detection = ball_detections.get(frame_idx)
            crop = self.crop(frame, detection, radius=radius)
            crops.append(crop)

        return np.stack(crops, axis=0)
