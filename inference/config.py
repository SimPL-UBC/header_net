"""Configuration for the inference pipeline."""

from dataclasses import dataclass, field
from typing import Optional, Literal
from pathlib import Path


@dataclass
class InferenceConfig:
    """Configuration for header detection inference pipeline.

    Attributes:
        video_path: Path to input video file (mp4/mkv)
        output_csv: Path for output CSV with predictions
        model_checkpoint: Path to trained model checkpoint (.pt file)
        backbone: Model backbone type ("vmae" or "csn")
        backbone_ckpt: Path to VideoMAE pretrained weights directory (for vmae only)
        window_mode: Sliding window mode ("dense", "every_n", "ball_only")
        window_stride: Frame stride for "every_n" mode
        num_frames: Temporal window size (frames per sample)
        input_size: CNN input resolution (pixels)
        batch_size: Inference batch size
        pre_xgb_model: Path to Pre-XGB model for filtering (optional)
        post_xgb_model: Path to Post-XGB model for temporal smoothing (optional)
        pre_xgb_threshold: Probability threshold for Pre-XGB filter
        rf_detr_weights: Path to RF-DETR weights (optional, uses default if None)
        rf_detr_variant: RF-DETR model variant
        ball_conf_threshold: Ball detection confidence threshold
        use_kalman: Whether to apply Kalman smoothing to ball detections
        confidence_threshold: Final prediction confidence threshold
        device: Device to use (auto-detect if None)
        save_intermediate: Whether to save intermediate results for debugging
        intermediate_dir: Directory for intermediate outputs
    """

    # Input/Output
    video_path: Path = field(default_factory=Path)
    output_csv: Path = field(default_factory=lambda: Path("predictions.csv"))

    # Model configuration
    model_checkpoint: Path = field(default_factory=Path)
    backbone: Literal["vmae", "csn"] = "vmae"
    backbone_ckpt: Optional[Path] = None

    # Sliding window mode
    window_mode: Literal["dense", "every_n", "ball_only"] = "dense"
    window_stride: int = 5

    # Processing parameters
    num_frames: int = 16
    input_size: int = 224
    batch_size: int = 4

    # XGB filters (optional)
    pre_xgb_model: Optional[Path] = None
    post_xgb_model: Optional[Path] = None
    pre_xgb_threshold: float = 0.3

    # Ball detection
    rf_detr_weights: Optional[Path] = None
    rf_detr_variant: str = "medium"
    ball_conf_threshold: float = 0.3
    use_kalman: bool = True

    # Output thresholds
    confidence_threshold: float = 0.5

    # Device
    device: Optional[str] = None

    # Debug options
    save_intermediate: bool = False
    intermediate_dir: Optional[Path] = None

    def __post_init__(self):
        """Convert string paths to Path objects if needed."""
        if isinstance(self.video_path, str):
            self.video_path = Path(self.video_path)
        if isinstance(self.output_csv, str):
            self.output_csv = Path(self.output_csv)
        if isinstance(self.model_checkpoint, str):
            self.model_checkpoint = Path(self.model_checkpoint)
        if isinstance(self.backbone_ckpt, str):
            self.backbone_ckpt = Path(self.backbone_ckpt)
        if isinstance(self.pre_xgb_model, str):
            self.pre_xgb_model = Path(self.pre_xgb_model)
        if isinstance(self.post_xgb_model, str):
            self.post_xgb_model = Path(self.post_xgb_model)
        if isinstance(self.rf_detr_weights, str):
            self.rf_detr_weights = Path(self.rf_detr_weights)
        if isinstance(self.intermediate_dir, str):
            self.intermediate_dir = Path(self.intermediate_dir)
