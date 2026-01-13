"""
Soccer header detection inference pipeline.

This module provides a modular inference pipeline for detecting headers in soccer videos.
The pipeline consists of multiple stages:
1. Ball Detection (RF-DETR + Kalman smoothing)
2. Pre-XGB Filter (optional - kinematic features)
3. CNN/VMAE Inference (main classification)
4. Post-XGB Filter (optional - temporal smoothing)

Usage:
    python -m inference.cli --video match.mp4 --checkpoint model.pt --output predictions.csv
"""

from .config import InferenceConfig
from .pipeline import HeaderDetectionPipeline, FramePrediction

__all__ = [
    "InferenceConfig",
    "HeaderDetectionPipeline",
    "FramePrediction",
]
