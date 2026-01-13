"""CNN/VMAE inference stage for header classification."""

from typing import List, Optional
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

import sys
from pathlib import Path

HEADER_NET_ROOT = Path(__file__).resolve().parents[2]
if str(HEADER_NET_ROOT) not in sys.path:
    sys.path.insert(0, str(HEADER_NET_ROOT))

from training.models.factory import build_model
from training.config import Config as TrainingConfig

from ..config import InferenceConfig
from ..preprocessing.transforms import get_inference_transforms


class CNNInference:
    """
    CNN/VMAE inference stage for header classification.

    Supports both CSN (ResNet3D-50) and VideoMAE backbones.
    Handles model loading from checkpoints and batched inference.

    Attributes:
        config: Inference configuration.
        device: Torch device for inference.
        model: Loaded model in eval mode.
        transform: Preprocessing transform pipeline.
    """

    def __init__(self, config: InferenceConfig, device: torch.device):
        """
        Initialize the CNN inference stage.

        Args:
            config: Inference configuration.
            device: Torch device for inference.
        """
        self.config = config
        self.device = device

        # Build and load model
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()

        # Get inference transforms
        self.transform = get_inference_transforms(
            input_size=config.input_size,
            backbone=config.backbone,
        )

    def _load_model(self) -> torch.nn.Module:
        """
        Load model from checkpoint.

        The checkpoint is expected to contain:
        - 'state_dict': Model weights
        - 'config': Training configuration (optional)

        Returns:
            Loaded model in eval mode.
        """
        checkpoint_path = self.config.model_checkpoint

        # Create training config for model building
        cfg = TrainingConfig()
        cfg.backbone = self.config.backbone
        cfg.num_frames = self.config.num_frames
        cfg.input_size = self.config.input_size

        if self.config.backbone == "vmae" and self.config.backbone_ckpt:
            cfg.backbone_ckpt = str(self.config.backbone_ckpt)

        # Always use full mode for inference (no freezing needed)
        cfg.finetune_mode = "full"

        # Build model architecture
        model, _ = build_model(cfg)

        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Extract state dict
        if isinstance(state, dict):
            state_dict = state.get("state_dict", state)
        else:
            state_dict = state

        # Handle DataParallel prefix (module.*)
        if list(state_dict.keys())[0].startswith("module."):
            state_dict = {
                k.replace("module.", ""): v for k, v in state_dict.items()
            }

        # Load weights
        model.load_state_dict(state_dict, strict=True)
        print(f"Model loaded successfully ({self.config.backbone} backbone)")

        return model

    def preprocess_window(self, window: np.ndarray) -> torch.Tensor:
        """
        Preprocess a single temporal window for inference.

        Args:
            window: Numpy array of shape (T, H, W, C) with uint8 RGB.

        Returns:
            Tensor of shape (1, C, T, H, W) ready for model input.
        """
        processed = []

        for i in range(window.shape[0]):
            frame = window[i]
            img = Image.fromarray(frame.astype("uint8"), "RGB")
            tensor = self.transform(img)
            processed.append(tensor)

        # Stack: (T, C, H, W) -> permute to (C, T, H, W)
        video = torch.stack(processed, dim=0)  # (T, C, H, W)
        video = video.permute(1, 0, 2, 3)  # (C, T, H, W)
        video = video.unsqueeze(0)  # (1, C, T, H, W)

        return video

    def predict_single(self, window: np.ndarray) -> float:
        """
        Run inference on a single temporal window.

        Args:
            window: Numpy array of shape (T, H, W, C).

        Returns:
            Header probability [0, 1].
        """
        tensor = self.preprocess_window(window).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)
            header_prob = probs[0, 1].item()

        return header_prob

    def predict_batch(
        self,
        batch_windows: List[Optional[np.ndarray]],
    ) -> List[float]:
        """
        Run inference on a batch of temporal windows.

        Args:
            batch_windows: List of numpy arrays (T, H, W, C) or None.
                          None entries will return 0.0 probability.

        Returns:
            List of header probabilities [0, 1] for each window.
        """
        # Identify valid windows
        valid_indices = [i for i, w in enumerate(batch_windows) if w is not None]
        valid_windows = [batch_windows[i] for i in valid_indices]

        if not valid_windows:
            return [0.0] * len(batch_windows)

        # Preprocess all valid windows
        tensors = []
        for window in valid_windows:
            processed = []
            for i in range(window.shape[0]):
                frame = window[i]
                img = Image.fromarray(frame.astype("uint8"), "RGB")
                tensor = self.transform(img)
                processed.append(tensor)

            video = torch.stack(processed, dim=0).permute(1, 0, 2, 3)
            tensors.append(video)

        # Stack into batch
        batch = torch.stack(tensors, dim=0).to(self.device)

        # Run inference
        with torch.no_grad():
            logits = self.model(batch)
            probs = F.softmax(logits, dim=1)[:, 1]  # Header probabilities

        # Map back to original indices
        results = [0.0] * len(batch_windows)
        for i, prob in zip(valid_indices, probs.cpu().numpy()):
            results[i] = float(prob)

        return results

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        return {
            "backbone": self.config.backbone,
            "num_frames": self.config.num_frames,
            "input_size": self.config.input_size,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "checkpoint": str(self.config.model_checkpoint),
        }
