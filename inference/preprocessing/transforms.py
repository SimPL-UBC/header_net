"""Inference transforms for CNN/VMAE models."""

from typing import Literal
import torch
from torchvision import transforms
from PIL import Image
import numpy as np


# Normalization constants for different backbones
VMAE_MEAN = [0.5, 0.5, 0.5]
VMAE_STD = [0.5, 0.5, 0.5]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_inference_transforms(
    input_size: int = 224,
    backbone: Literal["vmae", "csn"] = "vmae",
) -> transforms.Compose:
    """
    Get inference transforms for the specified backbone.

    Args:
        input_size: Target size for resizing (pixels).
        backbone: Model backbone type ("vmae" or "csn").

    Returns:
        Composed transform pipeline.

    Note:
        - VideoMAE uses [0.5, 0.5, 0.5] normalization
        - CSN uses ImageNet normalization [0.485, 0.456, 0.406]
    """
    if backbone == "vmae":
        mean, std = VMAE_MEAN, VMAE_STD
    else:  # csn
        mean, std = IMAGENET_MEAN, IMAGENET_STD

    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def preprocess_frames(
    frames: np.ndarray,
    transform: transforms.Compose,
) -> torch.Tensor:
    """
    Preprocess a batch of frames for model inference.

    Args:
        frames: Numpy array of shape (T, H, W, C) with uint8 RGB values.
        transform: Transform pipeline to apply.

    Returns:
        Tensor of shape (C, T, H, W) ready for model input.
    """
    processed = []

    for i in range(frames.shape[0]):
        frame = frames[i]
        # Convert to PIL Image
        img = Image.fromarray(frame.astype("uint8"), "RGB")
        # Apply transforms
        tensor = transform(img)
        processed.append(tensor)

    # Stack: (T, C, H, W) -> permute to (C, T, H, W)
    video = torch.stack(processed, dim=0)  # (T, C, H, W)
    video = video.permute(1, 0, 2, 3)  # (C, T, H, W)

    return video


def preprocess_batch(
    batch_frames: list,
    transform: transforms.Compose,
) -> torch.Tensor:
    """
    Preprocess a batch of temporal windows for model inference.

    Args:
        batch_frames: List of numpy arrays, each (T, H, W, C).
        transform: Transform pipeline to apply.

    Returns:
        Tensor of shape (B, C, T, H, W) ready for model input.
    """
    batch = []

    for frames in batch_frames:
        if frames is not None:
            video = preprocess_frames(frames, transform)
            batch.append(video)
        else:
            # Create placeholder (will be masked in inference)
            batch.append(None)

    # Stack valid tensors
    valid_tensors = [t for t in batch if t is not None]
    if not valid_tensors:
        return None

    return torch.stack(valid_tensors, dim=0)
