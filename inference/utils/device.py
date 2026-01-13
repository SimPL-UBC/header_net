"""Device detection and management utilities."""

from typing import Optional
import torch


def get_device(requested_device: Optional[str] = None) -> torch.device:
    """
    Get the appropriate torch device for inference.

    Args:
        requested_device: Explicit device string (e.g., "cuda", "cuda:0", "cpu").
                         If None, auto-detects CUDA availability.

    Returns:
        torch.device: The resolved device.

    Examples:
        >>> get_device()  # Auto-detect
        device(type='cuda')
        >>> get_device("cpu")  # Explicit CPU
        device(type='cpu')
        >>> get_device("cuda:1")  # Specific GPU
        device(type='cuda', index=1)
    """
    if requested_device is not None:
        return torch.device(requested_device)

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def get_device_info(device: torch.device) -> str:
    """
    Get human-readable device information.

    Args:
        device: The torch device.

    Returns:
        String describing the device.
    """
    if device.type == "cuda":
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(device.index or 0)
            gpu_memory = torch.cuda.get_device_properties(device.index or 0).total_memory
            gpu_memory_gb = gpu_memory / (1024 ** 3)
            return f"{gpu_name} ({gpu_memory_gb:.1f} GB)"
        return "CUDA (unavailable)"

    return device.type.upper()
