import logging
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


class VideoMAEBackbone(nn.Module):
    """
    Thin wrapper around a pretrained VideoMAE v2 model that exposes a feature-only forward.
    """

    def __init__(self, model: nn.Module, expected_frames: Optional[int] = None):
        super().__init__()
        self.model = model
        self.expected_frames = expected_frames
        self.hidden_dim = self._infer_hidden_dim(model)
        self._warned_frame_mismatch = False

    @staticmethod
    def _infer_hidden_dim(model: nn.Module) -> int:
        inner = getattr(model, "model", model)
        for attr in ("embed_dim", "num_features"):
            hidden = getattr(inner, attr, None)
            if hidden is not None:
                return int(hidden)
        raise ValueError("Unable to determine hidden_dim for VideoMAE backbone.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"VideoMAE backbone expects (B, C, T, H, W) input, got {x.shape}")
        if (
            self.expected_frames
            and x.shape[2] != self.expected_frames
            and not self._warned_frame_mismatch
        ):
            logging.warning(
                "Input clip length (%s) differs from pretrained expectation (%s). "
                "Ensure num_frames matches the checkpoint.",
                x.shape[2],
                self.expected_frames,
            )
            self._warned_frame_mismatch = True
        if hasattr(self.model, "extract_features"):
            feats = self.model.extract_features(x)
        else:
            feats = self.model(x)
        if isinstance(feats, tuple):
            feats = feats[0]
        return feats


def _resolve_backbone_path(backbone_ckpt: str) -> Path:
    path = Path(backbone_ckpt)
    if path.is_file():
        return path.parent
    return path


def build_vmae_backbone(config) -> VideoMAEBackbone:
    """
    Load a Kinetics-400–pretrained VideoMAE v2 backbone from the provided checkpoint path.
    """
    ckpt_path = _resolve_backbone_path(config.backbone_ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"VideoMAE checkpoint path does not exist: {ckpt_path}")

    try:
        # Force Hugging Face caches into a repo-local, writable directory before importing transformers.
        repo_cache = Path(__file__).resolve().parents[2] / ".cache" / "hf_local"
        repo_cache.mkdir(parents=True, exist_ok=True)
        modules_cache = repo_cache / "modules"
        modules_cache.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(repo_cache)
        os.environ["HF_MODULES_CACHE"] = str(modules_cache)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(repo_cache / "hub")
        os.environ["TRANSFORMERS_CACHE"] = str(repo_cache / "transformers")
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        from transformers import AutoModel

        model = AutoModel.from_pretrained(
            ckpt_path,
            trust_remote_code=True,
            local_files_only=True,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to load VideoMAE backbone from {ckpt_path}: {exc}") from exc

    expected_frames = None
    if getattr(model, "config", None) is not None:
        model_config = getattr(model.config, "model_config", None)
        if isinstance(model_config, dict):
            expected_frames = model_config.get("num_frames")

    if expected_frames and getattr(config, "num_frames", expected_frames) != expected_frames:
        raise ValueError(
            f"config.num_frames={config.num_frames} does not match backbone expectation "
            f"of {expected_frames}. Please resample clips to the pretrained length."
        )

    return VideoMAEBackbone(model, expected_frames=expected_frames)
