"""Utilities for working with detection dictionaries."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np

BallDetections = Dict[str, Dict[int, Dict[int, Dict[str, Any]]]]


def make_video_key(match_name: str, half: int) -> str:
    return f"{match_name}_half{int(half)}"


def normalise_id(value: str) -> str:
    return ''.join(ch for ch in value.lower() if ch.isalnum())


def video_ids_match(det_id: str, label_id: str) -> bool:
    det_norm = normalise_id(det_id)
    label_norm = normalise_id(label_id)
    return det_norm == label_norm or det_norm in label_norm or label_norm in det_norm


def load_ball_det_dict(path: Path) -> BallDetections:
    if not path.exists():
        raise FileNotFoundError(f"Ball detection dictionary not found: {path}")

    data = np.load(path, allow_pickle=True).item()
    result: BallDetections = {}
    for video_id, frame_dict in data.items():
        frame_entries: Dict[int, Dict[int, Dict[str, float]]] = {}
        for frame_key, detections in frame_dict.items():
            try:
                frame_id = int(frame_key)
            except (ValueError, TypeError):
                continue
            if isinstance(detections, dict):
                det_map = detections
            else:
                det_map = {idx: det for idx, det in enumerate(detections)}
            frame_entries[frame_id] = det_map
        result[str(video_id)] = frame_entries
    return result
