import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def get_header_transforms(input_size: int = 224, is_training: bool = True):
    """Image transforms matching the legacy CSN header setup."""
    if is_training:
        return transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class HeaderCacheDataset(Dataset):
    """
    Cache-driven dataset for header detection.
    Expects CSV columns: path,label,video_id,half,frame,(metadata optional).
    """

    def __init__(
        self,
        csv_path: str,
        num_frames: int = 11,
        input_size: int = 224,
        is_training: bool = True,
        frame_sampling: str = "center",
    ):
        self.csv_path = Path(csv_path)
        self.num_frames = num_frames
        self.input_size = input_size
        self.is_training = is_training
        self.frame_sampling = frame_sampling

        if self.frame_sampling != "center":
            raise ValueError(f"Unsupported frame_sampling={frame_sampling} for Phase 1.")

        self.records = self._load_records(self.csv_path)
        self.transform = get_header_transforms(input_size=input_size, is_training=is_training)

    @staticmethod
    def _load_records(csv_path: Path) -> List[Dict]:
        required_cols = {"path", "label", "video_id", "half", "frame"}
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            logging.error("Failed to read CSV %s: %s", csv_path, exc)
            raise

        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns in {csv_path}: {missing}")

        records = df.to_dict(orient="records")
        cleaned = []
        for row in records:
            try:
                row["label"] = int(row["label"])
            except Exception:
                logging.warning("Invalid label in row %s; skipping", row)
                continue
            cleaned.append(row)

        if not cleaned:
            raise ValueError(f"No valid rows parsed from {csv_path}")
        return cleaned

    def __len__(self) -> int:
        return len(self.records)

    def _load_cache(self, base_path: str) -> np.ndarray:
        cache_file = f"{base_path}_s.npy"
        try:
            return np.load(cache_file)
        except FileNotFoundError:
            logging.error("Cache file not found: %s", cache_file)
        except Exception as exc:
            logging.error("Error loading cache %s: %s", cache_file, exc)
        # Fallback zeros to keep pipeline running
        return np.zeros((self.num_frames, self.input_size, self.input_size, 3), dtype=np.uint8)

    def _sample_indices(self, total_frames: int) -> List[int]:
        if total_frames <= 0:
            return [0 for _ in range(self.num_frames)]
        if total_frames <= self.num_frames:
            indices = list(range(total_frames))
            indices += [indices[-1]] * (self.num_frames - len(indices))
            return indices

        if self.is_training:
            segment = total_frames // self.num_frames
            sampled = []
            for i in range(self.num_frames):
                start = i * segment
                end = total_frames if i == self.num_frames - 1 else (i + 1) * segment
                end = max(end, start + 1)
                sampled.append(random.randint(start, end - 1))
            return sampled
        return np.linspace(0, total_frames - 1, self.num_frames, dtype=int).tolist()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        record = self.records[idx]
        cache = self._load_cache(str(record["path"]))
        total_frames = cache.shape[0] if cache.ndim >= 1 else 0
        indices = self._sample_indices(total_frames)

        frames = []
        for ind in indices:
            if total_frames == 0:
                frame = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
            else:
                safe_idx = min(ind, total_frames - 1)
                frame = cache[safe_idx]
            if frame.ndim == 3 and frame.shape[-1] in (1, 3):
                pass
            elif frame.ndim == 3 and frame.shape[0] in (1, 3):
                frame = np.transpose(frame, (1, 2, 0))
            else:
                frame = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)

            frame_uint8 = frame.astype(np.uint8)
            pil_frame = Image.fromarray(frame_uint8)
            frames.append(self.transform(pil_frame))

        video = torch.stack(frames, dim=0).permute(1, 0, 2, 3)  # (C, T, H, W)
        label = int(record["label"])
        meta = {
            "video_id": record.get("video_id", ""),
            "half": record.get("half", ""),
            "frame": int(record.get("frame", -1)),
            "path": record.get("path", ""),
        }
        if "metadata" in record:
            meta["metadata"] = record.get("metadata", "")
        return video, label, meta


def build_dataloaders(config) -> Tuple[DataLoader, DataLoader]:
    """Construct train/val dataloaders from the provided config."""
    train_dataset = HeaderCacheDataset(
        config.train_csv,
        num_frames=config.num_frames,
        input_size=config.input_size,
        is_training=True,
        frame_sampling=config.frame_sampling,
    )
    val_dataset = HeaderCacheDataset(
        config.val_csv,
        num_frames=config.num_frames,
        input_size=config.input_size,
        is_training=False,
        frame_sampling=config.frame_sampling,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
