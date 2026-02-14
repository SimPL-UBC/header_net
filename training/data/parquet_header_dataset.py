import random
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler

from ..config import Config
from inference.preprocessing.frame_cropper import FrameCropper


VMAE_MEAN = [0.5, 0.5, 0.5]
VMAE_STD = [0.5, 0.5, 0.5]


REQUIRED_PARQUET_COLUMNS = [
    "video_id",
    "half",
    "frame",
    "label",
    "video_path",
    "ball_x",
    "ball_y",
    "ball_w",
    "ball_h",
]


def _parse_neg_pos_ratio(value: Union[str, int]) -> Optional[int]:
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("neg_pos_ratio must be > 0 or 'all'")
        return value

    text = str(value).strip().lower()
    if text == "all":
        return None

    ratio = int(text)
    if ratio <= 0:
        raise ValueError("neg_pos_ratio must be > 0 or 'all'")
    return ratio


def _build_window_offsets(num_frames: int) -> np.ndarray:
    if num_frames <= 0:
        raise ValueError("num_frames must be > 0")
    half = num_frames // 2
    if num_frames % 2 == 0:
        return np.arange(-half, half, dtype=np.int32)
    return np.arange(-half, half + 1, dtype=np.int32)


def _seed_worker(worker_id: int) -> None:
    # Keep dataloader workers deterministic across runs.
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class _VideoReaderPool:
    def __init__(self, max_open_videos: int = 8, frame_cache_size: int = 128):
        if max_open_videos <= 0:
            raise ValueError("max_open_videos must be > 0")
        self.max_open_videos = max_open_videos
        self.frame_cache_size = frame_cache_size
        self._readers: "OrderedDict[str, cv2.VideoCapture]" = OrderedDict()
        self._meta: Dict[str, Tuple[int, int, int]] = {}

    def _open(self, video_path: str) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._meta[video_path] = (frame_count, width, height)
        return cap

    def get(self, video_path: str) -> cv2.VideoCapture:
        reader = self._readers.get(video_path)
        if reader is not None:
            self._readers.move_to_end(video_path)
            return reader

        while len(self._readers) >= self.max_open_videos:
            old_path, old_reader = self._readers.popitem(last=False)
            old_reader.release()
            self._meta.pop(old_path, None)

        reader = self._open(video_path)
        self._readers[video_path] = reader
        return reader

    def get_meta(self, video_path: str) -> Tuple[int, int, int]:
        if video_path not in self._meta:
            self.get(video_path)
        return self._meta[video_path]

    def read_frame(self, video_path: str, frame_idx: int) -> Optional[np.ndarray]:
        cap = self.get(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        for reader in self._readers.values():
            reader.release()
        self._readers.clear()
        self._meta.clear()

    def __del__(self) -> None:
        self.close()


class DeterministicRatioSampler(Sampler[int]):
    def __init__(
        self,
        positive_indices: np.ndarray,
        negative_indices: np.ndarray,
        neg_pos_ratio: Union[str, int],
        seed: int = 42,
        shuffle: bool = True,
    ):
        self.positive_indices = np.asarray(positive_indices, dtype=np.int64)
        self.negative_indices = np.asarray(negative_indices, dtype=np.int64)
        self.neg_pos_ratio = _parse_neg_pos_ratio(neg_pos_ratio)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.current_epoch = 0
        self._indices = np.array([], dtype=np.int64)
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)
        rng = np.random.default_rng(self.seed + self.current_epoch)

        if self.neg_pos_ratio is None:
            neg = self.negative_indices
        else:
            target_neg = min(
                len(self.negative_indices),
                len(self.positive_indices) * self.neg_pos_ratio,
            )
            if target_neg > 0:
                neg = rng.choice(self.negative_indices, size=target_neg, replace=False)
            else:
                neg = np.array([], dtype=np.int64)

        indices = np.concatenate([self.positive_indices, neg])
        if self.shuffle and len(indices) > 0:
            rng.shuffle(indices)
        self._indices = indices.astype(np.int64, copy=False)

    def __iter__(self):
        return iter(self._indices.tolist())

    def __len__(self) -> int:
        return len(self._indices)

    def get_indices(self) -> np.ndarray:
        return self._indices.copy()

    def get_counts(self) -> Dict[str, int]:
        pos_count = int(len(self.positive_indices))
        neg_count = int(len(self._indices) - pos_count)
        return {
            "samples": int(len(self._indices)),
            "positives": pos_count,
            "negatives": neg_count,
        }


class ParquetHeaderDataset(Dataset):
    def __init__(
        self,
        parquet_path: Union[str, Path],
        num_frames: int = 16,
        input_size: int = 224,
        transform=None,
        strict_paths: bool = True,
        dataset_root: Optional[Union[str, Path]] = None,
        max_open_videos: int = 8,
        crop_scale_factor: float = 4.5,
        default_radius: int = 100,
    ):
        self.parquet_path = Path(parquet_path)
        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Parquet not found: {self.parquet_path}")

        self.num_frames = int(num_frames)
        self.input_size = int(input_size)
        self.transform = transform
        self.strict_paths = strict_paths
        self.dataset_root = Path(dataset_root) if dataset_root else None
        self.window_offsets = _build_window_offsets(self.num_frames)
        self.cropper = FrameCropper(
            crop_scale_factor=crop_scale_factor,
            output_size=self.input_size,
            default_radius=default_radius,
        )
        self.reader_pool = _VideoReaderPool(max_open_videos=max_open_videos)

        self.df = pd.read_parquet(self.parquet_path, columns=REQUIRED_PARQUET_COLUMNS)
        self._validate_schema()
        self._prepare_arrays()
        self._validate_paths()

        labels = self.labels
        self.positive_indices = np.where(labels == 1)[0].astype(np.int64)
        self.negative_indices = np.where(labels == 0)[0].astype(np.int64)

    def _validate_schema(self) -> None:
        missing = [col for col in REQUIRED_PARQUET_COLUMNS if col not in self.df.columns]
        if missing:
            raise ValueError(
                f"Parquet {self.parquet_path} missing required columns: {missing}"
            )

    def _prepare_arrays(self) -> None:
        self.video_ids = self.df["video_id"].astype(str).to_numpy()
        self.halves = self.df["half"].astype(np.int16).to_numpy()
        self.frames = self.df["frame"].astype(np.int64).to_numpy()
        self.labels = self.df["label"].astype(np.int8).to_numpy()
        self.video_paths = self.df["video_path"].astype(str).to_numpy()
        self.ball_x = pd.to_numeric(self.df["ball_x"], errors="coerce").to_numpy(np.float32)
        self.ball_y = pd.to_numeric(self.df["ball_y"], errors="coerce").to_numpy(np.float32)
        self.ball_w = pd.to_numeric(self.df["ball_w"], errors="coerce").to_numpy(np.float32)
        self.ball_h = pd.to_numeric(self.df["ball_h"], errors="coerce").to_numpy(np.float32)

        unique_labels = set(np.unique(self.labels).tolist())
        if not unique_labels.issubset({0, 1}):
            raise ValueError(f"Labels must be binary 0/1, got: {sorted(unique_labels)}")

    def _validate_paths(self) -> None:
        if self.dataset_root is not None and not self.dataset_root.exists():
            raise FileNotFoundError(f"dataset_root does not exist: {self.dataset_root}")

        if not self.strict_paths:
            return

        unique_paths = pd.unique(self.video_paths)
        missing = [path for path in unique_paths if not Path(path).exists()]
        if missing:
            preview = ", ".join(missing[:5])
            raise FileNotFoundError(
                f"{len(missing)} parquet video_path entries are missing. "
                f"Examples: {preview}"
            )

    def __len__(self) -> int:
        return len(self.labels)

    def _build_center_detection(self, row_idx: int) -> Optional[Dict]:
        x = float(self.ball_x[row_idx])
        y = float(self.ball_y[row_idx])
        w = float(self.ball_w[row_idx])
        h = float(self.ball_h[row_idx])

        if np.isnan([x, y, w, h]).any():
            return None
        if w <= 0.0 or h <= 0.0:
            return None

        return {"box": [x, y, w, h]}

    def _load_window_frames(self, video_path: str, center_frame: int) -> list:
        frame_count, width, height = self.reader_pool.get_meta(video_path)
        if frame_count <= 0:
            raise RuntimeError(f"Video has no frames: {video_path}")

        requested = center_frame + self.window_offsets
        clamped = np.clip(requested, 0, frame_count - 1)

        frames = []
        for idx in clamped.tolist():
            frame = self.reader_pool.read_frame(video_path, int(idx))
            if frame is None:
                frame = np.zeros((height, width, 3), dtype=np.uint8)
            frames.append(frame)
        return frames

    def _apply_spatial_policy(self, frames: list, center_det: Optional[Dict]) -> list:
        # If the center frame has no ball, use full-frame resize via transform path.
        if center_det is None:
            return frames

        center_idx = len(frames) // 2
        radius = self.cropper.compute_radius(
            center_det, frames[center_idx].shape[:2]
        )
        return [self.cropper.crop(frame, center_det, radius=radius) for frame in frames]

    def __getitem__(self, idx: int):
        row_idx = int(idx)
        video_path = self.video_paths[row_idx]
        center_frame = int(self.frames[row_idx])
        label = int(self.labels[row_idx])

        frames = self._load_window_frames(video_path, center_frame)
        center_det = self._build_center_detection(row_idx)
        frames = self._apply_spatial_policy(frames, center_det)

        processed = []
        for frame in frames:
            img = Image.fromarray(frame.astype("uint8"), "RGB")
            if self.transform is not None:
                tensor = self.transform(img)
            else:
                tensor = T.ToTensor()(img)
            processed.append(tensor)

        video = torch.stack(processed, dim=0).permute(1, 0, 2, 3)

        meta = {
            "video_id": str(self.video_ids[row_idx]),
            "half": str(self.halves[row_idx]),
            "frame": int(center_frame),
            "path": str(video_path),
            "row_idx": int(row_idx),
        }
        return video, label, meta

    def class_counts(self) -> Dict[str, int]:
        return {
            "samples": int(len(self.labels)),
            "positives": int(len(self.positive_indices)),
            "negatives": int(len(self.negative_indices)),
        }


def get_transforms(input_size: int = 224, is_training: bool = True):
    normalize = T.Normalize(mean=VMAE_MEAN, std=VMAE_STD)

    if is_training:
        return T.Compose(
            [
                T.Resize((input_size, input_size)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0.1,
                    hue=0.1,
                ),
                T.ToTensor(),
                normalize,
            ]
        )

    return T.Compose(
        [
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            normalize,
        ]
    )


def build_parquet_dataloaders(config: Config):
    train_transform = get_transforms(config.input_size, is_training=True)
    val_transform = get_transforms(config.input_size, is_training=False)

    train_dataset = ParquetHeaderDataset(
        parquet_path=config.train_parquet,
        num_frames=config.num_frames,
        input_size=config.input_size,
        transform=train_transform,
        strict_paths=True,
        dataset_root=config.dataset_root,
    )

    val_dataset = ParquetHeaderDataset(
        parquet_path=config.val_parquet,
        num_frames=config.num_frames,
        input_size=config.input_size,
        transform=val_transform,
        strict_paths=True,
        dataset_root=config.dataset_root,
    )

    train_sampler = DeterministicRatioSampler(
        positive_indices=train_dataset.positive_indices,
        negative_indices=train_dataset.negative_indices,
        neg_pos_ratio=config.neg_pos_ratio,
        seed=config.seed,
        shuffle=True,
    )
    if len(train_dataset.positive_indices) == 0:
        raise ValueError("Training parquet has no positive samples (label=1).")

    generator = torch.Generator()
    generator.manual_seed(config.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        worker_init_fn=_seed_worker,
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        worker_init_fn=_seed_worker,
        generator=generator,
    )

    return train_loader, val_loader, train_dataset, val_dataset, train_sampler
