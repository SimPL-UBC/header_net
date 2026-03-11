import random
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import decord
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler

# decord returns RGB by default when using the default bridge.
decord.bridge.set_bridge("native")

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

# Optional columns consumed by _prepare_arrays(); loaded when present in the file.
_OPTIONAL_PARQUET_COLUMNS = [
    "fps",
    "ball_confidence",
]


def _factorize_strings(values: pd.Series) -> tuple[np.ndarray, tuple[str, ...]]:
    codes, uniques = pd.factorize(values.astype(str), sort=False)
    codes = codes.astype(np.int32, copy=False)
    return codes, tuple(str(value) for value in uniques.tolist())


def _columns_to_load(available_columns: list[str]) -> list[str]:
    """Return the subset of *available_columns* that the dataset actually uses.

    This avoids loading heavyweight columns such as ``other_detections`` that
    are never accessed by ``__getitem__``, saving gigabytes of RAM on large
    parquet files.
    """
    wanted = set(REQUIRED_PARQUET_COLUMNS) | set(_OPTIONAL_PARQUET_COLUMNS)
    return [col for col in available_columns if col in wanted]


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
    # Re-initialise decord bridge in each forked worker — the module-level
    # set_bridge() call only runs in the main process and the internal state
    # may not survive fork.
    decord.bridge.set_bridge("native")


class _VideoReaderPool:
    def __init__(self, max_open_videos: int = 8, frame_cache_size: int = 128):
        if max_open_videos <= 0:
            raise ValueError("max_open_videos must be > 0")
        self.max_open_videos = max_open_videos
        self.frame_cache_size = frame_cache_size
        self._readers: "OrderedDict[str, decord.VideoReader]" = OrderedDict()
        self._meta: Dict[str, Tuple[int, int, int]] = {}

    def _open(self, video_path: str) -> decord.VideoReader:
        try:
            reader = decord.VideoReader(video_path, ctx=decord.cpu())
        except Exception as exc:
            raise RuntimeError(f"Unable to open video: {video_path}") from exc
        frame_count = len(reader)
        if frame_count == 0:
            raise RuntimeError(f"Video has no frames: {video_path}")
        h, w, _ = reader[0].shape
        self._meta[video_path] = (frame_count, int(w), int(h))
        return reader

    def get(self, video_path: str) -> decord.VideoReader:
        reader = self._readers.get(video_path)
        if reader is not None:
            self._readers.move_to_end(video_path)
            return reader

        while len(self._readers) >= self.max_open_videos:
            old_path, _ = self._readers.popitem(last=False)
            self._meta.pop(old_path, None)

        reader = self._open(video_path)
        self._readers[video_path] = reader
        return reader

    def get_meta(self, video_path: str) -> Tuple[int, int, int]:
        if video_path not in self._meta:
            self.get(video_path)
        return self._meta[video_path]

    def read_frame(self, video_path: str, frame_idx: int) -> Optional[np.ndarray]:
        reader = self.get(video_path)
        try:
            return reader[frame_idx].asnumpy()
        except Exception:
            return None

    def read_frames(self, video_path: str, frame_indices: list[int]) -> list[np.ndarray]:
        """Read multiple frames in one optimized batch call.

        Duplicate indices (common at video boundaries due to clamping) are
        deduplicated before the batch call because some decord versions
        mishandle repeated indices.
        """
        if len(frame_indices) == 0:
            return []

        reader = self.get(video_path)
        unique_indices, inverse = np.unique(
            np.asarray(frame_indices, dtype=np.int64), return_inverse=True
        )
        # decord 0.6.0 returns an NDArray that supports asnumpy() but not direct
        # Python indexing, so convert once before reconstructing duplicate indices.
        batch = reader.get_batch(unique_indices.tolist()).asnumpy()
        return [frame for frame in batch[inverse]]

    def close(self) -> None:
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
            target_neg = len(self.positive_indices) * self.neg_pos_ratio
            if target_neg > 0 and len(self.negative_indices) > 0:
                replace = len(self.negative_indices) < target_neg
                neg = rng.choice(self.negative_indices, size=target_neg, replace=replace)
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
        max_resample_attempts: int = 20,
        resample_on_decode_failure: bool = True,
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
        self.max_resample_attempts = max(1, int(max_resample_attempts))
        self.resample_on_decode_failure = bool(resample_on_decode_failure)
        self.cropper = FrameCropper(
            crop_scale_factor=crop_scale_factor,
            output_size=self.input_size,
            default_radius=default_radius,
        )
        self.reader_pool = _VideoReaderPool(max_open_videos=max_open_videos)

        # Only load the columns the dataset actually uses.  This skips
        # heavyweight columns like ``other_detections`` (JSON blobs with
        # per-frame bounding boxes) which can consume several GB of RAM on
        # large parquet files but are never accessed by __getitem__.
        import pyarrow.parquet as pq

        all_columns = pq.read_schema(self.parquet_path).names
        self.df = pd.read_parquet(
            self.parquet_path, columns=_columns_to_load(all_columns)
        )
        self._validate_schema()
        self._prepare_arrays()
        self._validate_paths()

        # Free the DataFrame now that all needed data has been extracted into
        # numpy arrays.  With forked DataLoader workers this prevents each
        # worker from holding a (copy-on-write-degraded) copy of the full
        # DataFrame, which can waste tens of GB on large parquet files.
        del self.df

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
        # Factorize string columns so forked DataLoader workers do not gradually
        # copy hundreds of thousands of Python string objects into private RSS.
        self.video_id_codes, self.video_id_table = _factorize_strings(self.df["video_id"])
        self.halves = self.df["half"].astype(np.int16).to_numpy()
        self.frames = self.df["frame"].astype(np.int64).to_numpy()
        self.labels = self.df["label"].astype(np.int8).to_numpy()
        self.video_path_codes, self.video_path_table = _factorize_strings(self.df["video_path"])
        self.ball_x = pd.to_numeric(self.df["ball_x"], errors="coerce").to_numpy(np.float32)
        self.ball_y = pd.to_numeric(self.df["ball_y"], errors="coerce").to_numpy(np.float32)
        self.ball_w = pd.to_numeric(self.df["ball_w"], errors="coerce").to_numpy(np.float32)
        self.ball_h = pd.to_numeric(self.df["ball_h"], errors="coerce").to_numpy(np.float32)
        self.fps = self._numeric_column("fps")
        self.ball_confidence = self._numeric_column("ball_confidence")

        unique_labels = set(np.unique(self.labels).tolist())
        if not unique_labels.issubset({0, 1}):
            raise ValueError(f"Labels must be binary 0/1, got: {sorted(unique_labels)}")

    def _numeric_column(self, column_name: str, default: float = np.nan) -> np.ndarray:
        if column_name not in self.df.columns:
            return np.full(len(self.df), default, dtype=np.float32)
        return pd.to_numeric(self.df[column_name], errors="coerce").to_numpy(np.float32)

    def _validate_paths(self) -> None:
        if self.dataset_root is not None and not self.dataset_root.exists():
            raise FileNotFoundError(f"dataset_root does not exist: {self.dataset_root}")

        if not self.strict_paths:
            return

        unique_paths = self.video_path_table
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

    def _video_id_at(self, row_idx: int) -> str:
        if hasattr(self, "video_id_codes") and hasattr(self, "video_id_table"):
            return self.video_id_table[int(self.video_id_codes[row_idx])]
        return str(self.video_ids[row_idx])

    def _video_path_at(self, row_idx: int) -> str:
        if hasattr(self, "video_path_codes") and hasattr(self, "video_path_table"):
            return self.video_path_table[int(self.video_path_codes[row_idx])]
        return str(self.video_paths[row_idx])

    def _load_window_frames(self, video_path: str, center_frame: int) -> list[np.ndarray]:
        frame_count, _, _ = self.reader_pool.get_meta(video_path)
        if frame_count <= 0:
            raise RuntimeError(f"Video has no frames: {video_path}")

        requested = center_frame + self.window_offsets
        clamped = np.clip(requested, 0, frame_count - 1)

        return self.reader_pool.read_frames(video_path, clamped.tolist())

    def _sample_replacement_row(self, label: int) -> Optional[int]:
        candidates = self.positive_indices if label == 1 else self.negative_indices
        if len(candidates) == 0:
            return None
        return int(candidates[np.random.randint(0, len(candidates))])

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
        target_label = int(self.labels[row_idx])

        for _ in range(self.max_resample_attempts):
            video_path = self._video_path_at(row_idx)
            center_frame = int(self.frames[row_idx])

            try:
                frames = self._load_window_frames(video_path, center_frame)
            except Exception as exc:
                if not self.resample_on_decode_failure:
                    raise RuntimeError(
                        "Unable to decode temporal window for "
                        f"row_idx={row_idx} video={video_path} frame={center_frame}"
                    ) from exc
                replacement = self._sample_replacement_row(target_label)
                if replacement is None:
                    break
                row_idx = replacement
                continue

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
                "video_id": self._video_id_at(row_idx),
                "half": str(self.halves[row_idx]),
                "frame": int(center_frame),
                "path": str(video_path),
                "video_path": str(video_path),
                "row_idx": int(row_idx),
                "fps": float(self.fps[row_idx]),
                "ball_x": float(self.ball_x[row_idx]),
                "ball_y": float(self.ball_y[row_idx]),
                "ball_w": float(self.ball_w[row_idx]),
                "ball_h": float(self.ball_h[row_idx]),
                "ball_confidence": float(self.ball_confidence[row_idx]),
            }
            return video, target_label, meta

        raise RuntimeError(
            f"Unable to decode a valid sample after {self.max_resample_attempts} attempts "
            f"for target label {target_label}"
        )

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


def _build_torch_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def build_parquet_train_dataloader(config: Config):
    train_transform = get_transforms(config.input_size, is_training=True)
    train_dataset = ParquetHeaderDataset(
        parquet_path=config.train_parquet,
        num_frames=config.num_frames,
        input_size=config.input_size,
        transform=train_transform,
        strict_paths=True,
        dataset_root=config.dataset_root,
    )

    if len(train_dataset.positive_indices) == 0:
        raise ValueError("Training parquet has no positive samples (label=1).")

    train_sampler = DeterministicRatioSampler(
        positive_indices=train_dataset.positive_indices,
        negative_indices=train_dataset.negative_indices,
        neg_pos_ratio=config.neg_pos_ratio,
        seed=config.seed,
        shuffle=True,
    )

    generator = _build_torch_generator(config.seed)
    num_workers = int(getattr(config, "num_workers", 0))
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_seed_worker,
        generator=generator,
        persistent_workers=bool(num_workers > 0),
    )
    return train_loader, train_dataset, train_sampler


def build_parquet_val_dataloader(
    config: Config,
    neg_pos_ratio: Union[str, int] = "all",
    seed_offset: int = 1000,
    shuffle: bool = True,
    pin_memory: Optional[bool] = None,
):
    val_transform = get_transforms(config.input_size, is_training=False)
    val_dataset = ParquetHeaderDataset(
        parquet_path=config.val_parquet,
        num_frames=config.num_frames,
        input_size=config.input_size,
        transform=val_transform,
        strict_paths=True,
        dataset_root=config.dataset_root,
    )

    val_sampler = None
    ratio = _parse_neg_pos_ratio(neg_pos_ratio)
    if ratio is not None:
        if len(val_dataset.positive_indices) == 0:
            raise ValueError(
                "Validation parquet has no positive samples but val_neg_pos_ratio was set."
            )
        val_sampler = DeterministicRatioSampler(
            positive_indices=val_dataset.positive_indices,
            negative_indices=val_dataset.negative_indices,
            neg_pos_ratio=ratio,
            seed=int(config.seed) + int(seed_offset),
            shuffle=shuffle,
        )

    generator = _build_torch_generator(config.seed)
    val_num_workers = int(getattr(config, "val_num_workers", 0))
    resolved_pin_memory = (
        bool(getattr(config, "val_pin_memory", False))
        if pin_memory is None
        else bool(pin_memory)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=val_num_workers,
        pin_memory=resolved_pin_memory,
        worker_init_fn=_seed_worker,
        generator=generator,
        persistent_workers=bool(val_num_workers > 0),
    )
    return val_loader, val_dataset, val_sampler


def build_parquet_dataloaders(config: Config):
    train_loader, train_dataset, train_sampler = build_parquet_train_dataloader(config)
    val_loader, val_dataset, _ = build_parquet_val_dataloader(
        config,
        neg_pos_ratio="all",
        seed_offset=1000,
        shuffle=True,
        pin_memory=False,
    )
    return train_loader, val_loader, train_dataset, val_dataset, train_sampler
