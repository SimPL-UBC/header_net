import random
from collections import OrderedDict
import hashlib
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

import cv2
import decord
import numpy as np
import pandas as pd
import pyarrow.dataset as pads
import torch
import torchvision.transforms as T
from torchvision.transforms import functional as TF
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

PREPROCESS_MODES = {"torchvision", "low_memory_eval"}
SPATIAL_MODES = {"ball_crop", "full_frame"}
TRAIN_AUGMENTATION_MODES = {"clip_consistent", "legacy_frame_random", "none"}


def _normalize_optional_strings(values: Optional[Iterable[Union[str, int]]]) -> tuple[str, ...]:
    if values is None:
        return ()
    normalized = []
    for value in values:
        text = str(value).strip()
        if text:
            normalized.append(text)
    return tuple(normalized)


def _normalize_optional_ints(values: Optional[Iterable[Union[str, int]]]) -> tuple[int, ...]:
    if values is None:
        return ()
    normalized = []
    for value in values:
        normalized.append(int(value))
    return tuple(normalized)


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


def _dataset_filter_expression(
    video_ids: tuple[str, ...],
    halves: tuple[int, ...],
):
    expression = None
    if video_ids:
        expression = pads.field("video_id").isin(list(video_ids))
    if halves:
        half_expression = pads.field("half").isin([int(value) for value in halves])
        expression = half_expression if expression is None else (expression & half_expression)
    return expression


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


def _stable_seed(*values: object) -> int:
    payload = "|".join(str(value) for value in values).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


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
        if frame_cache_size < 0:
            raise ValueError("frame_cache_size must be >= 0")
        self.max_open_videos = max_open_videos
        self.frame_cache_size = frame_cache_size
        self._readers: "OrderedDict[str, decord.VideoReader]" = OrderedDict()
        self._meta: Dict[str, Tuple[int, int, int]] = {}
        self._frame_cache: "OrderedDict[tuple[str, int], np.ndarray]" = OrderedDict()

    def _open(self, video_path: str) -> decord.VideoReader:
        try:
            reader = decord.VideoReader(video_path, ctx=decord.cpu())
        except Exception as exc:
            raise RuntimeError(f"Unable to open video: {video_path}") from exc
        frame_count = len(reader)
        if frame_count == 0:
            raise RuntimeError(f"Video has no frames: {video_path}")
        try:
            first_frame = reader.get_batch([0]).asnumpy()[0]
        except Exception as exc:
            raise RuntimeError(
                f"Unable to decode the first frame while opening video: {video_path}"
            ) from exc
        h, w, _ = first_frame.shape
        self._meta[video_path] = (frame_count, int(w), int(h))
        self._cache_put(video_path, 0, np.ascontiguousarray(first_frame))
        return reader

    def _cache_get(self, video_path: str, frame_idx: int) -> Optional[np.ndarray]:
        key = (video_path, int(frame_idx))
        frame = self._frame_cache.get(key)
        if frame is None:
            return None
        self._frame_cache.move_to_end(key)
        return frame

    def _cache_put(self, video_path: str, frame_idx: int, frame: np.ndarray) -> None:
        if self.frame_cache_size == 0:
            return
        key = (video_path, int(frame_idx))
        self._frame_cache[key] = frame
        self._frame_cache.move_to_end(key)
        while len(self._frame_cache) > self.frame_cache_size:
            self._frame_cache.popitem(last=False)

    def get(self, video_path: str) -> decord.VideoReader:
        reader = self._readers.get(video_path)
        if reader is not None:
            self._readers.move_to_end(video_path)
            return reader

        while len(self._readers) >= self.max_open_videos:
            old_path, _ = self._readers.popitem(last=False)
            self._meta.pop(old_path, None)
            stale_keys = [key for key in self._frame_cache.keys() if key[0] == old_path]
            for key in stale_keys:
                self._frame_cache.pop(key, None)

        reader = self._open(video_path)
        self._readers[video_path] = reader
        return reader

    def get_meta(self, video_path: str) -> Tuple[int, int, int]:
        if video_path not in self._meta:
            self.get(video_path)
        return self._meta[video_path]

    def read_frame(self, video_path: str, frame_idx: int) -> np.ndarray:
        cached = self._cache_get(video_path, frame_idx)
        if cached is not None:
            return cached

        reader = self.get(video_path)
        try:
            frame = reader[int(frame_idx)].asnumpy()
        except Exception as exc:
            raise RuntimeError(
                f"decord failed reading frame {int(frame_idx)} from {video_path}"
            ) from exc
        frame = np.ascontiguousarray(frame)
        self._cache_put(video_path, frame_idx, frame)
        return frame

    def read_frames(self, video_path: str, frame_indices: list[int]) -> list[np.ndarray]:
        """Read multiple frames in one optimized batch call.

        Duplicate indices (common at video boundaries due to clamping) are
        deduplicated before the batch call because some decord versions
        mishandle repeated indices.
        """
        if len(frame_indices) == 0:
            return []

        requested = [int(value) for value in frame_indices]
        frames_by_index: Dict[int, np.ndarray] = {}
        missing_indices: list[int] = []
        for frame_idx in requested:
            cached = self._cache_get(video_path, frame_idx)
            if cached is not None:
                frames_by_index[frame_idx] = cached
            elif frame_idx not in frames_by_index:
                missing_indices.append(frame_idx)

        if missing_indices:
            reader = self.get(video_path)
            unique_missing = sorted(set(missing_indices))
            try:
                batch = reader.get_batch(unique_missing).asnumpy()
            except Exception as exc:
                raise RuntimeError(
                    "decord failed reading frame batch "
                    f"{unique_missing[0]}..{unique_missing[-1]} from {video_path}"
                ) from exc
            for offset, frame_idx in enumerate(unique_missing):
                frame = np.ascontiguousarray(batch[offset])
                frames_by_index[frame_idx] = frame
                self._cache_put(video_path, frame_idx, frame)

        return [frames_by_index[int(frame_idx)] for frame_idx in requested]

    def close(self) -> None:
        self._readers.clear()
        self._meta.clear()
        self._frame_cache.clear()

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
        group_codes: Optional[np.ndarray] = None,
        order_values: Optional[np.ndarray] = None,
        num_replicas: int = 1,
        rank: int = 0,
    ):
        self.positive_indices = np.asarray(positive_indices, dtype=np.int64)
        self.negative_indices = np.asarray(negative_indices, dtype=np.int64)
        self.neg_pos_ratio = _parse_neg_pos_ratio(neg_pos_ratio)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.group_codes = (
            np.asarray(group_codes, dtype=np.int64) if group_codes is not None else None
        )
        self.order_values = (
            np.asarray(order_values, dtype=np.int64) if order_values is not None else None
        )
        self.num_replicas = max(1, int(num_replicas))
        self.rank = int(rank)
        if self.rank < 0 or self.rank >= self.num_replicas:
            raise ValueError(
                f"rank must be in [0, {self.num_replicas - 1}], got {self.rank}"
            )
        self.current_epoch = 0
        self.start_offset = 0
        self._global_indices = np.array([], dtype=np.int64)
        self._rank_indices = np.array([], dtype=np.int64)
        self._indices = np.array([], dtype=np.int64)
        self._sample_count = 0
        self._negative_sample_count = 0
        self.set_epoch(0)

    def _order_indices(
        self,
        indices: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if len(indices) == 0:
            return indices
        if self.group_codes is None or self.order_values is None:
            if self.shuffle:
                rng.shuffle(indices)
            return indices

        group_codes = self.group_codes[indices]
        unique_groups = np.unique(group_codes)
        if self.shuffle and len(unique_groups) > 1:
            rng.shuffle(unique_groups)

        ordered_chunks = []
        for group_code in unique_groups.tolist():
            group_mask = group_codes == group_code
            group_indices = indices[group_mask]
            group_order = np.argsort(self.order_values[group_indices], kind="stable")
            ordered_chunks.append(group_indices[group_order])

        if not ordered_chunks:
            return np.array([], dtype=np.int64)
        return np.concatenate(ordered_chunks).astype(np.int64, copy=False)

    def set_epoch(self, epoch: int, start_offset: int = 0) -> None:
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

        indices = np.concatenate([self.positive_indices, neg]).astype(np.int64, copy=False)
        self._global_indices = self._order_indices(
            indices,
            rng,
        )
        self._sample_count = int(len(self._global_indices))
        self._negative_sample_count = int(self._sample_count - len(self.positive_indices))
        self._rank_indices = self._shard_indices(self._global_indices)
        self.set_start_offset(start_offset)

    def set_start_offset(self, start_offset: int) -> None:
        offset = int(start_offset)
        if offset < 0:
            raise ValueError(f"start_offset must be >= 0, got {offset}")
        if offset > len(self._rank_indices):
            raise ValueError(
                f"start_offset={offset} exceeds available rank-local samples={len(self._rank_indices)}"
            )
        self.start_offset = offset
        self._indices = self._rank_indices[offset:].astype(np.int64, copy=False)

    def _shard_indices(self, global_indices: np.ndarray) -> np.ndarray:
        if self.num_replicas == 1:
            return global_indices.copy()
        if len(global_indices) == 0:
            return np.array([], dtype=np.int64)

        pad_size = (-len(global_indices)) % self.num_replicas
        if pad_size > 0:
            padded = np.concatenate([global_indices, global_indices[:pad_size]])
        else:
            padded = global_indices
        return padded[self.rank :: self.num_replicas].astype(np.int64, copy=False)

    def __iter__(self):
        return iter(self._indices.tolist())

    def __len__(self) -> int:
        return len(self._indices)

    def get_indices(self) -> np.ndarray:
        return self._indices.copy()

    def get_global_indices(self) -> np.ndarray:
        return self._global_indices.copy()

    def get_counts(self) -> Dict[str, int]:
        return {
            "samples": self._sample_count,
            "positives": int(len(self.positive_indices)),
            "negatives": self._negative_sample_count,
            "processed_rank_samples": int(self.start_offset),
            "remaining_rank_samples": int(len(self._indices)),
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
        preprocess_mode: str = "torchvision",
        spatial_mode: str = "ball_crop",
        is_training: bool = False,
        base_seed: int = 42,
        train_augmentation_mode: str = "clip_consistent",
        video_id_filters: Optional[Iterable[Union[str, int]]] = None,
        half_filters: Optional[Iterable[Union[str, int]]] = None,
        frame_cache_size: int = 128,
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
        self.preprocess_mode = str(preprocess_mode)
        self.spatial_mode = str(spatial_mode)
        self.is_training = bool(is_training)
        self.base_seed = int(base_seed)
        self.train_augmentation_mode = str(train_augmentation_mode)
        self.current_epoch = 0
        self.video_id_filters = _normalize_optional_strings(video_id_filters)
        self.half_filters = _normalize_optional_ints(half_filters)
        if self.preprocess_mode not in PREPROCESS_MODES:
            raise ValueError(
                f"preprocess_mode must be one of {sorted(PREPROCESS_MODES)}, "
                f"got {self.preprocess_mode!r}"
            )
        if self.spatial_mode not in SPATIAL_MODES:
            raise ValueError(
                f"spatial_mode must be one of {sorted(SPATIAL_MODES)}, "
                f"got {self.spatial_mode!r}"
            )
        if self.train_augmentation_mode not in TRAIN_AUGMENTATION_MODES:
            raise ValueError(
                "train_augmentation_mode must be one of "
                f"{sorted(TRAIN_AUGMENTATION_MODES)}, got {self.train_augmentation_mode!r}"
            )
        self.cropper = FrameCropper(
            crop_scale_factor=crop_scale_factor,
            output_size=self.input_size,
            default_radius=default_radius,
        )
        self.reader_pool = _VideoReaderPool(
            max_open_videos=max_open_videos,
            frame_cache_size=frame_cache_size,
        )
        self._norm_mean = torch.tensor(VMAE_MEAN, dtype=torch.float32).view(3, 1, 1, 1)
        self._norm_std = torch.tensor(VMAE_STD, dtype=torch.float32).view(3, 1, 1, 1)

        self.df = self._load_dataframe()
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

    def _load_dataframe(self) -> pd.DataFrame:
        dataset_kwargs = {"format": "parquet"}
        if self.parquet_path.is_dir():
            dataset_kwargs["partitioning"] = "hive"
        dataset = pads.dataset(str(self.parquet_path), **dataset_kwargs)
        all_columns = list(dataset.schema.names)
        columns = _columns_to_load(all_columns)
        filter_expression = _dataset_filter_expression(
            self.video_id_filters,
            self.half_filters,
        )
        table = dataset.to_table(columns=columns, filter=filter_expression)
        df = table.to_pandas()
        if df.empty:
            filter_parts = []
            if self.video_id_filters:
                filter_parts.append(f"video_id in {list(self.video_id_filters)}")
            if self.half_filters:
                filter_parts.append(f"half in {list(self.half_filters)}")
            filter_text = ", ".join(filter_parts) if filter_parts else "no filters"
            raise ValueError(
                f"No parquet rows matched {filter_text} from {self.parquet_path}"
            )
        return df.reset_index(drop=True)

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
        self.row_indices = np.arange(len(self.df), dtype=np.int64)
        self.labels = self.df["label"].astype(np.int8).to_numpy()
        self.video_path_codes, self.video_path_table = _factorize_strings(self.df["video_path"])
        self.ball_x = pd.to_numeric(self.df["ball_x"], errors="coerce").to_numpy(np.float32)
        self.ball_y = pd.to_numeric(self.df["ball_y"], errors="coerce").to_numpy(np.float32)
        self.ball_w = pd.to_numeric(self.df["ball_w"], errors="coerce").to_numpy(np.float32)
        self.ball_h = pd.to_numeric(self.df["ball_h"], errors="coerce").to_numpy(np.float32)
        self.fps = self._numeric_column("fps")
        self.ball_confidence = self._numeric_column("ball_confidence")
        self.sample_group_codes = self.video_path_codes.astype(np.int64, copy=False)
        self.sample_order_values = self.frames.astype(np.int64, copy=False)

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

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def get_epoch(self) -> int:
        return int(self.current_epoch)

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

    def _uses_low_memory_eval(self) -> bool:
        return getattr(self, "preprocess_mode", "torchvision") == "low_memory_eval"

    def _resize_full_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame.shape[0] == self.input_size and frame.shape[1] == self.input_size:
            return frame
        return cv2.resize(
            frame,
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_LINEAR,
        )

    def _apply_spatial_policy(self, frames: list, center_det: Optional[Dict]) -> list:
        if self.spatial_mode == "full_frame":
            if self._uses_low_memory_eval():
                return [self._resize_full_frame(frame) for frame in frames]
            return frames
        if self.spatial_mode != "ball_crop":
            raise RuntimeError(f"Unsupported spatial_mode: {self.spatial_mode}")

        # Shrink the no-ball path during inference to avoid feeding full-size
        # frames into PIL/torchvision before we downscale them anyway.
        if center_det is None:
            if self._uses_low_memory_eval():
                return [self._resize_full_frame(frame) for frame in frames]
            return frames

        center_idx = len(frames) // 2
        radius = self.cropper.compute_radius(
            center_det, frames[center_idx].shape[:2]
        )
        return [self.cropper.crop(frame, center_det, radius=radius) for frame in frames]

    def _sample_clip_augmentation_params(self, row_idx: int) -> Optional[Dict[str, object]]:
        if not self.is_training or self.train_augmentation_mode != "clip_consistent":
            return None

        rng = random.Random(_stable_seed(self.base_seed, self.get_epoch(), row_idx))
        brightness = rng.uniform(0.9, 1.1)
        contrast = rng.uniform(0.9, 1.1)
        saturation = rng.uniform(0.9, 1.1)
        hue = rng.uniform(-0.1, 0.1)
        color_ops = ["brightness", "contrast", "saturation", "hue"]
        rng.shuffle(color_ops)
        return {
            "flip": rng.random() < 0.5,
            "brightness": brightness,
            "contrast": contrast,
            "saturation": saturation,
            "hue": hue,
            "color_ops": tuple(color_ops),
        }

    def _apply_clip_augmentation_to_image(
        self,
        image: Image.Image,
        augmentation_params: Optional[Dict[str, object]],
    ) -> Image.Image:
        if augmentation_params is None:
            return image

        if bool(augmentation_params["flip"]):
            image = TF.hflip(image)

        for op_name in augmentation_params["color_ops"]:
            if op_name == "brightness":
                image = TF.adjust_brightness(image, float(augmentation_params["brightness"]))
            elif op_name == "contrast":
                image = TF.adjust_contrast(image, float(augmentation_params["contrast"]))
            elif op_name == "saturation":
                image = TF.adjust_saturation(image, float(augmentation_params["saturation"]))
            elif op_name == "hue":
                image = TF.adjust_hue(image, float(augmentation_params["hue"]))
        return image

    def _to_torchvision_video(
        self,
        frames: list[np.ndarray],
        augmentation_params: Optional[Dict[str, object]] = None,
    ) -> torch.Tensor:
        processed = []
        for frame in frames:
            frame_array = frame.astype("uint8", copy=False)
            img = Image.fromarray(frame_array, "RGB")
            img = self._apply_clip_augmentation_to_image(img, augmentation_params)
            if self.transform is not None:
                tensor = self.transform(img)
            else:
                tensor = T.ToTensor()(img)
            processed.append(tensor)
        return torch.stack(processed, dim=0).permute(1, 0, 2, 3)

    def _to_low_memory_video(self, frames: list[np.ndarray]) -> torch.Tensor:
        video = torch.empty(
            (3, len(frames), self.input_size, self.input_size),
            dtype=torch.float32,
        )
        for frame_idx, frame in enumerate(frames):
            frame_array = np.ascontiguousarray(frame.astype(np.uint8, copy=False))
            video[:, frame_idx].copy_(torch.from_numpy(frame_array).permute(2, 0, 1))
        video.div_(255.0)
        video.sub_(self._norm_mean).div_(self._norm_std)
        return video

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
            augmentation_params = self._sample_clip_augmentation_params(row_idx)
            if self._uses_low_memory_eval():
                video = self._to_low_memory_video(frames)
            else:
                video = self._to_torchvision_video(frames, augmentation_params)
            del frames
            meta = {
                "video_id": self._video_id_at(row_idx),
                "half": str(self.halves[row_idx]),
                "frame": int(center_frame),
                "path": str(video_path),
                "video_path": str(video_path),
                "row_idx": int(self.row_indices[row_idx]),
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


def get_transforms(
    input_size: int = 224,
    is_training: bool = True,
    augmentation_mode: str = "clip_consistent",
):
    normalize = T.Normalize(mean=VMAE_MEAN, std=VMAE_STD)

    if is_training and augmentation_mode == "legacy_frame_random":
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


def _dataloader_kwargs(
    num_workers: int,
    config: Config,
    *,
    persistent_workers: bool = True,
) -> Dict[str, object]:
    kwargs: Dict[str, object] = {}
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(persistent_workers)
        start_method = str(getattr(config, "loader_start_method", "spawn")).strip()
        if start_method:
            kwargs["multiprocessing_context"] = start_method
    return kwargs


def build_parquet_train_dataloader(
    config: Config,
    *,
    num_replicas: int = 1,
    rank: int = 0,
):
    augmentation_mode = str(getattr(config, "train_augmentation_mode", "clip_consistent"))
    train_transform = get_transforms(
        config.input_size,
        is_training=True,
        augmentation_mode=augmentation_mode,
    )
    train_dataset = ParquetHeaderDataset(
        parquet_path=config.train_parquet,
        num_frames=config.num_frames,
        input_size=config.input_size,
        transform=train_transform,
        strict_paths=True,
        dataset_root=config.dataset_root,
        max_open_videos=int(getattr(config, "max_open_videos", 8)),
        frame_cache_size=int(getattr(config, "frame_cache_size", 128)),
        spatial_mode=str(getattr(config, "spatial_mode", "ball_crop")),
        is_training=True,
        base_seed=int(getattr(config, "seed", 42)),
        train_augmentation_mode=augmentation_mode,
        resample_on_decode_failure=bool(
            getattr(config, "resample_on_decode_failure", True)
        ),
        video_id_filters=getattr(config, "train_video_ids", ()),
        half_filters=getattr(config, "train_halves", ()),
    )

    if len(train_dataset.positive_indices) == 0:
        raise ValueError("Training parquet has no positive samples (label=1).")

    train_sampler = DeterministicRatioSampler(
        positive_indices=train_dataset.positive_indices,
        negative_indices=train_dataset.negative_indices,
        neg_pos_ratio=config.neg_pos_ratio,
        seed=config.seed,
        shuffle=True,
        group_codes=train_dataset.sample_group_codes,
        order_values=train_dataset.sample_order_values,
        num_replicas=num_replicas,
        rank=rank,
    )

    generator = _build_torch_generator(int(config.seed) + int(rank))
    num_workers = int(getattr(config, "num_workers", 0))
    use_persistent_workers = augmentation_mode != "clip_consistent"
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_seed_worker,
        generator=generator,
        **_dataloader_kwargs(
            num_workers,
            config,
            persistent_workers=use_persistent_workers,
        ),
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
        max_open_videos=int(getattr(config, "max_open_videos", 8)),
        frame_cache_size=int(getattr(config, "frame_cache_size", 128)),
        spatial_mode=str(getattr(config, "spatial_mode", "ball_crop")),
        is_training=False,
        base_seed=int(getattr(config, "seed", 42)),
        train_augmentation_mode="none",
        resample_on_decode_failure=bool(
            getattr(config, "resample_on_decode_failure", True)
        ),
        video_id_filters=getattr(config, "val_video_ids", ()),
        half_filters=getattr(config, "val_halves", ()),
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
            group_codes=val_dataset.sample_group_codes,
            order_values=val_dataset.sample_order_values,
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
        **_dataloader_kwargs(val_num_workers, config),
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
