from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

HEADER_NET_ROOT = Path(__file__).resolve().parents[1]
if str(HEADER_NET_ROOT) not in sys.path:
    sys.path.append(str(HEADER_NET_ROOT))

pytest.importorskip("decord")

import training.data.parquet_header_dataset as parquet_dataset_module
from training.data.parquet_header_dataset import ParquetHeaderDataset, _VideoReaderPool


class FakeBatch:
    def __init__(self, array: np.ndarray):
        self._array = array

    def asnumpy(self) -> np.ndarray:
        return self._array

    def __getitem__(self, index):
        raise TypeError("'NDArray' object is not subscriptable")


class FakeReader:
    def __init__(self, batch_array: np.ndarray):
        self.batch_array = batch_array
        self.calls: list[list[int]] = []

    def get_batch(self, indices: list[int]) -> FakeBatch:
        self.calls.append(list(indices))
        return FakeBatch(self.batch_array)


def make_dataset_stub(
    *,
    labels: np.ndarray,
    video_paths: np.ndarray,
    frames: np.ndarray,
    resample_on_decode_failure: bool,
    max_resample_attempts: int,
    preprocess_mode: str = "torchvision",
    input_size: int = 4,
) -> ParquetHeaderDataset:
    dataset = object.__new__(ParquetHeaderDataset)
    dataset.labels = labels
    dataset.video_paths = video_paths
    dataset.frames = frames
    dataset.max_resample_attempts = max_resample_attempts
    dataset.resample_on_decode_failure = resample_on_decode_failure
    dataset.preprocess_mode = preprocess_mode
    dataset.input_size = input_size
    dataset.transform = None
    dataset.row_indices = np.arange(len(labels), dtype=np.int64)
    dataset._norm_mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(3, 1, 1, 1)
    dataset._norm_std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(3, 1, 1, 1)
    return dataset


def test_partitioned_parquet_filters_match_and_half(tmp_path):
    dataset_root = tmp_path / "dense_train"
    df = pd.DataFrame(
        [
            {
                "video_id": "match_a",
                "half": 1,
                "frame": 10,
                "label": 1,
                "video_path": "/tmp/a_1.mkv",
                "ball_x": 1.0,
                "ball_y": 2.0,
                "ball_w": 3.0,
                "ball_h": 4.0,
                "fps": 25.0,
                "ball_confidence": 0.9,
            },
            {
                "video_id": "match_a",
                "half": 2,
                "frame": 20,
                "label": 0,
                "video_path": "/tmp/a_2.mkv",
                "ball_x": 5.0,
                "ball_y": 6.0,
                "ball_w": 7.0,
                "ball_h": 8.0,
                "fps": 25.0,
                "ball_confidence": 0.8,
            },
            {
                "video_id": "match_b",
                "half": 1,
                "frame": 30,
                "label": 0,
                "video_path": "/tmp/b_1.mkv",
                "ball_x": 9.0,
                "ball_y": 10.0,
                "ball_w": 11.0,
                "ball_h": 12.0,
                "fps": 50.0,
                "ball_confidence": 0.7,
            },
        ]
    )
    df.to_parquet(dataset_root, index=False, partition_cols=["video_id", "half"])

    dataset = ParquetHeaderDataset(
        parquet_path=dataset_root,
        num_frames=16,
        input_size=224,
        transform=None,
        strict_paths=False,
        preprocess_mode="low_memory_eval",
        video_id_filters=["match_a"],
        half_filters=[1],
    )

    assert len(dataset) == 1
    assert dataset.class_counts() == {"samples": 1, "positives": 1, "negatives": 0}
    assert dataset._video_id_at(0) == "match_a"
    assert int(dataset.halves[0]) == 1
    assert int(dataset.frames[0]) == 10


def test_read_frames_reconstructs_duplicate_indices_from_numpy_batch(monkeypatch):
    pool = _VideoReaderPool(max_open_videos=1)
    unique_batch = np.stack(
        [np.full((2, 2, 3), fill_value=value, dtype=np.uint8) for value in (0, 1, 2)],
        axis=0,
    )
    reader = FakeReader(unique_batch)
    monkeypatch.setattr(pool, "get", lambda _video_path: reader)

    requested = [2, 2, 0, 1, 0]
    frames = pool.read_frames("match.mkv", requested)

    assert reader.calls == [[0, 1, 2]]
    assert len(frames) == len(requested)
    assert all(isinstance(frame, np.ndarray) for frame in frames)
    assert all(np.shares_memory(frame, unique_batch) for frame in frames)
    np.testing.assert_array_equal(frames[0], unique_batch[2])
    np.testing.assert_array_equal(frames[1], unique_batch[2])
    np.testing.assert_array_equal(frames[2], unique_batch[0])
    np.testing.assert_array_equal(frames[3], unique_batch[1])
    np.testing.assert_array_equal(frames[4], unique_batch[0])


def test_getitem_chains_decode_error_when_resampling_disabled():
    dataset = make_dataset_stub(
        labels=np.array([0], dtype=np.int8),
        video_paths=np.array(["broken.mkv"], dtype=object),
        frames=np.array([0], dtype=np.int64),
        resample_on_decode_failure=False,
        max_resample_attempts=1,
    )

    def fail_load(video_path: str, center_frame: int):
        raise TypeError("'NDArray' object is not subscriptable")

    dataset._load_window_frames = fail_load

    with pytest.raises(
        RuntimeError,
        match=r"Unable to decode temporal window for row_idx=0 video=broken\.mkv frame=0",
    ) as exc_info:
        dataset[0]

    assert isinstance(exc_info.value.__cause__, TypeError)
    assert "not subscriptable" in str(exc_info.value.__cause__)


def test_getitem_resamples_after_decode_exception():
    dataset = make_dataset_stub(
        labels=np.array([0, 0], dtype=np.int8),
        video_paths=np.array(["broken.mkv", "good.mkv"], dtype=object),
        frames=np.array([0, 10], dtype=np.int64),
        resample_on_decode_failure=True,
        max_resample_attempts=2,
    )
    dataset.video_ids = np.array(["match", "match"], dtype=object)
    dataset.halves = np.array([1, 1], dtype=np.int16)
    dataset.ball_x = np.array([np.nan, np.nan], dtype=np.float32)
    dataset.ball_y = np.array([np.nan, np.nan], dtype=np.float32)
    dataset.ball_w = np.array([np.nan, np.nan], dtype=np.float32)
    dataset.ball_h = np.array([np.nan, np.nan], dtype=np.float32)
    dataset.fps = np.array([25.0, 25.0], dtype=np.float32)
    dataset.ball_confidence = np.array([np.nan, np.nan], dtype=np.float32)
    dataset.transform = None
    dataset._sample_replacement_row = lambda _label: 1

    def load_window(video_path: str, center_frame: int):
        if video_path == "broken.mkv":
            raise TypeError("'NDArray' object is not subscriptable")
        frame = np.full((4, 4, 3), fill_value=center_frame, dtype=np.uint8)
        return [frame, frame]

    dataset._load_window_frames = load_window

    video, label, meta = dataset[0]

    assert label == 0
    assert tuple(video.shape) == (3, 2, 4, 4)
    assert meta["row_idx"] == 1
    assert meta["video_path"] == "good.mkv"
    assert meta["frame"] == 10


def test_low_memory_eval_resizes_missing_ball_frames_without_pil(monkeypatch):
    dataset = make_dataset_stub(
        labels=np.array([0], dtype=np.int8),
        video_paths=np.array(["video.mkv"], dtype=object),
        frames=np.array([42], dtype=np.int64),
        resample_on_decode_failure=False,
        max_resample_attempts=1,
        preprocess_mode="low_memory_eval",
        input_size=4,
    )
    dataset.video_ids = np.array(["match"], dtype=object)
    dataset.halves = np.array([1], dtype=np.int16)
    dataset.ball_x = np.array([np.nan], dtype=np.float32)
    dataset.ball_y = np.array([np.nan], dtype=np.float32)
    dataset.ball_w = np.array([np.nan], dtype=np.float32)
    dataset.ball_h = np.array([np.nan], dtype=np.float32)
    dataset.fps = np.array([25.0], dtype=np.float32)
    dataset.ball_confidence = np.array([np.nan], dtype=np.float32)

    frames = [
        np.full((12, 16, 3), fill_value=32, dtype=np.uint8),
        np.full((12, 16, 3), fill_value=96, dtype=np.uint8),
    ]
    dataset._load_window_frames = lambda _path, _center: frames

    def fail_fromarray(*_args, **_kwargs):
        raise AssertionError("low_memory_eval should not call PIL.Image.fromarray")

    monkeypatch.setattr(parquet_dataset_module.Image, "fromarray", fail_fromarray)

    video, label, meta = dataset[0]

    assert label == 0
    assert tuple(video.shape) == (3, 2, 4, 4)
    assert video.dtype == torch.float32
    assert torch.isfinite(video).all()
    assert meta["video_path"] == "video.mkv"
    assert meta["frame"] == 42
