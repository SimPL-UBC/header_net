from pathlib import Path
import sys

import numpy as np
import pytest

HEADER_NET_ROOT = Path(__file__).resolve().parents[1]
if str(HEADER_NET_ROOT) not in sys.path:
    sys.path.append(str(HEADER_NET_ROOT))

pytest.importorskip("decord")

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
) -> ParquetHeaderDataset:
    dataset = object.__new__(ParquetHeaderDataset)
    dataset.labels = labels
    dataset.video_paths = video_paths
    dataset.frames = frames
    dataset.max_resample_attempts = max_resample_attempts
    dataset.resample_on_decode_failure = resample_on_decode_failure
    return dataset


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
