import numpy as np
from pathlib import Path

from cache.create_cache_header import VideoSource, gather_ball_track, build_sample


def make_source(width: int = 640, height: int = 360) -> VideoSource:
    return VideoSource(
        match_name="TestMatch",
        half=1,
        key="TestMatch_half1",
        path=Path("/tmp/test.mp4"),
        frame_count=2000,
        width=width,
        height=height,
    )


def test_gather_ball_track_interpolates_between_detections():
    window = [-2, -1, 0, 1, 2]
    dets = {
        233: {
            0: {"box": [100.0, 200.0, 20.0, 20.0], "confidence": 0.9, "class_id": 0}
        },
        236: {
            0: {"box": [110.0, 205.0, 20.0, 20.0], "confidence": 0.8, "class_id": 0}
        },
    }
    frame_id = 235
    boxes, has_detection, entries = gather_ball_track(frame_id, window, dets, make_source())

    expected_flags = [(frame_id + offset) in dets for offset in window]
    assert has_detection == expected_flags
    assert len(entries) == len(window)

    # Interpolated centre frame should sit midway between neighbours
    centre_box = boxes[2]
    assert np.allclose(centre_box, [106.666667, 203.333333, 20.0, 20.0], atol=1e-3)


def test_build_sample_skips_when_no_detections():
    window = [-1, 0, 1]
    frames = {
        99: np.zeros((48, 64), dtype=np.uint8),
        100: np.zeros((48, 64), dtype=np.uint8),
        101: np.zeros((48, 64), dtype=np.uint8),
        97: np.zeros((48, 64), dtype=np.uint8),
        98: np.zeros((48, 64), dtype=np.uint8),
        102: np.zeros((48, 64), dtype=np.uint8),
        103: np.zeros((48, 64), dtype=np.uint8),
    }

    images, masks, empty_count, has_valid, metadata = build_sample(
        frame_id=100,
        window=window,
        dets={},
        frames=frames,
        source=make_source(width=64, height=48),
        crop_scale_factor=1.0,
        output_size=32,
    )

    assert images == []
    assert masks == []
    assert empty_count == len(window)
    assert has_valid is False
    assert metadata == []


def test_build_sample_returns_images_with_partial_detections():
    window = [-1, 0, 1]
    dets = {
        100: {
            0: {"box": [30.0, 40.0, 10.0, 10.0], "confidence": 0.95, "class_id": 0}
        }
    }
    frames = {
        idx: np.full((60, 80), fill_value=(idx % 255), dtype=np.uint8)
        for idx in range(95, 106)
    }

    images, masks, empty_count, has_valid, metadata = build_sample(
        frame_id=100,
        window=window,
        dets=dets,
        frames=frames,
        source=make_source(width=80, height=60),
        crop_scale_factor=1.0,
        output_size=32,
    )

    assert has_valid is True
    assert empty_count == 0
    assert len(images) == len(window)
    assert all(img.shape == (32, 32, 3) for img in images)
    assert len(metadata) == len(window)
