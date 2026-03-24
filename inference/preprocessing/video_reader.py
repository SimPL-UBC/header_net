"""Video reading utilities for inference pipeline."""

from pathlib import Path
from typing import Optional, Dict, Iterator, Tuple
from collections import OrderedDict
import decord
import numpy as np

decord.bridge.set_bridge("native")


class VideoReader:
    """
    Efficient video frame reader with optional caching.

    This class wraps decord's VideoReader with additional features:
    - Random access frame seeking
    - Optional LRU frame caching for repeated access
    - Context manager support

    Attributes:
        video_path: Path to the video file.
        frame_count: Total number of frames in the video.
        width: Video width in pixels.
        height: Video height in pixels.
        fps: Video frame rate.

    Example:
        >>> with VideoReader("match.mp4") as reader:
        ...     frame = reader.get_frame(100)
        ...     print(frame.shape)  # (H, W, 3) in RGB
    """

    def __init__(
        self,
        video_path: Path,
        cache_frames: bool = True,
        max_cache_size: int = 500,
    ):
        """
        Initialize the video reader.

        Args:
            video_path: Path to the video file (mp4, mkv, etc.)
            cache_frames: Whether to cache frames for repeated access.
            max_cache_size: Maximum number of frames to keep in cache.
        """
        self.video_path = Path(video_path)
        self.cache_frames = cache_frames
        self.max_cache_size = max_cache_size

        try:
            self._reader = decord.VideoReader(str(video_path), ctx=decord.cpu())
        except Exception as exc:
            raise RuntimeError(f"Unable to open video: {video_path}") from exc

        self.frame_count = int(len(self._reader))
        if self.frame_count <= 0:
            raise RuntimeError(f"Video has no frames: {video_path}")

        first_frame = self._reader.get_batch([0]).asnumpy()[0]
        self.height = int(first_frame.shape[0])
        self.width = int(first_frame.shape[1])
        try:
            self.fps = float(self._reader.get_avg_fps())
        except Exception:
            self.fps = 0.0

        # LRU cache using OrderedDict
        self._frame_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        if self.cache_frames:
            self._frame_cache[0] = np.ascontiguousarray(first_frame)

    def get_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """
        Get a frame by index (0-indexed).

        Args:
            frame_idx: The frame index to retrieve.

        Returns:
            RGB numpy array (H, W, 3) or None if frame unavailable.
        """
        if frame_idx < 0 or frame_idx >= self.frame_count:
            return None

        # Check cache first
        if self.cache_frames and frame_idx in self._frame_cache:
            # Move to end (most recently used)
            self._frame_cache.move_to_end(frame_idx)
            return self._frame_cache[frame_idx]

        try:
            frame_rgb = self._reader[int(frame_idx)].asnumpy()
        except Exception:
            return None
        frame_rgb = np.ascontiguousarray(frame_rgb)

        # Add to cache if enabled
        if self.cache_frames:
            # Evict oldest if cache is full
            while len(self._frame_cache) >= self.max_cache_size:
                self._frame_cache.popitem(last=False)
            self._frame_cache[frame_idx] = frame_rgb

        return frame_rgb

    def get_frames(self, frame_indices: list) -> Dict[int, np.ndarray]:
        """
        Get multiple frames efficiently.

        Optimizes by sorting indices and reading sequentially when possible.

        Args:
            frame_indices: List of frame indices to retrieve.

        Returns:
            Dictionary mapping frame_idx -> RGB frame array.
        """
        frames = {}
        sorted_indices = sorted(set(frame_indices))

        for frame_idx in sorted_indices:
            frame = self.get_frame(frame_idx)
            if frame is not None:
                frames[frame_idx] = frame

        return frames

    def iter_frames(
        self,
        start: int = 0,
        end: Optional[int] = None,
        step: int = 1
    ) -> Iterator[Tuple[int, np.ndarray]]:
        """
        Iterate over frames efficiently.

        Args:
            start: Starting frame index.
            end: Ending frame index (exclusive). None for end of video.
            step: Frame step size.

        Yields:
            Tuples of (frame_index, frame_array).
        """
        if end is None:
            end = self.frame_count

        for frame_idx in range(start, min(end, self.frame_count), step):
            frame = self.get_frame(frame_idx)
            if frame is not None:
                yield frame_idx, frame

    def release(self):
        """Release the video capture object."""
        if hasattr(self, "_reader"):
            self._reader = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def __del__(self):
        self.release()

    def __len__(self):
        return self.frame_count

    def __repr__(self):
        return (
            f"VideoReader({self.video_path.name}, "
            f"frames={self.frame_count}, "
            f"size={self.width}x{self.height}, "
            f"fps={self.fps:.2f})"
        )
