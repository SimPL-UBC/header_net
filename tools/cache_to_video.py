#!/usr/bin/env python3
"""Export cached header samples into per-match videos for quick inspection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd

HEADER_NET_ROOT = Path(__file__).resolve().parents[1]
if str(HEADER_NET_ROOT) not in sys.path:
    sys.path.append(str(HEADER_NET_ROOT))

from configs import header_default as cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stitch cached npy samples into per-match positive/negative videos."
    )
    parser.add_argument(
        "--cache-index",
        type=Path,
        required=True,
        help="CSV produced by create_cache_header.py",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        required=True,
        help="Directory containing *_s.npy cache tensors",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Where to store generated videos, grouped per match",
    )
    parser.add_argument(
        "--matches",
        nargs="+",
        help="Optional list of match ids (or substrings) to export. Case-insensitive.",
    )
    parser.add_argument(
        "--limit-per-label",
        type=int,
        help="Cap the number of samples per label per match (useful for spot checks).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Frames per second for output videos (default: %(default)s)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=4.0,
        help="Uniform resize factor applied to cached patches before writing video (default: %(default)s)",
    )
    parser.add_argument(
        "--pad-between-samples",
        type=int,
        default=0,
        help="How many blank frames to insert between samples (default: %(default)s)",
    )
    parser.add_argument(
        "--codec",
        default="mp4v",
        help="FourCC codec used by cv2.VideoWriter (default: %(default)s)",
    )
    parser.add_argument(
        "--ext",
        default=".mp4",
        help="File extension for generated videos (default: %(default)s)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing videos instead of skipping them.",
    )
    return parser.parse_args()


def load_index(index_path: Path) -> pd.DataFrame:
    if not index_path.exists():
        raise FileNotFoundError(f"Cache index not found: {index_path}")
    df = pd.read_csv(index_path)
    required_cols = {"path", "label", "video_id", "half", "frame"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{index_path} missing required columns: {sorted(missing)}")
    return df


def filter_matches(df: pd.DataFrame, matches: Iterable[str]) -> pd.DataFrame:
    terms = [m.lower() for m in matches]
    mask = df["video_id"].str.lower().apply(lambda vid: any(term in vid for term in terms))
    return df[mask]


def resolve_offsets(window_size: Sequence[int], temporal_len: int) -> np.ndarray:
    offsets = np.asarray(window_size, dtype=int)
    if offsets.shape[0] == temporal_len:
        return offsets
    center = temporal_len // 2
    return np.arange(-center, -center + temporal_len, dtype=int)


def ensure_color(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return np.repeat(array[..., None], 3, axis=2)
    if array.ndim == 3 and array.shape[2] == 1:
        return np.repeat(array, 3, axis=2)
    return array


def resize_frame(frame: np.ndarray, scale: float) -> np.ndarray:
    if np.isclose(scale, 1.0):
        return frame
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_NEAREST
    h, w = frame.shape[:2]
    new_size = (int(round(w * scale)), int(round(h * scale)))
    return cv2.resize(frame, new_size, interpolation=interpolation)


def annotate_frame(
    frame: np.ndarray,
    lines: Sequence[str],
    font_scale: float,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    shadow_color: Tuple[int, int, int] = (0, 0, 0),
    margin: int = 8,
) -> np.ndarray:
    annotated = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(round(font_scale)))

    y = margin
    for line in lines:
        line = line.strip()
        if not line:
            continue
        size, baseline = cv2.getTextSize(line, font, font_scale, thickness)
        y_line = y + size[1]
        cv2.putText(annotated, line, (margin, y_line), font, font_scale, shadow_color, thickness + 2, cv2.LINE_AA)
        cv2.putText(annotated, line, (margin, y_line), font, font_scale, text_color, thickness, cv2.LINE_AA)
        y += size[1] + baseline + 4
    return annotated


def video_writer(path: Path, codec: str, fps: float, frame_shape: Tuple[int, int]) -> cv2.VideoWriter:
    width, height = frame_shape
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Unable to create video writer for {path}")
    return writer


def collect_sample_path(cache_root: Path, raw_path: str) -> Path:
    stem = Path(raw_path).name
    filename = f"{stem}_s.npy"
    return cache_root / filename


def resolve_metadata_path(cache_root: Path, row: Any) -> Optional[Path]:
    meta_value = getattr(row, "metadata", None)
    if isinstance(meta_value, str) and meta_value:
        meta_path = Path(meta_value)
        if meta_path.is_absolute() and not meta_path.exists():
             # Fallback to cache_root if absolute path is missing (e.g. different machine)
             candidate = cache_root / meta_path.name
             if candidate.exists():
                 return candidate
        
        if not meta_path.is_absolute():
            meta_path = cache_root / meta_path.name
        return meta_path

    raw_path = getattr(row, "path", "")
    if raw_path:
        stem = Path(raw_path).name
        candidate = cache_root / f"{stem}_meta.json"
        return candidate
    return None


def load_metadata(meta_path: Optional[Path]) -> dict:
    if not meta_path:
        return {}
    try:
        with meta_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        print(f"[WARN] Failed to parse metadata at {meta_path}")
        return {}


CLASS_COLORS = {
    "ball": (0, 255, 0),
    "player": (0, 165, 255),
    "goalkeeper": (0, 255, 255),
    "referee": (255, 0, 0),
}


def draw_box_with_label(
    frame: np.ndarray,
    box: Sequence[float],
    color: Tuple[int, int, int],
    labels: Sequence[str],
) -> None:
    if not box:
        return
    x, y, w, h = box
    if w <= 0 or h <= 0:
        return
    p1 = (int(round(x)), int(round(y)))
    p2 = (int(round(x + w)), int(round(y + h)))
    cv2.rectangle(frame, p1, p2, color, max(1, int(round(frame.shape[0] / 256))))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.3, frame.shape[0] / 512.0)
    thickness = max(1, int(round(font_scale)))
    y_offset = p1[1] - 4
    for text in labels:
        text = text.strip()
        if not text:
            continue
        size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
        text_origin = (p1[0], max(size[1] + 2, y_offset))
        cv2.putText(frame, text, text_origin, font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(frame, text, text_origin, font, font_scale, color, thickness, cv2.LINE_AA)
        y_offset = text_origin[1] - (size[1] + baseline + 4)


def overlay_detections(frame: np.ndarray, frame_meta: dict) -> None:
    ball_meta = frame_meta.get("ball")
    if ball_meta:
        color = CLASS_COLORS.get("ball", (0, 255, 0))
        box = ball_meta.get("patch_box") or ball_meta.get("box")
        labels = ["Ball"]
        conf = ball_meta.get("confidence")
        if conf is not None and not np.isnan(conf):
            labels[0] = f"Ball {conf:.2f}"
        velocity = ball_meta.get("velocity")
        source = ball_meta.get("source")
        extra = []
        if isinstance(velocity, (list, tuple)) and len(velocity) == 2:
            extra.append(f"vx:{velocity[0]:+.2f}")
            extra.append(f"vy:{velocity[1]:+.2f}")
        if source:
            extra.append(str(source))
        if extra:
            labels.extend(extra)
        draw_box_with_label(frame, box, color, labels)

    for det in frame_meta.get("players", []):
        name = str(det.get("class_name", "player")).lower()
        color = CLASS_COLORS.get(name, (255, 140, 0))
        box = det.get("patch_box") or det.get("box")
        conf = det.get("confidence")
        label = name.capitalize()
        if conf is not None and not np.isnan(conf):
            label = f"{label} {conf:.2f}"
        draw_box_with_label(frame, box, color, [label])


def export_label_video(
    video_id: str,
    label_value: int,
    rows: pd.DataFrame,
    cache_root: Path,
    output_path: Path,
    codec: str,
    fps: float,
    scale: float,
    offsets: Sequence[int],
    pad_between_samples: int,
) -> Tuple[int, Path]:
    label_name = "positive" if label_value == 1 else "negative"
    sample_total = len(rows)
    writer: cv2.VideoWriter | None = None
    written_frames = 0
    first_frame_shape: Tuple[int, int] | None = None

    for sample_idx, row in enumerate(rows.itertuples(index=False), start=1):
        cache_path = collect_sample_path(cache_root, row.path)
        if not cache_path.exists():
            print(f"[WARN] Cache tensor missing for {cache_path}, skipping.")
            continue

        sample = np.load(cache_path, allow_pickle=False)
        if sample.ndim < 3:
            print(f"[WARN] Unexpected tensor shape {sample.shape} at {cache_path}, skipping.")
            continue

        sample = ensure_color(sample)
        temporal_len = sample.shape[0]
        if len(offsets) != temporal_len:
            local_offsets = resolve_offsets(offsets, temporal_len)
        else:
            local_offsets = np.asarray(offsets, dtype=int)

        meta_path = resolve_metadata_path(cache_root, row)
        metadata = load_metadata(meta_path)
        frame_meta_sequence = metadata.get("frames", []) if isinstance(metadata, dict) else []

        for frame_idx in range(temporal_len):
            frame = sample[frame_idx].copy()
            frame_meta = frame_meta_sequence[frame_idx] if frame_idx < len(frame_meta_sequence) else None
            if isinstance(frame_meta, dict):
                overlay_detections(frame, frame_meta)

            frame = resize_frame(frame, scale)
            if first_frame_shape is None:
                first_frame_shape = (frame.shape[1], frame.shape[0])
                output_path.parent.mkdir(parents=True, exist_ok=True)
                writer = video_writer(output_path, codec, fps, first_frame_shape)
            elif frame.shape[0] != first_frame_shape[1] or frame.shape[1] != first_frame_shape[0]:
                print(f"[WARN] Frame size mismatch in {cache_path}, skipping frame.")
                continue

            frame_number = int(row.frame) + int(local_offsets[frame_idx])
            extra_line = None
            if isinstance(frame_meta, dict) and frame_meta.get("ball"):
                ball_meta = frame_meta["ball"]
                conf = ball_meta.get("confidence")
                velocity = ball_meta.get("velocity")
                parts = []
                if conf is not None and not np.isnan(conf):
                    parts.append(f"conf {conf:.2f}")
                if isinstance(velocity, (list, tuple)) and len(velocity) == 2:
                    parts.append(f"vx {velocity[0]:+.2f} vy {velocity[1]:+.2f}")
                if ball_meta.get("source"):
                    parts.append(str(ball_meta["source"]))
                if parts:
                    extra_line = "ball " + " | ".join(parts)

            lines = (
                f"{label_name} sample {sample_idx}/{sample_total}",
                f"half {int(row.half)} frame {frame_number} (offset {int(local_offsets[frame_idx]):+d})",
            )
            if extra_line:
                lines = lines + (extra_line,)
            font_scale = max(0.3, 0.3 * scale)
            annotated = annotate_frame(frame, lines, font_scale=font_scale)
            writer.write(annotated)
            written_frames += 1

        if writer is not None and pad_between_samples > 0:
            blank = np.zeros((first_frame_shape[1], first_frame_shape[0], 3), dtype=np.uint8)
            for _ in range(pad_between_samples):
                writer.write(blank)
                written_frames += 1

    if writer is not None:
        writer.release()
    return written_frames, output_path


def main() -> None:
    args = parse_args()
    cache_index = args.cache_index.expanduser()
    cache_root = args.cache_root.expanduser()
    output_dir = args.output_dir.expanduser()

    df = load_index(cache_index)
    if args.matches:
        df = filter_matches(df, args.matches)
        if df.empty:
            print("No samples matched the provided match filters.")
            return

    offsets = resolve_offsets(cfg.WINDOW_SIZE, len(cfg.WINDOW_SIZE))

    grouped = df.groupby("video_id", sort=True)
    if not grouped.ngroups:
        print("No samples found in cache index.")
        return

    for video_id, group in grouped:
        group = group.sort_values(["half", "frame"])
        video_output_dir = output_dir / video_id

        for label_value in (1, 0):
            label_rows = group[group["label"] == label_value]
            if args.limit_per_label is not None:
                label_rows = label_rows.head(args.limit_per_label)
            if label_rows.empty:
                continue

            video_path = video_output_dir / f"{'pos' if label_value == 1 else 'neg'}{args.ext}"
            if video_path.exists() and not args.overwrite:
                print(f"[SKIP] {video_path} already exists (use --overwrite to rebuild).")
                continue

            written_frames, output_path = export_label_video(
                video_id=video_id,
                label_value=label_value,
                rows=label_rows,
                cache_root=cache_root,
                output_path=video_path,
                codec=args.codec,
                fps=args.fps,
                scale=args.scale,
                offsets=offsets,
                pad_between_samples=args.pad_between_samples,
            )

            if written_frames == 0 and output_path.exists():
                output_path.unlink(missing_ok=True)
            else:
                print(f"[OK] {video_id}: wrote {written_frames} frames to {output_path}")


if __name__ == "__main__":
    main()
