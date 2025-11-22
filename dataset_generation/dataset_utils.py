#!/usr/bin/env python3
"""Utility functions for header dataset generation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Set

from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import sys

HEADER_NET_ROOT = Path(__file__).resolve().parents[1]
if str(HEADER_NET_ROOT) not in sys.path:
    sys.path.append(str(HEADER_NET_ROOT))

from configs import header_default as cfg
from utils.labels import load_header_labels, build_half_frame_lookup, canonical_match_name
from utils.detections import (
    BallDetections,
    load_ball_det_dict,
    make_video_key,
)


def load_labels_dataframe(header_dataset: Path) -> Tuple[pd.DataFrame, Optional[Path]]:
    """Load header labels from the specified dataset path."""
    header_dataset = header_dataset.expanduser()
    df = load_header_labels(header_dataset)
    if not df.empty:
        return df, header_dataset
    return pd.DataFrame(), None


@dataclass
class VideoSource:
    match_name: str
    half: int
    key: str
    path: Path
    frame_count: int
    width: int
    height: int


def discover_video_sources(dataset_root: Path, matches: Optional[Iterable[str]] = None) -> Dict[str, VideoSource]:
    match_filter = set(matches) if matches is not None else None
    dataset_root = dataset_root.expanduser()
    
    # Try to find SoccerNet root, otherwise assume dataset_root is it
    search_root = dataset_root / "SoccerNet"
    if not search_root.exists():
        search_root = dataset_root

    print(f"[INFO] Searching for videos in {search_root}")
    sources: Dict[str, VideoSource] = {}

    # Search for match directories at different depths to handle various structures
    # Depth 1: SoccerNet/Match (User's structure)
    # Depth 2: SoccerNet/League/Match
    # Depth 3: SoccerNet/League/Season/Match (Standard structure)
    candidate_dirs = []
    candidate_dirs.extend(search_root.glob("*"))
    candidate_dirs.extend(search_root.glob("*/*"))
    candidate_dirs.extend(search_root.glob("*/*/*"))

    for match_path in candidate_dirs:
        if not match_path.is_dir():
            continue
            
        # Quick check if this directory contains video files
        has_video = False
        video_files = []
        for ext in ["*.mkv", "*.mp4", "*.MKV", "*.MP4"]:
            found = list(match_path.glob(ext))
            if found:
                has_video = True
                video_files.extend(found)
        
        if not has_video:
            continue

        match_name = canonical_match_name(match_path.name)
        if match_filter is not None and match_name not in match_filter:
            continue
            
        for video_file in video_files:
            stem = video_file.stem
            half = 1
            if stem.startswith("2") or "_2" in stem:
                half = 2
            key = make_video_key(match_name, half)

            cap = cv2.VideoCapture(str(video_file))
            if not cap.isOpened():
                print(f"Warning: unable to open video {video_file}")
                continue
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            sources[key] = VideoSource(
                match_name=match_name,
                half=half,
                key=key,
                path=video_file,
                frame_count=frame_count,
                width=width,
                height=height,
            )
    return sources


def maybe_denormalise_box(box: Sequence[float], source: VideoSource) -> List[float]:
    x, y, w, h = box
    if max(abs(x), abs(y), abs(w), abs(h)) <= 1.5:
        return [
            x * source.width,
            y * source.height,
            w * source.width,
            h * source.height,
        ]
    return [float(x), float(y), float(w), float(h)]


def normalise_frame_map(entry: Any) -> Dict[int, Dict[str, Any]]:
    if entry is None:
        return {}
    if isinstance(entry, dict):
        return {int(k): dict(v) for k, v in entry.items()}
    if isinstance(entry, list):
        return {idx: dict(det) for idx, det in enumerate(entry)}
    raise TypeError(f"Unsupported detection entry type: {type(entry)!r}")


def extract_ball_detection(frame_entry: Dict[int, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if 0 in frame_entry:
        det = frame_entry[0]
        class_id = int(det.get("class_id", -1))
        name = str(det.get("class_name", "")).lower()
        if class_id == 0 or name == "ball":
            result = dict(det)
            result.setdefault("class_name", "ball")
            return result

    for det in frame_entry.values():
        class_id = int(det.get("class_id", -1))
        name = str(det.get("class_name", "")).lower()
        if class_id == 0 or name == "ball":
            result = dict(det)
            result.setdefault("class_name", "ball")
            return result
    return None


PLAYER_CLASS_NAMES = {"player", "goalkeeper", "referee"}


def extract_player_detections(frame_entry: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    players: List[Dict[str, Any]] = []
    for det in frame_entry.values():
        class_id = int(det.get("class_id", -1))
        name = str(det.get("class_name", "")).lower()
        if class_id in {1, 2, 3} or name in PLAYER_CLASS_NAMES:
            result = dict(det)
            result.setdefault("class_name", name or "player")
            players.append(result)
    return players


def clamp(value: float, min_value: float, max_value: float) -> float:
    return float(max(min_value, min(max_value, value)))


def transform_box_to_patch(
    box: Sequence[float],
    center_x: float,
    center_y: float,
    radius: float,
    output_size: int,
) -> List[float]:
    if radius <= 0 or output_size <= 0:
        return [0.0, 0.0, 0.0, 0.0]
    x, y, w, h = box
    left = center_x - radius
    top = center_y - radius
    scale = output_size / (2.0 * radius)

    x1 = (x - left) * scale
    y1 = (y - top) * scale
    x2 = (x + w - left) * scale
    y2 = (y + h - top) * scale

    x1 = clamp(x1, 0.0, output_size)
    y1 = clamp(y1, 0.0, output_size)
    x2 = clamp(x2, 0.0, output_size)
    y2 = clamp(y2, 0.0, output_size)

    return [float(x1), float(y1), float(max(0.0, x2 - x1)), float(max(0.0, y2 - y1))]


def gather_ball_track(
    frame_id: int,
    window: Sequence[int],
    dets: Dict[int, Dict[int, Dict[str, Any]]],
    source: VideoSource,
) -> Tuple[np.ndarray, List[bool], List[Optional[Dict[str, Any]]]]:
    boxes = np.full((len(window), 4), np.nan, dtype=np.float32)
    has_detection = [False] * len(window)
    ball_entries: List[Optional[Dict[str, Any]]] = [None] * len(window)

    for idx, offset in enumerate(window):
        frame_key = frame_id + offset
        frame_entry = normalise_frame_map(dets.get(frame_key))
        if not frame_entry:
            continue
        ball_det = extract_ball_detection(frame_entry)
        if not ball_det:
            continue
        box = maybe_denormalise_box(ball_det.get("box", [0, 0, 0, 0]), source)
        boxes[idx] = box
        has_detection[idx] = True
        ball_det = dict(ball_det)
        ball_det["box"] = box
        ball_entries[idx] = ball_det

    if np.isnan(boxes).all():
        return boxes, has_detection, ball_entries

    df = pd.DataFrame(boxes, columns=["x", "y", "w", "h"])
    df.interpolate(inplace=True, limit_direction="both")
    boxes_interp = df.to_numpy(dtype=np.float32)

    for idx in range(len(window)):
        if ball_entries[idx] is not None:
            ball_entries[idx] = dict(ball_entries[idx])
            ball_entries[idx]["box"] = boxes_interp[idx].tolist()
            continue
        if np.isnan(boxes_interp[idx]).any():
            continue
        ball_entries[idx] = {
            "box": boxes_interp[idx].tolist(),
            "confidence": 0.0,
            "class_id": 0,
            "class_name": "ball",
            "source": "interpolated",
        }

    return boxes_interp, has_detection, ball_entries


def crop_patch(
    image: np.ndarray,
    mask: np.ndarray,
    center_x: float,
    center_y: float,
    radius: int,
    output_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    half = radius
    x1 = int(round(center_x - half))
    y1 = int(round(center_y - half))
    x2 = x1 + 2 * half
    y2 = y1 + 2 * half

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    if any((pad_left, pad_top, pad_right, pad_bottom)):
        image = cv2.copyMakeBorder(
            image,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=0,
        )
        mask = cv2.copyMakeBorder(
            mask,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=0,
        )
        x1 += pad_left
        x2 += pad_left
        y1 += pad_top
        y2 += pad_top

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.shape[1], x2)
    y2 = min(image.shape[0], y2)

    crop_img = image[y1:y2, x1:x2]
    crop_mask = mask[y1:y2, x1:x2]

    crop_img = cv2.resize(crop_img, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
    crop_mask = cv2.resize(crop_mask, (output_size, output_size), interpolation=cv2.INTER_LINEAR)

    return crop_img, crop_mask


def build_sample(
    frame_id: int,
    window: Sequence[int],
    dets: Dict[int, Dict[int, Dict[str, Any]]],
    frames: Dict[int, np.ndarray],
    source: VideoSource,
    crop_scale_factor: float,
    output_size: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], int, bool, List[Dict[str, Any]]]:
    boxes, has_detection, ball_entries = gather_ball_track(frame_id, window, dets, source)
    if np.isnan(boxes).all():
        return [], [], len(window), False, []

    default_radius = max(output_size // 2, 1)

    valid = boxes[~np.isnan(boxes).any(axis=1)]
    radius = default_radius
    if valid.size > 0:
        widths = valid[:, 2]
        heights = valid[:, 3]
        mean_size = max(float(widths.mean()), float(heights.mean()))
        radius = int(max(default_radius, crop_scale_factor * mean_size / 2))
        radius = int(min(radius, 0.75 * max(source.width, source.height)))

    images: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    empty_count = 0
    frame_metadata: List[Dict[str, Any]] = []

    for idx, offset in enumerate(window):
        target_frame = frame_id + offset
        base_frame = frames.get(target_frame)
        if base_frame is None:
            empty_count += 1
            continue

        prev_frame = frames.get(target_frame - 2, base_frame)
        next_frame = frames.get(target_frame + 2, base_frame)
        stack = np.stack([prev_frame, base_frame, next_frame], axis=-1)

        if not np.isnan(boxes[idx]).any():
            center_x = boxes[idx, 0] + boxes[idx, 2] / 2
            center_y = boxes[idx, 1] + boxes[idx, 3] / 2
        else:
            center_x = source.width / 2
            center_y = source.height / 2

        mask = np.zeros((source.height, source.width), dtype=np.uint8)
        ball_meta = ball_entries[idx]
        if ball_meta is not None and not np.isnan(boxes[idx]).any():
            box = boxes[idx]
            radius_mask = int(max(box[2], box[3]) * 0.35)
            cv2.circle(mask, (int(center_x), int(center_y)), max(radius_mask, 2), 255, thickness=-1)

        crop_img, crop_mask = crop_patch(stack, mask, center_x, center_y, radius, output_size)
        crop_img[crop_mask > 100] = 255
        images.append(crop_img.astype(np.uint8))
        masks.append(crop_mask.astype(np.uint8))

        frame_entry = normalise_frame_map(dets.get(target_frame))
        players_raw = extract_player_detections(frame_entry)
        players_processed: List[Dict[str, Any]] = []
        for det in players_raw:
            box = maybe_denormalise_box(det.get("box", [0, 0, 0, 0]), source)
            det_out = dict(det)
            det_out["box"] = box
            det_out["patch_box"] = transform_box_to_patch(box, center_x, center_y, radius, output_size)
            players_processed.append(det_out)

        ball_payload: Optional[Dict[str, Any]] = None
        if ball_meta is not None:
            ball_payload = dict(ball_meta)
            box = ball_payload.get("box", [0, 0, 0, 0])
            ball_payload["patch_box"] = transform_box_to_patch(box, center_x, center_y, radius, output_size)

        frame_metadata.append(
            {
                "offset": int(offset),
                "frame": int(target_frame),
                "center": [float(center_x), float(center_y)],
                "radius": int(radius),
                "ball": ball_payload,
                "players": players_processed,
            }
        )

    has_valid_detection = bool(np.isfinite(boxes).any())
    return images, masks, empty_count, has_valid_detection, frame_metadata


def load_selected_frames(source: VideoSource, indices: Sequence[int]) -> Dict[int, np.ndarray]:
    unique_indices = sorted(set(idx for idx in indices if 0 <= idx < source.frame_count))
    frames: Dict[int, np.ndarray] = {}
    if not unique_indices:
        return frames

    cap = cv2.VideoCapture(str(source.path))
    if not cap.isOpened():
        print(f'Warning: unable to open video {source.path}')
        return frames

    current_pos = 0
    for target in unique_indices:
        if target < current_pos:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            current_pos = target
        while current_pos <= target:
            ret, frame = cap.read()
            if not ret:
                break
            if current_pos == target:
                frames[target] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                break
            current_pos += 1
        current_pos = target + 1
    cap.release()
    return frames


def generate_negative_samples(
    labels_df: pd.DataFrame,
    detections: BallDetections,
    sources: Dict[str, VideoSource],
    negative_ratio: float,
    guard_frames: int,
    window: Sequence[int],
) -> pd.DataFrame:
    rng = np.random.default_rng(2024)
    records: List[dict] = []

    if window:
        window_min = min(window)
        window_max = max(window)
        window_span = window_max - window_min
        min_gap = max(1, window_span + 1)
    else:
        min_gap = 1

    for (match_name, half), group in labels_df.groupby(["video_id", "half"], sort=False):
        key = make_video_key(match_name, int(half))
        source = sources.get(key)
        if source is None:
            continue

        positive_frames = set(int(f) for f in group["frame"].values)
        det_frames = set(detections.get(key, {}).keys())
        if det_frames:
            available_frames = det_frames
        else:
            available_frames = set(range(source.frame_count))

        excluded = set()
        for frame in positive_frames:
            excluded.update(range(max(0, frame - guard_frames), frame + guard_frames + 1))

        candidates = sorted(available_frames - excluded)
        if not candidates or not positive_frames:
            continue

        n_samples = min(len(candidates), int(len(positive_frames) * negative_ratio))
        if n_samples <= 0:
            continue

        chosen: List[int] = []
        for frame in rng.permutation(candidates):
            if all(abs(frame - prev) >= min_gap for prev in chosen):
                chosen.append(int(frame))
            if len(chosen) >= n_samples:
                break

        if len(chosen) < n_samples:
            print(
                f"[INFO] Negative sampling constrained for {match_name}_half{half}: "
                f"requested {n_samples}, selected {len(chosen)} (min_gap={min_gap})"
            )

        for frame in chosen:
            records.append(
                {
                    "video_id": match_name,
                    "half": int(half),
                    "frame": int(frame),
                    "label": 0,
                }
            )

    return pd.DataFrame(records)


def create_header_cache(
    detections: BallDetections,
    labels_df: pd.DataFrame,
    sources: Dict[str, VideoSource],
    output_dir: Path,
    window: Sequence[int],
    crop_scale_factor: float,
    high_res_output: int,
    low_res_output: int,
    low_res_threshold: int,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: List[dict] = []
    skipped: List[dict] = []

    for (match_name, half), group in tqdm(labels_df.groupby(["video_id", "half"], sort=False), desc="Videos"):
        key = make_video_key(match_name, int(half))
        source = sources.get(key)
        if source is None:
            print(f"Warning: video source not found for {key}")
            continue

        det_map = detections.get(key, {})
        frames_needed = set()
        for frame_val in group['frame'].astype(int):
            for offset in window:
                frames_needed.add(frame_val + offset)
        frames = load_selected_frames(source, list(frames_needed))
        output_size = high_res_output if max(source.width, source.height) > low_res_threshold else low_res_output

        for _, row in group.iterrows():
            frame_id = int(row['frame'])
            label = int(row["label"])

            (
                images,
                masks,
                empty_count,
                has_valid_detection,
                frame_metadata,
            ) = build_sample(
                frame_id,
                window,
                det_map,
                frames,
                source,
                crop_scale_factor,
                output_size,
            )

            if not has_valid_detection:
                skipped.append(
                    {
                        "video_id": match_name,
                        "half": int(half),
                        "frame": frame_id,
                        "label": label,
                        "reason": "no_ball_detections_in_window",
                    }
                )
                continue
            if not images:
                continue
            if empty_count >= len(window) // 2:
                continue

            cache_name = f"{key}_{frame_id:06d}_{label}"
            cache_path = output_dir / cache_name
            np.save(str(cache_path) + "_s.npy", np.array(images, dtype=np.uint8))

            metadata = {
                "video_id": match_name,
                "half": int(half),
                "frame": frame_id,
                "label": label,
                "window": list(int(x) for x in window),
                "frames": frame_metadata,
            }

            meta_path = Path(str(cache_path) + "_meta.json")
            with meta_path.open("w", encoding="utf-8") as fh:
                json.dump(metadata, fh, ensure_ascii=False, indent=2)

            central_meta = next((fm for fm in frame_metadata if fm["offset"] == 0), None)
            if central_meta is None and frame_metadata:
                central_meta = frame_metadata[len(frame_metadata) // 2]
            ball_meta = central_meta.get("ball") if central_meta else None
            ball_conf = float(ball_meta.get("confidence", float("nan"))) if ball_meta else float("nan")
            velocity = ball_meta.get("velocity") if ball_meta else None
            if isinstance(velocity, (list, tuple)) and len(velocity) == 2:
                ball_vx, ball_vy = float(velocity[0]), float(velocity[1])
            else:
                ball_vx = float("nan")
                ball_vy = float("nan")
            player_count = len(central_meta.get("players", [])) if central_meta else 0

            records.append(
                {
                    "path": str(cache_path),
                    "label": label,
                    "video_id": match_name,
                    "half": int(half),
                    "frame": frame_id,
                    "metadata": str(meta_path),
                    "ball_conf": ball_conf,
                    "ball_vx": ball_vx,
                    "ball_vy": ball_vy,
                    "player_count": int(player_count),
                }
            )

    df = pd.DataFrame(records)
    if not df.empty:
        df.to_csv(output_dir / "train_cache_header.csv", index=False)

    skipped_df = pd.DataFrame(skipped)
    skipped_path = output_dir / "skipped_samples.csv"
    if not skipped_df.empty:
        skipped_df.to_csv(skipped_path, index=False)
        print(f"Skipped {len(skipped_df)} samples due to missing detections (logged to {skipped_path})")
    else:
        if skipped_path.exists():
            skipped_path.unlink()

    return df
