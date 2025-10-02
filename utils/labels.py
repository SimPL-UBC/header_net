"""Utilities for loading header annotations."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List

import pandas as pd


def _extract_numeric_frames(df: pd.DataFrame) -> List[int]:
    """Return numeric frame ids from a dataframe."""
    frame_cols = [col for col in df.columns if "frame" in str(col).lower()]
    candidates: Iterable[pd.Series]

    if frame_cols:
        candidates = (df[col] for col in frame_cols)
    else:
        candidates = (df.iloc[:, 0],)

    frames: List[int] = []
    for series in candidates:
        if series is None:
            continue
        series = pd.to_numeric(series, errors="coerce")
        frames.extend(series.dropna().astype(int).tolist())

    return frames


def _infer_half_from_name(name: str) -> int:
    match = re.search(r"(^|[^0-9])([12])([^0-9]|$)", name)
    if match:
        return int(match.group(2))
    return 1


def load_header_labels(header_dataset_root: Path) -> pd.DataFrame:
    """Load header annotations into a single dataframe.

    Returns a dataframe with at least the columns ``video_id``, ``half``, and
    ``frame``. The ``video_id`` uses the match directory (SoccerNetV2) or the
    file stem (SoccerDB) so downstream consumers can match by substring.
    """
    header_dataset_root = header_dataset_root.expanduser()
    entries: List[dict] = []

    soccernet_root = header_dataset_root / "SoccerNetV2"
    if soccernet_root.exists():
        for match_dir in soccernet_root.iterdir():
            if not match_dir.is_dir():
                continue
            video_id = match_dir.name
            for file_path in match_dir.glob("*.*"):
                if file_path.suffix.lower() not in {".xlsx", ".ods", ".csv"}:
                    continue
                try:
                    if file_path.suffix.lower() == ".csv":
                        df = pd.read_csv(file_path)
                    elif file_path.suffix.lower() == ".ods":
                        df = pd.read_excel(file_path, engine="odf")
                    else:
                        df = pd.read_excel(file_path)
                except Exception:
                    continue

                frames = _extract_numeric_frames(df)
                if not frames:
                    continue

                half = _infer_half_from_name(file_path.stem)
                for frame in frames:
                    entries.append(
                        {
                            "video_id": video_id,
                            "half": half,
                            "frame": int(frame),
                            "label": 1,
                        }
                    )

    soccerdb_root = header_dataset_root / "SoccerDB"
    if soccerdb_root.exists():
        for file_path in soccerdb_root.glob("*.*"):
            if file_path.suffix.lower() not in {".xlsx", ".csv"}:
                continue
            try:
                if file_path.suffix.lower() == ".csv":
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path)
            except Exception:
                continue

            frames = _extract_numeric_frames(df)
            if not frames:
                continue

            video_id = file_path.stem.replace("_framed", "")
            for frame in frames:
                entries.append(
                    {
                        "video_id": video_id,
                        "half": 1,
                        "frame": int(frame),
                        "label": 1,
                    }
                )

    df = pd.DataFrame(entries)
    if not df.empty:
        df.drop_duplicates(subset=["video_id", "half", "frame"], inplace=True)
        df.sort_values(["video_id", "half", "frame"], inplace=True)
        df.reset_index(drop=True, inplace=True)

    return df


def build_half_frame_lookup(df: pd.DataFrame) -> dict:
    lookup: dict[str, set[int]] = {}
    if df.empty:
        return lookup
    for _, row in df.iterrows():
        video_id = str(row['video_id'])
        half = int(row.get('half', 1))
        frame = int(row['frame'])
        key = f"{video_id}_half{half}"
        lookup.setdefault(key, set()).add(frame)
    return lookup


def canonical_match_name(name: str) -> str:
    return ''.join(ch for ch in name if ch.isalnum() or ch in '-_')
