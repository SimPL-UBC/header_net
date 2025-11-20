#!/usr/bin/env python3
"""Generate short uncropped video clips from header dataset annotations."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Set, Tuple

import cv2
from tqdm import tqdm

HEADER_NET_ROOT = Path(__file__).resolve().parents[1]
if str(HEADER_NET_ROOT) not in sys.path:
    sys.path.append(str(HEADER_NET_ROOT))

from utils.detections import make_video_key  # noqa: E402
from utils.labels import canonical_match_name, load_header_labels  # noqa: E402
from utils.videos import infer_half_from_stem  # noqa: E402


@dataclass
class VideoSource:
    match_name: str
    display_name: str
    half: int
    key: str
    path: Path
    frame_count: int
    width: int
    height: int
    fps: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create uncropped verification clips aligned with header dataset labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "header_dataset_path",
        help="Path to header dataset labels (e.g. DeepImpact/header_dataset/SoccerNetV2)",
    )
    parser.add_argument(
        "video_path",
        help="Path to raw SoccerNet videos root (either DeepImpact/ or DeepImpact/SoccerNet)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=str(HEADER_NET_ROOT / "outputs" / "uncropped_clips"),
        help="Where to write generated mp4 clips",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=11,
        help="Number of consecutive frames to include per clip",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate clips even if the output file already exists",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process at most N labelled frames (useful for smoke tests)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable progress bars",
    )
    return parser.parse_args()


def resolve_header_root(path: Path) -> Path:
    path = path.expanduser()
    if path.is_file():
        path = path.parent
    if path.name in {"SoccerNetV2", "SoccerDB"}:
        return path.parent
    return path


def resolve_video_roots(path: Path) -> Tuple[Path, Path]:
    """Return (dataset_root, soccer_root)."""
    path = path.expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Video path not found: {path}")

    if path.is_file():
        raise ValueError(f"Expected a directory for raw videos, received file: {path}")

    if (path / "SoccerNet").is_dir():
        return path, path / "SoccerNet"

    if path.name == "SoccerNet":
        return path.parent, path

    # Attempt to walk upwards until we locate a SoccerNet directory
    current = path
    while current != current.parent:
        candidate = current / "SoccerNet"
        if candidate.is_dir():
            return current, candidate
        current = current.parent

    raise FileNotFoundError(
        f"Unable to locate a 'SoccerNet' directory relative to {path}"
    )


def build_video_index(soccer_root: Path, matches: Set[str]) -> Dict[str, VideoSource]:
    """Create a lookup of match/half -> video metadata."""
    sources: Dict[str, VideoSource] = {}
    matches = set(matches)

    def consider_video(
        video_file: Path, canonical_match: str, display_name: str
    ) -> None:
        half = infer_half_from_stem(video_file.stem)
        key = make_video_key(canonical_match, half)

        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            print(f"Warning: unable to open video {video_file}", file=sys.stderr)
            return
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        cap.release()

        if frame_count <= 0 or width <= 0 or height <= 0:
            print(
                f"Warning: skipping invalid video metadata for {video_file}",
                file=sys.stderr,
            )
            return

        existing = sources.get(key)
        if existing:
            existing_pixels = existing.width * existing.height
            candidate_pixels = width * height
            if candidate_pixels <= existing_pixels:
                return

        sources[key] = VideoSource(
            match_name=canonical_match,
            display_name=display_name,
            half=half,
            key=key,
            path=video_file,
            frame_count=frame_count,
            width=width,
            height=height,
            fps=fps if fps > 0 else 25.0,
        )

    for league_dir in soccer_root.iterdir():
        if not league_dir.is_dir():
            continue
        for season_dir in league_dir.iterdir():
            if not season_dir.is_dir():
                continue
            for match_dir in season_dir.iterdir():
                if not match_dir.is_dir():
                    continue
                display_name = match_dir.name
                canonical_match = canonical_match_name(display_name)
                if matches and canonical_match not in matches:
                    continue
                for video_file in match_dir.iterdir():
                    if not video_file.is_file():
                        continue
                    if video_file.suffix.lower() not in {".mp4", ".mkv"}:
                        continue
                    consider_video(video_file, canonical_match, display_name)
    return sources


def annotate_frame(
    frame: "cv2.typing.MatLike", absolute_frame: int, offset: int
) -> None:
    text = f"frame:{{{absolute_frame}, {offset}}}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    color = (255, 255, 255)
    shadow_color = (0, 0, 0)

    text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
    x = max(5, frame.shape[1] - text_size[0] - 10)
    y = max(5 + text_size[1], 10 + baseline)
    top_left = (x - 6, y - text_size[1] - 6)
    bottom_right = (x + text_size[0] + 6, y + 6)

    cv2.rectangle(frame, top_left, bottom_right, shadow_color, thickness=-1)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def extract_clip(
    cap: cv2.VideoCapture,
    start_frame: int,
    num_frames: int,
    frame_count: int,
) -> Optional[Sequence["cv2.typing.MatLike"]]:
    clip_frames = []
    for offset in range(num_frames):
        target = start_frame + offset
        if target < 0 or target >= frame_count:
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ret, frame = cap.read()
        if not ret or frame is None:
            return None
        frame_disp = frame.copy()
        annotate_frame(frame_disp, target, offset)
        clip_frames.append(frame_disp)
    return clip_frames


def main() -> None:
    args = parse_args()

    header_root = resolve_header_root(Path(args.header_dataset_path))
    df = load_header_labels(header_root)
    if df.empty:
        print(
            "No header annotations found. Please verify the dataset path.",
            file=sys.stderr,
        )
        sys.exit(1)

    df = df.copy()
    df["frame"] = df["frame"].astype(int)
    df["half"] = df["half"].fillna(1).astype(int)
    df["canonical_match"] = df["video_id"].apply(
        lambda name: canonical_match_name(str(name))
    )
    df["video_key"] = df.apply(
        lambda row: make_video_key(row["canonical_match"], row["half"]), axis=1
    )

    if args.limit is not None and args.limit > 0:
        df = df.head(args.limit)

    _, soccer_root = resolve_video_roots(Path(args.video_path))
    required_matches = set(df["canonical_match"].unique())
    video_index = build_video_index(soccer_root, required_matches)

    missing_sources = sorted(set(df["video_key"].unique()) - set(video_index.keys()))
    if missing_sources:
        print(
            "Warning: missing videos for the following match halves:", file=sys.stderr
        )
        for key in missing_sources:
            print(f"  {key}", file=sys.stderr)

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped = df.groupby(["canonical_match", "half"])
    progress_iter = (
        grouped if args.quiet else tqdm(grouped, desc="Matches", unit="match")
    )

    clips_written = 0
    skipped_existing = 0
    skipped_missing_video = 0
    skipped_out_of_range = 0
    skipped_decode = 0

    for (canonical_match, half), group in progress_iter:
        key = make_video_key(canonical_match, half)
        source = video_index.get(key)
        if source is None:
            skipped_missing_video += len(group)
            continue

        display_name = str(group.iloc[0]["video_id"])
        match_dir = output_dir / display_name / f"half{half}"
        match_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(source.path))
        if not cap.isOpened():
            print(f"Warning: unable to read video {source.path}", file=sys.stderr)
            skipped_decode += len(group)
            cap.release()
            continue

        frames_iter = group.sort_values("frame").itertuples(index=False)
        frame_iterable = list(frames_iter)

        per_match_iter = (
            frame_iterable
            if args.quiet
            else tqdm(frame_iterable, desc=display_name, leave=False)
        )

        for row in per_match_iter:
            frame_id = int(row.frame)
            output_path = match_dir / f"frame_{frame_id:06d}.mp4"

            if output_path.exists() and not args.overwrite:
                skipped_existing += 1
                continue

            end_frame = frame_id + args.num_frames - 1
            if end_frame >= source.frame_count:
                skipped_out_of_range += 1
                continue

            clip_frames = extract_clip(
                cap, frame_id, args.num_frames, source.frame_count
            )
            if clip_frames is None:
                skipped_decode += 1
                continue

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = source.fps if source.fps > 0 else 25.0
            writer = cv2.VideoWriter(
                str(output_path), fourcc, fps, (source.width, source.height)
            )
            if not writer.isOpened():
                print(f"Warning: unable to create clip {output_path}", file=sys.stderr)
                skipped_decode += 1
                writer.release()
                continue
            for frame in clip_frames:
                writer.write(frame)
            writer.release()
            clips_written += 1

        cap.release()

    summary = (
        f"Finished. Clips written: {clips_written}. "
        f"Skipped (existing): {skipped_existing}, "
        f"missing video: {skipped_missing_video}, "
        f"out of range: {skipped_out_of_range}, "
        f"decode errors: {skipped_decode}."
    )
    print(summary)


if __name__ == "__main__":
    main()
