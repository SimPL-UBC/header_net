#!/usr/bin/env python3
"""Build a ball-detection dictionary limited to labelled header-dataset videos."""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Protocol, TYPE_CHECKING

HEADER_NET_ROOT = Path(__file__).resolve().parents[1]
if str(HEADER_NET_ROOT) not in sys.path:
    sys.path.append(str(HEADER_NET_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from utils.labels import canonical_match_name, load_header_labels  # noqa: E402
from utils.detections import make_video_key  # noqa: E402
from configs import header_default as cfg  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from cache import build_ball_det_dict as builder_module  # noqa: F401


LabelHalves = Dict[str, Set[int]]
LabelNames = Dict[str, Set[str]]
LabelSkips = Set[str]


class VideoLike(Protocol):
    video_id: str
    rel_dir: Path
    half: int


def _build_labelled_match_index(
    labels_df: pd.DataFrame,
    allowed_matches: Optional[Set[str]] = None,
) -> Tuple[LabelHalves, LabelNames, LabelSkips]:
    """Return canonical match -> halves and original match names."""
    match_halves: LabelHalves = {}
    name_map: LabelNames = {}
    skipped: LabelSkips = set()

    if labels_df.empty:
        return match_halves, name_map, skipped

    for _, row in labels_df.iterrows():
        raw_name = str(row["video_id"])
        canonical_name = canonical_match_name(raw_name)
        if allowed_matches and canonical_name not in allowed_matches:
            skipped.add(canonical_name)
            continue
        half = int(row.get("half", 1))
        match_halves.setdefault(canonical_name, set()).add(half)
        name_map.setdefault(canonical_name, set()).add(raw_name)

    return match_halves, name_map, skipped


def load_labels_dataframe(header_dataset: Path) -> Tuple[pd.DataFrame, Optional[Path]]:
    """Load header labels, accepting either the dataset root or a direct subdirectory."""
    header_dataset = header_dataset.expanduser()
    if header_dataset.is_file():
        header_dataset = header_dataset.parent

    attempts: List[Path] = []
    current = header_dataset

    while True:
        if current not in attempts:
            attempts.append(current)
        if (current / "SoccerNetV2").exists() or (current / "SoccerDB").exists():
            break
        parent = current.parent
        if parent == current:
            break
        current = parent

    seen: Set[Path] = set()
    for candidate in attempts:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        df = load_header_labels(candidate)
        if not df.empty:
            return df, candidate
    return pd.DataFrame(), None


def build_label_lookup(labels_df: pd.DataFrame) -> Dict[str, Sequence[int]]:
    """Construct a lookup of video key -> labelled frame ids using canonical naming."""
    lookup: Dict[str, Set[int]] = {}
    if labels_df.empty:
        return {}

    for _, row in labels_df.iterrows():
        canonical_name = canonical_match_name(str(row["video_id"]))
        half = int(row.get("half", 1))
        frame = int(row["frame"])
        key = make_video_key(canonical_name, half)
        lookup.setdefault(key, set()).add(frame)

    return {key: sorted(frames) for key, frames in lookup.items()}


def collect_labelled_videos(
    videos: Iterable[VideoLike],
    labels_df: pd.DataFrame,
) -> Tuple[List[VideoLike], Dict[str, Set[int]], LabelNames, LabelSkips]:
    """Filter SoccerNet videos so only labelled matches/halves remain."""
    video_list = list(videos)
    available_match_ids: Set[str] = set()
    for video in video_list:
        video_match = getattr(video, "video_id", "")
        if "_half" in video_match:
            video_match = video_match.rsplit("_half", 1)[0]
        else:
            video_match = canonical_match_name(video.rel_dir.name)
        available_match_ids.add(video_match)

    match_halves, name_map, skipped = _build_labelled_match_index(labels_df, allowed_matches=available_match_ids)
    selected: List[VideoLike] = []
    available_halves: Dict[str, Set[int]] = defaultdict(set)

    if not match_halves:
        return [], {}, name_map, skipped

    for video in video_list:
        video_match = getattr(video, "video_id", "")
        if "_half" in video_match:
            video_match = video_match.rsplit("_half", 1)[0]
        else:
            video_match = canonical_match_name(video.rel_dir.name)

        halves = match_halves.get(video_match)
        if not halves:
            continue
        if video.half in halves:
            selected.append(video)
            available_halves[video_match].add(video.half)

    missing: Dict[str, Set[int]] = {}
    for canonical_match, halves in match_halves.items():
        unmatched = halves - available_halves.get(canonical_match, set())
        if unmatched:
            missing[canonical_match] = unmatched

    return selected, missing, name_map, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ball detection dictionary restricted to labelled videos"
    )
    parser.add_argument(
        "--dataset-path",
        default=str(cfg.DATASET_PATH),
        help="Path to dataset root containing SoccerNet folders",
    )
    parser.add_argument(
        "--yolo-dir",
        default=str(cfg.YOLO_DETECTIONS_PATH),
        help="Directory containing YOLO detection files",
    )
    parser.add_argument(
        "--detector",
        default="yolo",
        choices=["yolo", "rf-detr"],
        help="Detector backend to use for ball localisation",
    )
    parser.add_argument(
        "--detector-weights",
        type=str,
        default=str(cfg.RF_DETR_WEIGHTS),
        help="Checkpoint file for RF-DETR weights (detector=rf-detr)",
    )
    parser.add_argument(
        "--rf-batch-size",
        type=int,
        default=cfg.RF_DETR_BATCH_SIZE,
        help="Batch size for RF-DETR inference",
    )
    parser.add_argument(
        "--rf-score-threshold",
        type=float,
        default=cfg.RF_DETR_SCORE_THRESHOLD,
        help="Score threshold for RF-DETR detections",
    )
    parser.add_argument(
        "--rf-frame-stride",
        type=int,
        default=cfg.RF_DETR_FRAME_STRIDE,
        help="Process every Nth frame with RF-DETR (>=1)",
    )
    parser.add_argument(
        "--rf-device",
        type=str,
        default=str(cfg.RF_DETR_DEVICE)
        if cfg.RF_DETR_DEVICE is not None
        else None,
        help="Torch device for RF-DETR inference (e.g., cuda:0)",
    )
    parser.add_argument(
        "--rf-variant",
        default="medium",
        choices=["nano", "small", "medium", "base", "large"],
        help="RF-DETR variant to instantiate",
    )
    parser.add_argument(
        "--rf-target-classes",
        nargs="+",
        default=["sports ball"],
        help="Class names to keep from RF-DETR predictions (COCO names)",
    )
    parser.add_argument(
        "--rf-optimize",
        dest="rf_optimize",
        action="store_true",
        help="Prepare the RF-DETR model for inference via torch.jit tracing (enabled by default)",
    )
    parser.add_argument(
        "--no-rf-optimize",
        dest="rf_optimize",
        action="store_false",
        help="Skip RF-DETR inference optimisation",
    )
    parser.add_argument(
        "--rf-optimize-batch-size",
        type=int,
        default=1,
        help="Batch size to use when tracing the optimized RF-DETR model",
    )
    parser.add_argument(
        "--rf-optimize-compile",
        action="store_true",
        help="Enable torch.jit compilation during RF-DETR optimization",
    )
    parser.add_argument(
        "--rf-topk",
        type=int,
        default=cfg.RF_DETR_TOPK,
        help="Maximum number of RF-DETR detections to keep per frame",
    )
    parser.add_argument(
        "--header-dataset",
        default=str(cfg.HEADER_DATASET_PATH),
        help="Path to header dataset annotations",
    )
    parser.add_argument(
        "--output",
        default=str(cfg.CACHE_PATH / "ball_det_dict_labelled_only.npy"),
        help="Output path for the ball detection dictionary",
    )
    parser.add_argument(
        "--no-kalman",
        action="store_true",
        help="Disable Kalman smoothing",
    )
    parser.add_argument(
        "--missing-report",
        type=str,
        default=str(cfg.CACHE_PATH / "missing_detections_labelled_only.csv"),
        help="CSV file to log frames without detections",
    )
    parser.set_defaults(rf_optimize=None)
    return parser.parse_args()


def main() -> None:
    from cache import build_ball_det_dict as builder  # Local import to avoid heavy deps at module load

    args = parse_args()
    rf_optimize = True if args.rf_optimize is None else args.rf_optimize

    dataset_path = Path(args.dataset_path)
    det_dir = Path(args.yolo_dir)
    header_dataset = Path(args.header_dataset)
    output_path = Path(args.output)

    videos = builder.discover_videos(dataset_path)
    if not videos:
        raise SystemExit("No videos found under SoccerNet. Check dataset path.")

    labels_df, resolved_header_root = load_labels_dataframe(header_dataset)
    if labels_df.empty:
        raise SystemExit(f"No labels found under header dataset path: {header_dataset}")
    if resolved_header_root and resolved_header_root != header_dataset:
        print(f"[INFO] Resolved header dataset root to {resolved_header_root}")

    filtered_videos, missing_halves, name_map, skipped_matches = collect_labelled_videos(videos, labels_df)
    if not filtered_videos:
        raise SystemExit(
            "No SoccerNet videos align with labelled matches. "
            "Verify header_dataset naming conventions."
        )

    print(f"[INFO] Processing dataset root: {dataset_path.resolve()}")
    if resolved_header_root:
        print(f"[INFO] Header annotations from: {resolved_header_root.resolve()}")
    else:
        print(f"[INFO] Header annotations from: {header_dataset.resolve()}")

    match_to_paths: Dict[str, Dict[int, Path]] = {}
    for video in sorted(filtered_videos, key=lambda v: (v.video_id, v.half)):
        match_id = video.video_id.rsplit("_half", 1)[0]
        match_to_paths.setdefault(match_id, {})[video.half] = video.path

    print(f"[INFO] Labelled matches to process: {len(match_to_paths)}")
    for match_id in sorted(match_to_paths):
        paths = match_to_paths[match_id]
        print(f"  {match_id}")
        for half in sorted(paths):
            print(f"    half{half}: {paths[half]}")

    if skipped_matches:
        preview = ", ".join(sorted(skipped_matches)[:10])
        suffix = " ..." if len(skipped_matches) > 10 else ""
        print(f"[INFO] Ignoring labels with no matching SoccerNet video: {preview}{suffix}")

    if missing_halves:
        print("[WARN] Missing video files for labelled matches:")
        for canonical_name, halves in sorted(missing_halves.items()):
            label_names = ", ".join(sorted(name_map.get(canonical_name, {canonical_name})))
            half_list = ", ".join(str(h) for h in sorted(halves))
            print(
                f"  {label_names}: missing half(s) {half_list}"
            )

    allowed_matches = set(name_map.keys())
    filtered_labels_df = labels_df[
        labels_df["video_id"].map(lambda x: canonical_match_name(str(x)) in allowed_matches)
    ].copy()
    label_lookup = build_label_lookup(filtered_labels_df)

    rf_inference: Optional[builder.RFDetrInference] = None
    if args.detector == "rf-detr":
        if args.rf_frame_stride < 1:
            raise ValueError("rf-frame-stride must be >= 1")
        print("Initialising RF-DETR detector")
        requested_weight = Path(args.detector_weights).expanduser() if args.detector_weights else None
        resolved_weight: Optional[str] = None
        if requested_weight:
            if requested_weight.exists():
                resolved_weight = str(requested_weight)
            else:
                alt_suffix = ".pth" if requested_weight.suffix == ".pt" else ".pt"
                alt_candidate = requested_weight.with_suffix(alt_suffix)
                if alt_candidate.exists():
                    print(f"RF-DETR weights not found at {requested_weight}; using {alt_candidate} instead")
                    resolved_weight = str(alt_candidate)
                else:
                    hosted_name = requested_weight.name
                    print(
                        f"RF-DETR weights not found at {requested_weight}; "
                        f"falling back to hosted name '{hosted_name}' if available"
                    )
                    resolved_weight = hosted_name
        target_names = tuple(args.rf_target_classes) if args.rf_target_classes else ()
        if any(name.lower() == "all" for name in target_names):
            target_names = ()
        rf_config = builder.RFDetrConfig(
            variant=args.rf_variant,
            weights_path=resolved_weight,
            device=args.rf_device,
            target_class_names=target_names,
            optimize=rf_optimize,
            optimize_batch_size=args.rf_optimize_batch_size,
            optimize_compile=args.rf_optimize_compile,
        )
        model = builder.build_rf_detr(rf_config)
        rf_inference = builder.RFDetrInference(model, rf_config)
        print(f"RF-DETR running on {rf_inference.device}")

    detections, missing_records = builder.build_ball_detection_dict(
        videos=filtered_videos,
        det_dir=det_dir,
        label_lookup=label_lookup,
        use_kalman=not args.no_kalman,
        detector=args.detector,
        rf_inference=rf_inference,
        rf_batch_size=args.rf_batch_size,
        rf_score_threshold=args.rf_score_threshold,
        rf_frame_stride=args.rf_frame_stride,
        rf_topk=args.rf_topk,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, detections)
    print(f"Saved detections for {len(detections)} videos to {output_path}")

    if missing_records:
        missing_path = Path(args.missing_report)
        missing_path.parent.mkdir(parents=True, exist_ok=True)
        missing_df = pd.DataFrame(missing_records)
        missing_df.sort_values(["video_id", "frame"], inplace=True)
        missing_df.to_csv(missing_path, index=False)
        print(f"Logged {len(missing_records)} missing detections to {missing_path}")
    else:
        print("All labelled frames had corresponding detections.")


if __name__ == "__main__":
    main()
