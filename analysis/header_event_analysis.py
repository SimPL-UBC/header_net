from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Iterable

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

DEFAULT_INFERENCE_ROOT = Path("output") / "2026_03_31_ball-frame_vmae2-base_inference_result"
DEFAULT_LABEL_ROOT = Path("SoccerNet") / "test" / "labelled_header"
INFERENCE_REQUIRED_COLUMNS = [
    "video_id",
    "half",
    "frame",
    "label",
    "prob_header",
]
LABEL_FRAME_COLUMN = "Frame Number"


@dataclass(frozen=True)
class EventMatchResult:
    pred_events: pd.DataFrame
    gt_events: pd.DataFrame
    matches: pd.DataFrame
    metrics: dict[str, float | int]


def resolve_inference_per_match_dir(
    inference_folder: str | Path,
    repo_root: str | Path | None = None,
) -> Path:
    repo_root_path = Path.cwd() if repo_root is None else Path(repo_root).expanduser().resolve()
    raw = Path(inference_folder)

    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.extend([raw, raw / "per_match"])
    else:
        candidates.extend(
            [
                repo_root_path / raw,
                repo_root_path / raw / "per_match",
                repo_root_path / DEFAULT_INFERENCE_ROOT / raw,
                repo_root_path / DEFAULT_INFERENCE_ROOT / raw / "per_match",
            ]
        )

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_dir() and candidate.name == "per_match":
            return candidate
        if candidate.is_dir() and any(candidate.glob("*.csv")):
            return candidate
        if candidate.is_dir() and (candidate / "per_match").is_dir():
            return candidate / "per_match"

    raise FileNotFoundError(
        f"Could not resolve inference folder {inference_folder!r} relative to {repo_root_path}."
    )


def load_inference_folder(
    inference_folder: str | Path,
    repo_root: str | Path | None = None,
) -> pd.DataFrame:
    per_match_dir = resolve_inference_per_match_dir(inference_folder, repo_root=repo_root)
    csv_paths = sorted(per_match_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No per-match CSVs found in {per_match_dir}")

    frames = [pd.read_csv(path) for path in csv_paths]
    inference_df = pd.concat(frames, ignore_index=True)
    missing = [column for column in INFERENCE_REQUIRED_COLUMNS if column not in inference_df.columns]
    if missing:
        raise ValueError(f"Inference CSVs are missing required columns: {missing}")

    inference_df = inference_df.copy()
    inference_df["video_id"] = inference_df["video_id"].astype(str)
    inference_df["half"] = pd.to_numeric(inference_df["half"], errors="raise").astype(int)
    inference_df["frame"] = pd.to_numeric(inference_df["frame"], errors="raise").astype(int)
    inference_df["label"] = pd.to_numeric(inference_df["label"], errors="raise").astype(int)
    inference_df["prob_header"] = pd.to_numeric(
        inference_df["prob_header"], errors="raise"
    ).astype(float)
    inference_df.sort_values(["video_id", "half", "frame"], inplace=True)
    inference_df.reset_index(drop=True, inplace=True)
    inference_df.attrs["per_match_dir"] = str(per_match_dir)
    return inference_df


def _infer_half_from_filename(path: Path) -> int:
    match = re.match(r"(\d+)", path.stem)
    if not match:
        raise ValueError(f"Could not infer half from label file name: {path.name}")
    half = int(match.group(1))
    if half not in {1, 2}:
        raise ValueError(f"Unsupported half value {half} in label file {path.name}")
    return half


def load_labelled_headers(label_dir: str | Path) -> pd.DataFrame:
    label_root = Path(label_dir).expanduser().resolve()
    if not label_root.exists():
        raise FileNotFoundError(f"Label directory not found: {label_root}")

    rows: list[dict[str, Any]] = []
    for match_dir in sorted(path for path in label_root.iterdir() if path.is_dir()):
        for label_path in sorted(match_dir.glob("*.xlsx")):
            label_df = pd.read_excel(label_path)
            if LABEL_FRAME_COLUMN not in label_df.columns:
                raise ValueError(
                    f"Label file {label_path} is missing required column {LABEL_FRAME_COLUMN!r}"
                )
            frame_numbers = pd.to_numeric(
                label_df[LABEL_FRAME_COLUMN], errors="coerce"
            ).dropna().astype(int)
            half = _infer_half_from_filename(label_path)
            for frame_number in frame_numbers.tolist():
                rows.append(
                    {
                        "video_id": match_dir.name,
                        "half": half,
                        "gt_frame": int(frame_number),
                        "label_path": str(label_path),
                    }
                )

    gt_df = pd.DataFrame(rows)
    if gt_df.empty:
        raise ValueError(f"No label rows found under {label_root}")

    gt_df.sort_values(["video_id", "half", "gt_frame"], inplace=True)
    gt_df.reset_index(drop=True, inplace=True)
    gt_df.attrs["label_dir"] = str(label_root)
    return gt_df


def dedupe_ground_truth_headers(gt_df: pd.DataFrame) -> pd.DataFrame:
    required = ["video_id", "half", "gt_frame"]
    missing = [column for column in required if column not in gt_df.columns]
    if missing:
        raise ValueError(f"Ground-truth dataframe is missing required columns: {missing}")

    deduped = gt_df.copy()
    deduped["video_id"] = deduped["video_id"].astype(str)
    deduped["half"] = pd.to_numeric(deduped["half"], errors="raise").astype(int)
    deduped["gt_frame"] = pd.to_numeric(deduped["gt_frame"], errors="raise").astype(int)
    deduped = deduped.drop_duplicates(subset=["video_id", "half", "gt_frame"], keep="first")
    deduped.sort_values(["video_id", "half", "gt_frame"], inplace=True)
    deduped.reset_index(drop=True, inplace=True)
    deduped["gt_event_id"] = np.arange(len(deduped), dtype=int)
    return deduped


def filter_rows_by_frame_windows(
    df: pd.DataFrame,
    frame_windows: dict[tuple[str, int], tuple[int, int]] | None,
    frame_column: str,
) -> pd.DataFrame:
    if not frame_windows:
        return df.copy()
    if frame_column not in df.columns:
        raise ValueError(f"Dataframe is missing frame column {frame_column!r}.")
    for column in ["video_id", "half"]:
        if column not in df.columns:
            raise ValueError(f"Dataframe is missing required column {column!r}.")

    filtered = df.copy()
    video_ids = filtered["video_id"].astype(str)
    halves = pd.to_numeric(filtered["half"], errors="raise").astype(int)
    frames = pd.to_numeric(filtered[frame_column], errors="raise").astype(int)
    keep_mask = pd.Series(True, index=filtered.index)

    for (video_id, half), (start_frame, end_frame) in frame_windows.items():
        if start_frame > end_frame:
            raise ValueError(
                f"Invalid frame window for {(video_id, half)}: start {start_frame} > end {end_frame}"
            )
        match_mask = (video_ids == str(video_id)) & (halves == int(half))
        keep_mask.loc[match_mask] = frames.loc[match_mask].between(
            int(start_frame), int(end_frame), inclusive="both"
        )

    filtered = filtered.loc[keep_mask].copy()
    filtered.attrs.update(df.attrs)
    return filtered


def cluster_predictions(
    inference_df: pd.DataFrame,
    threshold: float,
    merge_gap: int,
) -> pd.DataFrame:
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be in [0, 1].")
    if merge_gap < 0:
        raise ValueError("merge_gap must be >= 0.")

    positives = inference_df.loc[inference_df["prob_header"] >= float(threshold)].copy()
    if positives.empty:
        return pd.DataFrame(
            columns=[
                "pred_event_id",
                "video_id",
                "half",
                "start_frame",
                "end_frame",
                "peak_frame",
                "peak_prob",
                "n_positive_frames",
                "positive_frames",
            ]
        )

    positives.sort_values(["video_id", "half", "frame"], inplace=True)
    event_rows: list[dict[str, Any]] = []
    event_id = 0

    for (video_id, half), group in positives.groupby(["video_id", "half"], sort=False):
        group = group.sort_values("frame").copy()
        cluster_break = group["frame"].diff().gt(merge_gap).fillna(True)
        group["cluster_id"] = cluster_break.cumsum().astype(int)

        for _, cluster in group.groupby("cluster_id", sort=False):
            peak_idx = cluster["prob_header"].idxmax()
            peak_row = cluster.loc[peak_idx]
            positive_frames = tuple(int(frame) for frame in cluster["frame"].tolist())
            event_rows.append(
                {
                    "pred_event_id": event_id,
                    "video_id": str(video_id),
                    "half": int(half),
                    "start_frame": int(cluster["frame"].min()),
                    "end_frame": int(cluster["frame"].max()),
                    "peak_frame": int(peak_row["frame"]),
                    "peak_prob": float(peak_row["prob_header"]),
                    "n_positive_frames": int(len(cluster)),
                    "positive_frames": positive_frames,
                }
            )
            event_id += 1

    pred_events = pd.DataFrame(event_rows)
    pred_events.sort_values(["video_id", "half", "peak_frame"], inplace=True)
    pred_events.reset_index(drop=True, inplace=True)
    return pred_events


def _candidate_distance(positive_frames: Iterable[int], gt_frame: int) -> int:
    frame_array = np.asarray(list(positive_frames), dtype=int)
    return int(np.min(np.abs(frame_array - int(gt_frame))))


def _matches_ground_truth(
    positive_frames: Iterable[int],
    gt_frame: int,
    tol_left: int,
    tol_right: int,
) -> bool:
    frame_array = np.asarray(list(positive_frames), dtype=int)
    deltas = frame_array - int(gt_frame)
    return bool(np.any((deltas >= -tol_left) & (deltas <= tol_right)))


def match_events(
    pred_events: pd.DataFrame,
    gt_events: pd.DataFrame,
    tol_left: int = 7,
    tol_right: int = 8,
) -> EventMatchResult:
    if tol_left < 0 or tol_right < 0:
        raise ValueError("tol_left and tol_right must be >= 0.")

    pred_result = pred_events.copy()
    gt_result = gt_events.copy()

    if "pred_event_id" not in pred_result.columns:
        pred_result = pred_result.reset_index(drop=True)
        pred_result["pred_event_id"] = np.arange(len(pred_result), dtype=int)
    if "gt_event_id" not in gt_result.columns:
        gt_result = dedupe_ground_truth_headers(gt_result)

    pred_result["matched_gt_event_id"] = pd.Series([pd.NA] * len(pred_result), dtype="Int64")
    pred_result["match_status"] = "fp"
    gt_result["matched_pred_event_id"] = pd.Series([pd.NA] * len(gt_result), dtype="Int64")
    gt_result["match_status"] = "fn"

    matched_gt_ids: set[int] = set()
    match_rows: list[dict[str, Any]] = []
    match_columns = [
        "pred_event_id",
        "gt_event_id",
        "video_id",
        "half",
        "peak_frame",
        "peak_prob",
        "gt_frame",
        "frame_distance",
    ]

    gt_by_group: dict[tuple[str, int], list[pd.Series]] = {}
    for _, gt_row in gt_result.iterrows():
        key = (str(gt_row["video_id"]), int(gt_row["half"]))
        gt_by_group.setdefault(key, []).append(gt_row)

    sorted_pred = pred_result.sort_values(
        ["peak_prob", "video_id", "half", "peak_frame"],
        ascending=[False, True, True, True],
    )

    for pred_index, pred_row in sorted_pred.iterrows():
        key = (str(pred_row["video_id"]), int(pred_row["half"]))
        candidates: list[pd.Series] = []
        for gt_row in gt_by_group.get(key, []):
            gt_event_id = int(gt_row["gt_event_id"])
            if gt_event_id in matched_gt_ids:
                continue
            if _matches_ground_truth(pred_row["positive_frames"], gt_row["gt_frame"], tol_left, tol_right):
                candidates.append(gt_row)

        if not candidates:
            continue

        matched_gt = min(
            candidates,
            key=lambda row: (
                _candidate_distance(pred_row["positive_frames"], int(row["gt_frame"])),
                abs(int(pred_row["peak_frame"]) - int(row["gt_frame"])),
                int(row["gt_frame"]),
            ),
        )

        gt_event_id = int(matched_gt["gt_event_id"])
        pred_event_id = int(pred_row["pred_event_id"])
        matched_gt_ids.add(gt_event_id)
        pred_result.at[pred_index, "matched_gt_event_id"] = gt_event_id
        pred_result.at[pred_index, "match_status"] = "tp"

        gt_index = gt_result.index[gt_result["gt_event_id"] == gt_event_id][0]
        gt_result.at[gt_index, "matched_pred_event_id"] = pred_event_id
        gt_result.at[gt_index, "match_status"] = "tp"

        match_rows.append(
            {
                "pred_event_id": pred_event_id,
                "gt_event_id": gt_event_id,
                "video_id": str(pred_row["video_id"]),
                "half": int(pred_row["half"]),
                "peak_frame": int(pred_row["peak_frame"]),
                "peak_prob": float(pred_row["peak_prob"]),
                "gt_frame": int(matched_gt["gt_frame"]),
                "frame_distance": abs(int(pred_row["peak_frame"]) - int(matched_gt["gt_frame"])),
            }
        )

    matches = pd.DataFrame(match_rows, columns=match_columns)
    tp = int((pred_result["match_status"] == "tp").sum())
    fp = int((pred_result["match_status"] == "fp").sum())
    fn = int((gt_result["match_status"] == "fn").sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    metrics = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "pred_event_count": int(len(pred_result)),
        "gt_event_count": int(len(gt_result)),
        "tol_left": int(tol_left),
        "tol_right": int(tol_right),
    }
    return EventMatchResult(
        pred_events=pred_result.sort_values(["video_id", "half", "peak_frame"]).reset_index(drop=True),
        gt_events=gt_result.sort_values(["video_id", "half", "gt_frame"]).reset_index(drop=True),
        matches=matches.sort_values(["video_id", "half", "gt_frame"]).reset_index(drop=True),
        metrics=metrics,
    )


def evaluate_event_detection(
    inference_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    threshold: float,
    merge_gap: int = 3,
    tol_left: int = 7,
    tol_right: int = 8,
) -> EventMatchResult:
    gt_events = dedupe_ground_truth_headers(gt_df)
    pred_events = cluster_predictions(inference_df, threshold=threshold, merge_gap=merge_gap)
    result = match_events(pred_events, gt_events, tol_left=tol_left, tol_right=tol_right)

    metrics = dict(result.metrics)
    metrics["threshold"] = float(threshold)
    metrics["merge_gap"] = int(merge_gap)

    return EventMatchResult(
        pred_events=result.pred_events,
        gt_events=result.gt_events,
        matches=result.matches,
        metrics=metrics,
    )


def summarize_event_metrics_by_video(result: EventMatchResult) -> pd.DataFrame:
    pred_summary = (
        result.pred_events.assign(
            tp=lambda df: (df["match_status"] == "tp").astype(int),
            fp=lambda df: (df["match_status"] == "fp").astype(int),
        )
        .groupby(["video_id", "half"], as_index=False)[["tp", "fp"]]
        .sum()
    )
    gt_summary = (
        result.gt_events.assign(fn=lambda df: (df["match_status"] == "fn").astype(int))
        .groupby(["video_id", "half"], as_index=False)[["fn"]]
        .sum()
    )

    summary = pred_summary.merge(gt_summary, on=["video_id", "half"], how="outer").fillna(0)
    for column in ["tp", "fp", "fn"]:
        summary[column] = summary[column].astype(int)
    summary["precision"] = np.where(
        (summary["tp"] + summary["fp"]) > 0,
        summary["tp"] / (summary["tp"] + summary["fp"]),
        0.0,
    )
    summary["recall"] = np.where(
        (summary["tp"] + summary["fn"]) > 0,
        summary["tp"] / (summary["tp"] + summary["fn"]),
        0.0,
    )
    summary.sort_values(["video_id", "half"], inplace=True)
    summary.reset_index(drop=True, inplace=True)
    return summary


def compute_event_metrics_curve(
    inference_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    thresholds: Iterable[float],
    merge_gap: int = 3,
    tol_left: int = 7,
    tol_right: int = 8,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        result = evaluate_event_detection(
            inference_df,
            gt_df,
            threshold=float(threshold),
            merge_gap=merge_gap,
            tol_left=tol_left,
            tol_right=tol_right,
        )
        row = dict(result.metrics)
        row["threshold"] = float(threshold)
        rows.append(row)

    curve_df = pd.DataFrame(rows)
    curve_df.sort_values("threshold", inplace=True)
    curve_df.reset_index(drop=True, inplace=True)
    return curve_df


def plot_match_timeline(
    inference_df: pd.DataFrame,
    result: EventMatchResult,
    match_id: str,
    half: int,
    threshold: float,
    merge_gap: int,
    tol_left: int = 7,
    tol_right: int = 8,
    frame_window: tuple[int, int] | None = None,
    ax: Axes | None = None,
) -> Axes:
    subset = inference_df[
        (inference_df["video_id"].astype(str) == str(match_id))
        & (pd.to_numeric(inference_df["half"], errors="coerce").astype(int) == int(half))
    ].copy()
    if subset.empty:
        raise ValueError(f"No inference rows found for match={match_id!r}, half={half}.")

    pred_subset = result.pred_events[
        (result.pred_events["video_id"].astype(str) == str(match_id))
        & (pd.to_numeric(result.pred_events["half"], errors="coerce").astype(int) == int(half))
    ].copy()
    gt_subset = result.gt_events[
        (result.gt_events["video_id"].astype(str) == str(match_id))
        & (pd.to_numeric(result.gt_events["half"], errors="coerce").astype(int) == int(half))
    ].copy()

    if frame_window is not None:
        start_frame, end_frame = frame_window
        subset = subset[(subset["frame"] >= int(start_frame)) & (subset["frame"] <= int(end_frame))]
        pred_subset = pred_subset[
            (pred_subset["end_frame"] >= int(start_frame))
            & (pred_subset["start_frame"] <= int(end_frame))
        ]
        gt_subset = gt_subset[
            (gt_subset["gt_frame"] >= int(start_frame) - tol_right)
            & (gt_subset["gt_frame"] <= int(end_frame) + tol_left)
        ]

    if ax is None:
        _, ax = plt.subplots(figsize=(16, 5))

    ax.plot(
        subset["frame"],
        subset["prob_header"],
        color="#1f77b4",
        linewidth=1.0,
        label="P(header)",
    )
    ax.axhline(float(threshold), color="#7f7f7f", linestyle="--", linewidth=1.0, label="Threshold")

    gt_label_drawn = set()
    for gt_row in gt_subset.itertuples(index=False):
        color = "#2ca02c" if gt_row.match_status == "tp" else "#f0ad4e"
        label = "GT window (matched)" if gt_row.match_status == "tp" else "GT window (missed)"
        draw_label = label not in gt_label_drawn
        ax.axvspan(
            int(gt_row.gt_frame) - tol_left,
            int(gt_row.gt_frame) + tol_right,
            color=color,
            alpha=0.16,
            label=label if draw_label else None,
        )
        gt_label_drawn.add(label)

    pred_label_drawn = set()
    for pred_row in pred_subset.itertuples(index=False):
        color = "#1b9e77" if pred_row.match_status == "tp" else "#d62728"
        label = "Pred event (TP)" if pred_row.match_status == "tp" else "Pred event (FP)"
        draw_label = label not in pred_label_drawn
        ax.axvspan(
            int(pred_row.start_frame),
            int(pred_row.end_frame),
            color=color,
            alpha=0.18,
            label=label if draw_label else None,
        )
        ax.scatter(
            [int(pred_row.peak_frame)],
            [float(pred_row.peak_prob)],
            color=color,
            edgecolor="black",
            linewidth=0.5,
            s=32,
            zorder=3,
        )
        pred_label_drawn.add(label)

    x_min = int(subset["frame"].min())
    x_max = int(subset["frame"].max())
    if frame_window is not None:
        x_min, x_max = int(frame_window[0]), int(frame_window[1])
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Header probability")
    ax.set_title(
        f"{match_id} half {half} | threshold={threshold:.2f}, merge_gap={merge_gap}"
    )
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    return ax


def plot_event_metrics_vs_threshold(
    curve_df: pd.DataFrame,
    current_threshold: float | None = None,
    ax: Axes | None = None,
) -> Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    ax.plot(curve_df["threshold"], curve_df["precision"], label="Precision", color="#1b9e77")
    ax.plot(curve_df["threshold"], curve_df["recall"], label="Recall", color="#d95f02")
    if current_threshold is not None:
        ax.axvline(
            float(current_threshold),
            color="#7f7f7f",
            linestyle="--",
            linewidth=1.0,
            label=f"Current threshold = {float(current_threshold):.2f}",
        )
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Event-Level Precision and Recall vs Threshold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_frame_roc_curve(
    inference_df: pd.DataFrame,
    ax: Axes | None = None,
) -> float:
    labels = pd.to_numeric(inference_df["label"], errors="raise").astype(int).to_numpy()
    scores = pd.to_numeric(inference_df["prob_header"], errors="raise").astype(float).to_numpy()
    if np.unique(labels).size < 2:
        raise ValueError("Frame-level ROC requires both positive and negative labels.")

    fpr, tpr, _ = roc_curve(labels, scores)
    auc_value = float(auc(fpr, tpr))

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="#d62728", linewidth=2.0, label=f"AUC = {auc_value:.3f}")
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="#7f7f7f", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Frame-Level ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    return auc_value
