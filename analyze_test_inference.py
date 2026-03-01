#!/usr/bin/env python3
"""Analyze parquet inference output and render an annotated debug video."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze parquet inference output, save scored predictions, and "
            "render an annotated video for one selected video-half."
        )
    )
    parser.add_argument(
        "--predictions-csv",
        default=str(
            Path("output") / "vmae_parquet_ratio10" / "test_inference" / "test_predictions_raw.csv"
        ),
        help="Raw prediction CSV from inference_parquet_test.py (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path("output") / "vmae_parquet_ratio10" / "test_inference" / "analysis"),
        help="Output directory (default: %(default)s)",
    )
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=None,
        help="Explicit header threshold. If omitted, the script selects the best F1 threshold.",
    )
    parser.add_argument(
        "--f1-threshold-step",
        type=float,
        default=0.01,
        help="Threshold sweep step when auto-selecting the decision threshold (default: %(default)s)",
    )
    parser.add_argument(
        "--render-video-id",
        default=None,
        help="video_id to render (default: first sorted pair in the scored CSV)",
    )
    parser.add_argument(
        "--render-half",
        type=int,
        default=None,
        help="Half to render (default: first available half for the selected video_id)",
    )
    parser.add_argument(
        "--render-frame-stride",
        type=int,
        default=5,
        help="Write every Nth source frame to the annotated video (default: %(default)s)",
    )
    parser.add_argument(
        "--render-output",
        default=None,
        help="Optional explicit output path for the rendered video",
    )
    return parser.parse_args()


def validate_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Predictions CSV is missing required columns: {missing}")


def choose_threshold(labels: np.ndarray, probs: np.ndarray, step: float) -> float:
    if step <= 0.0 or step > 1.0:
        raise ValueError("f1-threshold-step must be in (0, 1].")

    thresholds = np.arange(0.0, 1.0 + 1e-12, step, dtype=np.float64)
    best_threshold = 0.5
    best_f1 = -1.0
    best_recall = -1.0

    for threshold in thresholds:
        preds = (probs >= threshold).astype(np.int64)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            preds,
            average="binary",
            pos_label=1,
            zero_division=0,
        )
        better = f1 > best_f1
        if not better and np.isclose(f1, best_f1):
            if recall > best_recall:
                better = True
            elif np.isclose(recall, best_recall):
                better = abs(threshold - 0.5) < abs(best_threshold - 0.5)
        if better:
            best_threshold = float(threshold)
            best_f1 = float(f1)
            best_recall = float(recall)

    return best_threshold


def compute_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float,
) -> tuple[dict[str, Any], np.ndarray]:
    preds = (probs >= threshold).astype(np.int64)
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        pos_label=1,
        zero_division=0,
    )

    tp = int(np.sum((labels == 1) & (preds == 1)))
    tn = int(np.sum((labels == 0) & (preds == 0)))
    fp = int(np.sum((labels == 0) & (preds == 1)))
    fn = int(np.sum((labels == 1) & (preds == 0)))

    metrics = {
        "sample_count": int(labels.shape[0]),
        "decision_threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
    return metrics, preds


def sanitize_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return stem.strip("_") or "video"


def select_render_rows(
    df: pd.DataFrame,
    requested_video_id: str | None,
    requested_half: int | None,
) -> tuple[pd.DataFrame, str, int]:
    available = (
        df[["video_id", "half"]]
        .drop_duplicates()
        .sort_values(["video_id", "half"])
        .reset_index(drop=True)
    )
    if available.empty:
        raise ValueError("No predictions available for rendering.")

    if requested_half is not None and requested_video_id is None:
        raise ValueError("--render-half requires --render-video-id.")

    if requested_video_id is None:
        chosen_video_id = str(available.iloc[0]["video_id"])
        chosen_half = int(available.iloc[0]["half"])
    else:
        matching = available[available["video_id"].astype(str) == str(requested_video_id)]
        if matching.empty:
            raise ValueError(f"No rows found for render video_id={requested_video_id!r}.")
        if requested_half is None:
            chosen_video_id = str(matching.iloc[0]["video_id"])
            chosen_half = int(matching.iloc[0]["half"])
        else:
            matching = matching[matching["half"].astype(int) == int(requested_half)]
            if matching.empty:
                raise ValueError(
                    f"No rows found for render target ({requested_video_id!r}, {requested_half})."
                )
            chosen_video_id = str(matching.iloc[0]["video_id"])
            chosen_half = int(matching.iloc[0]["half"])

    selected = df[
        (df["video_id"].astype(str) == chosen_video_id)
        & (df["half"].astype(int) == chosen_half)
    ].copy()
    selected.sort_values("frame", inplace=True)
    return selected, chosen_video_id, chosen_half


def put_text_lines(frame: np.ndarray, lines: list[str]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, frame.shape[0] / 900.0)
    thickness = max(1, int(round(font_scale * 2)))
    x = 12
    y = 24

    for line in lines:
        cv2.putText(
            frame,
            line,
            (x, y),
            font,
            font_scale,
            (0, 0, 0),
            thickness + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            line,
            (x, y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        y += int(26 * font_scale) + 12


def draw_ball_box(frame: np.ndarray, row: dict[str, Any]) -> None:
    values = [row.get("ball_x"), row.get("ball_y"), row.get("ball_w"), row.get("ball_h")]
    if any(value is None for value in values):
        return
    x, y, w, h = (float(value) for value in values)
    if not all(math.isfinite(value) for value in (x, y, w, h)):
        return
    if w <= 0.0 or h <= 0.0:
        return

    pt1 = (int(round(x)), int(round(y)))
    pt2 = (int(round(x + w)), int(round(y + h)))
    thickness = max(1, int(round(frame.shape[0] / 360)))
    cv2.rectangle(frame, pt1, pt2, (0, 255, 0), thickness)

    conf = row.get("ball_confidence")
    if conf is None or not math.isfinite(float(conf)):
        label = "Ball"
    else:
        label = f"Ball {float(conf):.2f}"
    text_origin = (pt1[0], max(18, pt1[1] - 8))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, frame.shape[0] / 900.0)
    text_thickness = max(1, int(round(font_scale * 2)))
    cv2.putText(
        frame,
        label,
        text_origin,
        font,
        font_scale,
        (0, 0, 0),
        text_thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        label,
        text_origin,
        font,
        font_scale,
        (0, 255, 0),
        text_thickness,
        cv2.LINE_AA,
    )


def render_video(
    rows: pd.DataFrame,
    output_path: Path,
    frame_stride: int,
) -> dict[str, Any]:
    if frame_stride < 1:
        raise ValueError("render-frame-stride must be >= 1.")

    video_path = Path(str(rows.iloc[0]["video_path"])).expanduser()
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found for rendering: {video_path}")

    frame_lookup: dict[int, dict[str, Any]] = {}
    deduped = rows.drop_duplicates(subset=["frame"], keep="first")
    for row in deduped.to_dict(orient="records"):
        frame_lookup[int(row["frame"])] = row

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video for rendering: {video_path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if source_fps <= 0.0:
        csv_fps = pd.to_numeric(rows["fps"], errors="coerce").dropna()
        source_fps = float(csv_fps.iloc[0]) if not csv_fps.empty else 25.0
    render_fps = max(1.0, source_fps / float(frame_stride))

    writer: cv2.VideoWriter | None = None
    frame_index = 0
    written_frames = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if frame_index % frame_stride != 0:
                frame_index += 1
                continue

            annotated = frame.copy()
            row = frame_lookup.get(frame_index)

            lines = [f"Frame: {frame_index}"]
            if row is None:
                lines.extend(
                    [
                        "GT: n/a",
                        "Pred: n/a",
                        "P(header): n/a",
                        "Confidence: n/a",
                    ]
                )
            else:
                draw_ball_box(annotated, row)
                lines.extend(
                    [
                        f"GT: {int(row['label'])}",
                        f"Pred: {int(row['pred_label'])}",
                        f"P(header): {float(row['prob_header']):.3f}",
                        f"Confidence: {float(row['pred_confidence']):.3f}",
                    ]
                )

            put_text_lines(annotated, lines)

            if writer is None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                height, width = annotated.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    render_fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Unable to create video writer: {output_path}")

            writer.write(annotated)
            written_frames += 1
            frame_index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    return {
        "video_path": str(video_path),
        "render_fps": float(render_fps),
        "frame_stride": int(frame_stride),
        "frames_written": int(written_frames),
        "output_path": str(output_path),
    }


def main() -> None:
    args = parse_args()

    predictions_csv = Path(args.predictions_csv).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    metrics_path = output_dir / "test_metrics.json"
    scored_csv_path = output_dir / "test_predictions_scored.csv"

    if not predictions_csv.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {predictions_csv}")

    predictions_df = pd.read_csv(predictions_csv)
    if predictions_df.empty:
        raise ValueError(f"Predictions CSV is empty: {predictions_csv}")

    validate_columns(
        predictions_df,
        [
            "video_id",
            "half",
            "frame",
            "label",
            "video_path",
            "prob_header",
            "prob_non_header",
            "ball_x",
            "ball_y",
            "ball_w",
            "ball_h",
        ],
    )

    labels = pd.to_numeric(predictions_df["label"], errors="raise").to_numpy(np.int64)
    probs = pd.to_numeric(predictions_df["prob_header"], errors="raise").to_numpy(np.float64)

    if args.decision_threshold is None:
        decision_threshold = choose_threshold(labels, probs, args.f1_threshold_step)
    else:
        decision_threshold = float(args.decision_threshold)
        if not 0.0 <= decision_threshold <= 1.0:
            raise ValueError("decision-threshold must be in [0, 1].")

    metrics, pred_labels = compute_metrics(labels, probs, decision_threshold)

    scored_df = predictions_df.copy()
    scored_df["decision_threshold"] = float(decision_threshold)
    scored_df["pred_label"] = pred_labels
    scored_df["pred_confidence"] = np.maximum(
        pd.to_numeric(scored_df["prob_header"], errors="coerce").to_numpy(np.float64),
        pd.to_numeric(scored_df["prob_non_header"], errors="coerce").to_numpy(np.float64),
    )
    scored_df["is_correct"] = (
        pd.to_numeric(scored_df["label"], errors="coerce").to_numpy(np.int64) == pred_labels
    )
    if "row_idx" in scored_df.columns:
        scored_df.sort_values("row_idx", inplace=True)

    render_rows, render_video_id, render_half = select_render_rows(
        scored_df,
        args.render_video_id,
        args.render_half,
    )

    if args.render_output:
        render_output = Path(args.render_output).expanduser()
    else:
        render_output = (
            output_dir
            / "videos"
            / f"{sanitize_stem(render_video_id)}_half{render_half}.mp4"
        )

    render_info = render_video(
        rows=render_rows,
        output_path=render_output,
        frame_stride=int(args.render_frame_stride),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    scored_df.to_csv(scored_csv_path, index=False)

    metrics_payload = {
        **metrics,
        "render_video_id": render_video_id,
        "render_half": int(render_half),
        "render": render_info,
    }
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics_payload, fh, indent=4)

    print(f"Samples:         {metrics['sample_count']}")
    print(f"Threshold:       {metrics['decision_threshold']:.4f}")
    print(f"Accuracy:        {metrics['accuracy']:.4f}")
    print(f"Recall:          {metrics['recall']:.4f}")
    print(f"F1:              {metrics['f1']:.4f}")
    print(f"Scored CSV:      {scored_csv_path}")
    print(f"Metrics JSON:    {metrics_path}")
    print(f"Rendered video:  {render_info['output_path']}")


if __name__ == "__main__":
    main()
