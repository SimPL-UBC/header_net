from pathlib import Path
import sys

import pandas as pd

HEADER_NET_ROOT = Path(__file__).resolve().parents[1]
if str(HEADER_NET_ROOT) not in sys.path:
    sys.path.append(str(HEADER_NET_ROOT))

from analysis.header_event_analysis import (
    cluster_predictions,
    dedupe_ground_truth_headers,
    evaluate_event_detection,
    filter_rows_by_frame_windows,
)


def make_inference_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"video_id": "match_a", "half": 1, "frame": 1000, "label": 0, "prob_header": 0.70},
            {"video_id": "match_a", "half": 1, "frame": 1003, "label": 0, "prob_header": 0.95},
            {"video_id": "match_a", "half": 1, "frame": 1004, "label": 0, "prob_header": 0.90},
            {"video_id": "match_a", "half": 1, "frame": 1500, "label": 0, "prob_header": 0.80},
            {"video_id": "match_a", "half": 1, "frame": 2008, "label": 0, "prob_header": 0.92},
            {"video_id": "match_a", "half": 1, "frame": 2500, "label": 0, "prob_header": 0.10},
            {"video_id": "match_a", "half": 2, "frame": 500, "label": 0, "prob_header": 0.85},
            {"video_id": "match_b", "half": 1, "frame": 300, "label": 0, "prob_header": 0.20},
        ]
    )


def make_gt_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"video_id": "match_a", "half": 1, "gt_frame": 1005},
            {"video_id": "match_a", "half": 1, "gt_frame": 1005},
            {"video_id": "match_a", "half": 1, "gt_frame": 2000},
            {"video_id": "match_a", "half": 2, "gt_frame": 498},
        ]
    )


def test_dedupe_ground_truth_headers_removes_exact_duplicates():
    deduped = dedupe_ground_truth_headers(make_gt_df())

    assert deduped["gt_frame"].tolist() == [1005, 2000, 498]
    assert deduped["gt_event_id"].tolist() == [0, 1, 2]


def test_cluster_predictions_merges_nearby_positive_frames():
    pred_events = cluster_predictions(make_inference_df(), threshold=0.5, merge_gap=3)
    subset = pred_events[(pred_events["video_id"] == "match_a") & (pred_events["half"] == 1)]

    assert subset["start_frame"].tolist() == [1000, 1500, 2008]
    assert subset["end_frame"].tolist() == [1004, 1500, 2008]
    assert subset["peak_frame"].tolist() == [1003, 1500, 2008]
    assert subset.iloc[0]["positive_frames"] == (1000, 1003, 1004)


def test_evaluate_event_detection_counts_tp_fp_fn_once_per_cluster():
    result = evaluate_event_detection(
        make_inference_df(),
        make_gt_df(),
        threshold=0.5,
        merge_gap=3,
        tol_left=7,
        tol_right=8,
    )

    assert result.metrics["tp"] == 3
    assert result.metrics["fp"] == 1
    assert result.metrics["fn"] == 0
    assert result.metrics["precision"] == 0.75
    assert result.metrics["recall"] == 1.0

    pred_status = result.pred_events.sort_values(["video_id", "half", "peak_frame"])["match_status"]
    assert pred_status.tolist() == ["tp", "fp", "tp", "tp"]


def test_evaluate_event_detection_keeps_one_to_one_matching():
    inference_df = pd.DataFrame(
        [
            {"video_id": "match_a", "half": 1, "frame": 1000, "label": 0, "prob_header": 0.95},
            {"video_id": "match_a", "half": 1, "frame": 1010, "label": 0, "prob_header": 0.90},
        ]
    )
    gt_df = pd.DataFrame([{"video_id": "match_a", "half": 1, "gt_frame": 1005}])

    result = evaluate_event_detection(
        inference_df,
        gt_df,
        threshold=0.5,
        merge_gap=3,
        tol_left=7,
        tol_right=8,
    )

    assert result.metrics["tp"] == 1
    assert result.metrics["fp"] == 1
    assert result.metrics["fn"] == 0


def test_filter_rows_by_frame_windows_only_trims_target_match_half():
    df = pd.DataFrame(
        [
            {"video_id": "match_a", "half": 1, "frame": 10},
            {"video_id": "match_a", "half": 1, "frame": 20},
            {"video_id": "match_a", "half": 2, "frame": 30},
            {"video_id": "match_b", "half": 1, "frame": 40},
        ]
    )

    filtered = filter_rows_by_frame_windows(
        df,
        {("match_a", 1): (15, 25)},
        frame_column="frame",
    )

    assert filtered[["video_id", "half", "frame"]].to_dict(orient="records") == [
        {"video_id": "match_a", "half": 1, "frame": 20},
        {"video_id": "match_a", "half": 2, "frame": 30},
        {"video_id": "match_b", "half": 1, "frame": 40},
    ]
