"""Analysis helpers for event-level header evaluation."""

from .header_event_analysis import (
    EventMatchResult,
    compute_event_metrics_curve,
    dedupe_ground_truth_headers,
    evaluate_event_detection,
    filter_rows_by_frame_windows,
    load_inference_folder,
    load_labelled_headers,
    plot_event_metrics_vs_threshold,
    plot_frame_roc_curve,
    plot_match_timeline,
    summarize_event_metrics_by_video,
)

__all__ = [
    "EventMatchResult",
    "compute_event_metrics_curve",
    "dedupe_ground_truth_headers",
    "evaluate_event_detection",
    "filter_rows_by_frame_windows",
    "load_inference_folder",
    "load_labelled_headers",
    "plot_event_metrics_vs_threshold",
    "plot_frame_roc_curve",
    "plot_match_timeline",
    "summarize_event_metrics_by_video",
]
