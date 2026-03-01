#!/usr/bin/env python3
"""Wrapper around generate_dense_dataset.py with SoccerNet test defaults."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_BASE = REPO_ROOT / "output" / "dense_dataset"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a dense per-frame parquet for SoccerNet/test. "
            "Any unrecognized arguments are forwarded to generate_dense_dataset.py."
        )
    )
    parser.add_argument(
        "--dataset-path",
        default=str(REPO_ROOT / "SoccerNet" / "test"),
        help="Path to the SoccerNet test split (default: %(default)s)",
    )
    parser.add_argument(
        "--labels-dir",
        default=str(REPO_ROOT / "SoccerNet" / "test" / "labelled_header"),
        help="Path to the test labelled_header directory (default: %(default)s)",
    )
    parser.add_argument(
        "--output-path",
        default=str(DEFAULT_OUTPUT_BASE / "dense_test.parquet"),
        help="Output parquet path (default: %(default)s)",
    )
    parser.add_argument(
        "--failed-log-path",
        default=str(DEFAULT_OUTPUT_BASE / "dense_test_failed_videos.csv"),
        help="Failed videos CSV path (default: %(default)s)",
    )
    parser.add_argument(
        "--failed-frame-log-path",
        default=str(DEFAULT_OUTPUT_BASE / "dense_test_failed_frames.csv"),
        help="Failed frames CSV path (default: %(default)s)",
    )
    return parser.parse_known_args()


def main() -> None:
    args, passthrough = parse_args()

    dense_argv = [
        "generate_dense_dataset.py",
        "--dataset-path",
        args.dataset_path,
        "--labels-dir",
        args.labels_dir,
        "--output-path",
        args.output_path,
        "--failed-log-path",
        args.failed_log_path,
        "--failed-frame-log-path",
        args.failed_frame_log_path,
    ]

    raw_args = sys.argv[1:]
    if (
        "--one-frame-header" not in raw_args
        and "--continuous-frame-header" not in raw_args
    ):
        dense_argv.append("--continuous-frame-header")

    dense_argv.extend(passthrough)

    original_argv = sys.argv
    try:
        sys.argv = dense_argv
        from generate_dense_dataset import main as dense_main

        dense_main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
