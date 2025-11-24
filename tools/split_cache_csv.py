#!/usr/bin/env python3
"""
Split a cache CSV into train/val sets with video-level grouping and label stratification.

Example:
python tools/split_cache_csv.py \
  --input_csv scratch_output/generate_dataset_test/16_frames_ver/dataset_generation/train_cache_header.csv \
  --train_csv scratch_output/generate_dataset_test/16_frames_ver/dataset_generation/train_split.csv \
  --val_csv scratch_output/generate_dataset_test/16_frames_ver/dataset_generation/val_split.csv \
  --val_frac 0.2 \
  --seed 42
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


def stratified_group_split(df: pd.DataFrame, val_frac: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split by video_id while approximately preserving label distribution."""
    rng = np.random.default_rng(seed)
    groups = df["video_id"].unique().tolist()
    rng.shuffle(groups)

    # Collect group-level label counts to stratify.
    group_stats = []
    for g in groups:
        subset = df[df["video_id"] == g]
        counts = subset["label"].value_counts().to_dict()
        group_stats.append((g, counts.get(1, 0), counts.get(0, 0)))

    # Greedy assign groups to train/val to match val_frac on positives.
    val_groups: List[str] = []
    train_groups: List[str] = []
    total_pos = sum(pos for _, pos, _ in group_stats)
    target_val_pos = val_frac * total_pos
    running_pos = 0

    for g, pos, neg in group_stats:
        # Decide whether to place this group into val based on remaining need.
        remaining_pos_need = target_val_pos - running_pos
        should_add_to_val = remaining_pos_need > 0 and pos > 0
        # If no positives, just balance sizes roughly.
        if not should_add_to_val and running_pos < target_val_pos and pos == 0:
            should_add_to_val = True

        if should_add_to_val:
            val_groups.append(g)
            running_pos += pos
        else:
            train_groups.append(g)

    # If we undershot val positives (e.g., few positive groups), move one more group.
    if running_pos < target_val_pos and train_groups:
        val_groups.append(train_groups.pop())

    train_df = df[df["video_id"].isin(train_groups)]
    val_df = df[df["video_id"].isin(val_groups)]
    return train_df, val_df


def main():
    parser = argparse.ArgumentParser(description="Split cache CSV into train/val splits with grouping.")
    parser.add_argument("--input_csv", required=True, type=str, help="Path to full cache CSV.")
    parser.add_argument("--train_csv", required=True, type=str, help="Output path for train CSV.")
    parser.add_argument("--val_csv", required=True, type=str, help="Output path for val CSV.")
    parser.add_argument("--val_frac", default=0.2, type=float, help="Fraction of data (by positives) to allocate to val.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument(
        "--override_root",
        type=str,
        default=None,
        help="If set, replace each sample path with override_root / basename(original path).",
    )
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    if "video_id" not in df.columns or "label" not in df.columns:
        raise ValueError("Input CSV must contain video_id and label columns.")

    if args.override_root:
        override_root = Path(args.override_root)
        if not override_root.exists():
            raise FileNotFoundError(f"override_root does not exist: {override_root}")
        df["path"] = df["path"].apply(lambda p: str(override_root / Path(p).name))

    train_df, val_df = stratified_group_split(df, args.val_frac, args.seed)

    Path(args.train_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.val_csv).parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(args.train_csv, index=False)
    val_df.to_csv(args.val_csv, index=False)

    print(f"Input rows: {len(df)} | Train: {len(train_df)} | Val: {len(val_df)}")
    print(f"Train label counts:\n{train_df['label'].value_counts()}")
    print(f"Val label counts:\n{val_df['label'].value_counts()}")
    print(f"Train videos: {train_df['video_id'].nunique()} | Val videos: {val_df['video_id'].nunique()}")


if __name__ == "__main__":
    main()
