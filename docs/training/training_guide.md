# Header Net Training Guide

This guide explains how to prepare the dataset and train the Header Net model.

## 1. Dataset Preparation

The training script expects a split dataset (train and validation CSVs) with relative paths to the video clips (`.npy` files).
Use the `tools/split_dataset.py` script to split your single CSV file and fix the paths.

### Usage

```bash
python tools/split_dataset.py \
  --input_csv <path_to_input_csv> \
  --output_dir <path_to_output_dir> \
  --root_dir <path_to_dataset_root> \
  --val_split 0.2 \
  --seed 42
```

### Example

```bash
python tools/split_dataset.py \
  --input_csv scratch_output/generate_dataset_test/16_frames_ver/dataset_generation/train_cache_header.csv \
  --output_dir cache/header_net_16frames \
  --root_dir scratch_output/generate_dataset_test/16_frames_ver/dataset_generation
```

This will create `train.csv` and `val.csv` in `cache/header_net_16frames/`.

## 2. Training

Use `training.cli_train_header` to train the model.

### Usage

```bash
conda activate deep_impact_env
python -m training.cli_train_header \
  --train_csv <path_to_train_csv> \
  --val_csv <path_to_val_csv> \
  --backbone csn \
  --finetune_mode full \
  --run_name <run_name> \
  --epochs <num_epochs> \
  --num_frames <num_frames> \
  --batch_size <batch_size>
```

### Example

```bash
conda activate deep_impact_env
python -m training.cli_train_header \
  --train_csv cache/header_net_16frames/train.csv \
  --val_csv cache/header_net_16frames/val.csv \
  --backbone csn \
  --finetune_mode full \
  --run_name csn_16frames_test \
  --epochs 50 \
  --num_frames 16 \
  --batch_size 16
```

## 3. Data Requirements

- **Video Clips**: The model requires `.npy` files containing the video clips.
- **Metadata**: The `.json` metadata files are **not** required for training. The training script reads labels and paths directly from the CSV.
