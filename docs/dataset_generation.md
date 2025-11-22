# Dataset Generation

This document describes how to generate the training dataset (positive and negative samples) for the HeaderNet model using the `generate_positive_negative_dataset.py` script.

## Overview

The `dataset_generation/generate_positive_negative_dataset.py` script performs **sparse detection** to efficiently create training samples. It:
1.  Loads positive header events from the dataset.
2.  Generates negative samples (random frames without header events).
3.  Identifies the specific frames required for these samples (including temporal windows).
4.  Runs the RF-DETR object detector *only* on these frames.
5.  Generates the final cache (cropped images and metadata) directly.

## Usage

### Basic Command

```bash
python dataset_generation/generate_positive_negative_dataset.py \
    --dataset-path /path/to/SoccerNet \
    --header-dataset /path/to/annotations \
    --weights /path/to/rtdetr_weights.pt \
    --output-dir /path/to/output/cache \
    --device cuda:0
```

### Key Arguments

-   `--negative-ratio`: The ratio of negative samples to positive samples (default: 3.0).
-   `--guard-frames`: Number of frames to exclude around positive events when sampling negatives (default: 10).
-   `--window`: The temporal window offsets to process around each event (default: `cfg.WINDOW_SIZE`, typically `[-10, ..., 10]`).
-   `--batch-size`: Batch size for the RF-DETR model (default: 4).
-   `--optimize`: Enable TensorRT/TorchScript optimization for the model (recommended for speed).

### Example with Custom Sampling

To generate a cache with 5x negative samples and a larger guard window:

```bash
python dataset_generation/generate_positive_negative_dataset.py \
    --negative-ratio 5.0 \
    --guard-frames 20 \
    --output-dir cache/my_custom_cache
```

## Output

The script produces a directory containing:
-   `train_cache_header.csv`: A CSV index of all generated samples.
-   `*_s.npy`: Stacked image crops for each sample.
-   `*_meta.json`: Metadata for each sample (bounding boxes, labels, etc.).
-   `skipped_samples.csv`: A log of samples that were skipped due to missing detections (e.g., ball not found).
