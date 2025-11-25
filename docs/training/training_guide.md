# Header Net Training Guide

This guide explains how to prepare the dataset and train the Header Net model with either CSN or VideoMAE v2 backbones.

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

Use `training.cli_train_header` to train the model with either CSN or VideoMAE v2 backbones.

### Available Backbones

- **CSN** (ResNet3D-CSN): Convolutional 3D backbone, requires full fine-tuning
- **VideoMAE v2**: Vision Transformer backbone pretrained on Kinetics-400, supports frozen and full fine-tuning modes

### Common Arguments

```bash
conda activate deep_impact_env
python -m training.cli_train_header \
  --train_csv <path_to_train_csv> \
  --val_csv <path_to_val_csv> \
  --backbone <csn|vmae> \
  --finetune_mode <full|frozen> \
  --run_name <run_name> \
  --epochs <num_epochs> \
  --num_frames <num_frames> \
  --batch_size <batch_size>
```

### CSN Training Example

```bash
conda activate deep_impact_env
python -m training.cli_train_header \
  --train_csv cache/header_net_16frames/train.csv \
  --val_csv cache/header_net_16frames/val.csv \
  --backbone csn \
  --finetune_mode full \
  --run_name csn_16frames_baseline \
  --epochs 50 \
  --num_frames 16 \
  --batch_size 16 \
  --lr_backbone 0.001
```

### VideoMAE v2 Frozen Mode Example

Train only the classification head while keeping the VideoMAE backbone frozen:

```bash
conda activate deep_impact_env
python -m training.cli_train_header \
  --train_csv cache/header_net_16frames/train.csv \
  --val_csv cache/header_net_16frames/val.csv \
  --backbone vmae \
  --finetune_mode frozen \
  --backbone_ckpt checkpoints/VideoMAEv2-Base \
  --lr_head 1e-3 \
  --run_name vmae_frozen_base \
  --epochs 50 \
  --num_frames 16 \
  --batch_size 4
```

**Note**: VideoMAE models are memory-intensive. Use smaller batch sizes (2-4) to avoid OOM errors.

### VideoMAE v2 Full Fine-Tuning Example

Fine-tune both the backbone and classification head:

```bash
conda activate deep_impact_env
python -m training.cli_train_header \
  --train_csv cache/header_net_16frames/train.csv \
  --val_csv cache/header_net_16frames/val.csv \
  --backbone vmae \
  --finetune_mode full \
  --backbone_ckpt checkpoints/VideoMAEv2-Base \
  --lr_backbone 1e-5 \
  --lr_head 1e-3 \
  --run_name vmae_full_base \
  --epochs 30 \
  --num_frames 16 \
  --batch_size 2
```

## 3. VideoMAE Backbone Variants

Three VideoMAE v2 variants are available:

| Variant | Hidden Dim | Checkpoint Path |
|---------|-----------|-----------------|
| Base | 768 | `checkpoints/VideoMAEv2-Base` |
| Large | 1024 | `checkpoints/VideoMAEv2-Large` |
| Giant | 1408 | `checkpoints/VideoMAEv2-giant` |

Larger variants provide better performance but require more memory.

## 4. Data Requirements

- **Video Clips**: The model requires `.npy` files containing the video clips in shape `(T, H, W, C)`.
- **Frame Count**: VideoMAE models are optimized for 16 frames. CSN can work with different frame counts.
- **Normalization**: 
  - CSN uses ImageNet normalization (mean/std: `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`)
  - VideoMAE uses standard normalization (mean/std: `[0.5, 0.5, 0.5]` / `[0.5, 0.5, 0.5]`)
- **Metadata**: The `.json` metadata files are **not** required for training. The training script reads labels and paths directly from the CSV.

## 5. Multi-GPU Training

The training script supports multi-GPU training on a single node using `torch.nn.DataParallel`.

### Usage

To use multiple GPUs, specify the GPU IDs with the `--gpus` argument:

```bash
conda activate deep_impact_env
python -m training.cli_train_header \
  --train_csv cache/header_net_16frames/train.csv \
  --val_csv cache/header_net_16frames/val.csv \
  --backbone csn \
  --finetune_mode full \
  --run_name csn_16frames_multi_gpu \
  --epochs 50 \
  --num_frames 16 \
  --batch_size 32 \
  --gpus 0 1 2 3
```

**Note**: When using multiple GPUs, you can typically increase the `batch_size`.

## 6. Output Artifacts

All training runs save the following artifacts to `report/header_experiments/<run_name>/`:

- `config.yaml` - Full configuration used for the run
- `metrics_epoch.csv` - Per-epoch training and validation metrics
- `best_metrics.json` - Best validation metrics and checkpoint path
- `val_predictions.csv` - Validation set predictions (video_id, label, probabilities, predictions)
- `checkpoints/best_epoch_<N>.pt` - Model checkpoint from best epoch

## 7. Learning Rate Guidelines

| Backbone | Mode | lr_backbone | lr_head |
|----------|------|-------------|---------|
| CSN | Full | 1e-3 | N/A |
| VideoMAE | Frozen | N/A | 1e-3 |
| VideoMAE | Full | 1e-5 to 1e-4 | 1e-3 |

**Tip**: When fine-tuning VideoMAE, use a much lower learning rate for the backbone (1e-5) than the head (1e-3) to preserve pretrained features.
