# Optuna VMAE Parquet Tuning

This guide explains how to run Optuna hyperparameter tuning for the VMAE2 Base and VMAE2 Giant parquet training pipeline.

The tuning workflow trains on `output/dense_dataset/dense_train` and evaluates every trial on `output/dense_dataset/dense_val`. It does not use the test dataset for model selection.

## Prerequisites

Generate the dense train and validation parquet datasets first:

```bash
job_script/generate_dense_train_val.sh
```

Install the project dependencies in `deep_impact_env`:

```bash
conda activate deep_impact_env
pip install -r requirements.txt
```

Confirm these paths exist before starting a study:

```bash
ls output/dense_dataset/dense_train
ls output/dense_dataset/dense_val
ls checkpoints/VideoMAEv2-Base
ls checkpoints/VideoMAEv2-giant
```

## Quick Start

Run 20 trials for both Base and Giant:

```bash
N_TRIALS=20 BACKBONES="base giant" job_script/optuna_vmae_parquet.sh
```

Run only Base:

```bash
N_TRIALS=20 BACKBONES=base job_script/optuna_vmae_parquet.sh
```

Run only Giant:

```bash
N_TRIALS=20 BACKBONES=giant job_script/optuna_vmae_parquet.sh
```

Run a dry-run first to verify generated commands without launching training:

```bash
DRY_RUN=1 BACKBONES=base N_TRIALS=1 job_script/optuna_vmae_parquet.sh
```

## What Is Tuned

The default search space is intentionally small because VMAE2 trials are expensive:

| Parameter | Default Search Space | Meaning |
|-----------|----------------------|---------|
| `FOCAL_ALPHA` | `0.55..0.95` | Positive-class focal loss weight |
| `FOCAL_GAMMA` | `0.5..5.0` | Focal loss focusing strength |
| `NEG_POS_RATIO` | `3 5 8 10 15 20` | Train negative:positive sampling ratio |

The objective metric is validation positive-class F1:

```bash
OBJECTIVE_METRIC=val_f1
```

Validation is handled by `training.cli_train_header_parquet_eval`, which performs a threshold sweep and reports `val_f1`, `val_precision`, `val_recall`, `val_auc`, and related metrics.

## Custom Search Spaces

Change focal alpha range:

```bash
ALPHA_MIN=0.6 ALPHA_MAX=0.9 job_script/optuna_vmae_parquet.sh
```

Change focal gamma range:

```bash
GAMMA_MIN=1.0 GAMMA_MAX=4.0 job_script/optuna_vmae_parquet.sh
```

Change sampled negative:positive ratios:

```bash
NEG_POS_RATIOS="5 8 10 15" job_script/optuna_vmae_parquet.sh
```

Optuna does not allow changing categorical choices inside an existing study. The launcher handles this by default with `STUDY_NAME_SUFFIX=auto`: when `NEG_POS_RATIOS` differs from the default list, it creates a separate study name and separate output directory.

To force a custom study identity:

```bash
STUDY_NAME_SUFFIX=wide_ratio_search NEG_POS_RATIOS="3 5 8 10 12 15" job_script/optuna_vmae_parquet.sh
```

To intentionally reuse the exact old study name, keep the same `NEG_POS_RATIOS` as the existing study or set:

```bash
STUDY_NAME_SUFFIX=none job_script/optuna_vmae_parquet.sh
```

Only use `STUDY_NAME_SUFFIX=none` when the categorical search space matches the previous study.

Use AUC instead of F1:

```bash
OBJECTIVE_METRIC=val_auc job_script/optuna_vmae_parquet.sh
```

## Common Runtime Controls

Use fewer epochs for smoke tests:

```bash
BACKBONES=base N_TRIALS=1 EPOCHS=1 job_script/optuna_vmae_parquet.sh
```

Override GPU selection:

```bash
GPUS="0 1" job_script/optuna_vmae_parquet.sh
```

Disable DDP and use the trainer's non-DDP behavior:

```bash
DDP_MODE=off job_script/optuna_vmae_parquet.sh
```

Force DDP when multiple GPUs are provided:

```bash
DDP_MODE=on GPUS="0 1" job_script/optuna_vmae_parquet.sh
```

Use a custom output root:

```bash
OUTPUT_ROOT=output/my_optuna_study job_script/optuna_vmae_parquet.sh
```

Use a custom Optuna SQLite database:

```bash
OPTUNA_STORAGE=sqlite:///output/my_optuna_study/optuna.db job_script/optuna_vmae_parquet.sh
```

## Base And Giant Defaults

Base defaults are tuned for the existing `job_script/train_vmae_parquet.sh` behavior:

```bash
BASE_BATCH_SIZE=16
BASE_GRADIENT_ACCUMULATION_STEPS=1
BASE_NUM_WORKERS=4
BASE_MAX_OPEN_VIDEOS=4
BASE_BACKBONE_CKPT=checkpoints/VideoMAEv2-Base
```

Giant defaults are tuned for the existing `job_script/train_vmae_parquet_giant.sh` behavior:

```bash
GIANT_BATCH_SIZE=1
GIANT_GRADIENT_ACCUMULATION_STEPS=2
GIANT_NUM_WORKERS=1
GIANT_MAX_OPEN_VIDEOS=1
GIANT_BACKBONE_CKPT=checkpoints/VideoMAEv2-giant
DDP_MODE=auto
```

`DDP_MODE=auto` uses DDP for Giant when multiple GPUs are provided. Base runs without DDP by default.

## Outputs

Default outputs are written under:

```text
output/optuna_vmae_parquet/
```

Per-backbone outputs:

```text
output/optuna_vmae_parquet/base/
output/optuna_vmae_parquet/giant/
```

When `STUDY_NAME_SUFFIX=auto` creates a suffix for a changed ratio list, outputs are nested under the suffix:

```text
output/optuna_vmae_parquet/base/np_<hash>/
output/optuna_vmae_parquet/giant/np_<hash>/
```

Each trial writes:

```text
trial_0000/checkpoints/last.pt
trial_0000/logs/train.log
trial_0000/logs/validate.log
trial_0000/validation/final_val_metrics.json
trial_0000/validation/final_val_predictions.csv
```

Each study writes:

```text
study_results.csv
best_trial.json
```

`best_trial.json` is the main file to inspect after a study. It includes the best params, validation score, checkpoint path, and metrics path.

## Optional Best-Checkpoint Validation Inference

To run raw parquet inference on the validation parquet for the best checkpoint after the study:

```bash
RUN_BEST_VAL_INFERENCE=true job_script/optuna_vmae_parquet.sh
```

The output is:

```text
output/optuna_vmae_parquet/<backbone>/best_val_inference/val_predictions_raw.csv
```

This uses `job_script/inference_parquet_val.sh`, whose defaults point to `dense_val`.

## Recommended First Runs

Run this first to check command wiring:

```bash
DRY_RUN=1 BACKBONES="base giant" N_TRIALS=1 job_script/optuna_vmae_parquet.sh
```

Run a small Base smoke study:

```bash
BACKBONES=base N_TRIALS=1 EPOCHS=1 job_script/optuna_vmae_parquet.sh
```

Run the first real studies:

```bash
BACKBONES=base N_TRIALS=20 job_script/optuna_vmae_parquet.sh
BACKBONES=giant N_TRIALS=20 job_script/optuna_vmae_parquet.sh
```

## Notes

Optuna trials are sequential by default. This avoids oversubscribing GPUs while each trial may already use multiple GPUs.

`VALIDATE_VIDEO_LOAD=false` by default because checking all video paths before every trial is expensive. If you suspect corrupted paths, run:

```bash
VALIDATE_VIDEO_LOAD=true BACKBONES=base N_TRIALS=1 EPOCHS=1 job_script/optuna_vmae_parquet.sh
```

Use the validation metric to select hyperparameters. Keep test inference separate for final reporting only.
