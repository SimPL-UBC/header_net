# Future Optuna Tuning Ideas

The first Optuna workflow tunes only focal loss alpha, focal loss gamma, and the train negative:positive ratio. That is the right first pass because those parameters directly control class imbalance and false-positive pressure.

This document lists good candidates for future tuning once the core 3 are stable.

## High-Value Next Parameters

| Parameter | Suggested Search Space | Why Tune It |
|-----------|------------------------|-------------|
| `BASE_LR` | log-uniform `1e-5..3e-3` | Most important optimizer parameter after class-balance settings |
| `WEIGHT_DECAY` | log-uniform `1e-5..0.2` | Controls regularization and can reduce overfitting on dense negatives |
| `LAYER_LR_DECAY` | uniform `0.60..0.95` | Controls how aggressively lower VideoMAE layers are updated |
| `GRADIENT_ACCUMULATION_STEPS` | categorical `1 2 4 8` | Changes effective batch size without changing microbatch memory |
| `BATCH_SIZE` | categorical per backbone | Changes gradient noise and throughput, but is memory-limited |

These should usually be added after establishing a good range for `NEG_POS_RATIO`, `FOCAL_ALPHA`, and `FOCAL_GAMMA`.

## Recommended Phase 2 Search Space

For Base:

```text
FOCAL_ALPHA: 0.55..0.95
FOCAL_GAMMA: 0.5..5.0
NEG_POS_RATIO: 3, 5, 8, 10, 15, 20
BASE_LR: log 1e-5..3e-3
WEIGHT_DECAY: log 1e-5..0.2
LAYER_LR_DECAY: 0.60..0.95
```

For Giant:

```text
FOCAL_ALPHA: 0.55..0.95
FOCAL_GAMMA: 0.5..5.0
NEG_POS_RATIO: 3, 5, 8, 10, 15, 20
BASE_LR: log 3e-6..1e-3
WEIGHT_DECAY: log 1e-5..0.2
LAYER_LR_DECAY: 0.65..0.95
```

Giant should use a more conservative learning-rate range because the model is larger and more sensitive to unstable updates.

## Parameters To Tune Later

### Fine-Tuning Depth

Tune:

```text
FINETUNE_MODE: full, partial
UNFREEZE_BLOCKS: 2, 4, 6, 8, 12
```

Use this when full fine-tuning overfits or is too expensive. `partial` can be useful for Giant if training is unstable or slow.

### Training Length

Tune:

```text
EPOCHS: 5, 10, 20, 30
```

This is expensive because it changes trial runtime. Prefer using fixed shorter studies first, then retrain the best settings longer.

### Augmentation

Tune:

```text
TRAIN_AUGMENTATION_MODE: clip_consistent, legacy_frame_random, none
```

Use this only after optimizer and class-imbalance settings are reasonable. Augmentation can interact strongly with focal loss and sampling ratio.

### Validation Sampling

Keep validation on the full validation set by default:

```text
VAL_NEG_POS_RATIO=all
```

Only tune or reduce validation sampling for quick experiments. Model selection should use full validation whenever possible.

### Threshold Sweep

The evaluator currently optimizes positive-class F1 over thresholds using:

```text
F1_THRESHOLD_STEP=0.01
```

This is usually fine. A smaller step, such as `0.005`, may slightly improve reported best F1 but increases validation post-processing cost.

## Parameters To Avoid Tuning Early

Avoid tuning too many of these in the first expanded study:

| Parameter | Reason |
|-----------|--------|
| `NUM_FRAMES` | Requires regenerated dense parquet windows to be fully consistent |
| `SPATIAL_MODE` | Changes the data view and should be compared as a separate experiment |
| `LOSS` | Cross entropy and focal loss answer different imbalance assumptions |
| `VAL_NEG_POS_RATIO` | Can bias model selection if not using full validation |
| `NUM_WORKERS` | Mostly throughput/stability, not model quality |
| `MAX_OPEN_VIDEOS` | Mostly memory/file-handle behavior, not model quality |

## Suggested Study Progression

1. Tune the core 3 for Base and Giant separately.
2. Retrain the best core-3 settings for a few extra epochs and confirm validation stability.
3. Add `BASE_LR`, `WEIGHT_DECAY`, and `LAYER_LR_DECAY`.
4. If Giant is unstable or slow, add `FINETUNE_MODE=partial` and `UNFREEZE_BLOCKS`.
5. Only after those are stable, compare augmentation modes and spatial modes as separate studies.

## Trial Budget Guidance

Use rough minimums:

| Search Space | Suggested Trials Per Backbone |
|--------------|-------------------------------|
| Core 3 only | 20 |
| Core 3 plus LR/regularization | 40-80 |
| Add fine-tuning depth | 60-120 |
| Add augmentation or spatial mode | Separate study |

If GPU time is limited, use Base to narrow broad ranges, then run a smaller Giant study around the best Base-informed ranges.

## How To Add These Later

The implementation point is `tools/optuna_vmae_parquet.py`, in `suggest_params()`.

Example future extension:

```python
params = {
    "focal_alpha": trial.suggest_float("focal_alpha", args.alpha_min, args.alpha_max),
    "focal_gamma": trial.suggest_float("focal_gamma", args.gamma_min, args.gamma_max),
    "neg_pos_ratio": trial.suggest_categorical("neg_pos_ratio", args.neg_pos_ratios),
    "base_lr": trial.suggest_float("base_lr", 1e-5, 3e-3, log=True),
    "weight_decay": trial.suggest_float("weight_decay", 1e-5, 0.2, log=True),
    "layer_lr_decay": trial.suggest_float("layer_lr_decay", 0.60, 0.95),
}
```

Then pass the sampled values into the train command instead of the fixed `args.base_lr`, `args.weight_decay`, and `args.layer_lr_decay`.

Keep the first expanded implementation explicit. Avoid hidden automatic search-space changes because they make studies harder to compare.
