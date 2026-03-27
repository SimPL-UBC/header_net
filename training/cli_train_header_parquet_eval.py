import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from .config import Config
from .data.parquet_header_dataset import build_parquet_val_dataloader
from .engine.supervised_trainer import Trainer
from .eval.predictions import save_predictions
from .models.factory import build_model
from .run_utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Header Net parquet checkpoint on validation parquet."
    )
    parser.add_argument("--checkpoint_path", required=True, help="Path to checkpoint .pt")
    parser.add_argument(
        "--val_parquet",
        required=True,
        help="Path to val parquet file or partitioned parquet dataset directory",
    )
    parser.add_argument(
        "--dataset_root",
        default="SoccerNet",
        help="SoccerNet root directory (validated in strict path mode)",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory for metrics/predictions (default: checkpoint run directory)",
    )
    parser.add_argument(
        "--video_id",
        default=None,
        help="Optional validation video_id filter",
    )
    parser.add_argument(
        "--half",
        type=int,
        default=None,
        help="Optional validation half filter",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Eval batch size (default: checkpoint config batch_size or 4)",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=None,
        help="Frames per clip (default: checkpoint config num_frames or 16)",
    )
    parser.add_argument(
        "--val_num_workers",
        type=int,
        default=8,
        help="Validation DataLoader workers",
    )
    parser.add_argument(
        "--val_pin_memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use pin_memory for validation DataLoader",
    )
    parser.add_argument(
        "--val_progress_every",
        type=int,
        default=1000,
        help="Print progress every N validation batches",
    )
    parser.add_argument(
        "--max_open_videos",
        type=int,
        default=8,
        help="Per-worker cap on open decord readers",
    )
    parser.add_argument(
        "--frame_cache_size",
        type=int,
        default=128,
        help="Per-worker frame cache size (frames)",
    )
    parser.add_argument(
        "--loader_start_method",
        choices=("spawn", "fork", "forkserver"),
        default="spawn",
        help="Multiprocessing start method for validation workers",
    )
    parser.add_argument(
        "--val_neg_pos_ratio",
        default="all",
        help=(
            "Validation negative:positive ratio. "
            "Use 'all' for full set, or a positive integer (e.g., 15)."
        ),
    )

    parser.add_argument(
        "--backbone",
        default=None,
        help="Model backbone (default: checkpoint config value)",
    )
    parser.add_argument(
        "--finetune_mode",
        default=None,
        help="Finetune mode used for model rebuild (default: checkpoint config value)",
    )
    parser.add_argument(
        "--unfreeze_blocks",
        type=int,
        default=None,
        help="Unfreeze blocks used for model rebuild (default: checkpoint config value)",
    )
    parser.add_argument(
        "--backbone_ckpt",
        default=None,
        help="VideoMAE checkpoint directory (default: checkpoint config value)",
    )
    parser.add_argument(
        "--base_lr",
        type=float,
        default=None,
        help="Base LR used for model rebuild param groups (default: checkpoint config value)",
    )
    parser.add_argument(
        "--layer_lr_decay",
        type=float,
        default=None,
        help="Layer LR decay used for model rebuild (default: checkpoint config value)",
    )
    parser.add_argument(
        "--loss",
        choices=["focal", "ce"],
        default=None,
        help="Loss function for eval loss reporting (default: checkpoint config value)",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=None,
        help="Focal gamma (default: checkpoint config value)",
    )
    parser.add_argument(
        "--focal_alpha",
        type=float,
        default=None,
        help="Focal alpha (default: checkpoint config value)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Eval seed (default: checkpoint config value or 42)",
    )
    parser.add_argument("--gpus", type=int, nargs="+", help="GPU IDs")

    parser.add_argument(
        "--f1_threshold_step",
        type=float,
        default=0.01,
        help="Threshold sweep step for positive-class F1 selection",
    )
    parser.add_argument(
        "--save_predictions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write per-sample predictions CSV",
    )
    parser.add_argument(
        "--predictions_path",
        default=None,
        help="Custom predictions output path (default: <output_dir>/final_test_predictions.csv)",
    )
    parser.add_argument(
        "--metrics_path",
        default=None,
        help="Custom metrics output path (default: <output_dir>/final_test_metrics.json)",
    )
    parser.add_argument(
        "--skip_existing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip evaluation when metrics output already exists",
    )
    parser.add_argument(
        "--reuse_predictions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse an existing predictions CSV to rebuild metrics when possible",
    )
    return parser.parse_args()


def _parse_optional_ratio(value, arg_name):
    text = str(value).strip().lower()
    if text == "all":
        return None
    try:
        ratio = int(text)
    except ValueError as exc:
        raise ValueError(f"{arg_name} must be 'all' or a positive integer.") from exc
    if ratio <= 0:
        raise ValueError(f"{arg_name} must be 'all' or a positive integer.")
    return ratio


def _threshold_sweep_positive_f1(labels, probs, step):
    if step <= 0.0 or step > 1.0:
        raise ValueError("f1_threshold_step must be in (0, 1]")

    thresholds = np.arange(0.0, 1.0 + 1e-12, step, dtype=np.float64)
    best = {
        "threshold": 0.5,
        "pos_precision": 0.0,
        "pos_recall": 0.0,
        "pos_f1": -1.0,
        "pred_labels": np.zeros_like(labels),
    }

    for threshold in thresholds:
        pred_labels = (probs >= threshold).astype(np.int64)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            pred_labels,
            average="binary",
            pos_label=1,
            zero_division=0,
        )

        better = f1 > best["pos_f1"]
        if not better and np.isclose(f1, best["pos_f1"]):
            if recall > best["pos_recall"]:
                better = True
            elif np.isclose(recall, best["pos_recall"]):
                better = abs(threshold - 0.5) < abs(best["threshold"] - 0.5)

        if better:
            best = {
                "threshold": float(threshold),
                "pos_precision": float(precision),
                "pos_recall": float(recall),
                "pos_f1": float(f1),
                "pred_labels": pred_labels,
            }

    acc = accuracy_score(labels, best["pred_labels"])
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        labels,
        best["pred_labels"],
        average="weighted",
        zero_division=0,
    )
    if len(np.unique(labels)) > 1:
        try:
            auc = roc_auc_score(labels, probs)
        except ValueError:
            auc = 0.0
    else:
        auc = 0.0

    best.update(
        {
            "acc": float(acc),
            "weighted_precision": float(weighted_precision),
            "weighted_recall": float(weighted_recall),
            "weighted_f1": float(weighted_f1),
            "auc": float(auc),
        }
    )
    return best


def _resolve(cli_value, checkpoint_config, key, default):
    if cli_value is not None:
        return cli_value
    if isinstance(checkpoint_config, dict) and key in checkpoint_config:
        return checkpoint_config[key]
    return default


def _resolve_output_dir(checkpoint_path: Path, output_dir_arg: str):
    if output_dir_arg:
        return Path(output_dir_arg)
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent
    return checkpoint_path.parent


def _load_saved_predictions(predictions_path: Path):
    predictions = []
    with open(predictions_path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return predictions
        for row in reader:
            if not row:
                continue
            predictions.append(
                {
                    "row_idx": int(row["row_idx"]) if row.get("row_idx") not in (None, "") else -1,
                    "video_id": row.get("video_id", ""),
                    "half": row.get("half", ""),
                    "frame": int(row["frame"]),
                    "path": row.get("path", ""),
                    "label": int(row["label"]),
                    "loss": float(row["loss"]) if row.get("loss") not in (None, "") else None,
                    "prob_header": float(row["prob_header"]),
                    "prob_non_header": float(row["prob_non_header"]),
                    "pred_label": int(row["pred_label"]) if row.get("pred_label") not in (None, "") else 0,
                }
            )
    return predictions


def _finalize_metrics_from_predictions(val_preds, f1_threshold_step):
    if not val_preds:
        raise RuntimeError("Validation produced no predictions.")

    loss_values = [pred.get("loss") for pred in val_preds]
    if any(loss is None for loss in loss_values):
        raise RuntimeError(
            "Predictions CSV is missing per-sample loss values and cannot be reused "
            "to compute exact validation metrics."
        )

    labels = np.array([int(pred["label"]) for pred in val_preds], dtype=np.int64)
    probs = np.array([float(pred["prob_header"]) for pred in val_preds], dtype=np.float64)
    sweep = _threshold_sweep_positive_f1(
        labels=labels,
        probs=probs,
        step=f1_threshold_step,
    )
    pred_labels = sweep["pred_labels"]
    for idx, pred in enumerate(val_preds):
        pred["pred_label"] = int(pred_labels[idx])

    final_metrics = {
        "val_loss": float(np.mean(np.array(loss_values, dtype=np.float64))),
        "val_acc": float(sweep["acc"]),
        "val_precision": float(sweep["pos_precision"]),
        "val_recall": float(sweep["pos_recall"]),
        "val_f1": float(sweep["pos_f1"]),
        "val_auc": float(sweep["auc"]),
        "val_weighted_precision": float(sweep["weighted_precision"]),
        "val_weighted_recall": float(sweep["weighted_recall"]),
        "val_weighted_f1": float(sweep["weighted_f1"]),
        "val_best_threshold": float(sweep["threshold"]),
    }
    eval_counts = {
        "samples": int(labels.shape[0]),
        "positives": int(labels.sum()),
        "negatives": int(labels.shape[0] - labels.sum()),
    }
    return final_metrics, eval_counts


def _run_validation_with_retry(trainer, model, val_loader, epoch_label):
    try:
        return trainer.validate(model, val_loader, epoch_label)
    except RuntimeError as err:
        err_text = str(err)
        worker_crash = "DataLoader worker" in err_text and "exited unexpectedly" in err_text
        if not worker_crash:
            raise

        print(
            "Validation DataLoader worker crashed. "
            "Retrying validation with num_workers=0 and pin_memory=False."
        )
        retry_sampler = getattr(val_loader, "sampler", None)
        retry_collate_fn = getattr(val_loader, "collate_fn", None)
        retry_worker_init_fn = getattr(val_loader, "worker_init_fn", None)
        retry_drop_last = bool(getattr(val_loader, "drop_last", False))
        retry_batch_size = int(getattr(val_loader, "batch_size", 1))

        retry_loader = torch.utils.data.DataLoader(
            val_loader.dataset,
            batch_size=retry_batch_size,
            shuffle=False,
            sampler=retry_sampler,
            num_workers=0,
            pin_memory=False,
            drop_last=retry_drop_last,
            collate_fn=retry_collate_fn,
            worker_init_fn=retry_worker_init_fn,
        )
        return trainer.validate(model, retry_loader, epoch_label)


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict")
    if state_dict is None:
        raise ValueError(f"Checkpoint missing 'state_dict': {checkpoint_path}")
    checkpoint_config = checkpoint.get("config", {})
    if checkpoint_config is None:
        checkpoint_config = {}

    config = Config()
    config.train_parquet = ""
    config.val_parquet = args.val_parquet
    config.dataset_root = args.dataset_root
    config.neg_pos_ratio = "all"
    config.train_video_ids = []
    config.train_halves = []
    config.val_video_ids = [str(args.video_id)] if args.video_id else []
    config.val_halves = [int(args.half)] if args.half is not None else []
    config.num_frames = int(_resolve(args.num_frames, checkpoint_config, "num_frames", 16))
    config.input_size = int(_resolve(None, checkpoint_config, "input_size", 224))
    config.backbone = str(_resolve(args.backbone, checkpoint_config, "backbone", "vmae"))
    config.finetune_mode = str(
        _resolve(args.finetune_mode, checkpoint_config, "finetune_mode", "full")
    )
    config.unfreeze_blocks = int(
        _resolve(args.unfreeze_blocks, checkpoint_config, "unfreeze_blocks", 4)
    )
    config.backbone_ckpt = _resolve(
        args.backbone_ckpt, checkpoint_config, "backbone_ckpt", None
    )
    config.optimizer_type = str(_resolve(None, checkpoint_config, "optimizer", "adamw"))
    config.base_lr = float(_resolve(args.base_lr, checkpoint_config, "base_lr", 1e-3))
    config.layer_lr_decay = float(
        _resolve(args.layer_lr_decay, checkpoint_config, "layer_lr_decay", 0.75)
    )
    config.loss_type = str(_resolve(args.loss, checkpoint_config, "loss", "focal"))
    config.focal_gamma = float(
        _resolve(args.focal_gamma, checkpoint_config, "focal_gamma", 2.0)
    )
    config.focal_alpha = float(
        _resolve(args.focal_alpha, checkpoint_config, "focal_alpha", 0.75)
    )
    config.seed = int(_resolve(args.seed, checkpoint_config, "seed", 42))
    config.batch_size = int(_resolve(args.batch_size, checkpoint_config, "batch_size", 4))
    config.num_workers = 0
    config.val_num_workers = int(args.val_num_workers)
    config.val_pin_memory = bool(args.val_pin_memory)
    config.val_progress_every = int(args.val_progress_every)
    config.max_open_videos = int(args.max_open_videos)
    config.frame_cache_size = int(args.frame_cache_size)
    config.loader_start_method = str(args.loader_start_method)
    config.gpus = args.gpus if args.gpus is not None else checkpoint_config.get("gpus")
    config.f1_threshold_step = float(args.f1_threshold_step)
    config.val_neg_pos_ratio = str(args.val_neg_pos_ratio)

    if config.f1_threshold_step <= 0.0 or config.f1_threshold_step > 1.0:
        raise ValueError("f1_threshold_step must be in (0, 1]")
    _parse_optional_ratio(config.val_neg_pos_ratio, "val_neg_pos_ratio")

    output_dir = _resolve_output_dir(checkpoint_path, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics_path) if args.metrics_path else output_dir / "final_test_metrics.json"
    predictions_path = (
        Path(args.predictions_path)
        if args.predictions_path
        else output_dir / "final_test_predictions.csv"
    )

    if args.skip_existing and metrics_path.exists() and metrics_path.stat().st_size > 0:
        print(f"Skipping validation because metrics already exist: {metrics_path}")
        return

    if (
        args.reuse_predictions
        and args.save_predictions
        and predictions_path.exists()
        and predictions_path.stat().st_size > 0
    ):
        try:
            reused_predictions = _load_saved_predictions(predictions_path)
            final_metrics, eval_counts = _finalize_metrics_from_predictions(
                reused_predictions,
                config.f1_threshold_step,
            )
        except Exception as exc:
            print(
                "Existing predictions could not be reused; running full validation. "
                f"Reason: {exc}"
            )
        else:
            with open(metrics_path, "w") as f:
                json.dump(
                    {
                        "checkpoint_path": str(checkpoint_path),
                        "val_parquet": str(config.val_parquet),
                        "val_counts": eval_counts,
                        "metrics": final_metrics,
                    },
                    f,
                    indent=4,
                )
            print(f"Reused predictions: {predictions_path}")
            print(f"Saved metrics: {metrics_path}")
            print(
                f"Eval complete. Pos F1={final_metrics['val_f1']:.4f}, "
                f"P={final_metrics['val_precision']:.4f}, "
                f"R={final_metrics['val_recall']:.4f}, "
                f"thr={final_metrics['val_best_threshold']:.2f}"
            )
            return

    set_seed(config.seed)
    if config.gpus and torch.cuda.is_available():
        device = torch.device(f"cuda:{config.gpus[0]}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Val parquet: {config.val_parquet}")
    print(f"Output directory: {output_dir}")
    print(
        "Dataloader workers: "
        f"val={config.val_num_workers}, pin_memory={config.val_pin_memory}"
    )
    print(
        "Loader settings: "
        f"max_open_videos={config.max_open_videos}, "
        f"frame_cache_size={config.frame_cache_size}, "
        f"start_method={config.loader_start_method}"
    )
    if config.val_video_ids or config.val_halves:
        print(
            "Validation filters: "
            f"video_ids={config.val_video_ids or 'all'}, "
            f"halves={config.val_halves or 'all'}"
        )
    print(f"Using device: {device}")

    ratio = _parse_optional_ratio(config.val_neg_pos_ratio, "val_neg_pos_ratio")
    val_loader, val_dataset, val_sampler = build_parquet_val_dataloader(
        config,
        neg_pos_ratio=config.val_neg_pos_ratio,
        seed_offset=1000,
        shuffle=True,
        pin_memory=config.val_pin_memory,
    )
    if ratio is None:
        eval_counts = val_dataset.class_counts()
        print(f"Validation sample counts (full set): {eval_counts}")
    else:
        eval_counts = val_sampler.get_counts()
        print(
            "Validation sample counts (ratio-sampled): "
            f"{eval_counts} (neg:pos={ratio})"
        )

    model, _ = build_model(config)
    model = model.to(device)
    if config.gpus and len(config.gpus) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpus)

    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)

    trainer = Trainer(config, device)
    val_metrics_raw, val_preds = _run_validation_with_retry(
        trainer=trainer,
        model=model,
        val_loader=val_loader,
        epoch_label="eval",
    )
    final_metrics, eval_counts_from_preds = _finalize_metrics_from_predictions(
        val_preds,
        config.f1_threshold_step,
    )
    final_metrics["val_loss"] = float(val_metrics_raw["val_loss"])

    if args.save_predictions:
        save_predictions(val_preds, predictions_path)
        print(f"Saved predictions: {predictions_path}")

    eval_counts = eval_counts_from_preds

    with open(metrics_path, "w") as f:
        json.dump(
            {
                "checkpoint_path": str(checkpoint_path),
                "val_parquet": str(config.val_parquet),
                "val_counts": eval_counts,
                "metrics": final_metrics,
            },
            f,
            indent=4,
        )
    print(f"Saved metrics: {metrics_path}")

    print(
        f"Eval complete. Pos F1={final_metrics['val_f1']:.4f}, "
        f"P={final_metrics['val_precision']:.4f}, "
        f"R={final_metrics['val_recall']:.4f}, "
        f"thr={final_metrics['val_best_threshold']:.2f}"
    )


if __name__ == "__main__":
    main()
