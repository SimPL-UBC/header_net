import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from .config import merge_cli_args
from .data.parquet_header_dataset import build_parquet_dataloaders
from .engine.supervised_trainer import Trainer
from .eval.plots import generate_all_plots
from .eval.predictions import save_predictions
from .models.factory import build_model
from .run_utils import create_run_dir, save_config, set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Header Net on dense parquet metadata with raw videos (VMAE2)."
    )
    parser.add_argument("--train_parquet", required=True, help="Path to train parquet")
    parser.add_argument("--val_parquet", required=True, help="Path to val parquet")
    parser.add_argument(
        "--dataset_root",
        default="SoccerNet",
        help="SoccerNet root directory (validated in strict path mode)",
    )
    parser.add_argument(
        "--neg_pos_ratio",
        default="all",
        choices=["10", "20", "30", "all"],
        help="Train negative:positive ratio; 'all' uses full dataset",
    )

    parser.add_argument(
        "--backbone",
        default="vmae",
        choices=["vmae"],
        help="Backbone model (parquet trainer currently supports only vmae)",
    )
    parser.add_argument(
        "--finetune_mode",
        default="full",
        choices=["full", "frozen", "partial"],
        help="Finetune mode: full, frozen, or partial",
    )
    parser.add_argument(
        "--unfreeze_blocks",
        type=int,
        default=4,
        help="Number of last VideoMAE blocks to unfreeze in partial mode",
    )
    parser.add_argument(
        "--backbone_ckpt", default=None, help="Path to VideoMAE checkpoint directory"
    )
    parser.add_argument("--run_name", required=True, help="Run name")
    parser.add_argument("--output_root", default="output/vmae", help="Output root")

    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers")
    parser.add_argument(
        "--optimizer",
        default="adamw",
        choices=["adamw", "sgd"],
        help="Optimizer type",
    )
    parser.add_argument("--base_lr", type=float, default=1e-3, help="Base learning rate")
    parser.add_argument(
        "--layer_lr_decay",
        type=float,
        default=0.75,
        help="Layer-wise learning rate decay for VideoMAE",
    )
    parser.add_argument(
        "--betas",
        type=float,
        nargs=2,
        default=(0.9, 0.999),
        help="AdamW betas",
    )
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--num_frames", type=int, default=16, help="Number of frames per clip"
    )
    parser.add_argument(
        "--loss",
        default="focal",
        choices=["focal", "ce"],
        help="Loss function type",
    )
    parser.add_argument(
        "--focal_gamma", type=float, default=2.0, help="Focal loss gamma"
    )
    parser.add_argument(
        "--focal_alpha",
        type=float,
        default=0.75,
        help="Focal loss alpha (positive class weight)",
    )
    parser.add_argument(
        "--f1_threshold_step",
        type=float,
        default=0.01,
        help="Validation threshold sweep step for positive-class F1 selection",
    )
    parser.add_argument(
        "--save_epoch_indices",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save sampled train row indices per epoch for reproducibility",
    )
    parser.add_argument("--gpus", type=int, nargs="+", help="GPU IDs")
    return parser.parse_args()


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


def _get_group_lr(optimizer, group_name):
    for group in optimizer.param_groups:
        if group.get("name") == group_name:
            return group.get("lr")
    return None


def main():
    args = parse_args()
    config = merge_cli_args(args)

    scale = config.batch_size / 256.0
    scaled_lr = config.base_lr * scale
    config.base_lr = scaled_lr

    set_seed(config.seed)
    run_dir = create_run_dir(config.output_root, config.run_name)
    save_config(config, run_dir)

    print(f"Starting run: {config.run_name}")
    print(f"Output directory: {run_dir}")
    print(f"Backbone: {config.backbone}")
    print(f"Negative:positive ratio: {config.neg_pos_ratio}")
    print(f"Base LR scaled: {args.base_lr} * ({config.batch_size}/256) = {scaled_lr:.6g}")

    if config.gpus and torch.cuda.is_available():
        device = torch.device(f"cuda:{config.gpus[0]}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    print("Building parquet dataloaders...")
    train_loader, val_loader, train_dataset, val_dataset, train_sampler = (
        build_parquet_dataloaders(config)
    )
    print(f"Train parquet class counts: {train_dataset.class_counts()}")
    print(f"Val parquet class counts:   {val_dataset.class_counts()}")

    print("Building model...")
    model, param_groups = build_model(config)
    model = model.to(device)
    if config.gpus and len(config.gpus) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpus)

    if config.optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(
            param_groups,
            betas=config.betas,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer_type == "sgd":
        optimizer = torch.optim.SGD(
            param_groups,
            momentum=0.9,
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer_type}")

    trainer = Trainer(config, device)

    metrics_path = run_dir / "metrics_epoch.csv"
    best_metrics_path = run_dir / "best_metrics.json"
    predictions_path = run_dir / "val_predictions.csv"
    epoch_indices_dir = run_dir / "epoch_indices"
    if config.save_epoch_indices:
        epoch_indices_dir.mkdir(parents=True, exist_ok=True)

    with open(metrics_path, "w") as f:
        f.write(
            "epoch,train_loss,val_loss,val_acc,val_precision,val_recall,val_f1,val_auc,"
            "val_weighted_precision,val_weighted_recall,val_weighted_f1,val_best_threshold,"
            "lr_backbone,lr_head,train_samples,train_pos,train_neg\n"
        )

    best_val_f1 = -1.0
    best_val_recall = -1.0

    for epoch in range(1, config.epochs + 1):
        train_sampler.set_epoch(epoch)
        train_counts = train_sampler.get_counts()
        if config.save_epoch_indices:
            np.save(
                epoch_indices_dir / f"epoch_{epoch:03d}_indices.npy",
                train_sampler.get_indices(),
            )

        print(f"\nEpoch {epoch}/{config.epochs}")
        print(f"Train sample counts: {train_counts}")

        train_metrics = trainer.train_one_epoch(model, train_loader, optimizer, epoch)
        print(
            f"Train Loss: {train_metrics['train_loss']:.4f} "
            f"Acc: {train_metrics['train_acc']:.4f} "
            f"F1: {train_metrics['train_f1']:.4f}"
        )

        val_metrics_raw, val_preds = trainer.validate(model, val_loader, epoch)
        labels = np.array([int(p["label"]) for p in val_preds], dtype=np.int64)
        probs = np.array([float(p["prob_header"]) for p in val_preds], dtype=np.float64)
        sweep = _threshold_sweep_positive_f1(
            labels=labels,
            probs=probs,
            step=config.f1_threshold_step,
        )

        pred_labels = sweep["pred_labels"]
        for i, pred in enumerate(val_preds):
            pred["pred_label"] = int(pred_labels[i])

        val_metrics = {
            "val_loss": float(val_metrics_raw["val_loss"]),
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
        print(
            f"Val Loss: {val_metrics['val_loss']:.4f} "
            f"Pos F1: {val_metrics['val_f1']:.4f} "
            f"(P={val_metrics['val_precision']:.4f}, R={val_metrics['val_recall']:.4f}, "
            f"thr={val_metrics['val_best_threshold']:.2f})"
        )

        lr_backbone = _get_group_lr(optimizer, "backbone")
        if lr_backbone is None:
            lr_backbone = _get_group_lr(optimizer, "block_0")
        if lr_backbone is None:
            lr_backbone = config.base_lr
        lr_head = _get_group_lr(optimizer, "head")
        if lr_head is None:
            lr_head = 0.0

        with open(metrics_path, "a") as f:
            f.write(
                f"{epoch},{train_metrics['train_loss']:.6f},"
                f"{val_metrics['val_loss']:.6f},{val_metrics['val_acc']:.6f},"
                f"{val_metrics['val_precision']:.6f},{val_metrics['val_recall']:.6f},"
                f"{val_metrics['val_f1']:.6f},{val_metrics['val_auc']:.6f},"
                f"{val_metrics['val_weighted_precision']:.6f},{val_metrics['val_weighted_recall']:.6f},"
                f"{val_metrics['val_weighted_f1']:.6f},{val_metrics['val_best_threshold']:.2f},"
                f"{lr_backbone:.6f},{lr_head:.6f},"
                f"{train_counts['samples']},{train_counts['positives']},{train_counts['negatives']}\n"
            )

        better = val_metrics["val_f1"] > best_val_f1
        if not better and np.isclose(val_metrics["val_f1"], best_val_f1):
            better = val_metrics["val_recall"] > best_val_recall

        if better:
            best_val_f1 = val_metrics["val_f1"]
            best_val_recall = val_metrics["val_recall"]
            print(f"New best positive F1: {best_val_f1:.4f}")

            checkpoint_name = f"best_epoch_{epoch}.pt"
            best_data = {
                "epoch": int(epoch),
                "checkpoint": f"checkpoints/{checkpoint_name}",
                "val_loss": val_metrics["val_loss"],
                "val_acc": val_metrics["val_acc"],
                "val_auc": val_metrics["val_auc"],
                "val_best_threshold": val_metrics["val_best_threshold"],
                "val_pos_precision": val_metrics["val_precision"],
                "val_pos_recall": val_metrics["val_recall"],
                "val_pos_f1": val_metrics["val_f1"],
                "val_weighted_precision": val_metrics["val_weighted_precision"],
                "val_weighted_recall": val_metrics["val_weighted_recall"],
                "val_weighted_f1": val_metrics["val_weighted_f1"],
                "train_samples": train_counts["samples"],
                "train_positives": train_counts["positives"],
                "train_negatives": train_counts["negatives"],
            }
            with open(best_metrics_path, "w") as f:
                json.dump(best_data, f, indent=4)

            save_predictions(val_preds, predictions_path)

            checkpoint_path = run_dir / "checkpoints" / checkpoint_name
            if isinstance(model, torch.nn.DataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": state_dict,
                    "optimizer_state": optimizer.state_dict(),
                    "config": vars(args),
                },
                checkpoint_path,
            )

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    generate_all_plots(run_dir, metrics_path, predictions_path)


if __name__ == "__main__":
    main()
