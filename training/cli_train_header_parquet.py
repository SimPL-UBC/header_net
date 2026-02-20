import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from .config import merge_cli_args
from .data.parquet_header_dataset import (
    DeterministicRatioSampler,
    build_parquet_dataloaders,
)
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
        "--val_num_workers",
        type=int,
        default=0,
        help="Validation DataLoader workers (use 0 for stability)",
    )
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
    parser.add_argument(
        "--run_intermediate_validation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable validation during training epochs",
    )
    parser.add_argument(
        "--validate_every_n_epochs",
        type=int,
        default=1,
        help="When intermediate validation is enabled, run it every N epochs",
    )
    parser.add_argument(
        "--val_neg_pos_ratio",
        default="all",
        help=(
            "Intermediate validation negative:positive ratio. "
            "Use 'all' for full validation set, or a positive integer (e.g., 15)."
        ),
    )
    parser.add_argument(
        "--run_final_test",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run one parquet evaluation after training finishes",
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


def _run_validation_with_retry(trainer, model, val_loader, config, epoch_label):
    try:
        val_metrics_raw, val_preds = trainer.validate(model, val_loader, epoch_label)
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
        retry_batch_size = int(getattr(val_loader, "batch_size", config.batch_size))

        val_loader = torch.utils.data.DataLoader(
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
        val_metrics_raw, val_preds = trainer.validate(model, val_loader, epoch_label)

    if not val_preds:
        raise RuntimeError("Validation produced no predictions.")

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
    return val_loader, val_metrics, val_preds


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
    print(
        "Dataloader workers: "
        f"train={config.num_workers}, val={config.val_num_workers}"
    )
    print(f"Run intermediate validation: {config.run_intermediate_validation}")
    print(f"Validation cadence (if enabled): every {config.validate_every_n_epochs} epoch(s)")
    print(f"Intermediate val neg:pos ratio: {config.val_neg_pos_ratio}")
    print(f"Run final test: {config.run_final_test}")

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

    validate_every_n_epochs = int(config.validate_every_n_epochs)
    if config.run_intermediate_validation and validate_every_n_epochs <= 0:
        raise ValueError("validate_every_n_epochs must be >= 1")
    if validate_every_n_epochs <= 0:
        validate_every_n_epochs = 1
    intermediate_val_ratio = _parse_optional_ratio(
        config.val_neg_pos_ratio,
        "val_neg_pos_ratio",
    )

    best_val_f1 = -1.0
    best_val_recall = -1.0
    best_checkpoint_path = None
    intermediate_val_loader = val_loader
    intermediate_val_sampler = None

    if config.run_intermediate_validation and intermediate_val_ratio is not None:
        if len(val_dataset.positive_indices) == 0:
            raise ValueError(
                "Validation parquet has no positive samples but val_neg_pos_ratio was set."
            )
        intermediate_val_sampler = DeterministicRatioSampler(
            positive_indices=val_dataset.positive_indices,
            negative_indices=val_dataset.negative_indices,
            neg_pos_ratio=intermediate_val_ratio,
            seed=config.seed + 1000,
            shuffle=True,
        )
        intermediate_val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            sampler=intermediate_val_sampler,
            shuffle=False,
            num_workers=config.val_num_workers,
            pin_memory=False,
        )
        print(
            "Intermediate validation sampling enabled: "
            f"all positives + {intermediate_val_ratio} negatives per positive."
        )
    elif config.run_intermediate_validation:
        print("Intermediate validation sampling disabled: using full validation set.")

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

        should_validate = (
            bool(config.run_intermediate_validation)
            and (epoch % validate_every_n_epochs == 0)
        )
        if should_validate:
            active_val_loader = val_loader
            uses_ratio_sample = intermediate_val_sampler is not None
            if uses_ratio_sample:
                intermediate_val_sampler.set_epoch(epoch)
                active_val_loader = intermediate_val_loader
                print(
                    "Intermediate val sample counts: "
                    f"{intermediate_val_sampler.get_counts()}"
                )
            else:
                print(
                    "Intermediate val sample counts: "
                    f"{val_dataset.class_counts()}"
                )

            active_val_loader, val_metrics, val_preds = _run_validation_with_retry(
                trainer=trainer,
                model=model,
                val_loader=active_val_loader,
                config=config,
                epoch_label=epoch,
            )
            if uses_ratio_sample:
                intermediate_val_loader = active_val_loader
            else:
                val_loader = active_val_loader

            print(
                f"Val Loss: {val_metrics['val_loss']:.4f} "
                f"Pos F1: {val_metrics['val_f1']:.4f} "
                f"(P={val_metrics['val_precision']:.4f}, R={val_metrics['val_recall']:.4f}, "
                f"thr={val_metrics['val_best_threshold']:.2f})"
            )
        else:
            if config.run_intermediate_validation:
                print(
                    f"Skipping validation at epoch {epoch} "
                    f"(validate_every_n_epochs={validate_every_n_epochs})."
                )
            else:
                print(
                    f"Skipping validation at epoch {epoch} "
                    "(run_intermediate_validation=False)."
                )
            val_metrics = {
                "val_loss": float("nan"),
                "val_acc": float("nan"),
                "val_precision": float("nan"),
                "val_recall": float("nan"),
                "val_f1": float("nan"),
                "val_auc": float("nan"),
                "val_weighted_precision": float("nan"),
                "val_weighted_recall": float("nan"),
                "val_weighted_f1": float("nan"),
                "val_best_threshold": float("nan"),
            }
            val_preds = None

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

        if should_validate:
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
                best_checkpoint_path = checkpoint_path

    if config.run_final_test:
        print("\nRunning final test after training...")
        print(
            "Final test sample counts (full validation set): "
            f"{val_dataset.class_counts()}"
        )
        checkpoint_used = "final_in_memory_state"
        if best_checkpoint_path is not None and best_checkpoint_path.exists():
            checkpoint = torch.load(best_checkpoint_path, map_location=device)
            state_dict = checkpoint["state_dict"]
            if isinstance(model, torch.nn.DataParallel):
                model.module.load_state_dict(state_dict)
            else:
                model.load_state_dict(state_dict)
            checkpoint_used = str(best_checkpoint_path.name)
            print(f"Loaded best checkpoint for final test: {checkpoint_used}")
        else:
            print(
                "Best checkpoint not found; final test will use the last in-memory model state."
            )

        val_loader, final_test_metrics, final_test_preds = _run_validation_with_retry(
            trainer=trainer,
            model=model,
            val_loader=val_loader,
            config=config,
            epoch_label="final_test",
        )

        final_test_predictions_path = run_dir / "final_test_predictions.csv"
        final_test_metrics_path = run_dir / "final_test_metrics.json"
        save_predictions(final_test_preds, final_test_predictions_path)
        with open(final_test_metrics_path, "w") as f:
            json.dump(
                {
                    "checkpoint_used": checkpoint_used,
                    "metrics": final_test_metrics,
                },
                f,
                indent=4,
            )
        print(
            f"Final test complete. Metrics: {final_test_metrics_path}, "
            f"Predictions: {final_test_predictions_path}"
        )

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    generate_all_plots(run_dir, metrics_path, predictions_path)


if __name__ == "__main__":
    main()
