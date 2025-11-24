import argparse
import csv
import json
import logging
from pathlib import Path

import torch

from training.config import Config, build_config_from_args
from training.data.header_dataset import build_dataloaders
from training.engine.supervised_trainer import Trainer
from training.eval.predictions import save_predictions
from training.models.factory import build_model
from training.run_utils import create_run_dir, save_config, set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="VideoMAE v2 Header Training (frozen backbone)"
    )
    parser.add_argument("--train_csv", required=True, type=str)
    parser.add_argument("--val_csv", required=True, type=str)
    parser.add_argument("--backbone_ckpt", default=Config.backbone_ckpt, type=str)
    parser.add_argument("--lr_head", default=Config.lr_head, type=float)
    parser.add_argument("--num_frames", default=Config.num_frames, type=int)
    parser.add_argument("--input_size", default=Config.input_size, type=int)
    parser.add_argument(
        "--frame_sampling", default=Config.frame_sampling, type=str, choices=["center"]
    )
    parser.add_argument("--num_classes", default=Config.num_classes, type=int)
    parser.add_argument("--run_name", required=True, type=str)
    parser.add_argument("--output_root", default=Config.output_root, type=str)
    parser.add_argument("--epochs", default=Config.epochs, type=int)
    parser.add_argument("--batch_size", default=Config.batch_size, type=int)
    parser.add_argument("--num_workers", default=Config.num_workers, type=int)
    parser.add_argument("--weight_decay", default=Config.weight_decay, type=float)
    parser.add_argument("--optimizer_type", default=Config.optimizer_type, type=str)
    parser.add_argument("--seed", default=Config.seed, type=int)
    parser.add_argument("--gpus", nargs="+", type=int, default=None)
    return parser.parse_args()


def _setup_device(gpus):
    if gpus and torch.cuda.is_available():
        torch.cuda.set_device(gpus[0])
        return torch.device(f"cuda:{gpus[0]}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _write_metrics_row(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_acc",
        "val_precision",
        "val_recall",
        "val_f1",
        "val_auc",
        "lr_head",
    ]
    write_header = not csv_path.exists()
    try:
        with csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if write_header:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in header})
    except Exception as exc:
        logging.error("Failed to write metrics to %s: %s", csv_path, exc)
        raise


def _validate_param_freezing(model: torch.nn.Module) -> None:
    """Ensure the backbone is frozen and only the head is trainable."""
    if not hasattr(model, "backbone") or not hasattr(model, "head"):
        return

    backbone_trainable = any(p.requires_grad for p in model.backbone.parameters())
    head_trainable = any(p.requires_grad for p in model.head.parameters())

    if backbone_trainable:
        raise RuntimeError("Backbone parameters must be frozen for Phase 2 (VideoMAE v2).")
    if not head_trainable:
        raise RuntimeError("Classification head parameters are not trainable.")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = parse_args()

    config = build_config_from_args(args)
    set_seed(config.seed)

    run_dir = create_run_dir(config.output_root, config.run_name)
    logging.info("Created run directory at %s", run_dir)
    save_config(config, run_dir)
    if not (run_dir / "config.yaml").exists():
        raise RuntimeError(f"config.yaml not found in {run_dir} after save_config")

    device = _setup_device(args.gpus)
    logging.info("Using device: %s", device)

    train_loader, val_loader = build_dataloaders(config)
    model, param_groups = build_model(config)
    _validate_param_freezing(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    logging.info(
        "Parameter counts - trainable (head): %d | frozen (backbone): %d",
        trainable_params,
        frozen_params,
    )
    model.to(device)
    if args.gpus and len(args.gpus) > 1 and torch.cuda.device_count() >= len(args.gpus):
        model = torch.nn.DataParallel(model, device_ids=args.gpus)
        logging.info("Wrapped model in DataParallel on GPUs: %s", args.gpus)

    optimizer_type = str(config.optimizer_type).lower()
    if optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(
            param_groups, lr=config.lr_head, weight_decay=config.weight_decay
        )
    elif optimizer_type == "adam":
        optimizer = torch.optim.Adam(
            param_groups, lr=config.lr_head, weight_decay=config.weight_decay
        )
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(
            param_groups, lr=config.lr_head, momentum=0.9, weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer type for Phase 2: {config.optimizer_type}")

    trainer = Trainer(config, device)

    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = run_dir / "metrics_epoch.csv"
    best_metrics_path = run_dir / "best_metrics.json"
    val_pred_path = run_dir / "val_predictions.csv"

    best_f1 = -float("inf")

    for epoch in range(config.epochs):
        train_metrics = trainer.train_one_epoch(model, train_loader, optimizer, epoch + 1)
        val_metrics, predictions = trainer.validate(model, val_loader, epoch + 1)

        metrics_row = {
            "epoch": epoch + 1,
            "train_loss": float(train_metrics["train_loss"]),
            "val_loss": float(val_metrics.get("val_loss", 0.0)),
            "val_acc": float(val_metrics.get("val_acc", 0.0)),
            "val_precision": float(val_metrics.get("val_precision", 0.0)),
            "val_recall": float(val_metrics.get("val_recall", 0.0)),
            "val_f1": float(val_metrics.get("val_f1", 0.0)),
            "val_auc": float(val_metrics.get("val_auc", 0.0)),
            "lr_head": float(optimizer.param_groups[0].get("lr", config.lr_head)),
        }
        _write_metrics_row(metrics_csv, metrics_row)
        if not metrics_csv.exists():
            raise RuntimeError(f"metrics_epoch.csv not written to {metrics_csv}")
        logging.info("Epoch %d metrics appended to %s", epoch + 1, metrics_csv)

        if metrics_row["val_f1"] > best_f1:
            best_f1 = metrics_row["val_f1"]
            checkpoint_path = checkpoints_dir / f"best_epoch_{epoch + 1}.pt"
            try:
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "state_dict": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "config": config.__dict__,
                    },
                    checkpoint_path,
                )
            except Exception as exc:
                logging.error("Failed to save checkpoint %s: %s", checkpoint_path, exc)
                raise
            if not checkpoint_path.exists():
                raise RuntimeError(f"Checkpoint not saved to {checkpoint_path}")
            logging.info("Saved checkpoint for epoch %d to %s", epoch + 1, checkpoint_path)

            best_payload = {
                "epoch": epoch + 1,
                "val_loss": metrics_row["val_loss"],
                "val_acc": metrics_row["val_acc"],
                "val_precision": metrics_row["val_precision"],
                "val_recall": metrics_row["val_recall"],
                "val_f1": metrics_row["val_f1"],
                "val_auc": metrics_row["val_auc"],
                "checkpoint": str(checkpoint_path.relative_to(run_dir)),
            }
            try:
                with best_metrics_path.open("w") as f:
                    json.dump(best_payload, f, indent=2)
            except Exception as exc:
                logging.error("Failed to write best_metrics.json to %s: %s", best_metrics_path, exc)
                raise
            if not best_metrics_path.exists():
                raise RuntimeError(f"best_metrics.json not written to {best_metrics_path}")
            logging.info("Updated best metrics at %s", best_metrics_path)

            save_predictions(predictions, val_pred_path)
            if not val_pred_path.exists():
                raise RuntimeError(f"val_predictions.csv not written to {val_pred_path}")
            logging.info("Saved validation predictions for epoch %d to %s", epoch + 1, val_pred_path)


if __name__ == "__main__":
    main()
