import argparse
import json

import numpy as np
import torch

from .config import merge_cli_args
from .data.parquet_header_dataset import build_parquet_train_dataloader
from .engine.supervised_trainer import Trainer
from .models.factory import build_model
from .run_utils import create_run_dir, save_config, set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Header Net on parquet metadata (training only)."
    )
    parser.add_argument(
        "--train_parquet",
        required=True,
        help="Path to train parquet file or partitioned parquet dataset directory",
    )
    parser.add_argument(
        "--dataset_root",
        default="SoccerNet",
        help="SoccerNet root directory (validated in strict path mode)",
    )
    parser.add_argument(
        "--neg_pos_ratio",
        default="all",
        help="Train negative:positive ratio; 'all' or a positive integer",
    )
    parser.add_argument(
        "--train_video_ids",
        nargs="+",
        default=None,
        help="Optional train video_id filter(s)",
    )
    parser.add_argument(
        "--train_halves",
        type=int,
        nargs="+",
        default=None,
        help="Optional train half filter(s)",
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
        "--backbone_ckpt",
        default=None,
        help="Path to VideoMAE checkpoint directory",
    )
    parser.add_argument("--run_name", required=True, help="Run name")
    parser.add_argument("--output_root", default="output/vmae", help="Output root")

    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers")
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
        help="Multiprocessing start method for DataLoader workers",
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
        "--save_epoch_indices",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save sampled train row indices per epoch for reproducibility",
    )
    parser.add_argument(
        "--save_every_n_epochs",
        type=int,
        default=1,
        help="Save model checkpoint every N epochs (N must be >= 1)",
    )
    parser.add_argument("--gpus", type=int, nargs="+", help="GPU IDs")
    return parser.parse_args()


def _get_group_lr(optimizer, group_name):
    for group in optimizer.param_groups:
        if group.get("name") == group_name:
            return group.get("lr")
    return None


def _validate_optional_ratio(value, arg_name):
    text = str(value).strip().lower()
    if text == "all":
        return
    try:
        ratio = int(text)
    except ValueError as exc:
        raise ValueError(f"{arg_name} must be 'all' or a positive integer.") from exc
    if ratio <= 0:
        raise ValueError(f"{arg_name} must be 'all' or a positive integer.")


def _save_checkpoint(model, optimizer, args, path, epoch):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    torch.save(
        {
            "epoch": int(epoch),
            "state_dict": state_dict,
            "optimizer_state": optimizer.state_dict(),
            "config": vars(args),
        },
        path,
    )


def main():
    args = parse_args()
    config = merge_cli_args(args)

    _validate_optional_ratio(config.neg_pos_ratio, "neg_pos_ratio")
    if int(config.save_every_n_epochs) < 1:
        raise ValueError("save_every_n_epochs must be >= 1")

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
    print(
        f"Base LR scaled: {args.base_lr} * ({config.batch_size}/256) = {scaled_lr:.6g}"
    )
    print(f"Dataloader workers: train={config.num_workers}")
    print(
        "Loader settings: "
        f"max_open_videos={config.max_open_videos}, "
        f"frame_cache_size={config.frame_cache_size}, "
        f"start_method={config.loader_start_method}"
    )
    print(f"Save checkpoint every N epochs: {config.save_every_n_epochs}")

    if config.gpus and torch.cuda.is_available():
        device = torch.device(f"cuda:{config.gpus[0]}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    print("Building parquet train dataloader...")
    train_loader, train_dataset, train_sampler = build_parquet_train_dataloader(config)
    print(f"Train parquet class counts: {train_dataset.class_counts()}")

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

    metrics_path = run_dir / "metrics_train_epoch.csv"
    artifacts_path = run_dir / "train_artifacts.json"
    epoch_indices_dir = run_dir / "epoch_indices"
    if config.save_epoch_indices:
        epoch_indices_dir.mkdir(parents=True, exist_ok=True)

    with open(metrics_path, "w") as f:
        f.write(
            "epoch,train_loss,train_acc,train_f1,lr_backbone,lr_head,"
            "train_samples,train_pos,train_neg,checkpoint\n"
        )

    last_epoch = 0
    for epoch in range(1, config.epochs + 1):
        last_epoch = epoch
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

        lr_backbone = _get_group_lr(optimizer, "backbone")
        if lr_backbone is None:
            lr_backbone = _get_group_lr(optimizer, "block_0")
        if lr_backbone is None:
            lr_backbone = config.base_lr
        lr_head = _get_group_lr(optimizer, "head")
        if lr_head is None:
            lr_head = 0.0

        checkpoint_rel = ""
        if epoch % int(config.save_every_n_epochs) == 0:
            checkpoint_name = f"epoch_{epoch:03d}.pt"
            checkpoint_path = run_dir / "checkpoints" / checkpoint_name
            _save_checkpoint(model, optimizer, args, checkpoint_path, epoch)
            checkpoint_rel = f"checkpoints/{checkpoint_name}"
            print(f"Saved periodic checkpoint: {checkpoint_path}")

        with open(metrics_path, "a") as f:
            f.write(
                f"{epoch},{train_metrics['train_loss']:.6f},"
                f"{train_metrics['train_acc']:.6f},{train_metrics['train_f1']:.6f},"
                f"{lr_backbone:.6f},{lr_head:.6f},"
                f"{train_counts['samples']},{train_counts['positives']},{train_counts['negatives']},"
                f"{checkpoint_rel}\n"
            )

    final_checkpoint_path = run_dir / "checkpoints" / "last.pt"
    _save_checkpoint(model, optimizer, args, final_checkpoint_path, last_epoch)
    print(f"Saved final checkpoint: {final_checkpoint_path}")

    with open(artifacts_path, "w") as f:
        json.dump(
            {
                "last_epoch": int(last_epoch),
                "final_checkpoint": "checkpoints/last.pt",
                "metrics_file": metrics_path.name,
            },
            f,
            indent=4,
        )
    print(f"Training complete. Artifacts: {artifacts_path}")


if __name__ == "__main__":
    main()
