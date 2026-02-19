from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import argparse


@dataclass
class Config:
    # Data
    train_csv: str = ""
    val_csv: str = ""
    train_parquet: str = ""
    val_parquet: str = ""
    dataset_root: str = ""
    neg_pos_ratio: str = "all"
    num_frames: int = 16  # Default to current cache value
    input_size: int = 224
    frame_sampling: str = "center"  # Phase 1: only "center" supported

    # Model
    backbone: str = "csn"  # Phase 1: only "csn"; Phase 2: "csn" or "vmae"
    finetune_mode: str = "full"  # Phase 1: only "full"; Phase 2: "full" or "frozen"; Phase 3: add "partial"
    unfreeze_blocks: int = 4  # Number of last transformer blocks to unfreeze for VideoMAE partial fine-tuning
    backbone_ckpt: Optional[str] = None  # Path to VideoMAE checkpoint

    # Optimization
    optimizer_type: str = "adamw"
    base_lr: float = 1e-3
    lr_backbone: float = 0.001
    lr_head: float = 1e-3  # Learning rate for VideoMAE head
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.05
    epochs: int = 50
    batch_size: int = 16
    num_workers: int = 8
    val_num_workers: int = 0

    # Loss
    loss_type: str = "focal"
    focal_gamma: float = 2.0
    focal_alpha: float = 0.75

    # Run
    run_name: str = ""
    seed: int = 42
    output_root: str = "report/header_experiments"
    gpus: Optional[List[int]] = None
    layer_lr_decay: float = 0.75
    f1_threshold_step: float = 0.01
    save_epoch_indices: bool = True


def merge_cli_args(args: argparse.Namespace) -> Config:
    """
    Merges CLI arguments into a Config instance.
    """
    config = Config()

    # Data
    if hasattr(args, "train_csv"):
        config.train_csv = args.train_csv
    if hasattr(args, "val_csv"):
        config.val_csv = args.val_csv
    if hasattr(args, "train_parquet"):
        config.train_parquet = args.train_parquet
    if hasattr(args, "val_parquet"):
        config.val_parquet = args.val_parquet
    if hasattr(args, "dataset_root"):
        config.dataset_root = args.dataset_root
    if hasattr(args, "neg_pos_ratio"):
        config.neg_pos_ratio = str(args.neg_pos_ratio)

    # Model
    if hasattr(args, "backbone"):
        config.backbone = args.backbone
    if hasattr(args, "finetune_mode"):
        config.finetune_mode = args.finetune_mode
    if hasattr(args, "unfreeze_blocks"):
        config.unfreeze_blocks = args.unfreeze_blocks
    if hasattr(args, "backbone_ckpt"):
        config.backbone_ckpt = args.backbone_ckpt

    # Optimization
    if hasattr(args, "lr_backbone"):
        config.lr_backbone = args.lr_backbone
    if hasattr(args, "lr_head"):
        config.lr_head = args.lr_head
    if hasattr(args, "base_lr"):
        config.base_lr = args.base_lr
    if hasattr(args, "betas"):
        config.betas = tuple(args.betas)
    if hasattr(args, "weight_decay"):
        config.weight_decay = args.weight_decay
    if hasattr(args, "optimizer"):
        config.optimizer_type = args.optimizer
    if hasattr(args, "epochs"):
        config.epochs = args.epochs
    if hasattr(args, "batch_size"):
        config.batch_size = args.batch_size
    if hasattr(args, "num_workers"):
        config.num_workers = args.num_workers
    if hasattr(args, "val_num_workers"):
        config.val_num_workers = args.val_num_workers

    # Run
    if hasattr(args, "run_name"):
        config.run_name = args.run_name
    if hasattr(args, "seed"):
        config.seed = args.seed
    if hasattr(args, "output_root"):
        config.output_root = args.output_root
    if hasattr(args, "gpus"):
        config.gpus = args.gpus
    if hasattr(args, "num_frames"):
        config.num_frames = args.num_frames
    if hasattr(args, "layer_lr_decay"):
        config.layer_lr_decay = args.layer_lr_decay
    if hasattr(args, "loss"):
        config.loss_type = args.loss
    if hasattr(args, "focal_gamma"):
        config.focal_gamma = args.focal_gamma
    if hasattr(args, "focal_alpha"):
        config.focal_alpha = args.focal_alpha
    if hasattr(args, "f1_threshold_step"):
        config.f1_threshold_step = args.f1_threshold_step
    if hasattr(args, "save_epoch_indices"):
        config.save_epoch_indices = args.save_epoch_indices

    return config
