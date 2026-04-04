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
    train_video_ids: List[str] = field(default_factory=list)
    train_halves: List[int] = field(default_factory=list)
    val_video_ids: List[str] = field(default_factory=list)
    val_halves: List[int] = field(default_factory=list)
    num_frames: int = 16  # Default to current cache value
    input_size: int = 224
    frame_sampling: str = "center"  # Phase 1: only "center" supported
    spatial_mode: str = "ball_crop"

    # Model
    backbone: str = "csn"  # Phase 1: only "csn"; Phase 2: "csn" or "vmae"
    finetune_mode: str = "full"  # Phase 1: only "full"; Phase 2: "full" or "frozen"; Phase 3: add "partial"
    unfreeze_blocks: int = 4  # Number of last transformer blocks to unfreeze for VideoMAE partial fine-tuning
    backbone_ckpt: Optional[str] = None  # Path to VideoMAE checkpoint
    gradient_checkpointing: Optional[bool] = None  # Runtime override for VideoMAE with_cp

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
    max_open_videos: int = 8
    frame_cache_size: int = 128
    loader_start_method: str = "spawn"

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
    amp: bool = False
    f1_threshold_step: float = 0.01
    save_epoch_indices: bool = True
    save_every_n_epochs: int = 1
    run_intermediate_validation: bool = True
    validate_every_n_epochs: int = 1
    val_neg_pos_ratio: str = "all"
    run_final_test: bool = True
    val_pin_memory: bool = False
    val_progress_every: int = 1000


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
    if hasattr(args, "spatial_mode"):
        config.spatial_mode = str(args.spatial_mode)
    if hasattr(args, "neg_pos_ratio"):
        config.neg_pos_ratio = str(args.neg_pos_ratio)
    if hasattr(args, "train_video_ids") and args.train_video_ids is not None:
        config.train_video_ids = [str(value) for value in args.train_video_ids]
    if hasattr(args, "train_halves") and args.train_halves is not None:
        config.train_halves = [int(value) for value in args.train_halves]
    if hasattr(args, "val_video_ids") and args.val_video_ids is not None:
        config.val_video_ids = [str(value) for value in args.val_video_ids]
    if hasattr(args, "val_halves") and args.val_halves is not None:
        config.val_halves = [int(value) for value in args.val_halves]

    # Model
    if hasattr(args, "backbone"):
        config.backbone = args.backbone
    if hasattr(args, "finetune_mode"):
        config.finetune_mode = args.finetune_mode
    if hasattr(args, "unfreeze_blocks"):
        config.unfreeze_blocks = args.unfreeze_blocks
    if hasattr(args, "backbone_ckpt"):
        config.backbone_ckpt = args.backbone_ckpt
    if hasattr(args, "gradient_checkpointing"):
        config.gradient_checkpointing = args.gradient_checkpointing

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
    if hasattr(args, "max_open_videos"):
        config.max_open_videos = args.max_open_videos
    if hasattr(args, "frame_cache_size"):
        config.frame_cache_size = args.frame_cache_size
    if hasattr(args, "loader_start_method"):
        config.loader_start_method = args.loader_start_method

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
    if hasattr(args, "amp"):
        config.amp = args.amp
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
    if hasattr(args, "save_every_n_epochs"):
        config.save_every_n_epochs = args.save_every_n_epochs
    if hasattr(args, "run_intermediate_validation"):
        config.run_intermediate_validation = args.run_intermediate_validation
    if hasattr(args, "validate_every_n_epochs"):
        config.validate_every_n_epochs = args.validate_every_n_epochs
    if hasattr(args, "val_neg_pos_ratio"):
        config.val_neg_pos_ratio = str(args.val_neg_pos_ratio)
    if hasattr(args, "run_final_test"):
        config.run_final_test = args.run_final_test
    if hasattr(args, "val_pin_memory"):
        config.val_pin_memory = args.val_pin_memory
    if hasattr(args, "val_progress_every"):
        config.val_progress_every = args.val_progress_every

    return config
