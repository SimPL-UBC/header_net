from dataclasses import dataclass, field
from typing import Optional, List
import argparse

@dataclass
class Config:
    # Data
    train_csv: str = ""
    val_csv: str = ""
    num_frames: int = 11  # Default to current cache value
    input_size: int = 224
    frame_sampling: str = "center"  # Phase 1: only "center" supported

    # Model
    backbone: str = "csn"  # Phase 1: only "csn"
    finetune_mode: str = "full"  # Phase 1: only "full"

    # Optimization
    optimizer_type: str = "sgd"  # Phase 1: "sgd"
    lr_backbone: float = 0.001
    weight_decay: float = 1e-4
    epochs: int = 50
    batch_size: int = 16
    num_workers: int = 8
    
    # Run
    run_name: str = ""
    seed: int = 42
    output_root: str = "report/header_experiments"
    gpus: Optional[List[int]] = None

def merge_cli_args(args: argparse.Namespace) -> Config:
    """
    Merges CLI arguments into a Config instance.
    """
    config = Config()
    
    # Data
    if hasattr(args, 'train_csv'): config.train_csv = args.train_csv
    if hasattr(args, 'val_csv'): config.val_csv = args.val_csv
    
    # Model
    if hasattr(args, 'backbone'): config.backbone = args.backbone
    if hasattr(args, 'finetune_mode'): config.finetune_mode = args.finetune_mode
    
    # Optimization
    if hasattr(args, 'lr_backbone'): config.lr_backbone = args.lr_backbone
    if hasattr(args, 'weight_decay'): config.weight_decay = args.weight_decay
    if hasattr(args, 'epochs'): config.epochs = args.epochs
    if hasattr(args, 'batch_size'): config.batch_size = args.batch_size
    if hasattr(args, 'num_workers'): config.num_workers = args.num_workers
    
    # Run
    if hasattr(args, 'run_name'): config.run_name = args.run_name
    if hasattr(args, 'seed'): config.seed = args.seed
    if hasattr(args, 'output_root'): config.output_root = args.output_root
    if hasattr(args, 'gpus'): config.gpus = args.gpus
    if hasattr(args, 'num_frames'): config.num_frames = args.num_frames
    
    return config
