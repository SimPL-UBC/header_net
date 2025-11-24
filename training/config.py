from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class Config:
    # Data
    train_csv: str
    val_csv: str
    backbone_ckpt: str = "checkpoints/VideoMAEv2-Base"
    num_frames: int = 16
    input_size: int = 224
    frame_sampling: str = "center"
    num_classes: int = 2

    # Model
    lr_head: float = 1e-3

    # Optimization
    optimizer_type: str = "adamw"
    weight_decay: float = 1e-4
    epochs: int = 50
    batch_size: int = 16
    num_workers: int = 8

    # Run
    run_name: str = ""
    seed: int = 42
    output_root: str = "report/header_experiments"


def build_config_from_args(args: Any) -> Config:
    """
    Merge CLI args into a Config instance. Assumes argparse.Namespace-like input.
    """
    return Config(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        backbone_ckpt=args.backbone_ckpt,
        num_frames=getattr(args, "num_frames", Config.num_frames),
        input_size=getattr(args, "input_size", Config.input_size),
        frame_sampling=getattr(args, "frame_sampling", Config.frame_sampling),
        num_classes=getattr(args, "num_classes", Config.num_classes),
        lr_head=args.lr_head,
        optimizer_type=getattr(args, "optimizer_type", Config.optimizer_type),
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        run_name=args.run_name,
        seed=args.seed,
        output_root=args.output_root,
    )


def config_to_dict(config: Config) -> dict:
    """Convert Config dataclass to a dictionary for serialization."""
    return asdict(config)
