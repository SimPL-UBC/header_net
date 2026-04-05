import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from .config import merge_cli_args


@dataclass(frozen=True)
class DistributedRuntime:
    is_distributed: bool
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


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
        "--spatial_mode",
        choices=("ball_crop", "full_frame"),
        default="ball_crop",
        help="Spatial preprocessing policy for parquet clips",
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
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override VideoMAE gradient checkpointing (with_cp). "
            "Default uses the checkpoint config."
        ),
    )
    parser.add_argument("--run_name", required=True, help="Run name")
    parser.add_argument("--output_root", default="output/vmae", help="Output root")
    parser.add_argument(
        "--resume_checkpoint",
        default=None,
        help="Optional checkpoint path to resume model and optimizer state from",
    )

    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of microbatches to accumulate before each optimizer step",
    )
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
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable CUDA automatic mixed precision training",
    )
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


def _effective_batch_size(config, world_size=1):
    return (
        int(config.batch_size)
        * int(config.gradient_accumulation_steps)
        * int(world_size)
    )


def _should_use_data_parallel(config):
    return bool(
        config.gpus
        and len(config.gpus) > 1
        and int(config.batch_size) >= len(config.gpus)
    )


def _save_checkpoint(model, optimizer, args, path, epoch, scaler=None):
    checkpoint_model = _checkpoint_model(model)
    state_dict = checkpoint_model.state_dict()

    checkpoint = {
        "epoch": int(epoch),
        "state_dict": state_dict,
        "optimizer_state": optimizer.state_dict(),
        "config": vars(args),
    }
    if scaler is not None:
        checkpoint["scaler_state"] = scaler.state_dict()

    torch.save(checkpoint, path)


def _checkpoint_model(model):
    parallel_types = [torch.nn.DataParallel]
    distributed_parallel = getattr(torch.nn.parallel, "DistributedDataParallel", None)
    if distributed_parallel is not None:
        parallel_types.append(distributed_parallel)
    if isinstance(model, tuple(parallel_types)):
        return model.module
    return model


def _move_optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _load_resume_checkpoint(model, optimizer, checkpoint_path, device, scaler=None):
    resume_path = Path(checkpoint_path)
    if not resume_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

    checkpoint = torch.load(resume_path, map_location="cpu")
    checkpoint_model = _checkpoint_model(model)
    load_result = checkpoint_model.load_state_dict(checkpoint["state_dict"], strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            "Strict checkpoint load reported key mismatches: "
            f"missing={load_result.missing_keys}, "
            f"unexpected={load_result.unexpected_keys}"
        )

    optimizer_state = checkpoint.get("optimizer_state")
    if optimizer_state is None:
        raise RuntimeError(
            f"Resume checkpoint is missing optimizer_state: {resume_path}"
        )
    optimizer.load_state_dict(optimizer_state)
    _move_optimizer_state_to_device(optimizer, device)
    if scaler is not None and "scaler_state" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state"])

    completed_epoch = int(checkpoint.get("epoch", 0))
    return resume_path, completed_epoch


def _ensure_metrics_file(metrics_path, append=False):
    if append and metrics_path.exists():
        return

    with open(metrics_path, "w") as f:
        f.write(
            "epoch,train_loss,train_acc,train_f1,lr_backbone,lr_head,"
            "train_samples,train_pos,train_neg,checkpoint\n"
        )


def _detect_distributed_runtime():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return DistributedRuntime(False)
    return DistributedRuntime(
        is_distributed=True,
        rank=int(os.environ.get("RANK", "0")),
        local_rank=int(os.environ.get("LOCAL_RANK", "0")),
        world_size=world_size,
    )


def _initialize_distributed_runtime():
    runtime = _detect_distributed_runtime()
    if not runtime.is_distributed:
        return runtime
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed parquet training requires CUDA.")
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this build.")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(runtime.local_rank)
    return runtime


def _distributed_barrier(runtime):
    if runtime.is_distributed and dist.is_available() and dist.is_initialized():
        dist.barrier()


def _destroy_distributed_runtime(runtime):
    if runtime.is_distributed and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _log(runtime, message):
    if runtime.is_main_process:
        print(message)


def _reduce_train_stats(stats, device, runtime):
    reduced = {
        "loss_sum": float(stats["loss_sum"]),
        "sample_count": int(stats["sample_count"]),
        "tp": int(stats["tp"]),
        "fp": int(stats["fp"]),
        "fn": int(stats["fn"]),
        "tn": int(stats["tn"]),
    }
    if not runtime.is_distributed:
        return reduced

    packed = torch.tensor(
        [
            reduced["loss_sum"],
            float(reduced["sample_count"]),
            float(reduced["tp"]),
            float(reduced["fp"]),
            float(reduced["fn"]),
            float(reduced["tn"]),
        ],
        device=device,
        dtype=torch.float64,
    )
    dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    return {
        "loss_sum": float(packed[0].item()),
        "sample_count": int(round(packed[1].item())),
        "tp": int(round(packed[2].item())),
        "fp": int(round(packed[3].item())),
        "fn": int(round(packed[4].item())),
        "tn": int(round(packed[5].item())),
    }


def _finalize_train_metrics(train_stats):
    sample_count = int(train_stats["sample_count"])
    tp = int(train_stats["tp"])
    fp = int(train_stats["fp"])
    fn = int(train_stats["fn"])
    tn = int(train_stats["tn"])
    train_loss = float(train_stats["loss_sum"]) / sample_count if sample_count > 0 else 0.0
    train_acc = float(tp + tn) / sample_count if sample_count > 0 else 0.0
    f1_denominator = (2 * tp) + fp + fn
    train_f1 = (2.0 * tp) / float(f1_denominator) if f1_denominator > 0 else 0.0
    return {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "train_f1": train_f1,
    }


def main():
    from .data.parquet_header_dataset import build_parquet_train_dataloader
    from .engine.supervised_trainer import Trainer
    from .models.factory import build_model
    from .run_utils import create_run_dir, save_config, set_seed

    runtime = _initialize_distributed_runtime()
    try:
        args = parse_args()
        config = merge_cli_args(args)
        config.is_main_process = runtime.is_main_process
        config.is_distributed = runtime.is_distributed
        config.distributed_rank = runtime.rank
        config.distributed_world_size = runtime.world_size

        _validate_optional_ratio(config.neg_pos_ratio, "neg_pos_ratio")
        if int(config.save_every_n_epochs) < 1:
            raise ValueError("save_every_n_epochs must be >= 1")
        if int(config.gradient_accumulation_steps) < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1")

        effective_batch_size = _effective_batch_size(config, runtime.world_size)
        scale = effective_batch_size / 256.0
        scaled_lr = config.base_lr * scale
        config.base_lr = scaled_lr

        set_seed(config.seed)
        run_dir = Path(config.output_root) / config.run_name
        if runtime.is_main_process:
            run_dir = create_run_dir(config.output_root, config.run_name)
            save_config(config, run_dir)
        _distributed_barrier(runtime)

        _log(runtime, f"Starting run: {config.run_name}")
        _log(runtime, f"Output directory: {run_dir}")
        _log(runtime, f"Backbone: {config.backbone}")
        _log(runtime, f"Spatial mode: {config.spatial_mode}")
        _log(runtime, f"Negative:positive ratio: {config.neg_pos_ratio}")
        _log(
            runtime,
            f"Base LR scaled: {args.base_lr} * ({effective_batch_size}/256) = {scaled_lr:.6g}",
        )
        _log(
            runtime,
            f"Microbatch size: {config.batch_size}; "
            f"gradient accumulation steps: {config.gradient_accumulation_steps}; "
            f"effective batch size: {effective_batch_size}",
        )
        _log(runtime, f"Dataloader workers: train={config.num_workers}")
        _log(
            runtime,
            "Loader settings: "
            f"max_open_videos={config.max_open_videos}, "
            f"frame_cache_size={config.frame_cache_size}, "
            f"start_method={config.loader_start_method}",
        )
        _log(runtime, f"AMP: {config.amp}")
        _log(
            runtime,
            "Gradient checkpointing override: "
            f"{config.gradient_checkpointing if config.gradient_checkpointing is not None else 'checkpoint_default'}",
        )
        _log(runtime, f"Save checkpoint every N epochs: {config.save_every_n_epochs}")

        if runtime.is_distributed:
            device = torch.device(f"cuda:{runtime.local_rank}")
            _log(
                runtime,
                f"Using DDP runtime: rank={runtime.rank}, local_rank={runtime.local_rank}, "
                f"world_size={runtime.world_size}",
            )
        elif config.gpus and torch.cuda.is_available():
            device = torch.device(f"cuda:{config.gpus[0]}")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        _log(runtime, f"Using device: {device}")

        _log(runtime, "Building parquet train dataloader...")
        train_loader, train_dataset, train_sampler = build_parquet_train_dataloader(
            config,
            num_replicas=runtime.world_size,
            rank=runtime.rank,
        )
        _log(runtime, f"Train parquet class counts: {train_dataset.class_counts()}")

        _log(runtime, "Building model...")
        model, param_groups = build_model(config)
        model = model.to(device)
        if runtime.is_distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[runtime.local_rank],
                output_device=runtime.local_rank,
                find_unused_parameters=False,
            )
            _log(runtime, f"Using DistributedDataParallel across {runtime.world_size} ranks")
        elif _should_use_data_parallel(config):
            model = torch.nn.DataParallel(model, device_ids=config.gpus)
            _log(runtime, f"Using DataParallel on GPUs: {config.gpus}")
        elif config.gpus and len(config.gpus) > 1:
            _log(
                runtime,
                "Skipping DataParallel because microbatch size is smaller than the "
                f"number of requested GPUs: batch_size={config.batch_size}, gpus={config.gpus}",
            )

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
        if config.save_epoch_indices and runtime.is_main_process:
            epoch_indices_dir.mkdir(parents=True, exist_ok=True)

        start_epoch = 1
        resumed_from = None
        if args.resume_checkpoint:
            resumed_from, completed_epoch = _load_resume_checkpoint(
                model,
                optimizer,
                args.resume_checkpoint,
                device,
                trainer.scaler,
            )
            start_epoch = completed_epoch + 1
            if start_epoch > config.epochs:
                raise ValueError(
                    f"Resume checkpoint {resumed_from} is already at epoch {completed_epoch}, "
                    f"which is beyond requested total epochs={config.epochs}."
                )
            _log(
                runtime,
                f"Resuming from checkpoint: {resumed_from} "
                f"(completed epoch {completed_epoch}, next epoch {start_epoch})",
            )

        if runtime.is_main_process:
            _ensure_metrics_file(metrics_path, append=start_epoch > 1)
        _distributed_barrier(runtime)

        last_epoch = start_epoch - 1
        for epoch in range(start_epoch, config.epochs + 1):
            last_epoch = epoch
            train_sampler.set_epoch(epoch)
            train_counts = train_sampler.get_counts()
            if config.save_epoch_indices and runtime.is_main_process:
                epoch_indices = (
                    train_sampler.get_global_indices()
                    if hasattr(train_sampler, "get_global_indices")
                    else train_sampler.get_indices()
                )
                np.save(
                    epoch_indices_dir / f"epoch_{epoch:03d}_indices.npy",
                    epoch_indices,
                )

            _log(runtime, f"\nEpoch {epoch}/{config.epochs}")
            _log(runtime, f"Train sample counts: {train_counts}")

            train_stats = trainer.train_one_epoch(model, train_loader, optimizer, epoch)
            reduced_train_stats = _reduce_train_stats(train_stats, device, runtime)
            train_metrics = _finalize_train_metrics(reduced_train_stats)
            _log(
                runtime,
                f"Train Loss: {train_metrics['train_loss']:.4f} "
                f"Acc: {train_metrics['train_acc']:.4f} "
                f"F1: {train_metrics['train_f1']:.4f}",
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
            if runtime.is_main_process and epoch % int(config.save_every_n_epochs) == 0:
                checkpoint_name = f"epoch_{epoch:03d}.pt"
                checkpoint_path = run_dir / "checkpoints" / checkpoint_name
                _save_checkpoint(
                    model,
                    optimizer,
                    args,
                    checkpoint_path,
                    epoch,
                    trainer.scaler,
                )
                checkpoint_rel = f"checkpoints/{checkpoint_name}"
                print(f"Saved periodic checkpoint: {checkpoint_path}")

            if runtime.is_main_process:
                with open(metrics_path, "a") as f:
                    f.write(
                        f"{epoch},{train_metrics['train_loss']:.6f},"
                        f"{train_metrics['train_acc']:.6f},{train_metrics['train_f1']:.6f},"
                        f"{lr_backbone:.6f},{lr_head:.6f},"
                        f"{train_counts['samples']},{train_counts['positives']},{train_counts['negatives']},"
                        f"{checkpoint_rel}\n"
                    )

            _distributed_barrier(runtime)

        if runtime.is_main_process:
            final_checkpoint_path = run_dir / "checkpoints" / "last.pt"
            _save_checkpoint(
                model,
                optimizer,
                args,
                final_checkpoint_path,
                last_epoch,
                trainer.scaler,
            )
            print(f"Saved final checkpoint: {final_checkpoint_path}")

            with open(artifacts_path, "w") as f:
                json.dump(
                    {
                        "last_epoch": int(last_epoch),
                        "final_checkpoint": "checkpoints/last.pt",
                        "metrics_file": metrics_path.name,
                        "resumed_from": str(resumed_from) if resumed_from is not None else "",
                    },
                    f,
                    indent=4,
                )
            print(f"Training complete. Artifacts: {artifacts_path}")

        _distributed_barrier(runtime)
    finally:
        _destroy_distributed_runtime(runtime)


if __name__ == "__main__":
    main()
