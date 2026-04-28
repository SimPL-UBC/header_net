import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist

from .config import merge_cli_args


LATEST_CHECKPOINT_MANIFEST = "latest_train_checkpoint.json"


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
        help="Optional checkpoint path to resume model and optimizer state from, or 'latest'",
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
        "--resample_on_decode_failure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Randomly resample replacement rows on decode failure",
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
    parser.add_argument(
        "--save_every_n_steps",
        type=int,
        default=0,
        help="Save model checkpoint every N optimizer steps (0 disables step checkpoints)",
    )
    parser.add_argument(
        "--keep_last_n_step_checkpoints",
        type=int,
        default=2,
        help="Keep only the newest N step checkpoints (N must be >= 1)",
    )
    parser.add_argument(
        "--train_augmentation_mode",
        choices=["clip_consistent", "legacy_frame_random", "none"],
        default="clip_consistent",
        help="Training augmentation policy",
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


def _empty_train_stats():
    return {
        "loss_sum": 0.0,
        "sample_count": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": 0,
    }


def _normalize_train_stats(stats):
    normalized = _empty_train_stats()
    if not stats:
        return normalized
    normalized["loss_sum"] = float(stats.get("loss_sum", 0.0))
    normalized["sample_count"] = int(stats.get("sample_count", 0))
    normalized["tp"] = int(stats.get("tp", 0))
    normalized["fp"] = int(stats.get("fp", 0))
    normalized["fn"] = int(stats.get("fn", 0))
    normalized["tn"] = int(stats.get("tn", 0))
    return normalized


def _combine_train_stats(lhs, rhs):
    left = _normalize_train_stats(lhs)
    right = _normalize_train_stats(rhs)
    return {
        "loss_sum": float(left["loss_sum"]) + float(right["loss_sum"]),
        "sample_count": int(left["sample_count"]) + int(right["sample_count"]),
        "tp": int(left["tp"]) + int(right["tp"]),
        "fp": int(left["fp"]) + int(right["fp"]),
        "fn": int(left["fn"]) + int(right["fn"]),
        "tn": int(left["tn"]) + int(right["tn"]),
    }


def _capture_rng_state():
    rng_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng_state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return rng_state


def _restore_rng_state(rng_state):
    if not rng_state:
        return
    python_state = rng_state.get("python")
    if python_state is not None:
        random.setstate(python_state)
    numpy_state = rng_state.get("numpy")
    if numpy_state is not None:
        np.random.set_state(numpy_state)
    torch_state = rng_state.get("torch")
    if torch_state is not None:
        torch.random.set_rng_state(torch_state)
    cuda_state = rng_state.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def _manifest_path(run_dir: Path) -> Path:
    return run_dir / LATEST_CHECKPOINT_MANIFEST


def _relative_to_run_dir(path: Path, run_dir: Path) -> str:
    return str(path.relative_to(run_dir))


def _save_json(path: Path, payload) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=4)


def _update_latest_checkpoint_manifest(
    run_dir: Path,
    *,
    latest_resume_checkpoint=None,
    latest_resume_kind=None,
    latest_step_checkpoint=None,
    latest_epoch_checkpoint=None,
    epoch=None,
    global_step=None,
    rank_sample_offset=None,
):
    manifest_file = _manifest_path(run_dir)
    manifest = {}
    if manifest_file.exists():
        with open(manifest_file) as f:
            manifest = json.load(f)

    if latest_resume_checkpoint is not None:
        manifest["latest_resume_checkpoint"] = latest_resume_checkpoint
    if latest_resume_kind is not None:
        manifest["latest_resume_kind"] = latest_resume_kind
    if latest_step_checkpoint is not None:
        manifest["latest_step_checkpoint"] = latest_step_checkpoint
    if latest_epoch_checkpoint is not None:
        manifest["latest_epoch_checkpoint"] = latest_epoch_checkpoint
    if epoch is not None:
        manifest["epoch"] = int(epoch)
    if global_step is not None:
        manifest["global_step"] = int(global_step)
    if rank_sample_offset is not None:
        manifest["rank_sample_offset"] = int(rank_sample_offset)

    _save_json(manifest_file, manifest)


def _prune_old_step_checkpoints(run_dir: Path, keep_last_n: int) -> None:
    if keep_last_n < 1:
        return
    checkpoint_dir = run_dir / "checkpoints"
    step_checkpoints = sorted(checkpoint_dir.glob("step_ep*_gstep*.pt"))
    stale = step_checkpoints[:-keep_last_n]
    for path in stale:
        if path.exists():
            path.unlink()


def _resolve_resume_checkpoint_path(resume_checkpoint, run_dir: Path) -> Path:
    requested = str(resume_checkpoint).strip()
    if requested.lower() != "latest":
        return Path(requested).expanduser()

    manifest_file = _manifest_path(run_dir)
    if not manifest_file.exists():
        raise FileNotFoundError(
            f"Requested RESUME_CHECKPOINT=latest, but manifest not found: {manifest_file}"
        )
    with open(manifest_file) as f:
        manifest = json.load(f)

    relative_checkpoint = str(manifest.get("latest_resume_checkpoint", "")).strip()
    if not relative_checkpoint:
        raise RuntimeError(
            f"Manifest {manifest_file} does not contain latest_resume_checkpoint"
        )
    resolved = run_dir / relative_checkpoint
    if not resolved.exists():
        raise FileNotFoundError(
            "Manifest points to a missing resume checkpoint: "
            f"{resolved} (from {manifest_file})"
        )
    return resolved


def _save_checkpoint(
    model,
    optimizer,
    args,
    path,
    *,
    epoch,
    checkpoint_kind,
    global_step,
    rank_sample_offset=0,
    partial_train_stats=None,
    scaler=None,
):
    checkpoint_model = _checkpoint_model(model)
    state_dict = checkpoint_model.state_dict()

    checkpoint = {
        "epoch": int(epoch),
        "checkpoint_kind": str(checkpoint_kind),
        "global_step": int(global_step),
        "rank_sample_offset": int(rank_sample_offset),
        "state_dict": state_dict,
        "optimizer_state": optimizer.state_dict(),
        "partial_train_stats": _normalize_train_stats(partial_train_stats),
        "rng_state": _capture_rng_state(),
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

    checkpoint = torch.load(
        resume_path,
        map_location="cpu",
        weights_only=False,
    )
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

    checkpoint_kind = str(checkpoint.get("checkpoint_kind", "epoch")).strip().lower()
    checkpoint_epoch = int(checkpoint.get("epoch", 0))
    global_step = int(checkpoint.get("global_step", 0))
    rank_sample_offset = int(checkpoint.get("rank_sample_offset", 0))
    partial_train_stats = _normalize_train_stats(checkpoint.get("partial_train_stats"))
    _restore_rng_state(checkpoint.get("rng_state"))

    if checkpoint_kind == "step":
        start_epoch = checkpoint_epoch
        epoch_sample_offset = rank_sample_offset
        initial_train_stats = partial_train_stats
    else:
        start_epoch = checkpoint_epoch + 1
        epoch_sample_offset = 0
        initial_train_stats = _empty_train_stats()

    return {
        "resume_path": resume_path,
        "checkpoint_kind": checkpoint_kind,
        "checkpoint_epoch": checkpoint_epoch,
        "start_epoch": start_epoch,
        "global_step": global_step,
        "epoch_sample_offset": epoch_sample_offset,
        "initial_train_stats": initial_train_stats,
    }


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
        if int(config.save_every_n_steps) < 0:
            raise ValueError("save_every_n_steps must be >= 0")
        if int(config.keep_last_n_step_checkpoints) < 1:
            raise ValueError("keep_last_n_step_checkpoints must be >= 1")
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
            "Train persistent workers: "
            f"{'disabled' if config.train_augmentation_mode == 'clip_consistent' else 'enabled'}",
        )
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
        _log(runtime, f"Save checkpoint every N optimizer steps: {config.save_every_n_steps}")
        _log(
            runtime,
            f"Training augmentation mode: {config.train_augmentation_mode}; "
            f"resample_on_decode_failure={config.resample_on_decode_failure}",
        )

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
        resume_checkpoint_kind = ""
        epoch_resume_sample_offset = 0
        resume_epoch_train_stats = _empty_train_stats()
        global_step = 0
        if args.resume_checkpoint:
            resolved_resume_path = _resolve_resume_checkpoint_path(
                args.resume_checkpoint,
                run_dir,
            )
            resume_state = _load_resume_checkpoint(
                model,
                optimizer,
                resolved_resume_path,
                device,
                trainer.scaler,
            )
            resumed_from = resume_state["resume_path"]
            resume_checkpoint_kind = str(resume_state["checkpoint_kind"])
            start_epoch = int(resume_state["start_epoch"])
            epoch_resume_sample_offset = int(resume_state["epoch_sample_offset"])
            resume_epoch_train_stats = _normalize_train_stats(
                resume_state["initial_train_stats"]
            )
            global_step = int(resume_state["global_step"])
            if start_epoch > config.epochs:
                raise ValueError(
                    f"Resume checkpoint {resumed_from} starts at epoch {start_epoch}, "
                    f"which is beyond requested total epochs={config.epochs}."
                )
            if resume_checkpoint_kind == "step":
                _log(
                    runtime,
                    f"Resuming from step checkpoint: {resumed_from} "
                    f"(epoch {start_epoch}, rank sample offset {epoch_resume_sample_offset}, "
                    f"global optimizer step {global_step})",
                )
            else:
                _log(
                    runtime,
                    f"Resuming from epoch checkpoint: {resumed_from} "
                    f"(next epoch {start_epoch}, global optimizer step {global_step})",
                )

        if runtime.is_main_process:
            _ensure_metrics_file(
                metrics_path,
                append=bool(args.resume_checkpoint and metrics_path.exists()),
            )
        _distributed_barrier(runtime)

        last_epoch = start_epoch - 1
        for epoch in range(start_epoch, config.epochs + 1):
            last_epoch = epoch
            start_offset = epoch_resume_sample_offset if epoch == start_epoch else 0
            train_sampler.set_epoch(epoch, start_offset=start_offset)
            if hasattr(train_dataset, "set_epoch"):
                train_dataset.set_epoch(epoch)
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

            initial_epoch_train_stats = (
                resume_epoch_train_stats if epoch == start_epoch else _empty_train_stats()
            )
            starting_epoch_sample_offset = start_offset

            def on_optimizer_step(step_state):
                nonlocal global_step
                global_step += 1

                if int(config.save_every_n_steps) <= 0:
                    return
                if global_step % int(config.save_every_n_steps) != 0:
                    return
                if step_state["epoch_complete"]:
                    return

                reduced_segment_stats = _reduce_train_stats(
                    step_state["stats"],
                    device,
                    runtime,
                )
                partial_train_stats = _combine_train_stats(
                    initial_epoch_train_stats,
                    reduced_segment_stats,
                )
                rank_sample_offset = (
                    int(starting_epoch_sample_offset) + int(step_state["sample_count"])
                )

                _distributed_barrier(runtime)
                if runtime.is_main_process:
                    checkpoint_name = (
                        f"step_ep{epoch:03d}_gstep{global_step:08d}.pt"
                    )
                    checkpoint_path = run_dir / "checkpoints" / checkpoint_name
                    _save_checkpoint(
                        model,
                        optimizer,
                        args,
                        checkpoint_path,
                        epoch=epoch,
                        checkpoint_kind="step",
                        global_step=global_step,
                        rank_sample_offset=rank_sample_offset,
                        partial_train_stats=partial_train_stats,
                        scaler=trainer.scaler,
                    )
                    checkpoint_rel = _relative_to_run_dir(checkpoint_path, run_dir)
                    _prune_old_step_checkpoints(
                        run_dir,
                        int(config.keep_last_n_step_checkpoints),
                    )
                    _update_latest_checkpoint_manifest(
                        run_dir,
                        latest_resume_checkpoint=checkpoint_rel,
                        latest_resume_kind="step",
                        latest_step_checkpoint=checkpoint_rel,
                        epoch=epoch,
                        global_step=global_step,
                        rank_sample_offset=rank_sample_offset,
                    )
                    print(f"Saved step checkpoint: {checkpoint_path}")
                _distributed_barrier(runtime)

            train_stats = trainer.train_one_epoch(
                model,
                train_loader,
                optimizer,
                epoch,
                on_optimizer_step=on_optimizer_step,
            )
            reduced_segment_train_stats = _reduce_train_stats(train_stats, device, runtime)
            full_epoch_train_stats = _combine_train_stats(
                initial_epoch_train_stats,
                reduced_segment_train_stats,
            )
            train_metrics = _finalize_train_metrics(full_epoch_train_stats)
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
                    epoch=epoch,
                    checkpoint_kind="epoch",
                    global_step=global_step,
                    scaler=trainer.scaler,
                )
                checkpoint_rel = _relative_to_run_dir(checkpoint_path, run_dir)
                _update_latest_checkpoint_manifest(
                    run_dir,
                    latest_resume_checkpoint=checkpoint_rel,
                    latest_resume_kind="epoch",
                    latest_epoch_checkpoint=checkpoint_rel,
                    epoch=epoch,
                    global_step=global_step,
                    rank_sample_offset=0,
                )
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
            resume_epoch_train_stats = _empty_train_stats()
            epoch_resume_sample_offset = 0

        if runtime.is_main_process:
            final_checkpoint_path = run_dir / "checkpoints" / "last.pt"
            _save_checkpoint(
                model,
                optimizer,
                args,
                final_checkpoint_path,
                epoch=last_epoch,
                checkpoint_kind="epoch",
                global_step=global_step,
                scaler=trainer.scaler,
            )
            final_checkpoint_rel = _relative_to_run_dir(final_checkpoint_path, run_dir)
            _update_latest_checkpoint_manifest(
                run_dir,
                latest_resume_checkpoint=final_checkpoint_rel,
                latest_resume_kind="epoch",
                latest_epoch_checkpoint=final_checkpoint_rel,
                epoch=last_epoch,
                global_step=global_step,
                rank_sample_offset=0,
            )
            print(f"Saved final checkpoint: {final_checkpoint_path}")

            _save_json(
                artifacts_path,
                {
                    "last_epoch": int(last_epoch),
                    "global_step": int(global_step),
                    "final_checkpoint": final_checkpoint_rel,
                    "metrics_file": metrics_path.name,
                    "latest_checkpoint_manifest": LATEST_CHECKPOINT_MANIFEST,
                    "resumed_from": str(resumed_from) if resumed_from is not None else "",
                    "resume_checkpoint_kind": resume_checkpoint_kind,
                },
            )
            print(f"Training complete. Artifacts: {artifacts_path}")

        _distributed_barrier(runtime)
    finally:
        _destroy_distributed_runtime(runtime)


if __name__ == "__main__":
    main()
