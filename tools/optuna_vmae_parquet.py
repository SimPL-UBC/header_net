#!/usr/bin/env python3
"""Optuna hyperparameter tuning for VMAE2 parquet header training."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class BackboneSettings:
    name: str
    backbone_ckpt: Path
    batch_size: int
    gradient_accumulation_steps: int
    num_workers: int
    max_open_videos: int


def _str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune VMAE2 Base/Giant parquet header training with Optuna. "
            "Each trial trains on dense_train and evaluates on dense_val."
        )
    )

    parser.add_argument(
        "--backbones",
        nargs="+",
        default=["base", "giant"],
        help="Backbones to tune: base, giant, or both (default: %(default)s)",
    )
    parser.add_argument("--n-trials", type=int, default=20, help="Trials per backbone")
    parser.add_argument(
        "--objective-metric",
        default="val_f1",
        help="Metric key under validation metrics to maximize (default: %(default)s)",
    )
    parser.add_argument(
        "--study-name-prefix",
        default="optuna_vmae_parquet",
        help="Prefix for per-backbone Optuna study names",
    )
    parser.add_argument(
        "--storage",
        default="",
        help=(
            "Optuna storage URL. Empty uses sqlite:///<output-root>/optuna.db. "
            "Use sqlite:///path/to/file.db for resumable studies."
        ),
    )
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "output" / "optuna_vmae_parquet"),
        help="Root directory for trial runs, metrics, and study artifacts",
    )
    parser.add_argument("--sampler-seed", type=int, default=42)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print one representative train/eval command per backbone and exit",
    )

    parser.add_argument(
        "--train-parquet",
        default=str(REPO_ROOT / "output" / "dense_dataset" / "dense_train"),
    )
    parser.add_argument(
        "--val-parquet",
        default=str(REPO_ROOT / "output" / "dense_dataset" / "dense_val"),
    )
    parser.add_argument("--dataset-root", default=str(REPO_ROOT / "SoccerNet"))
    parser.add_argument("--spatial-mode", choices=("ball_crop", "full_frame"), default="ball_crop")

    parser.add_argument("--base-backbone-ckpt", default="")
    parser.add_argument("--giant-backbone-ckpt", default="")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size for all backbones")
    parser.add_argument("--base-batch-size", type=int, default=16)
    parser.add_argument("--giant-batch-size", type=int, default=1)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="Override gradient accumulation for all backbones",
    )
    parser.add_argument("--base-gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--giant-gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=None, help="Override train workers for all backbones")
    parser.add_argument("--base-num-workers", type=int, default=4)
    parser.add_argument("--giant-num-workers", type=int, default=1)
    parser.add_argument(
        "--max-open-videos",
        type=int,
        default=None,
        help="Override max open videos per worker for all backbones",
    )
    parser.add_argument("--base-max-open-videos", type=int, default=4)
    parser.add_argument("--giant-max-open-videos", type=int, default=1)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--frame-cache-size", type=int, default=128)
    parser.add_argument("--loader-start-method", choices=("spawn", "fork", "forkserver"), default="spawn")
    parser.add_argument("--optimizer", choices=("adamw", "sgd"), default="adamw")
    parser.add_argument("--base-lr", type=float, default=1e-3)
    parser.add_argument("--layer-lr-decay", type=float, default=0.75)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999))
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--loss", choices=("focal", "ce"), default="focal")
    parser.add_argument("--finetune-mode", choices=("full", "frozen", "partial"), default="full")
    parser.add_argument("--unfreeze-blocks", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpus", type=int, nargs="+", default=None)
    parser.add_argument(
        "--ddp-mode",
        choices=("auto", "on", "off"),
        default="auto",
        help="DDP launch policy. Auto enables DDP for Giant with multiple GPUs.",
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Override VMAE gradient checkpointing during training",
    )
    parser.add_argument(
        "--save-epoch-indices",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save sampled train row indices per trial epoch",
    )
    parser.add_argument("--save-every-n-epochs", type=int, default=1)
    parser.add_argument("--save-every-n-steps", type=int, default=0)
    parser.add_argument("--keep-last-n-step-checkpoints", type=int, default=2)
    parser.add_argument("--train-augmentation-mode", choices=("clip_consistent", "legacy_frame_random", "none"), default="clip_consistent")
    parser.add_argument("--resample-on-decode-failure", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--validate-video-load",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Validate train parquet video readability before every trial",
    )
    parser.add_argument("--validate-video-load-max-errors", type=int, default=20)

    parser.add_argument("--alpha-min", type=float, default=0.55)
    parser.add_argument("--alpha-max", type=float, default=0.95)
    parser.add_argument("--gamma-min", type=float, default=0.5)
    parser.add_argument("--gamma-max", type=float, default=5.0)
    parser.add_argument(
        "--neg-pos-ratios",
        nargs="+",
        default=["3", "5", "8", "10", "15", "20"],
        help="Categorical train negative:positive ratios to sample",
    )

    parser.add_argument("--val-batch-size", type=int, default=None)
    parser.add_argument("--val-num-workers", type=int, default=8)
    parser.add_argument("--val-pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--val-progress-every", type=int, default=1000)
    parser.add_argument("--val-neg-pos-ratio", default="all")
    parser.add_argument("--f1-threshold-step", type=float, default=0.01)
    parser.add_argument("--save-val-predictions", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--reuse-val-predictions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow eval to rebuild metrics from an existing trial predictions CSV",
    )

    parser.add_argument(
        "--run-best-val-inference",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="After each study, run raw validation inference for the best checkpoint",
    )
    parser.add_argument("--best-inference-batch-size", type=int, default=128)
    parser.add_argument("--best-inference-num-workers", type=int, default=6)
    parser.add_argument("--best-inference-max-open-videos", type=int, default=1)

    args = parser.parse_args()
    args.backbones = [str(value).strip().lower() for value in args.backbones]
    invalid_backbones = sorted(set(args.backbones) - {"base", "giant"})
    if invalid_backbones:
        raise ValueError(f"Unsupported backbone(s): {invalid_backbones}")
    if args.n_trials < 1:
        raise ValueError("--n-trials must be >= 1")
    if args.alpha_min > args.alpha_max:
        raise ValueError("--alpha-min must be <= --alpha-max")
    if args.gamma_min > args.gamma_max:
        raise ValueError("--gamma-min must be <= --gamma-max")
    if not args.neg_pos_ratios:
        raise ValueError("--neg-pos-ratios cannot be empty")
    return args


def resolve_backbone_settings(args: argparse.Namespace, backbone: str) -> BackboneSettings:
    if backbone == "base":
        default_ckpt = REPO_ROOT / "checkpoints" / "VideoMAEv2-Base"
        explicit_ckpt = args.base_backbone_ckpt
        default_batch_size = args.base_batch_size
        default_grad_accum = args.base_gradient_accumulation_steps
        default_workers = args.base_num_workers
        default_max_open = args.base_max_open_videos
    else:
        default_ckpt = REPO_ROOT / "checkpoints" / "VideoMAEv2-giant"
        explicit_ckpt = args.giant_backbone_ckpt
        default_batch_size = args.giant_batch_size
        default_grad_accum = args.giant_gradient_accumulation_steps
        default_workers = args.giant_num_workers
        default_max_open = args.giant_max_open_videos

    return BackboneSettings(
        name=backbone,
        backbone_ckpt=_path(explicit_ckpt) if explicit_ckpt else default_ckpt,
        batch_size=args.batch_size if args.batch_size is not None else default_batch_size,
        gradient_accumulation_steps=(
            args.gradient_accumulation_steps
            if args.gradient_accumulation_steps is not None
            else default_grad_accum
        ),
        num_workers=args.num_workers if args.num_workers is not None else default_workers,
        max_open_videos=(
            args.max_open_videos
            if args.max_open_videos is not None
            else default_max_open
        ),
    )


def should_use_ddp(args: argparse.Namespace, settings: BackboneSettings) -> bool:
    gpu_count = len(args.gpus or [])
    if gpu_count <= 1:
        return False
    if args.ddp_mode == "on":
        return True
    if args.ddp_mode == "off":
        return False
    return settings.name == "giant"


def command_to_text(cmd: list[str], env_overrides: dict[str, str] | None = None) -> str:
    prefix = ""
    if env_overrides:
        prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in sorted(env_overrides.items()))
        prefix += " "
    return prefix + shlex.join(cmd)


def run_command(
    cmd: list[str],
    *,
    cwd: Path,
    log_path: Path,
    env_overrides: dict[str, str] | None = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)

    cmd_text = command_to_text(cmd, env_overrides)
    print(f"[CMD] {cmd_text}", flush=True)
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"$ {cmd_text}\n\n")
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return_code = process.wait()
        log_file.write(f"\n[exit_code] {return_code}\n")

    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}: {cmd_text}")


def build_train_command(
    args: argparse.Namespace,
    settings: BackboneSettings,
    params: dict[str, Any],
    *,
    run_dir: Path,
) -> tuple[list[str], dict[str, str]]:
    output_root = run_dir.parent
    run_name = run_dir.name
    module_args = [
        "-m",
        "training.cli_train_header_parquet_train",
        "--train_parquet",
        str(_path(args.train_parquet)),
        "--dataset_root",
        str(_path(args.dataset_root)),
        "--spatial_mode",
        args.spatial_mode,
        "--neg_pos_ratio",
        str(params["neg_pos_ratio"]),
        "--backbone",
        "vmae",
        "--finetune_mode",
        args.finetune_mode,
        "--unfreeze_blocks",
        str(args.unfreeze_blocks),
        "--backbone_ckpt",
        str(settings.backbone_ckpt),
        "--run_name",
        run_name,
        "--output_root",
        str(output_root),
        "--epochs",
        str(args.epochs),
        "--num_frames",
        str(args.num_frames),
        "--batch_size",
        str(settings.batch_size),
        "--gradient_accumulation_steps",
        str(settings.gradient_accumulation_steps),
        "--num_workers",
        str(settings.num_workers),
        "--max_open_videos",
        str(settings.max_open_videos),
        "--frame_cache_size",
        str(args.frame_cache_size),
        "--loader_start_method",
        args.loader_start_method,
        "--optimizer",
        args.optimizer,
        "--base_lr",
        str(args.base_lr),
        "--layer_lr_decay",
        str(args.layer_lr_decay),
        "--betas",
        str(args.betas[0]),
        str(args.betas[1]),
        "--weight_decay",
        str(args.weight_decay),
        "--loss",
        args.loss,
        "--focal_gamma",
        str(params["focal_gamma"]),
        "--focal_alpha",
        str(params["focal_alpha"]),
        "--save_every_n_epochs",
        str(args.save_every_n_epochs),
        "--save_every_n_steps",
        str(args.save_every_n_steps),
        "--keep_last_n_step_checkpoints",
        str(args.keep_last_n_step_checkpoints),
        "--seed",
        str(args.seed),
        "--train_augmentation_mode",
        args.train_augmentation_mode,
    ]

    module_args.append("--amp" if args.amp else "--no-amp")
    module_args.append(
        "--gradient_checkpointing"
        if args.gradient_checkpointing
        else "--no-gradient_checkpointing"
    )
    module_args.append(
        "--save_epoch_indices" if args.save_epoch_indices else "--no-save_epoch_indices"
    )
    module_args.append(
        "--resample_on_decode_failure"
        if args.resample_on_decode_failure
        else "--no-resample_on_decode_failure"
    )
    if args.gpus:
        module_args.extend(["--gpus", *[str(gpu_id) for gpu_id in args.gpus]])

    cmd = [sys.executable, *module_args]
    env_overrides: dict[str, str] = {}
    if should_use_ddp(args, settings):
        launcher = (
            ["torchrun"]
            if shutil.which("torchrun")
            else [sys.executable, "-m", "torch.distributed.run"]
        )
        cmd = [
            *launcher,
            "--standalone",
            "--nproc_per_node",
            str(len(args.gpus or [])),
            *module_args,
        ]
        env_overrides["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in args.gpus)

    return cmd, env_overrides


def build_eval_command(
    args: argparse.Namespace,
    settings: BackboneSettings,
    params: dict[str, Any],
    *,
    checkpoint_path: Path,
    output_dir: Path,
    metrics_path: Path,
    predictions_path: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "training.cli_train_header_parquet_eval",
        "--checkpoint_path",
        str(checkpoint_path),
        "--val_parquet",
        str(_path(args.val_parquet)),
        "--dataset_root",
        str(_path(args.dataset_root)),
        "--spatial_mode",
        args.spatial_mode,
        "--output_dir",
        str(output_dir),
        "--val_num_workers",
        str(args.val_num_workers),
        "--val_progress_every",
        str(args.val_progress_every),
        "--max_open_videos",
        str(settings.max_open_videos),
        "--frame_cache_size",
        str(args.frame_cache_size),
        "--loader_start_method",
        args.loader_start_method,
        "--val_neg_pos_ratio",
        str(args.val_neg_pos_ratio),
        "--backbone",
        "vmae",
        "--finetune_mode",
        args.finetune_mode,
        "--unfreeze_blocks",
        str(args.unfreeze_blocks),
        "--backbone_ckpt",
        str(settings.backbone_ckpt),
        "--base_lr",
        str(args.base_lr),
        "--layer_lr_decay",
        str(args.layer_lr_decay),
        "--loss",
        args.loss,
        "--focal_gamma",
        str(params["focal_gamma"]),
        "--focal_alpha",
        str(params["focal_alpha"]),
        "--seed",
        str(args.seed),
        "--f1_threshold_step",
        str(args.f1_threshold_step),
        "--predictions_path",
        str(predictions_path),
        "--metrics_path",
        str(metrics_path),
    ]
    if args.val_batch_size is not None:
        cmd.extend(["--batch_size", str(args.val_batch_size)])
    if args.gpus:
        cmd.extend(["--gpus", *[str(gpu_id) for gpu_id in args.gpus]])
    cmd.append("--val_pin_memory" if args.val_pin_memory else "--no-val_pin_memory")
    cmd.append("--save_predictions" if args.save_val_predictions else "--no-save_predictions")
    cmd.append("--no-skip_existing")
    cmd.append("--reuse_predictions" if args.reuse_val_predictions else "--no-reuse_predictions")
    return cmd


def build_best_val_inference_command(
    args: argparse.Namespace,
    *,
    checkpoint_path: Path,
    output_dir: Path,
    output_csv: Path,
) -> tuple[list[str], dict[str, str]]:
    script = REPO_ROOT / "job_script" / "inference_parquet_val.sh"
    env_overrides = {
        "PARQUET": str(_path(args.val_parquet)),
        "CHECKPOINT": str(checkpoint_path),
        "OUTPUT_DIR": str(output_dir),
        "OUTPUT_CSV": str(output_csv),
        "DATASET_ROOT": str(_path(args.dataset_root)),
        "SPATIAL_MODE": args.spatial_mode,
        "BATCH_SIZE": str(args.best_inference_batch_size),
        "NUM_WORKERS": str(args.best_inference_num_workers),
        "MAX_OPEN_VIDEOS": str(args.best_inference_max_open_videos),
        "FRAME_CACHE_SIZE": str(args.frame_cache_size),
        "LOADER_START_METHOD": args.loader_start_method,
        "PIN_MEMORY": "on" if args.val_pin_memory else "off",
        "SEED": str(args.seed),
    }
    if args.gpus:
        env_overrides["GPUS"] = " ".join(str(gpu_id) for gpu_id in args.gpus)
    return ["bash", str(script)], env_overrides


def build_video_readability_command(args: argparse.Namespace) -> list[str]:
    code = r"""
import sys
from pathlib import Path

import decord
import pandas as pd

decord.bridge.set_bridge("native")

train_parquet = Path(sys.argv[1])
max_errors = int(sys.argv[2])

bad = []
if not train_parquet.exists():
    bad.append((str(train_parquet), "parquet_missing"))
else:
    df = pd.read_parquet(train_parquet, columns=["video_path"])
    for video_path in pd.unique(df["video_path"].astype(str)):
        try:
            reader = decord.VideoReader(video_path, ctx=decord.cpu())
        except Exception:
            bad.append((video_path, "open_failed"))
            continue
        frame_count = len(reader)
        if frame_count <= 0:
            bad.append((video_path, f"invalid_frame_count={frame_count}"))
            continue
        try:
            reader.get_batch([0]).asnumpy()
        except Exception:
            bad.append((video_path, "first_frame_decode_failed"))

if bad:
    print(
        f"[ERROR] Found {len(bad)} unreadable video path(s) referenced by the train parquet.",
        file=sys.stderr,
    )
    for path, reason in bad[:max_errors]:
        print(f"  - {path} ({reason})", file=sys.stderr)
    if len(bad) > max_errors:
        print(f"  ... and {len(bad) - max_errors} more", file=sys.stderr)
    sys.exit(1)

print("[INFO] Video readability check passed.")
"""
    return [
        sys.executable,
        "-c",
        code,
        str(_path(args.train_parquet)),
        str(args.validate_video_load_max_errors),
    ]


def load_metric(metrics_path: Path, objective_metric: str) -> tuple[float, dict[str, Any]]:
    with open(metrics_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    metrics = payload.get("metrics", {})
    if objective_metric not in metrics:
        raise RuntimeError(
            f"Objective metric '{objective_metric}' not found in {metrics_path}. "
            f"Available metrics: {sorted(metrics)}"
        )
    return float(metrics[objective_metric]), payload


def write_best_trial(study: Any, backbone_dir: Path) -> None:
    import optuna

    completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print(f"[WARN] No completed trials for study {study.study_name}; skipping best_trial.json")
        return

    best = study.best_trial
    payload = {
        "study_name": study.study_name,
        "trial_number": best.number,
        "objective_value": best.value,
        "params": best.params,
        "checkpoint_path": best.user_attrs.get("checkpoint_path", ""),
        "metrics_path": best.user_attrs.get("metrics_path", ""),
        "predictions_path": best.user_attrs.get("predictions_path", ""),
        "validation_metrics": best.user_attrs.get("validation_metrics", {}),
        "validation_counts": best.user_attrs.get("validation_counts", {}),
    }
    output_path = backbone_dir / "best_trial.json"
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"[INFO] Best trial summary: {output_path}")


def export_study_results(study: Any, backbone_dir: Path) -> None:
    results_path = backbone_dir / "study_results.csv"
    study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs")).to_csv(
        results_path,
        index=False,
    )
    print(f"[INFO] Study results: {results_path}")


def suggest_params(trial: Any, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "focal_alpha": trial.suggest_float("focal_alpha", args.alpha_min, args.alpha_max),
        "focal_gamma": trial.suggest_float("focal_gamma", args.gamma_min, args.gamma_max),
        "neg_pos_ratio": trial.suggest_categorical("neg_pos_ratio", args.neg_pos_ratios),
    }


def representative_params(args: argparse.Namespace) -> dict[str, Any]:
    alpha = min(max(0.75, args.alpha_min), args.alpha_max)
    gamma = min(max(2.0, args.gamma_min), args.gamma_max)
    return {
        "focal_alpha": alpha,
        "focal_gamma": gamma,
        "neg_pos_ratio": args.neg_pos_ratios[0],
    }


def make_objective(args: argparse.Namespace, settings: BackboneSettings, backbone_dir: Path):
    def objective(trial: Any) -> float:
        params = suggest_params(trial, args)
        trial_dir = backbone_dir / f"trial_{trial.number:04d}"
        logs_dir = trial_dir / "logs"
        validation_dir = trial_dir / "validation"
        checkpoint_path = trial_dir / "checkpoints" / "last.pt"
        metrics_path = validation_dir / "final_val_metrics.json"
        predictions_path = validation_dir / "final_val_predictions.csv"

        trial.set_user_attr("backbone", settings.name)
        trial.set_user_attr("run_dir", str(trial_dir))
        trial.set_user_attr("checkpoint_path", str(checkpoint_path))
        trial.set_user_attr("metrics_path", str(metrics_path))
        trial.set_user_attr("predictions_path", str(predictions_path))
        trial.set_user_attr("batch_size", settings.batch_size)
        trial.set_user_attr(
            "gradient_accumulation_steps",
            settings.gradient_accumulation_steps,
        )

        train_cmd, train_env = build_train_command(
            args,
            settings,
            params,
            run_dir=trial_dir,
        )
        eval_cmd = build_eval_command(
            args,
            settings,
            params,
            checkpoint_path=checkpoint_path,
            output_dir=validation_dir,
            metrics_path=metrics_path,
            predictions_path=predictions_path,
        )

        print("")
        print("=" * 72)
        print(
            f"[TRIAL] backbone={settings.name} trial={trial.number} "
            f"alpha={params['focal_alpha']:.6g} gamma={params['focal_gamma']:.6g} "
            f"neg_pos_ratio={params['neg_pos_ratio']}"
        )
        print("=" * 72)

        if args.validate_video_load:
            run_command(
                build_video_readability_command(args),
                cwd=REPO_ROOT,
                log_path=logs_dir / "validate_video_load.log",
            )

        run_command(
            train_cmd,
            cwd=REPO_ROOT,
            log_path=logs_dir / "train.log",
            env_overrides=train_env,
        )
        if not checkpoint_path.exists():
            raise RuntimeError(f"Expected checkpoint not found after training: {checkpoint_path}")

        run_command(
            eval_cmd,
            cwd=REPO_ROOT,
            log_path=logs_dir / "validate.log",
        )
        score, metrics_payload = load_metric(metrics_path, args.objective_metric)
        metrics = metrics_payload.get("metrics", {})
        counts = metrics_payload.get("val_counts", {})
        trial.set_user_attr("validation_metrics", metrics)
        trial.set_user_attr("validation_counts", counts)

        print(
            f"[RESULT] backbone={settings.name} trial={trial.number} "
            f"{args.objective_metric}={score:.6f}"
        )
        return score

    return objective


def run_dry_run(args: argparse.Namespace) -> None:
    print("[DRY_RUN] No training, validation, or Optuna storage writes will be performed.")
    params = representative_params(args)
    for backbone in args.backbones:
        settings = resolve_backbone_settings(args, backbone)
        backbone_dir = _path(args.output_root) / backbone
        trial_dir = backbone_dir / "trial_0000"
        checkpoint_path = trial_dir / "checkpoints" / "last.pt"
        validation_dir = trial_dir / "validation"
        train_cmd, train_env = build_train_command(
            args,
            settings,
            params,
            run_dir=trial_dir,
        )
        eval_cmd = build_eval_command(
            args,
            settings,
            params,
            checkpoint_path=checkpoint_path,
            output_dir=validation_dir,
            metrics_path=validation_dir / "final_val_metrics.json",
            predictions_path=validation_dir / "final_val_predictions.csv",
        )
        print("")
        print(f"[DRY_RUN] Backbone: {backbone}")
        print(f"[DRY_RUN] Train: {command_to_text(train_cmd, train_env)}")
        print(f"[DRY_RUN] Eval:  {command_to_text(eval_cmd)}")
        if args.run_best_val_inference:
            infer_cmd, infer_env = build_best_val_inference_command(
                args,
                checkpoint_path=checkpoint_path,
                output_dir=backbone_dir / "best_val_inference",
                output_csv=backbone_dir / "best_val_inference" / "val_predictions_raw.csv",
            )
            print(f"[DRY_RUN] Best val inference: {command_to_text(infer_cmd, infer_env)}")


def run_best_val_inference(args: argparse.Namespace, study: Any, backbone_dir: Path) -> None:
    import optuna

    completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print(f"[WARN] No completed trials for {study.study_name}; skipping best validation inference")
        return

    checkpoint_text = str(study.best_trial.user_attrs.get("checkpoint_path", ""))
    if not checkpoint_text:
        print(f"[WARN] Best trial has no checkpoint path; skipping best validation inference")
        return
    checkpoint_path = Path(checkpoint_text)
    output_dir = backbone_dir / "best_val_inference"
    output_csv = output_dir / "val_predictions_raw.csv"
    cmd, env_overrides = build_best_val_inference_command(
        args,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        output_csv=output_csv,
    )
    run_command(
        cmd,
        cwd=REPO_ROOT,
        log_path=output_dir / "inference.log",
        env_overrides=env_overrides,
    )


def run_study(args: argparse.Namespace, backbone: str) -> None:
    import optuna

    settings = resolve_backbone_settings(args, backbone)
    output_root = _path(args.output_root)
    backbone_dir = output_root / backbone
    backbone_dir.mkdir(parents=True, exist_ok=True)

    storage = args.storage.strip()
    if not storage:
        storage = f"sqlite:///{(output_root / 'optuna.db').resolve()}"

    study_name = f"{args.study_name_prefix}_{backbone}"
    sampler = optuna.samplers.TPESampler(seed=args.sampler_seed)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
    )

    print("")
    print("=" * 72)
    print(f"[STUDY] {study_name}")
    print(f"[STUDY] storage={storage}")
    print(f"[STUDY] output={backbone_dir}")
    print(f"[STUDY] objective={args.objective_metric}")
    print(f"[STUDY] trials_to_run={args.n_trials}")
    print("=" * 72)

    study.optimize(
        make_objective(args, settings, backbone_dir),
        n_trials=args.n_trials,
        catch=(RuntimeError, subprocess.SubprocessError),
    )
    export_study_results(study, backbone_dir)
    write_best_trial(study, backbone_dir)
    if args.run_best_val_inference:
        run_best_val_inference(args, study, backbone_dir)


def main() -> None:
    args = parse_args()
    if args.dry_run:
        run_dry_run(args)
        return

    try:
        import optuna  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Optuna is not installed. Install dependencies with requirements.txt "
            "or run `pip install optuna==4.5.0` inside deep_impact_env."
        ) from exc

    output_root = _path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for required_path, label in (
        (_path(args.train_parquet), "train parquet"),
        (_path(args.val_parquet), "validation parquet"),
        (_path(args.dataset_root), "dataset root"),
    ):
        if not required_path.exists():
            raise FileNotFoundError(f"Missing {label}: {required_path}")

    for backbone in args.backbones:
        run_study(args, backbone)


if __name__ == "__main__":
    main()
