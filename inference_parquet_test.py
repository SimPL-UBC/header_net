#!/usr/bin/env python3
"""Run VMAE inference over a dense parquet test dataset."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


HEADER_NET_ROOT = Path(__file__).resolve().parent
if str(HEADER_NET_ROOT) not in sys.path:
    sys.path.insert(0, str(HEADER_NET_ROOT))

from training.config import Config as TrainingConfig
from training.data.parquet_header_dataset import ParquetHeaderDataset, get_transforms
from training.models.factory import build_model
from training.run_utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run parquet-based VMAE inference on a dense test parquet."
    )
    parser.add_argument(
        "--parquet",
        default=str(HEADER_NET_ROOT / "output" / "dense_dataset" / "dense_test.parquet"),
        help="Input dense parquet (default: %(default)s)",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(
            HEADER_NET_ROOT
            / "output"
            / "vmae_parquet_ratio10"
            / "checkpoints"
            / "last.pt"
        ),
        help="Model checkpoint (.pt) (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            HEADER_NET_ROOT / "output" / "vmae_parquet_ratio10" / "test_inference"
        ),
        help="Output directory (default: %(default)s)",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional explicit output CSV path (default: <output-dir>/test_predictions_raw.csv)",
    )
    parser.add_argument(
        "--dataset-root",
        default=str(HEADER_NET_ROOT / "SoccerNet"),
        help="SoccerNet root for path validation (default: %(default)s)",
    )
    parser.add_argument("--video-id", default=None, help="Optional video_id filter")
    parser.add_argument("--half", type=int, default=None, help="Optional half filter")
    parser.add_argument(
        "--backbone-ckpt",
        default=None,
        help="Override VideoMAE pretrained weights directory",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Override temporal window length",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=None,
        help="Override model input size",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override inference batch size",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers (default: %(default)s)",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=None,
        help="GPU IDs for VMAE inference (e.g. --gpus 0 1). Uses DataParallel when >1 GPU is supplied.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device override when --gpus is not set (default: auto)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Inference seed (default: checkpoint config or 42)",
    )
    return parser.parse_args()


def resolve_from_checkpoint(
    explicit_value: Any,
    checkpoint_config: dict[str, Any],
    key: str,
    default_value: Any,
) -> Any:
    if explicit_value is not None:
        return explicit_value
    value = checkpoint_config.get(key)
    if value is None or value == "":
        return default_value
    return value


def resolve_backbone_ckpt(
    explicit_value: str | None,
    checkpoint_config: dict[str, Any],
) -> Path | None:
    candidates: list[Path] = []
    if explicit_value:
        candidates.append(Path(explicit_value).expanduser())

    checkpoint_value = checkpoint_config.get("backbone_ckpt")
    if checkpoint_value:
        raw_path = Path(str(checkpoint_value)).expanduser()
        candidates.append(raw_path)
        candidates.append(HEADER_NET_ROOT / "checkpoints" / raw_path.name)

    candidates.append(HEADER_NET_ROOT / "checkpoints" / "VideoMAEv2-Base")

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Unable to resolve a valid VideoMAE backbone checkpoint directory."
    )


def strip_module_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    if not state_dict:
        return state_dict
    first_key = next(iter(state_dict))
    if not first_key.startswith("module."):
        return state_dict
    return {key.replace("module.", "", 1): value for key, value in state_dict.items()}


def build_model_from_checkpoint(
    checkpoint_payload: dict[str, Any],
    backbone: str,
    num_frames: int,
    input_size: int,
    backbone_ckpt: Path | None,
) -> torch.nn.Module:
    cfg = TrainingConfig()
    cfg.backbone = backbone
    cfg.num_frames = int(num_frames)
    cfg.input_size = int(input_size)
    cfg.finetune_mode = "full"
    cfg.backbone_ckpt = str(backbone_ckpt) if backbone_ckpt is not None else None
    cfg.base_lr = 1e-3
    cfg.layer_lr_decay = 0.75

    model, _ = build_model(cfg)

    state_dict = checkpoint_payload.get("state_dict", checkpoint_payload)
    state_dict = strip_module_prefix(state_dict)
    model.load_state_dict(state_dict, strict=True)
    return model


def batch_item(value: Any, index: int) -> Any:
    if torch.is_tensor(value):
        return value[index].item()
    if isinstance(value, (list, tuple)):
        return value[index]
    return value


def resolve_device_config(
    gpus_arg: list[int] | None,
    device_arg: str | None,
) -> tuple[torch.device, list[int]]:
    if gpus_arg:
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"GPU IDs requested ({gpus_arg}) but torch.cuda.is_available() is False."
            )
        available_count = torch.cuda.device_count()
        invalid = [gpu_id for gpu_id in gpus_arg if gpu_id < 0 or gpu_id >= available_count]
        if invalid:
            raise RuntimeError(
                f"Requested GPU IDs {invalid} are unavailable. "
                f"Visible CUDA device count: {available_count}."
            )
        primary = torch.device(f"cuda:{gpus_arg[0]}")
        return primary, list(gpus_arg)

    if device_arg:
        device = torch.device(device_arg)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device requested ({device}) but torch.cuda.is_available() is False."
            )
        gpu_index = 0 if device.index is None else int(device.index)
        available_count = torch.cuda.device_count()
        if gpu_index < 0 or gpu_index >= available_count:
            raise RuntimeError(
                f"Requested CUDA device index {gpu_index} is unavailable. "
                f"Visible CUDA device count: {available_count}."
            )
        return device, [gpu_index]

    return device, []


def main() -> None:
    args = parse_args()

    parquet_path = Path(args.parquet).expanduser()
    checkpoint_path = Path(args.checkpoint).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_csv = (
        Path(args.output_csv).expanduser()
        if args.output_csv
        else output_dir / "test_predictions_raw.csv"
    )

    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint_payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    checkpoint_config = checkpoint_payload.get("config", {}) or {}

    checkpoint_backbone = str(checkpoint_config.get("backbone", "vmae"))
    if checkpoint_backbone != "vmae":
        raise ValueError(
            f"inference_parquet_test.py is VMAE-only, but checkpoint backbone is '{checkpoint_backbone}'."
        )
    backbone = "vmae"
    num_frames = int(resolve_from_checkpoint(args.num_frames, checkpoint_config, "num_frames", 16))
    input_size = int(resolve_from_checkpoint(args.input_size, checkpoint_config, "input_size", 224))
    batch_size = int(resolve_from_checkpoint(args.batch_size, checkpoint_config, "batch_size", 8))
    seed = int(resolve_from_checkpoint(args.seed, checkpoint_config, "seed", 42))
    backbone_ckpt = resolve_backbone_ckpt(args.backbone_ckpt, checkpoint_config)
    device, gpu_ids = resolve_device_config(args.gpus, args.device)

    set_seed(seed)

    print(f"Parquet:        {parquet_path}")
    print(f"Checkpoint:     {checkpoint_path}")
    print(f"Backbone:       {backbone}")
    if backbone_ckpt is not None:
        print(f"Backbone ckpt:  {backbone_ckpt}")
    print(f"Num frames:     {num_frames}")
    print(f"Input size:     {input_size}")
    print(f"Batch size:     {batch_size}")
    if gpu_ids:
        print(f"GPUs:           {gpu_ids}")
    print(f"Device:         {device}")

    filter_df = pd.read_parquet(parquet_path, columns=["video_id", "half"])
    mask = pd.Series(True, index=filter_df.index)
    if args.video_id:
        mask &= filter_df["video_id"].astype(str) == str(args.video_id)
    if args.half is not None:
        mask &= filter_df["half"].astype(int) == int(args.half)

    subset_indices = filter_df.index[mask].tolist()
    if not subset_indices:
        raise ValueError("No parquet rows matched the requested filters.")

    transform = get_transforms(input_size=input_size, is_training=False)
    base_dataset = ParquetHeaderDataset(
        parquet_path=parquet_path,
        num_frames=num_frames,
        input_size=input_size,
        transform=transform,
        strict_paths=True,
        dataset_root=Path(args.dataset_root).expanduser(),
        max_open_videos=4,
        resample_on_decode_failure=False,
    )
    dataset = (
        Subset(base_dataset, subset_indices)
        if len(subset_indices) != len(base_dataset)
        else base_dataset
    )

    num_workers = int(args.num_workers)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        prefetch_factor=1 if num_workers > 0 else None,
    )

    model = build_model_from_checkpoint(
        checkpoint_payload=checkpoint_payload,
        backbone=backbone,
        num_frames=num_frames,
        input_size=input_size,
        backbone_ckpt=backbone_ckpt,
    )
    model = model.to(device)
    if len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    model.eval()

    print(f"Evaluating {len(dataset)} samples...")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    csv_fieldnames = [
        "row_idx", "video_id", "half", "frame", "label", "video_path",
        "fps", "ball_x", "ball_y", "ball_w", "ball_h", "ball_confidence",
        "prob_header", "prob_non_header", "pred_label_0p5", "pred_confidence_0p5",
    ]

    rows_written = 0
    prob_header_sum = 0.0
    prob_header_sq_sum = 0.0

    with open(output_csv, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames)
        writer.writeheader()

        with torch.inference_mode():
            for batch_idx, (inputs, targets, meta) in enumerate(
                tqdm(loader, desc="Parquet inference", unit="batch")
            ):
                inputs = inputs.to(device, non_blocking=device.type == "cuda")
                logits = model(inputs)
                probs = F.softmax(logits, dim=1).cpu()

                batch_size_actual = inputs.shape[0]
                for item_idx in range(batch_size_actual):
                    prob_non_header = float(probs[item_idx, 0].item())
                    prob_header = float(probs[item_idx, 1].item()) if probs.shape[1] > 1 else 0.0
                    pred_label = int(prob_header >= 0.5)
                    pred_confidence = max(prob_header, prob_non_header)

                    writer.writerow(
                        {
                            "row_idx": int(batch_item(meta["row_idx"], item_idx)),
                            "video_id": str(batch_item(meta["video_id"], item_idx)),
                            "half": int(batch_item(meta["half"], item_idx)),
                            "frame": int(batch_item(meta["frame"], item_idx)),
                            "label": int(targets[item_idx].item()),
                            "video_path": str(batch_item(meta["video_path"], item_idx)),
                            "fps": float(batch_item(meta["fps"], item_idx)),
                            "ball_x": float(batch_item(meta["ball_x"], item_idx)),
                            "ball_y": float(batch_item(meta["ball_y"], item_idx)),
                            "ball_w": float(batch_item(meta["ball_w"], item_idx)),
                            "ball_h": float(batch_item(meta["ball_h"], item_idx)),
                            "ball_confidence": float(batch_item(meta["ball_confidence"], item_idx)),
                            "prob_header": prob_header,
                            "prob_non_header": prob_non_header,
                            "pred_label_0p5": pred_label,
                            "pred_confidence_0p5": pred_confidence,
                        }
                    )
                    prob_header_sum += prob_header
                    prob_header_sq_sum += prob_header * prob_header
                    rows_written += 1

                del inputs, logits, probs
                if batch_idx % 50 == 0:
                    torch.cuda.empty_cache()
                if batch_idx % 100 == 0:
                    csv_file.flush()

    print("")
    print("Inference complete")
    print(f"Rows written:   {rows_written}")
    print(f"Output CSV:     {output_csv}")
    if rows_written > 0:
        mean_ph = prob_header_sum / rows_written
        std_ph = (prob_header_sq_sum / rows_written - mean_ph ** 2) ** 0.5
        print(f"Header prob:    mean={mean_ph:.4f} std={std_ph:.4f}")


if __name__ == "__main__":
    main()
