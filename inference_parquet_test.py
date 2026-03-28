#!/usr/bin/env python3
"""Run VMAE inference over a dense parquet test dataset."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


HEADER_NET_ROOT = Path(__file__).resolve().parent
if str(HEADER_NET_ROOT) not in sys.path:
    sys.path.insert(0, str(HEADER_NET_ROOT))

from training.config import Config as TrainingConfig
from training.data.parquet_header_dataset import ParquetHeaderDataset, _seed_worker
from training.models.factory import build_model
from training.run_utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run parquet-based VMAE inference on a dense test parquet."
    )
    parser.add_argument(
        "--parquet",
        default=str(HEADER_NET_ROOT / "output" / "dense_dataset" / "dense_test"),
        help="Input dense parquet file or partitioned parquet dataset directory (default: %(default)s)",
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
    parser.add_argument(
        "--spatial-mode",
        choices=("ball_crop", "full_frame"),
        default=None,
        help="Spatial preprocessing policy (default: checkpoint config or ball_crop)",
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
        "--max-open-videos",
        type=int,
        default=4,
        help="Per-worker cap on simultaneously open video readers (default: %(default)s)",
    )
    parser.add_argument(
        "--pin-memory",
        choices=("auto", "on", "off"),
        default="auto",
        help="DataLoader pinned-memory mode: auto uses CUDA-only pinning (default: %(default)s)",
    )
    parser.add_argument(
        "--frame-cache-size",
        type=int,
        default=128,
        help="Per-worker frame cache size (frames)",
    )
    parser.add_argument(
        "--loader-start-method",
        choices=("spawn", "fork", "forkserver"),
        default="spawn",
        help="Multiprocessing start method for DataLoader workers",
    )
    parser.add_argument(
        "--debug-memory",
        action="store_true",
        help="Log main/worker RSS during inference for diagnosing host RAM usage",
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
    parser.add_argument(
        "--resume-output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Resume from an existing output CSV by skipping row_idx values already written",
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


def resolve_pin_memory(pin_memory_arg: str, device: torch.device) -> bool:
    if pin_memory_arg == "on":
        return True
    if pin_memory_arg == "off":
        return False
    return device.type == "cuda"


def _read_kib_fields(pid: int) -> dict[str, int]:
    fields: dict[str, int] = {}
    try:
        with open(f"/proc/{pid}/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith(("VmRSS:", "VmHWM:", "VmSize:")):
                    key, value = line.split(":", 1)
                    fields[key] = int(value.strip().split()[0])
    except FileNotFoundError:
        return {}
    return fields


def _read_direct_children(pid: int) -> list[int]:
    try:
        with open(f"/proc/{pid}/task/{pid}/children", "r", encoding="utf-8") as handle:
            text = handle.read().strip()
    except FileNotFoundError:
        return []
    if not text:
        return []
    return [int(value) for value in text.split()]


def _read_mem_available_kib() -> tuple[int, int]:
    total_kib = 0
    available_kib = 0
    with open("/proc/meminfo", "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("MemTotal:"):
                total_kib = int(line.split()[1])
            elif line.startswith("MemAvailable:"):
                available_kib = int(line.split()[1])
            if total_kib and available_kib:
                break
    return total_kib, available_kib


def _format_gib(kib: int) -> str:
    return f"{kib / 1024 / 1024:.2f} GiB"


def _load_processed_row_indices(output_csv: Path) -> set[int]:
    processed: set[int] = set()
    with open(output_csv, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "row_idx" not in reader.fieldnames:
            raise RuntimeError(
                f"Existing output CSV is missing required row_idx column: {output_csv}"
            )
        for row in reader:
            if not row:
                continue
            row_idx_text = str(row.get("row_idx", "")).strip()
            if not row_idx_text:
                continue
            processed.add(int(row_idx_text))
    return processed


def log_memory_snapshot(root_pid: int, *, batch_idx: int | None = None) -> None:
    main_fields = _read_kib_fields(root_pid)
    child_rows = []
    for child_pid in _read_direct_children(root_pid):
        fields = _read_kib_fields(child_pid)
        if not fields:
            continue
        child_rows.append((child_pid, fields.get("VmRSS", 0)))
    child_rows.sort(key=lambda row: row[1], reverse=True)

    total_kib, available_kib = _read_mem_available_kib()
    used_kib = max(total_kib - available_kib, 0)
    prefix = "[memory]" if batch_idx is None else f"[memory] batch={batch_idx}"
    main_rss = main_fields.get("VmRSS", 0)
    child_rss = sum(rss for _, rss in child_rows)
    print(
        f"{prefix} main_pid={root_pid} main_rss={_format_gib(main_rss)} "
        f"children={len(child_rows)} children_rss={_format_gib(child_rss)} "
        f"system_used={_format_gib(used_kib)}/{_format_gib(total_kib)}"
    )
    if child_rows:
        top_children = ", ".join(
            f"{pid}:{_format_gib(rss)}" for pid, rss in child_rows[:8]
        )
        print(f"{prefix} child_rss {top_children}")


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
    spatial_mode = str(
        resolve_from_checkpoint(args.spatial_mode, checkpoint_config, "spatial_mode", "ball_crop")
    )
    num_frames = int(resolve_from_checkpoint(args.num_frames, checkpoint_config, "num_frames", 16))
    input_size = int(resolve_from_checkpoint(args.input_size, checkpoint_config, "input_size", 224))
    batch_size = int(resolve_from_checkpoint(args.batch_size, checkpoint_config, "batch_size", 8))
    seed = int(resolve_from_checkpoint(args.seed, checkpoint_config, "seed", 42))
    backbone_ckpt = resolve_backbone_ckpt(args.backbone_ckpt, checkpoint_config)
    device, gpu_ids = resolve_device_config(args.gpus, args.device)
    pin_memory = resolve_pin_memory(args.pin_memory, device)
    max_open_videos = int(args.max_open_videos)
    preprocess_mode = "low_memory_eval"

    set_seed(seed)

    print(f"Parquet:        {parquet_path}")
    print(f"Checkpoint:     {checkpoint_path}")
    print(f"Backbone:       {backbone}")
    print(f"Spatial mode:   {spatial_mode}")
    if backbone_ckpt is not None:
        print(f"Backbone ckpt:  {backbone_ckpt}")
    print(f"Num frames:     {num_frames}")
    print(f"Input size:     {input_size}")
    print(f"Batch size:     {batch_size}")
    print(f"Workers:        {args.num_workers}")
    print(f"Pin memory:     {pin_memory}")
    print(f"Max open vids:  {max_open_videos}")
    print(f"Frame cache:    {args.frame_cache_size}")
    print(f"Start method:   {args.loader_start_method}")
    print(f"Preprocess:     {preprocess_mode}")
    print(f"Debug memory:   {args.debug_memory}")
    if gpu_ids:
        print(f"GPUs:           {gpu_ids}")
    print(f"Device:         {device}")

    dataset = ParquetHeaderDataset(
        parquet_path=parquet_path,
        num_frames=num_frames,
        input_size=input_size,
        transform=None,
        strict_paths=True,
        dataset_root=Path(args.dataset_root).expanduser(),
        max_open_videos=max_open_videos,
        frame_cache_size=int(args.frame_cache_size),
        resample_on_decode_failure=False,
        preprocess_mode=preprocess_mode,
        spatial_mode=spatial_mode,
        video_id_filters=[args.video_id] if args.video_id else (),
        half_filters=[args.half] if args.half is not None else (),
    )

    dataset_for_loader = dataset
    existing_rows = 0
    csv_mode = "w"
    write_header = True
    if args.resume_output and output_csv.exists() and output_csv.stat().st_size > 0:
        processed_row_indices = _load_processed_row_indices(output_csv)
        existing_rows = len(processed_row_indices)
        remaining_indices = [
            idx
            for idx, row_idx in enumerate(dataset.row_indices.tolist())
            if int(row_idx) not in processed_row_indices
        ]
        if not remaining_indices:
            print(f"Existing output already covers all {len(dataset)} samples: {output_csv}")
            return
        dataset_for_loader = Subset(dataset, remaining_indices)
        csv_mode = "a"
        write_header = False
        print(
            "Resuming from existing output: "
            f"{existing_rows} rows already present, {len(remaining_indices)} samples remaining."
        )

    num_workers = int(args.num_workers)
    loader_kwargs = {}
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = False
        loader_kwargs["prefetch_factor"] = 1
        loader_kwargs["multiprocessing_context"] = args.loader_start_method
    loader = DataLoader(
        dataset_for_loader,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker,
        **loader_kwargs,
    )

    model = build_model_from_checkpoint(
        checkpoint_payload=checkpoint_payload,
        backbone=backbone,
        num_frames=num_frames,
        input_size=input_size,
        backbone_ckpt=backbone_ckpt,
    )
    del checkpoint_payload
    model = model.to(device)
    if len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    model.eval()

    print(f"Evaluating {len(dataset_for_loader)} samples...")
    if args.debug_memory:
        log_memory_snapshot(os.getpid())

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

    with open(output_csv, csv_mode, newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames)
        if write_header:
            writer.writeheader()

        with torch.inference_mode():
            for batch_idx, (inputs, targets, meta) in enumerate(
                tqdm(loader, desc="Parquet inference", unit="batch")
            ):
                inputs = inputs.to(device, non_blocking=device.type == "cuda")
                logits = model(inputs)
                probs = F.softmax(logits, dim=1).cpu()
                if args.debug_memory and (batch_idx == 0 or batch_idx % 10 == 0):
                    log_memory_snapshot(os.getpid(), batch_idx=batch_idx)

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
                if batch_idx % 100 == 0:
                    csv_file.flush()

    print("")
    print("Inference complete")
    print(f"Rows written:   {rows_written}")
    if existing_rows:
        print(f"Rows reused:    {existing_rows}")
        print(f"Rows total:     {existing_rows + rows_written}")
    print(f"Output CSV:     {output_csv}")
    if rows_written > 0:
        mean_ph = prob_header_sum / rows_written
        std_ph = (prob_header_sq_sum / rows_written - mean_ph ** 2) ** 0.5
        print(f"Header prob:    mean={mean_ph:.4f} std={std_ph:.4f}")


if __name__ == "__main__":
    main()
