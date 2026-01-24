#!/usr/bin/env python3
"""Create an offline augmented dataset from cached clips."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageOps
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Augment cached dataset clips and produce a combined CSV."
    )
    parser.add_argument("--input_csv", required=True, help="Input cache CSV")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument(
        "--output_name",
        default="train_with_aug.csv",
        help="Output CSV filename",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--flip_p", type=float, default=0.5, help="Flip probability")
    parser.add_argument("--crop_p", type=float, default=0.5, help="Random crop probability")
    parser.add_argument(
        "--crop_scale_min", type=float, default=0.8, help="Min crop scale"
    )
    parser.add_argument(
        "--crop_scale_max", type=float, default=1.0, help="Max crop scale"
    )
    parser.add_argument(
        "--rotation_p", type=float, default=0.2, help="Rotation probability"
    )
    parser.add_argument(
        "--rotation_deg",
        type=float,
        default=10.0,
        help="Rotation range in degrees (+/-)",
    )
    parser.add_argument("--blur_p", type=float, default=0.2, help="Blur probability")
    parser.add_argument(
        "--blur_radius", type=float, default=1.0, help="Gaussian blur radius"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing augmented outputs",
    )
    return parser.parse_args()


def resolve_cache_path(path_str: str, csv_dir: Path) -> Path:
    candidate = Path(f"{path_str}_s.npy")
    if candidate.exists():
        return candidate
    filename = Path(path_str).name
    candidate = csv_dir / f"{filename}_s.npy"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Cache file not found for {path_str}")


def resolve_meta_path(row: pd.Series, csv_dir: Path, cache_path: Path) -> Optional[Path]:
    if "metadata" in row and isinstance(row["metadata"], str) and row["metadata"]:
        meta_candidate = Path(row["metadata"])
        if meta_candidate.exists():
            return meta_candidate
        filename = Path(row["metadata"]).name
        alt = csv_dir / filename
        if alt.exists():
            return alt

    base_str = str(cache_path)
    if base_str.endswith("_s.npy"):
        base_str = base_str[:-6]
    meta_candidate = Path(base_str + "_meta.json")
    if meta_candidate.exists():
        return meta_candidate
    return None


def base_from_cache_path(cache_path: Path) -> Path:
    base_str = str(cache_path)
    if base_str.endswith("_s.npy"):
        base_str = base_str[:-6]
    return Path(base_str)


def sample_crop_params(
    rng: np.random.Generator, width: int, height: int, scale_min: float, scale_max: float
) -> Tuple[int, int, int, int, float]:
    scale = float(rng.uniform(scale_min, scale_max))
    crop_w = max(1, int(round(width * scale)))
    crop_h = max(1, int(round(height * scale)))
    left = int(rng.integers(0, max(1, width - crop_w + 1)))
    top = int(rng.integers(0, max(1, height - crop_h + 1)))
    return left, top, crop_w, crop_h, scale


def apply_augmentations(
    clip: np.ndarray,
    rng: np.random.Generator,
    flip_p: float,
    crop_p: float,
    crop_scale_min: float,
    crop_scale_max: float,
    rotation_p: float,
    rotation_deg: float,
    blur_p: float,
    blur_radius: float,
) -> Tuple[np.ndarray, Dict[str, object]]:
    if clip.ndim != 4:
        raise ValueError(f"Expected clip shape (T,H,W,C), got {clip.shape}")

    height, width = clip.shape[1], clip.shape[2]

    do_flip = rng.random() < flip_p
    do_crop = rng.random() < crop_p
    do_rotate = rng.random() < rotation_p
    do_blur = rng.random() < blur_p

    crop_params = None
    if do_crop:
        crop_params = sample_crop_params(rng, width, height, crop_scale_min, crop_scale_max)

    angle = 0.0
    if do_rotate:
        angle = float(rng.uniform(-rotation_deg, rotation_deg))

    aug_meta: Dict[str, object] = {
        "flip": bool(do_flip),
        "crop": None,
        "rotation_deg": angle if do_rotate else 0.0,
        "blur_radius": blur_radius if do_blur else 0.0,
    }

    if crop_params:
        left, top, crop_w, crop_h, scale = crop_params
        aug_meta["crop"] = {
            "left": int(left),
            "top": int(top),
            "width": int(crop_w),
            "height": int(crop_h),
            "scale": float(scale),
        }

    frames: List[np.ndarray] = []
    for frame in clip:
        img = Image.fromarray(frame.astype("uint8"), "RGB")
        if do_flip:
            img = ImageOps.mirror(img)
        if crop_params:
            left, top, crop_w, crop_h, _ = crop_params
            img = img.crop((left, top, left + crop_w, top + crop_h))
            img = img.resize((width, height), resample=Image.BILINEAR)
        if do_rotate:
            img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
        if do_blur:
            img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        frames.append(np.asarray(img, dtype=np.uint8))

    return np.stack(frames, axis=0), aug_meta


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_name = args.output_name

    output_dir.mkdir(parents=True, exist_ok=True)
    if Path(output_name).name != output_name:
        raise ValueError("output_name must be a filename, not a path")

    df = pd.read_csv(input_csv)
    if df.empty:
        raise SystemExit("Input CSV is empty.")

    rng = np.random.default_rng(args.seed)
    csv_dir = input_csv.parent

    output_records: List[Dict[str, object]] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Augmenting clips"):
        row_dict = row.to_dict()
        cache_path = resolve_cache_path(str(row_dict["path"]), csv_dir)
        source_base = base_from_cache_path(cache_path)
        meta_path = resolve_meta_path(row, csv_dir, cache_path)

        output_base = output_dir / source_base.name
        output_cache = Path(str(output_base) + "_s.npy")
        output_meta = Path(str(output_base) + "_meta.json")

        if args.overwrite or not output_cache.exists():
            shutil.copy2(cache_path, output_cache)

        if meta_path and (args.overwrite or not output_meta.exists()):
            shutil.copy2(meta_path, output_meta)
        elif not output_meta.exists():
            minimal_meta = {
                "video_id": row_dict.get("video_id"),
                "half": int(row_dict.get("half", 1)),
                "frame": int(row_dict.get("frame", 0)),
                "label": int(row_dict.get("label", 0)),
                "frames": [],
            }
            with output_meta.open("w", encoding="utf-8") as fh:
                json.dump(minimal_meta, fh, ensure_ascii=False, indent=2)

        base_record = dict(row_dict)
        base_record["path"] = str(output_base)
        if "metadata" in base_record:
            base_record["metadata"] = str(output_meta)
        base_record["augmented"] = 0
        base_record["augmentation"] = ""
        output_records.append(base_record)

        clip = np.load(cache_path)
        aug_clip, aug_meta = apply_augmentations(
            clip,
            rng,
            args.flip_p,
            args.crop_p,
            args.crop_scale_min,
            args.crop_scale_max,
            args.rotation_p,
            args.rotation_deg,
            args.blur_p,
            args.blur_radius,
        )

        aug_base = output_dir / f"{source_base.name}_aug"
        aug_cache = Path(str(aug_base) + "_s.npy")
        aug_meta_path = Path(str(aug_base) + "_meta.json")

        if args.overwrite or not aug_cache.exists():
            np.save(aug_cache, aug_clip.astype(np.uint8))

        aug_meta_payload: Dict[str, object] = {}
        if meta_path and meta_path.exists():
            try:
                with meta_path.open("r", encoding="utf-8") as fh:
                    aug_meta_payload = json.load(fh)
            except json.JSONDecodeError:
                aug_meta_payload = {}

        aug_meta_payload["augmentation"] = aug_meta
        if args.overwrite or not aug_meta_path.exists():
            with aug_meta_path.open("w", encoding="utf-8") as fh:
                json.dump(aug_meta_payload, fh, ensure_ascii=False, indent=2)

        aug_record = dict(base_record)
        aug_record["path"] = str(aug_base)
        if "metadata" in aug_record:
            aug_record["metadata"] = str(aug_meta_path)
        aug_record["augmented"] = 1
        aug_record["augmentation"] = json.dumps(aug_meta, ensure_ascii=False)
        output_records.append(aug_record)

    output_df = pd.DataFrame(output_records)
    output_df.to_csv(output_dir / output_name, index=False)
    print(f"[INFO] Wrote {len(output_df)} rows to {output_dir / output_name}")


if __name__ == "__main__":
    main()
