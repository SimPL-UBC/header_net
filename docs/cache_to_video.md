# cache_to_video.py

Utility script to convert cached header samples (`*_s.npy`) into label-wise inspection videos.

## Requirements

- Cache index CSV produced by `cache/create_cache_header.py` (defaults to `cache/cache_header/train_cache_header.csv`).
- Cache tensor directory containing the `*_s.npy` files (defaults to `cache/cache_header`).
- OpenCV (`cv2`), pandas, and NumPy already included in the project env.

## Basic usage

```bash
python tools/cache_to_video.py \
  --cache-index /path/to/train_cache_header.csv \
  --cache-root /path/to/cache_header \
  --output-dir /path/to/output_videos
```

Writes per-match folders under `--output-dir`, each containing:

- `pos.mp4` – concatenated positive (label `1`) samples.
- `neg.mp4` – concatenated negative (label `0`) samples.

Frames include top-left overlays with sample index, half, absolute frame, and temporal offset.

## Inspect a single match

Use `--matches` with the match identifier (case-insensitive substring). Example:

```bash
python tools/cache_to_video.py \
  --cache-index /path/to/train_cache_header.csv \
  --cache-root /path/to/cache_header \
  --output-dir /path/to/output_videos \
  --matches 2015-04-11-16-30BayernMunich3-0EintrachtFrankfurt
```

You can also provide partial names:

```bash
python tools/cache_to_video.py \
  --cache-index /path/to/train_cache_header.csv \
  --cache-root /path/to/cache_header \
  --output-dir /path/to/output_videos \
  --matches bayernm
```

Multiple matches can be passed by listing additional identifiers:

```bash
python tools/cache_to_video.py \
  --cache-index /path/to/train_cache_header.csv \
  --cache-root /path/to/cache_header \
  --output-dir /path/to/output_videos \
  --matches bayernm parisSG
```

## Helpful options

- `--limit-per-label N` – cap samples per label (e.g. 50) for quick spot checks.
- `--fps 8` – change playback speed.
- `--scale 3.0` – resize cached 64×64 patches for larger previews.
- `--pad-between-samples 5` – insert blank frames between samples.
- `--output-dir some/path` – write videos to a different directory.
- `--overwrite` – regenerate videos if they already exist.

Run `python tools/cache_to_video.py --help` for the full list of flags.
