"""
Auto-populated context for optional visualizations (Grad-CAM, embeddings, galleries).
Loads the best checkpoint from scratch_output/csn_16frames_test and a small validation
split for quick demos. Adjust paths or dataset choices as needed.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from training.config import Config
from training.data.header_dataset import HeaderCacheDataset, get_transforms
from training.models.factory import build_model

# --------------------
# Paths and config
# --------------------
RUN_DIR = Path(__file__).resolve().parents[1] / "scratch_output" / "csn_16frames_test"
CONFIG_PATH = RUN_DIR / "config.yaml"
BEST_METRICS_PATH = RUN_DIR / "best_metrics.json"
CHECKPOINT_PATH = RUN_DIR / "checkpoints" / "best_epoch_48.pt"

# Local dataset (generated cache)
DATA_ROOT = Path("scratch_output/generate_dataset_test/16_frames_ver/dataset_generation")
VAL_CSV = DATA_ROOT / "val_split.csv"
VAL_CSV_DEBUG = DATA_ROOT / "val_split_debug.csv"


def load_config() -> Config:
    """Load YAML config into Config dataclass."""
    cfg = Config()
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            data = yaml.safe_load(f) or {}
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    # Override CSVs to point to local cache
    if VAL_CSV.exists():
        cfg.val_csv = str(VAL_CSV)
    return cfg


def load_model(cfg: Config, device: torch.device) -> torch.nn.Module:
    """Build CSN model and load best checkpoint."""
    model, _ = build_model(cfg)
    ckpt_path = CHECKPOINT_PATH
    if not ckpt_path.exists() and BEST_METRICS_PATH.exists():
        import json

        with open(BEST_METRICS_PATH, "r") as f:
            meta = json.load(f)
        ckpt_rel = meta.get("checkpoint")
        if ckpt_rel:
            ckpt_path = RUN_DIR / ckpt_rel
    state = torch.load(ckpt_path, map_location=device)
    state_dict = state.get("state_dict", state)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def make_loader(cfg: Config):
    """Build a small validation loader (debug split if available)."""
    val_csv = VAL_CSV_DEBUG if VAL_CSV_DEBUG.exists() else VAL_CSV
    transform = get_transforms(cfg.input_size, is_training=False)
    dataset = HeaderCacheDataset(
        csv_path=val_csv,
        num_frames=cfg.num_frames,
        input_size=cfg.input_size,
        transform=transform,
        is_training=False,
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=False)
    return dataset, loader, transform


# --------------------
# Build context
# --------------------
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = load_config()
model = load_model(config, device)
gradcam_target_layer = model.layer4 if hasattr(model, "layer4") else None
dataset, test_loader, _val_transform = make_loader(config)


def preprocess_frame(frame: np.ndarray) -> torch.Tensor:
    """Apply the same val transforms used during training to a single frame."""
    img = Image.fromarray(frame.astype("uint8"), "RGB")
    return _val_transform(img)


# Prepare a small subset for Grad-CAM and galleries
samples = []
for idx, row in dataset.df.iterrows():
    base_path = Path(row["path"])
    raw_video = np.load(str(base_path) + "_s.npy")  # (T, H, W, 3) uint8
    indices = dataset.sample_indices(raw_video.shape[0])
    sampled_raw = raw_video[indices]

    processed = [preprocess_frame(f) for f in sampled_raw]
    video_tensor = torch.stack(processed, dim=0).permute(1, 0, 2, 3).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(video_tensor)
        prob_header = torch.softmax(logits, dim=1)[0, 1].item()
        pred_label = int(prob_header >= 0.5)

    samples.append(
        {
            "raw": sampled_raw,
            "true": int(row["label"]),
            "prob": prob_header,
            "pred": pred_label,
            "id": str(base_path.name),
        }
    )

# Choose a balanced subset: correct headers, misclassified, then fallback.
max_subset = 6
correct_headers = [s for s in samples if s["true"] == 1 and s["pred"] == 1]
misclassified = [s for s in samples if s["true"] != s["pred"]]
fallback = samples

selected = correct_headers[: max_subset // 2] + misclassified[: max_subset // 2]
if not selected:
    selected = fallback[:max_subset]

test_videos_subset = [s["raw"] for s in selected]
y_true_subset = [s["true"] for s in selected]
y_pred_proba_subset = [s["prob"] for s in selected]
test_video_ids_subset = [s["id"] for s in selected]

# Expose preprocess for Grad-CAM to use normalized inputs
preprocess_fn = preprocess_frame
