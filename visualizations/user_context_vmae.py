"""
Context for VideoMAE visualizations (Grad-CAM, embeddings, galleries).
Loads the best checkpoint from your VideoMAE training run.

Usage:
1. After training completes, update RUN_DIR to point to your output directory
2. Rename this file to user_context.py (or update the import in run_all.py)
3. Run visualizations with --enable-gradcam --enable-embedding --enable-galleries
"""

from pathlib import Path
from typing import Optional
import json

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from training.config import Config
from training.data.header_dataset import HeaderCacheDataset, get_transforms
from training.models.factory import build_model

# --------------------
# Paths and config - UPDATE THESE AFTER TRAINING
# --------------------
RUN_DIR = Path("/scratch/st-lyndiacw-1/gyan/vmae_result/vmae_full_base")
CONFIG_PATH = RUN_DIR / "config.yaml"
BEST_METRICS_PATH = RUN_DIR / "best_metrics.json"

# Validation dataset (used for test loader and galleries)
VAL_CSV = Path("/scratch/st-lyndiacw-1/gyan/train_dataset/val.csv")


def load_config() -> Config:
    """Load YAML config into Config dataclass."""
    cfg = Config()
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            data = yaml.safe_load(f) or {}
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    return cfg


def load_model(cfg: Config, device: torch.device) -> torch.nn.Module:
    """Build VideoMAE model and load best checkpoint."""
    model, _ = build_model(cfg)
    
    # Get checkpoint path from best_metrics.json
    ckpt_path = None
    if BEST_METRICS_PATH.exists():
        with open(BEST_METRICS_PATH, "r") as f:
            meta = json.load(f)
        ckpt_rel = meta.get("checkpoint")
        if ckpt_rel:
            ckpt_path = RUN_DIR / ckpt_rel
    
    if ckpt_path and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        state_dict = state.get("state_dict", state)
        
        # Handle DataParallel wrapped models
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint from {ckpt_path}")
    else:
        print("WARNING: No checkpoint found, using random initialization!")
    
    model.to(device)
    model.eval()
    return model


def make_loader(cfg: Config):
    """Build validation loader for embeddings."""
    transform = get_transforms(cfg.input_size, is_training=False, backbone=cfg.backbone)
    dataset = HeaderCacheDataset(
        csv_path=str(VAL_CSV),
        num_frames=cfg.num_frames,
        input_size=cfg.input_size,
        transform=transform,
        is_training=False,
    )
    # Use smaller batch size for VideoMAE due to memory
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)
    return dataset, loader, transform


# --------------------
# Build context
# --------------------
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = load_config()
model = load_model(config, device)

# Grad-CAM target layer for VideoMAE
# For VideoMAE, we want to target the last transformer block
if hasattr(model, 'backbone'):  # VideoMAEModel wrapper
    if hasattr(model.backbone, 'vmae'):
        # Get the last transformer block
        if hasattr(model.backbone.vmae, 'blocks'):
            gradcam_target_layer = model.backbone.vmae.blocks[-1]
        else:
            gradcam_target_layer = None
    else:
        gradcam_target_layer = None
else:
    gradcam_target_layer = None

dataset, test_loader, _val_transform = make_loader(config)


def preprocess_frame(frame: np.ndarray) -> torch.Tensor:
    """Apply the same val transforms used during training to a single frame."""
    img = Image.fromarray(frame.astype("uint8"), "RGB")
    return _val_transform(img)


print(f"Loading subset of validation data for galleries and Grad-CAM...")
# Prepare a subset for Grad-CAM and galleries (limit to avoid long loading times)
MAX_SAMPLES = 50  # Limit total samples to process
samples = []

for idx, row in dataset.df.head(MAX_SAMPLES).iterrows():
    try:
        # Get cache path
        csv_dir = Path(dataset.csv_path).parent
        filename = Path(row['path']).name
        cache_path = csv_dir / f"{filename}_s.npy"
        
        if not cache_path.exists():
            continue
            
        raw_video = np.load(str(cache_path))  # (T, H, W, 3) uint8
        indices = dataset.sample_indices(raw_video.shape[0])
        sampled_raw = raw_video[indices]

        # Process for inference
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
                "id": str(filename),
            }
        )
    except Exception as e:
        print(f"Error processing sample {idx}: {e}")
        continue

print(f"Loaded {len(samples)} samples for visualization")

# Choose a balanced subset: correct headers, misclassified, then fallback
max_subset = 12
correct_headers = [s for s in samples if s["true"] == 1 and s["pred"] == 1]
misclassified = [s for s in samples if s["true"] != s["pred"]]
fallback = samples

selected = correct_headers[: max_subset // 2] + misclassified[: max_subset // 2]
if len(selected) < max_subset:
    # Fill remaining with any samples
    remaining = [s for s in fallback if s not in selected]
    selected.extend(remaining[: max_subset - len(selected)])

test_videos_subset = [s["raw"] for s in selected]
y_true_subset = [s["true"] for s in selected]
y_pred_proba_subset = [s["prob"] for s in selected]
test_video_ids_subset = [s["id"] for s in selected]

# Expose preprocess for Grad-CAM to use normalized inputs
preprocess_fn = preprocess_frame

print(f"Prepared {len(selected)} samples for Grad-CAM and galleries")
print(f"  - Correct predictions: {sum(1 for s in selected if s['true'] == s['pred'])}")
print(f"  - Misclassifications: {sum(1 for s in selected if s['true'] != s['pred'])}")
