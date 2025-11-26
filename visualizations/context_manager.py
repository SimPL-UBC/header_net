import json
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Any, Callable

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from training.config import Config
from training.data.header_dataset import HeaderCacheDataset, get_transforms
from training.models.factory import build_model

@dataclass
class VisualizationContext:
    model: torch.nn.Module
    device: torch.device
    test_loader: DataLoader
    gradcam_target_layer: Optional[Any]
    test_videos_subset: List[np.ndarray]
    y_true_subset: List[int]
    y_pred_proba_subset: List[float]
    preprocess_fn: Callable[[np.ndarray], torch.Tensor]

def load_config(config_path: Path, val_csv_path: Optional[Path] = None) -> Config:
    """Load YAML config into Config dataclass."""
    cfg = Config()
    if config_path.exists():
        with open(config_path, "r") as f:
            data = yaml.safe_load(f) or {}
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    
    # Override val_csv if provided
    if val_csv_path:
        cfg.val_csv = str(val_csv_path)
        
    return cfg

def load_model_from_run(run_dir: Path, cfg: Config, device: torch.device, checkpoint_path: Optional[Path] = None) -> torch.nn.Module:
    """Build model and load best checkpoint."""
    model, _ = build_model(cfg)
    
    # Determine checkpoint path
    if checkpoint_path is None:
        best_metrics_path = run_dir / "best_metrics.json"
        if best_metrics_path.exists():
            with open(best_metrics_path, "r") as f:
                meta = json.load(f)
            ckpt_rel = meta.get("checkpoint")
            if ckpt_rel:
                checkpoint_path = run_dir / ckpt_rel
    
    if checkpoint_path and checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device)
        state_dict = state.get("state_dict", state)
        
        # Handle DataParallel wrapped models
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
    else:
        print("WARNING: No checkpoint found, using random initialization!")
    
    model.to(device)
    model.eval()
    return model

def get_gradcam_target_layer(model: torch.nn.Module) -> Optional[Any]:
    """Determine target layer for Grad-CAM based on model architecture."""
    # VideoMAE
    if hasattr(model, 'backbone'):  # VideoMAEModel wrapper
        if hasattr(model.backbone, 'vmae'):
            # Get the last transformer block
            if hasattr(model.backbone.vmae, 'blocks'):
                return model.backbone.vmae.blocks[-1]
    
    # CSN (ResNet3dCSN)
    if hasattr(model, 'layer4'):
        return model.layer4
        
    return None

def build_context(
    run_dir: Path,
    val_csv_path: Optional[Path] = None,
    checkpoint_path: Optional[Path] = None,
    max_samples: int = 50,
    subset_size: int = 12,
    batch_size: int = 2
) -> VisualizationContext:
    """Build the visualization context dynamically."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_path = run_dir / "config.yaml"
    
    # Load config
    cfg = load_config(config_path, val_csv_path)
    
    # If val_csv not provided in args or config, try to find it in default location relative to run
    if not cfg.val_csv or not Path(cfg.val_csv).exists():
        # Fallback logic could go here, but for now rely on config or arg
        pass

    print(f"Using validation CSV: {cfg.val_csv}")
    
    # Load model
    model = load_model_from_run(run_dir, cfg, device, checkpoint_path)
    
    # Get Grad-CAM target layer
    gradcam_target_layer = get_gradcam_target_layer(model)
    
    # Build loader
    transform = get_transforms(cfg.input_size, is_training=False, backbone=cfg.backbone)
    dataset = HeaderCacheDataset(
        csv_path=cfg.val_csv,
        num_frames=cfg.num_frames,
        input_size=cfg.input_size,
        transform=transform,
        is_training=False,
    )
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Preprocess function for Grad-CAM
    def preprocess_fn(frame: np.ndarray) -> torch.Tensor:
        img = Image.fromarray(frame.astype("uint8"), "RGB")
        return transform(img)

    # Prepare subset for galleries/Grad-CAM
    print(f"Loading subset of validation data (max {max_samples} samples)...")
    samples = []
    
    # Iterate through dataset to find valid samples
    # We access the dataframe directly to get paths
    for idx, row in dataset.df.head(max_samples).iterrows():
        try:
            # Handle path resolution similar to dataset class
            path_str = str(row['path'])
            filename = Path(path_str).name
            
            # Try to find the cache file
            # 1. Try path as is (if absolute/relative and correct)
            cache_path = Path(path_str + "_s.npy")
            if not cache_path.exists():
                # 2. Try in same dir as CSV (common pattern in this project)
                csv_dir = Path(cfg.val_csv).parent
                cache_path = csv_dir / f"{filename}_s.npy"
            
            if not cache_path.exists():
                continue
                
            raw_video = np.load(str(cache_path))
            indices = dataset.sample_indices(raw_video.shape[0])
            sampled_raw = raw_video[indices]

            # Process for inference
            processed = [preprocess_fn(f) for f in sampled_raw]
            video_tensor = torch.stack(processed, dim=0).permute(1, 0, 2, 3).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(video_tensor)
                prob_header = torch.softmax(logits, dim=1)[0, 1].item()
                pred_label = int(prob_header >= 0.5)

            samples.append({
                "raw": sampled_raw,
                "true": int(row["label"]),
                "prob": prob_header,
                "pred": pred_label,
                "id": str(filename),
            })
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue

    print(f"Loaded {len(samples)} samples.")
    
    # Select balanced subset
    correct_headers = [s for s in samples if s["true"] == 1 and s["pred"] == 1]
    misclassified = [s for s in samples if s["true"] != s["pred"]]
    fallback = samples
    
    selected = correct_headers[: subset_size // 2] + misclassified[: subset_size // 2]
    # Use IDs for filtering to avoid numpy array comparison issues
    selected_ids = {s["id"] for s in selected}
    if len(selected) < subset_size:
        remaining = [s for s in fallback if s["id"] not in selected_ids]
        selected.extend(remaining[: subset_size - len(selected)])
        
    test_videos_subset = [s["raw"] for s in selected]
    y_true_subset = [s["true"] for s in selected]
    y_pred_proba_subset = [s["prob"] for s in selected]
    
    return VisualizationContext(
        model=model,
        device=device,
        test_loader=test_loader,
        gradcam_target_layer=gradcam_target_layer,
        test_videos_subset=test_videos_subset,
        y_true_subset=y_true_subset,
        y_pred_proba_subset=y_pred_proba_subset,
        preprocess_fn=preprocess_fn
    )
