import torch
import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import torchvision.transforms as T
from PIL import Image
from ..config import Config

class HeaderCacheDataset(Dataset):
    def __init__(self, csv_path, num_frames=11, input_size=224, transform=None, is_training=True):
        self.csv_path = csv_path  # Store for path resolution
        self.df = pd.read_csv(csv_path)
        self.num_frames = num_frames
        self.input_size = input_size
        self.transform = transform
        self.is_training = is_training

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row['path']
        label = row['label']
        
        # Handle path resolution:
        # Try absolute path from CSV first, then fall back to basename approach
        path_str = str(path)
        
        # Option 1: Try absolute path from CSV with _s.npy suffix
        cache_path_absolute = Path(f"{path_str}_s.npy")
        
        if cache_path_absolute.exists():
            cache_path = cache_path_absolute
        else:
            # Option 2: Fall back to basename in CSV directory
            # The CSV may contain absolute paths from another machine
            filename = os.path.basename(path_str)
            csv_dir = Path(self.csv_path).parent
            cache_path = csv_dir / f"{filename}_s.npy"
        
        try:
            # Load numpy array. Assuming shape (T, H, W, C) RGB.
            video_data = np.load(cache_path)
        except Exception as e:
            print(f"Error loading {cache_path}: {e}")
            raise e

        total_frames = video_data.shape[0]
        indices = self.sample_indices(total_frames)
        
        frames = video_data[indices] # (T, H, W, C)
        
        processed_frames = []
        for i in range(self.num_frames):
            frame = frames[i]
            # Convert to PIL for torchvision transforms
            img = Image.fromarray(frame.astype('uint8'), 'RGB')
            if self.transform:
                img = self.transform(img)
            processed_frames.append(img)
            
        # Stack: (T, C, H, W)
        video = torch.stack(processed_frames)
        # Convert to (C, T, H, W) for CSN
        video = video.permute(1, 0, 2, 3)
        
        meta = {
            "video_id": str(row['video_id']),
            "half": str(row['half']),
            "frame": int(row['frame']),
            "path": str(path)
        }
        
        return video, int(label), meta

    def sample_indices(self, total_frames):
        if total_frames == 0:
            return np.zeros(self.num_frames, dtype=int)
            
        if total_frames < self.num_frames:
            # Loop to fill
            indices = np.arange(total_frames)
            # Pad with wrap
            pad = self.num_frames - total_frames
            extra = np.resize(indices, pad)
            indices = np.concatenate([indices, extra])
            indices = np.sort(indices)
        else:
            # Uniform sampling (center of segments) or linspace
            # "matches current HeaderDataset behavior (11 segments)"
            # Usually segment-based sampling: split T into N segments, pick center/random.
            # Here we'll use linspace for simplicity and "center" sampling equivalent
            indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
            
        return indices

def get_transforms(input_size=224, is_training=True, backbone="csn"):
    # Use ImageNet normalization for both backbones.
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    
    if is_training:
        return T.Compose([
            T.Resize((input_size, input_size)), # Or Resize(256) then Crop? 
            # "224x224 resize" - usually implies resizing to input_size directly or RandomResizedCrop
            # Given "reuse CSN transforms", and typical video classification:
            # Often: Resize(256), RandomCrop(224) or RandomResizedCrop(224).
            # I will use Resize((input_size, input_size)) + RandomHorizontalFlip + ColorJitter as requested.
            # "random flip, jitter"
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), # Light jitter
            T.ToTensor(),
            normalize
        ])
    else:
        return T.Compose([
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            normalize
        ])

def build_dataloaders(config: Config):
    train_transform = get_transforms(config.input_size, is_training=True, backbone=config.backbone)
    val_transform = get_transforms(config.input_size, is_training=False, backbone=config.backbone)
    
    train_dataset = HeaderCacheDataset(
        config.train_csv,
        num_frames=config.num_frames,
        input_size=config.input_size,
        transform=train_transform,
        is_training=True
    )
    
    val_dataset = HeaderCacheDataset(
        config.val_csv,
        num_frames=config.num_frames,
        input_size=config.input_size,
        transform=val_transform,
        is_training=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
