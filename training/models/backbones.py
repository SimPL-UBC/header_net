import sys
import os
import torch
import torch.nn as nn
from pathlib import Path
from safetensors.torch import load_file

from .resnet3d_csn_local import ResNet3dCSN

def build_csn_backbone(config):
        
    # Constructs model: num_classes=2, depth 50 (as per train_header.py)
    model = ResNet3dCSN(
        pretrained2d=False,
        pretrained=None,
        depth=50,
        with_pool2=False,
        bottleneck_mode='ir',
        norm_eval=False,
        zero_init_residual=False,
        bn_frozen=False,
        num_classes=2
    )
    return model


class VideoMAEBackbone(nn.Module):
    """Wrapper for VideoMAE model to use as a feature extractor."""
    
    def __init__(self, vmae_model, hidden_dim):
        super().__init__()
        self.vmae = vmae_model
        self.hidden_dim = hidden_dim
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, T, H, W)
        Returns:
            features: Tensor of shape (B, hidden_dim)
        """
        # VideoMAE expects (B, C, T, H, W) which matches our format
        features = self.vmae.forward_features(x)
        return features


def build_vmae_backbone(config):
    """
    Build VideoMAE v2 backbone from checkpoint.
    
    Args:
        config: Config object with backbone_ckpt path
        
    Returns:
        VideoMAEBackbone module
    """
    # Default to Base if not specified
    if config.backbone_ckpt is None:
        ckpt_dir = Path("checkpoints/VideoMAEv2-Base")
    else:
        ckpt_dir = Path(config.backbone_ckpt)
    
    if not ckpt_dir.exists():
        raise ValueError(f"Checkpoint directory not found: {ckpt_dir}")
    
    # Add checkpoint directory to sys.path to allow imports
    ckpt_dir_abs = str(ckpt_dir.absolute())
    sys.path.insert(0, ckpt_dir_abs)
    try:
        import modeling_videomaev2
        vit_base_patch16_224 = modeling_videomaev2.vit_base_patch16_224
    finally:
        # Clean up sys.path
        if ckpt_dir_abs in sys.path:
            sys.path.remove(ckpt_dir_abs)
    
    # Determine hidden dimension based on model variant
    # Base: 768, Large: 1024, Giant: 1408
    model_name = ckpt_dir.name.lower()
    if "giant" in model_name:
        hidden_dim = 1408
    elif "large" in model_name:
        hidden_dim = 1024
    else:  # base
        hidden_dim = 768
    
    # Build model with correct parameters for video
    # The function vit_base_patch16_224 already sets norm_layer and other defaults
    model = vit_base_patch16_224(
        num_classes=0,  # No classifier head
        num_frames=config.num_frames,  # Use config num_frames
        tubelet_size=2,
        drop_path_rate=0.0,
        use_mean_pooling=True
    )
    
    # Load pretrained weights from safetensors
    safetensors_path = ckpt_dir / "model.safetensors"
    if safetensors_path.exists():
        state_dict = load_file(str(safetensors_path))
        
        # Remove 'model.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("model.", "") if k.startswith("model.") else k
            new_state_dict[new_key] = v
        
        # Load with strict=False to handle missing head weights
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded VideoMAE checkpoint from {safetensors_path}")
        if missing:
            print(f"Missing keys: {len(missing)} total")
        if unexpected:
            print(f"Unexpected keys: {len(unexpected)} total")
    else:
        print(f"Warning: No safetensors checkpoint found at {safetensors_path}, using random initialization")
    
    return VideoMAEBackbone(model, hidden_dim)
