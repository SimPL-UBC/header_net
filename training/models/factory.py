import torch.nn as nn
from ..config import Config
from .backbones import build_csn_backbone, build_vmae_backbone
from .heads import VideoMAEHead


class VideoMAEModel(nn.Module):
    """Combined VideoMAE backbone + classification head."""
    
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head
    
    def forward(self, x):
        features = self.backbone(x)
        logits = self.head(features)
        return logits


def build_model(config: Config):
    """
    Builds the model and parameter groups based on configuration.
    """
    if config.backbone == "csn":
        if config.finetune_mode != "full":
            raise ValueError(f"CSN only supports finetune_mode='full', got '{config.finetune_mode}'")
        
        model = build_csn_backbone(config)
        param_groups = [
            {'params': model.parameters(), 'lr': config.lr_backbone}
        ]
        
    elif config.backbone == "vmae":
        # Build backbone
        backbone = build_vmae_backbone(config)
        
        # Build head
        head = VideoMAEHead(
            hidden_dim=backbone.hidden_dim,
            num_classes=2,
            dropout=0.5
        )
        
        # Combine into single model
        model = VideoMAEModel(backbone, head)
        
        if config.finetune_mode == "frozen":
            # Freeze all backbone parameters
            for param in backbone.parameters():
                param.requires_grad = False
            
            # Only train the head
            param_groups = [
                {'params': head.parameters(), 'lr': config.lr_head}
            ]
            print(f"VideoMAE backbone frozen. Training only head with lr={config.lr_head}")
            
        elif config.finetune_mode == "full":
            # Train both backbone and head (Phase 3)
            param_groups = [
                {'params': backbone.parameters(), 'lr': config.lr_backbone},
                {'params': head.parameters(), 'lr': config.lr_head}
            ]
            print(f"VideoMAE full fine-tuning: backbone lr={config.lr_backbone}, head lr={config.lr_head}")
        else:
            raise ValueError(f"Unsupported finetune_mode for vmae: {config.finetune_mode}")
    else:
        raise ValueError(f"Unsupported backbone: {config.backbone}")
    
    return model, param_groups
