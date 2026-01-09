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


def _get_vmae_blocks(backbone):
    vmae = backbone.vmae if hasattr(backbone, "vmae") else backbone

    if hasattr(vmae, "blocks"):
        return vmae.blocks

    if hasattr(vmae, "encoder"):
        encoder = vmae.encoder
        if hasattr(encoder, "blocks"):
            return encoder.blocks
        if hasattr(encoder, "layers"):
            return encoder.layers

    if hasattr(vmae, "transformer"):
        transformer = vmae.transformer
        if hasattr(transformer, "blocks"):
            return transformer.blocks
        if hasattr(transformer, "layers"):
            return transformer.layers

    if hasattr(vmae, "model") and hasattr(vmae.model, "blocks"):
        return vmae.model.blocks

    raise ValueError("Unable to locate transformer blocks on the VideoMAE backbone.")


def build_model(config: Config):
    """
    Builds the model and parameter groups based on configuration.
    """
    if config.backbone == "csn":
        if config.finetune_mode != "full":
            raise ValueError(f"CSN only supports finetune_mode='full', got '{config.finetune_mode}'")
        
        model = build_csn_backbone(config)
        param_groups = [
            {'params': model.parameters(), 'lr': config.lr_backbone, 'name': 'backbone'}
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
                {'params': head.parameters(), 'lr': config.lr_head, 'name': 'head'}
            ]
            print(f"VideoMAE backbone frozen. Training only head with lr={config.lr_head}")
            
        elif config.finetune_mode == "partial":
            if config.unfreeze_blocks < 0:
                raise ValueError("unfreeze_blocks must be >= 0 for partial fine-tuning.")

            # Freeze all backbone parameters first
            for param in backbone.parameters():
                param.requires_grad = False

            blocks = list(_get_vmae_blocks(backbone))
            num_blocks = len(blocks)
            num_unfreeze = min(config.unfreeze_blocks, num_blocks)

            if num_unfreeze > 0:
                for block in blocks[-num_unfreeze:]:
                    for param in block.parameters():
                        param.requires_grad = True

            backbone_params = [p for p in backbone.parameters() if p.requires_grad]
            param_groups = []
            if backbone_params:
                param_groups.append(
                    {'params': backbone_params, 'lr': config.lr_backbone, 'name': 'backbone'}
                )
            param_groups.append({'params': head.parameters(), 'lr': config.lr_head, 'name': 'head'})
            print(
                "VideoMAE partial fine-tuning: "
                f"unfreeze_blocks={num_unfreeze}/{num_blocks}, "
                f"backbone lr={config.lr_backbone}, head lr={config.lr_head}"
            )

        elif config.finetune_mode == "full":
            # Train both backbone and head (Phase 3)
            for param in backbone.parameters():
                param.requires_grad = True
            param_groups = [
                {'params': backbone.parameters(), 'lr': config.lr_backbone, 'name': 'backbone'},
                {'params': head.parameters(), 'lr': config.lr_head, 'name': 'head'}
            ]
            print(f"VideoMAE full fine-tuning: backbone lr={config.lr_backbone}, head lr={config.lr_head}")
        else:
            raise ValueError(f"Unsupported finetune_mode for vmae: {config.finetune_mode}")
    else:
        raise ValueError(f"Unsupported backbone: {config.backbone}")
    
    return model, param_groups
