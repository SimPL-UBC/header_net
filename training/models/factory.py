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


def _build_vmae_param_groups(backbone, head, base_lr, layer_lr_decay):
    blocks = list(_get_vmae_blocks(backbone))
    num_blocks = len(blocks)
    param_groups = []
    used = set()

    def add_group(params, lr, name):
        filtered = [p for p in params if p.requires_grad and id(p) not in used]
        if not filtered:
            return
        used.update(id(p) for p in filtered)
        param_groups.append({"params": filtered, "lr": lr, "name": name})

    # Head gets full LR.
    add_group(head.parameters(), base_lr, "head")

    # Transformer blocks get decayed LR, last block closest to head.
    for idx, block in enumerate(blocks):
        depth = num_blocks - idx
        lr = base_lr * (layer_lr_decay ** depth)
        add_group(block.parameters(), lr, f"block_{idx}")

    # Remaining backbone params (embeddings/norms) get the smallest LR.
    remaining = [p for p in backbone.parameters() if p.requires_grad and id(p) not in used]
    if remaining:
        lr = base_lr * (layer_lr_decay ** (num_blocks + 1))
        add_group(remaining, lr, "backbone_other")

    return param_groups


def build_model(config: Config):
    """
    Builds the model and parameter groups based on configuration.
    """
    if config.backbone == "csn":
        if config.finetune_mode != "full":
            raise ValueError(f"CSN only supports finetune_mode='full', got '{config.finetune_mode}'")
        
        model = build_csn_backbone(config)
        param_groups = [
            {'params': model.parameters(), 'lr': config.base_lr, 'name': 'backbone'}
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
            param_groups = _build_vmae_param_groups(
                backbone, head, config.base_lr, config.layer_lr_decay
            )
            print(f"VideoMAE backbone frozen. Training only head with lr={config.base_lr}")
            
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

            param_groups = _build_vmae_param_groups(
                backbone, head, config.base_lr, config.layer_lr_decay
            )
            print(
                "VideoMAE partial fine-tuning: "
                f"unfreeze_blocks={num_unfreeze}/{num_blocks}, "
                f"base lr={config.base_lr}, layer decay={config.layer_lr_decay}"
            )

        elif config.finetune_mode == "full":
            # Train both backbone and head (Phase 3)
            for param in backbone.parameters():
                param.requires_grad = True
            param_groups = _build_vmae_param_groups(
                backbone, head, config.base_lr, config.layer_lr_decay
            )
            print(
                f"VideoMAE full fine-tuning: base lr={config.base_lr}, "
                f"layer decay={config.layer_lr_decay}"
            )
        else:
            raise ValueError(f"Unsupported finetune_mode for vmae: {config.finetune_mode}")
    else:
        raise ValueError(f"Unsupported backbone: {config.backbone}")
    
    return model, param_groups
