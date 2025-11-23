from ..config import Config
from .backbones import build_csn_backbone

def build_model(config: Config):
    """
    Builds the model and parameter groups based on configuration.
    """
    if config.backbone == "csn":
        model = build_csn_backbone(config)
    else:
        raise ValueError(f"Unsupported backbone: {config.backbone}")
        
    if config.finetune_mode != "full":
        raise ValueError(f"Unsupported finetune_mode: {config.finetune_mode}")

    # Return model and param_groups
    # Phase 1: Single param group for optimizer, LR=config.lr_backbone
    param_groups = [
        {'params': model.parameters(), 'lr': config.lr_backbone}
    ]
    
    return model, param_groups
