from .backbones import build_csn_backbone


def build_model(config):
    """
    Build model and optimizer parameter groups based on the configuration.
    Phase 1 supports CSN backbone with full finetuning only.
    """
    if config.backbone != "csn":
        raise ValueError(f"Unsupported backbone for Phase 1: {config.backbone}")
    if config.finetune_mode != "full":
        raise ValueError(f"Unsupported finetune_mode for Phase 1: {config.finetune_mode}")

    model = build_csn_backbone(config)
    param_groups = [{"params": model.parameters(), "lr": config.lr_backbone}]
    return model, param_groups
