import torch.nn as nn

from .backbones import build_vmae_backbone
from .heads import VideoMAEHead


class VideoMAEClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        feats = self.backbone(x)  # (B, hidden_dim)
        return self.head(feats)  # (B, num_classes)

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep backbone in eval mode even when the wrapper is set to train.
        self.backbone.eval()
        return self


def build_model(config):
    """Build a frozen VideoMAE backbone with a small classification head."""
    backbone = build_vmae_backbone(config)
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval()

    head = VideoMAEHead(backbone.hidden_dim, num_classes=config.num_classes)
    model = VideoMAEClassifier(backbone, head)
    param_groups = [{"params": head.parameters(), "lr": config.lr_head}]
    return model, param_groups
