import torch.nn as nn


class VideoMAEHead(nn.Module):
    """Lightweight classification head on top of frozen VideoMAE features."""

    def __init__(self, hidden_dim: int, num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        layers = []
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier(x)
