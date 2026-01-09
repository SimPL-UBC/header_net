"""
Classification heads for header detection models.
"""
import torch
import torch.nn as nn


class VideoMAEHead(nn.Module):
    """Simple classification head for VideoMAE backbone."""
    
    def __init__(self, hidden_dim, num_classes=2, dropout=0.5):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = self.dropout(x)
        return self.fc(x)
