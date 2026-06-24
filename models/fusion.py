"""
Fusion and classifier head utilities.
"""
import torch
import torch.nn as nn


class AttentionPoolingHead(nn.Module):
    """Apply temporal attention pooling and produce a pooled context."""

    def __init__(self, d_model: int = 256, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden, 1)

    def forward(self, attended: torch.Tensor, frame_wts: torch.Tensor):
        # attended: (B, T, d_model), frame_wts: (B, T)
        ctx = (attended * frame_wts.unsqueeze(-1)).sum(dim=1)
        x = self.mlp(ctx)
        logit = self.classifier(x)
        return logit, ctx


def get_classification_head(d_model: int = 256, hidden: int = 128, dropout: float = 0.3):
    return AttentionPoolingHead(d_model=d_model, hidden=hidden, dropout=dropout)
