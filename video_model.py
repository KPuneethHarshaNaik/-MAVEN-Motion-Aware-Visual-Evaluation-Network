"""
video_model.py
==============
Modular CNN + Temporal Transformer video classifier for ASD detection.
Uses EfficientNetV2-S (or MobileNetV3-Small for legacy checkpoints) as a per-frame
feature extractor and a factorized temporal Transformer encoder with attention pooling.

Input shape : (B, T, 3, H, W)   e.g. (16, 30, 3, 112, 112)
Output      : dict with logit, prob, label, confidence, frame_attention
"""

import torch
import torch.nn as nn
from typing import Optional

# Import modular components
from models.frame_backbones import get_frame_encoder
from models.temporal_transformer import TemporalTransformer, TemporalSelfAttention
from models.fusion import get_classification_head


class VideoASDClassifier(nn.Module):
    """
    End-to-end video ASD classifier composed from modular parts.
    """

    def __init__(
        self,
        frame_dim: int = 256,
        n_classes: int = 1,
        dropout: float = 0.4,
        pretrained: bool = True,
        transformer_depth: int = 4,
        n_heads: int = 8,
        ff_mult: int = 4,
        max_len: int = 300,
        backbone: str = "efficientnet",
    ):
        super().__init__()

        # Per-frame CNN encoder
        self.encoder = get_frame_encoder(backbone, out_dim=frame_dim, pretrained=pretrained)

        # Temporal transformer
        self.temporal = TemporalTransformer(d_model=frame_dim, n_heads=n_heads,
                                           depth=transformer_depth, ff_mult=ff_mult,
                                           dropout=dropout, max_len=max_len)

        # Attention head and classification pooling
        self.attn = TemporalSelfAttention(d_model=frame_dim, n_heads=max(4, n_heads//2), dropout=0.1)
        self.head = get_classification_head(d_model=frame_dim, hidden=128, dropout=dropout)

        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def encode_frames(self, videos: torch.Tensor) -> torch.Tensor:
        """Encode per-frame features.

        Args:
            videos: (B, T, 3, H, W)
        Returns:
            (B, T, frame_dim)
        """
        B, T, C, H, W = videos.shape
        frames = videos.view(B * T, C, H, W)
        feats = self.encoder(frames)
        return feats.view(B, T, -1)

    def forward(self, videos: torch.Tensor):
        B, T = videos.shape[:2]
        feats = self.encode_frames(videos)          # (B, T, D)
        temporal_out = self.temporal(feats)         # (B, T, D)
        attended, frame_wts = self.attn(temporal_out)
        logit, ctx = self.head(attended, frame_wts) # (B,1), (B,hidden)
        logit = logit / self.temperature.clamp(0.5, 5.0)
        return logit, frame_wts

    @torch.no_grad()
    def predict(self, videos: torch.Tensor) -> dict:
        logit, frame_wts = self(videos)
        prob = torch.sigmoid(logit).item()
        label = int(prob >= 0.5)
        conf = prob if label == 1 else 1.0 - prob

        # Compute per-frame feature norms for energy visualization
        feats = self.encode_frames(videos)
        feat_norms = feats.squeeze(0).norm(dim=-1).cpu().tolist()

        fw = frame_wts.squeeze(0).cpu().tolist()
        top_frames = sorted(range(len(fw)), key=lambda i: fw[i], reverse=True)[:3]

        return {
            "logit": float(logit.item()),
            "prob": float(prob),
            "label": label,
            "label_name": "ASD" if label == 1 else "TD",
            "confidence": float(conf),
            "frame_weights": fw,
            "top_frames": top_frames,
            "frame_energies": feat_norms,
        }


def model_factory(name: str = "option_a", backbone: str = "efficientnet", **kwargs) -> nn.Module:
    """Factory to create pre-configured models for the new architecture.

    name: identifier of preset (only 'option_a' implemented)
    backbone: 'efficientnet' or 'mobilenet'
    kwargs: forwarded to VideoASDClassifier
    """
    if name == "option_a":
        return VideoASDClassifier(backbone=backbone, **kwargs)
    raise ValueError(f"Unknown model preset: {name}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = model_factory("option_a", pretrained=False)
    print(f"Parameters: {count_parameters(m):,}")

    # Quick forward-pass smoke test
    dummy = torch.randn(2, 16, 3, 96, 96)
    with torch.no_grad():
        logit, fw = m(dummy)
    print(f"logit: {logit.shape}, frame_wts: {fw.shape}")
