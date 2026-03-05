"""
video_model.py
==============
CNN + LSTM video classifier for ASD detection from raw MP4 frames.
Uses MobileNetV3-Small as a per-frame feature extractor (pretrained on
ImageNet) and a 2-layer Bidirectional LSTM for temporal modelling.

Input shape : (B, T, 3, H, W)   e.g. (16, 30, 3, 112, 112)
Output      : dict with logit, prob, label, confidence, frame_attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


# ─────────────────────────────────────────────────────────────────────────────
class FrameEncoder(nn.Module):
    """MobileNetV3-Small backbone that returns per-frame feature vectors."""

    def __init__(self, out_dim: int = 256, pretrained: bool = True, freeze_bn: bool = False):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        base = mobilenet_v3_small(weights=weights)

        # Keep everything up to (but not including) the classifier head
        self.features = base.features          # → (B, 576, 4, 4) for 112×112 input
        self.avgpool  = base.avgpool           # → (B, 576, 1, 1)
        in_dim        = base.classifier[0].in_features  # 576

        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        if freeze_bn:
            for m in self.features.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            (B, out_dim)
        """
        feat = self.features(x)               # (B, 576, h, w)
        feat = self.avgpool(feat)             # (B, 576, 1, 1)
        feat = feat.flatten(1)               # (B, 576)
        return self.proj(feat)               # (B, out_dim)


# ─────────────────────────────────────────────────────────────────────────────
class TemporalSelfAttention(nn.Module):
    """Lightweight multi-head self-attention over frame sequence."""

    def __init__(self, d_model: int = 256, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm  = nn.LayerNorm(d_model)
        self.proj  = nn.Linear(d_model, 1)   # for frame importance weights

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, T, d_model)
        Returns:
            attended  : (B, T, d_model)
            frame_wts : (B, T)  importance per frame
        """
        out, _ = self.attn(x, x, x)         # (B, T, d_model)
        out    = self.norm(out + x)          # residual
        # Compute per-frame importance score
        frame_wts = torch.softmax(self.proj(out).squeeze(-1), dim=-1)  # (B, T)
        return out, frame_wts


# ─────────────────────────────────────────────────────────────────────────────
class VideoASDClassifier(nn.Module):
    """
    End-to-end video ASD classifier.

    Architecture:
        FrameEncoder → BiLSTM → TemporalSelfAttention
        → weighted pool → MLP → sigmoid probability
    """

    def __init__(
        self,
        frame_dim  : int   = 256,
        lstm_hidden: int   = 256,
        lstm_layers: int   = 2,
        n_classes  : int   = 1,
        dropout    : float = 0.4,
        pretrained : bool  = True,
    ):
        super().__init__()

        self.encoder  = FrameEncoder(out_dim=frame_dim, pretrained=pretrained)

        self.pos_emb  = nn.Parameter(torch.zeros(1, 200, frame_dim))  # learnable positional
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        self.lstm = nn.LSTM(
            input_size  = frame_dim,
            hidden_size = lstm_hidden,
            num_layers  = lstm_layers,
            batch_first = True,
            bidirectional = True,
            dropout     = dropout if lstm_layers > 1 else 0.0,
        )

        lstm_out_dim = lstm_hidden * 2   # bidirectional

        self.attn = TemporalSelfAttention(lstm_out_dim, n_heads=4, dropout=0.1)

        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    # ──────────────────────────────────────────────────────────
    def encode_frames(self, videos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            videos: (B, T, 3, H, W)
        Returns:
            frame_feats: (B, T, frame_dim)
        """
        B, T, C, H, W = videos.shape
        frames       = videos.view(B * T, C, H, W)   # (B*T, 3, H, W)
        feats        = self.encoder(frames)           # (B*T, frame_dim)
        return feats.view(B, T, -1)                  # (B, T, frame_dim)

    # ──────────────────────────────────────────────────────────
    def forward(self, videos: torch.Tensor):
        """
        Args:
            videos: (B, T, 3, H, W)
        Returns:
            logit        : (B, 1)
            frame_weights: (B, T)
        """
        B, T = videos.shape[:2]

        # 1. Per-frame CNN features
        feats = self.encode_frames(videos)            # (B, T, frame_dim)

        # 2. Add positional encoding
        feats = feats + self.pos_emb[:, :T, :]

        # 3. Temporal LSTM
        lstm_out, _ = self.lstm(feats)                # (B, T, lstm_hidden*2)

        # 4. Self-attention + frame importance
        attended, frame_wts = self.attn(lstm_out)     # (B,T,*), (B,T)

        # 5. Weighted pool over time
        ctx = (attended * frame_wts.unsqueeze(-1)).sum(dim=1)  # (B, hidden*2)

        # 6. Classification
        logit = self.classifier(ctx) / self.temperature.clamp(0.5, 5.0)  # (B, 1)
        return logit, frame_wts

    # ──────────────────────────────────────────────────────────
    @torch.no_grad()
    def predict(self, videos: torch.Tensor) -> dict:
        """
        Full inference with human-readable output.

        Args:
            videos: (1, T, 3, H, W)  single-video batch
        Returns:
            dict with prob, label, confidence, frame_weights
        """
        self.eval()
        logit, frame_wts = self(videos)
        prob  = torch.sigmoid(logit).item()
        label = int(prob >= 0.5)
        conf  = prob if label == 1 else 1.0 - prob

        # Top-3 most discriminative frame indices
        fw = frame_wts.squeeze(0).cpu().tolist()
        top_frames = sorted(range(len(fw)), key=lambda i: fw[i], reverse=True)[:3]

        return {
            "logit"        : logit.item(),
            "prob"         : prob,
            "label"        : label,
            "label_name"   : "ASD" if label == 1 else "TD",
            "confidence"   : conf,
            "frame_weights": fw,
            "top_frames"   : top_frames,
        }


# ─────────────────────────────────────────────────────────────────────────────
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = VideoASDClassifier(pretrained=False)
    print(f"Parameters: {count_parameters(model):,}")

    # Quick forward-pass test
    dummy = torch.randn(4, 30, 3, 112, 112)
    with torch.no_grad():
        logit, fw = model(dummy)
    print(f"logit shape : {logit.shape}")    # (4, 1)
    print(f"frame_wts   : {fw.shape}")       # (4, 30)
    print("Model OK")
