"""
Attention-Enhanced ST-GCN for Binary ASD Classification.

Architecture overview:
  Input  : (B, 2, T, 24)  — batch × channels × frames × joints

  ┌─────────────────────────────────────────────────────────┐
  │  Input BN + linear projection to 64 channels            │
  ├─────────────────────────────────────────────────────────┤
  │  6 × ST-GCN Block (spatial graph conv + temporal conv)  │
  │     channels: 64 → 64 → 128 → 128 → 256 → 256          │
  │     temporal stride=2 at blocks 3 & 5 (time compression)│
  ├─────────────────────────────────────────────────────────┤
  │  Temporal Self-Attention Pooling  →  (B, 256, 24)       │
  ├─────────────────────────────────────────────────────────┤
  │  Joint Attention Pooling           →  (B, 256)          │
  ├─────────────────────────────────────────────────────────┤
  │  MLP Head: 256 → 128 → 1 (logit)                        │
  │  Temperature-scaled sigmoid probability                  │
  └─────────────────────────────────────────────────────────┘

  Output: {
    'logit':           (B,)    — raw classifier score
    'prob':            (B,)    — ASD probability after temperature scaling
    'temporal_attn':   (B, T') — normalised temporal attention weights
    'joint_attn':      (B, 24) — normalised joint importance weights
  }

The attention maps are used directly for explainability:
  • joint_attn   → which body joints drove the decision
  • temporal_attn → which time-windows were most diagnostic
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    NUM_JOINTS, IN_CHANNELS, MAX_FRAMES,
    HIDDEN_CHANNELS, DROPOUT, TEMPORAL_KERNEL_SIZE,
    SMPL24_EDGES, JOINT_NAMES,
)


# ─── Adjacency matrix ─────────────────────────────────────────────────────────

def _build_smpl24_adjacency() -> np.ndarray:
    """Return symmetric, normalised adjacency for the SMPL-24 body graph."""
    A = np.zeros((NUM_JOINTS, NUM_JOINTS), dtype=np.float32)
    for i, j in SMPL24_EDGES:
        A[i, j] = 1.0
        A[j, i] = 1.0
    A += np.eye(NUM_JOINTS, dtype=np.float32)   # self-loops
    # Symmetric Laplacian normalisation: D^{-1/2} A D^{-1/2}
    D = A.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D + 1e-8))
    return (D_inv_sqrt @ A @ D_inv_sqrt).astype(np.float32)


# ─── Building blocks ──────────────────────────────────────────────────────────

class SpatialGraphConv(nn.Module):
    """
    Spatial graph convolution with a learnable residual weight.
    x: (B, C_in, T, V) → (B, C_out, T, V)
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn   = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # A: (V, V)  — fixed normalised adjacency
        # Spatial aggregate: x @ A  →  (B, C, T, V)
        x = torch.einsum("bctv,vw->bctw", x, A)
        x = self.bn(self.conv(x))
        return x


class STGCNBlock(nn.Module):
    """
    One Spatial-Temporal Graph Convolutional block.
    """
    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        A:            torch.Tensor,
        kernel_size:  int = TEMPORAL_KERNEL_SIZE,
        stride:       int = 1,
        dropout:      float = DROPOUT,
    ):
        super().__init__()
        self.register_buffer("A", A)

        # Spatial GCN
        self.sgcn   = SpatialGraphConv(in_channels, out_channels)
        self.relu_s = nn.ReLU(inplace=True)

        # Temporal conv  (conv along T dimension)
        pad = (kernel_size - 1) // 2
        self.tcn = nn.Sequential(
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=(kernel_size, 1),
                      stride=(stride, 1),
                      padding=(pad, 0),
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
        )

        # Residual
        if in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=(stride, 1), bias=False),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        x   = self.relu_s(self.sgcn(x, self.A))
        x   = self.tcn(x)
        return self.relu(x + res)


class TemporalAttention(nn.Module):
    """
    Self-attention pool over the temporal dimension.
    Input:  (B, C, T, V)
    Output: (B, C, V), attn_weights (B, T)
    """
    def __init__(self, channels: int):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor):
        # x: (B, C, T, V)
        scores = self.scorer(x)          # (B, 1, T, V)
        scores = scores.mean(dim=3)      # (B, 1, T)
        weights = F.softmax(scores, dim=2)  # (B, 1, T)
        # Weighted sum over T
        out = (x * weights.unsqueeze(3)).sum(dim=2)  # (B, C, V)
        return out, weights.squeeze(1).squeeze(1) if weights.dim() == 3 else weights.squeeze()


class JointAttention(nn.Module):
    """
    Attention pool over the joint dimension.
    Input:  (B, C, V)
    Output: (B, C), attn_weights (B, V)
    """
    def __init__(self, channels: int, num_joints: int = NUM_JOINTS):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(channels, channels // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, 1, bias=True),
        )

    def forward(self, x: torch.Tensor):
        # x: (B, C, V)
        B, C, V = x.shape
        scores  = self.scorer(x.permute(0, 2, 1))   # (B, V, 1)
        weights = F.softmax(scores, dim=1)            # (B, V, 1)
        out = (x * weights.permute(0, 2, 1)).sum(dim=2)  # (B, C)
        return out, weights.squeeze(2)                # (B, C), (B, V)


# ─── Main Model ───────────────────────────────────────────────────────────────

class ASDClassifier(nn.Module):
    """
    Attention-Enhanced ST-GCN for binary ASD vs TD classification.
    """

    def __init__(
        self,
        in_channels:      int   = IN_CHANNELS,
        num_joints:       int   = NUM_JOINTS,
        hidden_channels:  list  = None,
        dropout:          float = DROPOUT,
        temperature:      float = 1.0,
    ):
        super().__init__()

        if hidden_channels is None:
            hidden_channels = list(HIDDEN_CHANNELS)

        # Normalised adjacency
        A_np = _build_smpl24_adjacency()
        A    = torch.tensor(A_np, dtype=torch.float32)
        self.register_buffer("A", A)

        # Input normalisation
        self.input_bn = nn.BatchNorm1d(in_channels * num_joints)
        # Project input channels to first hidden channels
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels[0], kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels[0]),
            nn.ReLU(inplace=True),
        )

        # ST-GCN blocks
        strides = [1, 1, 2, 1, 2, 1]     # temporal down-sample at blocks 3, 5
        channels = hidden_channels
        blocks  = []
        for i in range(len(channels)):
            in_c  = channels[i - 1] if i > 0 else channels[0]
            out_c = channels[i]
            blocks.append(STGCNBlock(in_c, out_c, A,
                                     stride=strides[i], dropout=dropout))
        self.stgcn_blocks = nn.ModuleList(blocks)

        # Attention pooling
        self.temporal_attn = TemporalAttention(channels[-1])
        self.joint_attn    = JointAttention(channels[-1], num_joints)

        # MLP classification head
        self.classifier = nn.Sequential(
            nn.Linear(channels[-1], 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

        # Learnable temperature for probability calibration
        self.log_temperature = nn.Parameter(torch.tensor(float(temperature)).log())

        self._init_weights()

    # ──────────────────────────────────────────────────────────────────────────
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ──────────────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: (B, C, T, V)
        Returns:
            dict with keys:
              'logit'         (B,)
              'prob'          (B,)
              'temporal_attn' (B, T')   — attention over compressed time axis
              'joint_attn'    (B, 24)
        """
        B, C, T, V = x.shape

        # Input normalisation
        x_flat = x.permute(0, 2, 1, 3).reshape(B * T, C * V)
        x_flat = self.input_bn(x_flat)
        x = x_flat.reshape(B, T, C, V).permute(0, 2, 1, 3)   # (B, C, T, V)

        # Project to hidden dimension
        x = self.input_proj(x)

        # ST-GCN backbone
        for block in self.stgcn_blocks:
            x = block(x)                     # (B, C', T', V)

        # Temporal attention pooling  →  (B, C', V)
        x, t_attn = self.temporal_attn(x)    # (B, C', V), (B, T')

        # Joint attention pooling  →  (B, C')
        x, j_attn = self.joint_attn(x)       # (B, C'), (B, V)

        # Classification
        logit = self.classifier(x).squeeze(1)           # (B,)
        temp  = self.log_temperature.exp().clamp(0.1, 10.0)
        prob  = torch.sigmoid(logit / temp)              # (B,)

        return {
            "logit":         logit,
            "prob":          prob,
            "temporal_attn": t_attn,
            "joint_attn":    j_attn,
        }

    # ──────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> dict:
        """
        Single-sample or batch inference with human-readable output.

        Args:
            x: (B, 2, T, 24) or (2, T, 24)
        Returns:
            dict with all forward keys plus:
              'label'        (B,) int  — 0=TD, 1=ASD
              'confidence'   (B,) float — confidence in the predicted class
              'top_joints'   list[str]  — top 5 most attended joints per sample
        """
        self.eval()
        if x.dim() == 3:
            x = x.unsqueeze(0)

        out = self.forward(x)
        prob   = out["prob"]           # (B,)
        labels = (prob >= 0.5).long()  # (B,)
        # Confidence = probability of the predicted class
        conf   = torch.where(labels.bool(), prob, 1.0 - prob)

        # Top-5 joints per sample
        j_attn = out["joint_attn"]     # (B, 24)
        top5_ids = j_attn.topk(5, dim=1).indices.cpu().tolist()
        top5_names = [[JOINT_NAMES[i] for i in ids] for ids in top5_ids]

        return {
            **out,
            "label":      labels.cpu(),
            "confidence": conf.cpu(),
            "top_joints": top5_names,
        }


# ─── Utility ──────────────────────────────────────────────────────────────────

def model_summary(model: nn.Module) -> str:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return (f"ASDClassifier  —  "
            f"{total:,} params total  ({trainable:,} trainable)")


if __name__ == "__main__":
    model = ASDClassifier()
    print(model_summary(model))

    dummy = torch.randn(4, IN_CHANNELS, MAX_FRAMES, NUM_JOINTS)
    out   = model(dummy)
    print(f"Input shape  : {dummy.shape}")
    print(f"Logit shape  : {out['logit'].shape}")
    print(f"Prob shape   : {out['prob'].shape}")
    print(f"Temporal attn: {out['temporal_attn'].shape}")
    print(f"Joint attn   : {out['joint_attn'].shape}")
    print(f"Probs sample : {out['prob'].detach()}")

    # Prediction API
    pred = model.predict(dummy)
    print(f"\nPredicted labels : {pred['label'].tolist()}")
    print(f"Confidences      : {pred['confidence'].tolist()}")
    print(f"Top joints [0]   : {pred['top_joints'][0]}")
