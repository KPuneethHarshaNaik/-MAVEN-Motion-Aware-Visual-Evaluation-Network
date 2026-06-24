"""
Temporal transformer and lightweight attention utilities.
"""
import torch
import torch.nn as nn


class TemporalTransformer(nn.Module):
    """Factorized temporal Transformer encoder for frame sequences."""

    def __init__(self, d_model: int = 256, n_heads: int = 8, depth: int = 4, ff_mult: int = 4, dropout: float = 0.1, max_len: int = 300):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def forward(self, x: torch.Tensor):
        # x: (B, T, d_model)
        B, T, _ = x.shape
        pe = self.pos_emb[:, :T, :]
        x = x + pe
        return self.encoder(x)


class TemporalSelfAttention(nn.Module):
    """Lightweight attention head producing frame importance weights."""

    def __init__(self, d_model: int = 256, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor):
        # x: (B, T, d_model)
        out, _ = self.attn(x, x, x)
        out = self.norm(out + x)
        frame_wts = torch.softmax(self.proj(out).squeeze(-1), dim=-1)
        return out, frame_wts


def get_temporal_transformer(**kwargs):
    return TemporalTransformer(**kwargs)
