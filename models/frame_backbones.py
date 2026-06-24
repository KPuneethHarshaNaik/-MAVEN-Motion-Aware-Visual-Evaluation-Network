"""
Frame backbone utilities — EfficientNetV2-S wrapper returning per-frame vectors.
"""
import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights


class MobileNetV3Encoder(nn.Module):
    """MobileNetV3-Small backbone returning per-frame feature vectors."""
    def __init__(self, out_dim: int = 256, pretrained: bool = True):
        super().__init__()
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        base = mobilenet_v3_small(weights=weights)
        self.features = base.features
        self.avgpool = base.avgpool
        in_dim = 576  # MobileNetV3-Small output
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )
    def forward(self, x):
        feat = self.features(x)
        feat = self.avgpool(feat)
        feat = feat.flatten(1)
        return self.proj(feat)


class FrameEncoder(nn.Module):
    """EfficientNetV2-S backbone returning per-frame feature vectors."""

    def __init__(self, out_dim: int = 256, pretrained: bool = True, freeze_bn: bool = False):
        super().__init__()
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        base = efficientnet_v2_s(weights=weights)

        # Keep feature extractor and avgpool
        self.features = base.features
        self.avgpool  = base.avgpool
        # EfficientNetV2-S output feature dimension before classification head is 1280
        in_dim = 1280

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
        # x: (B, 3, H, W) -> (B, out_dim)
        feat = self.features(x)
        feat = self.avgpool(feat)
        feat = feat.flatten(1)
        return self.proj(feat)


def get_frame_encoder(name: str = "efficientnet", **kwargs) -> nn.Module:
    if name.lower().startswith("mobilenet"):
        return MobileNetV3Encoder(**kwargs)
    # Default to EfficientNet
    return FrameEncoder(**kwargs)
