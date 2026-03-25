"""
ST-GCN (Spatial-Temporal Graph Convolutional Network) Model.

A lightweight implementation for skeleton-based action recognition
optimized for the MMASD Autism Intervention Dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def build_openpose_adjacency():
    """
    Build the adjacency matrix for the OpenPose BODY_25 skeleton graph.
    Returns a (25, 25) adjacency matrix including self-loops.
    """
    # OpenPose BODY_25 bone connections (parent -> child)
    edges = [
        (0, 1),   # Nose -> Neck
        (1, 2),   # Neck -> RShoulder
        (2, 3),   # RShoulder -> RElbow
        (3, 4),   # RElbow -> RWrist
        (1, 5),   # Neck -> LShoulder
        (5, 6),   # LShoulder -> LElbow
        (6, 7),   # LElbow -> LWrist
        (1, 8),   # Neck -> MidHip
        (8, 9),   # MidHip -> RHip
        (9, 10),  # RHip -> RKnee
        (10, 11), # RKnee -> RAnkle
        (8, 12),  # MidHip -> LHip
        (12, 13), # LHip -> LKnee
        (13, 14), # LKnee -> LAnkle
        (0, 15),  # Nose -> REye
        (15, 17), # REye -> REar
        (0, 16),  # Nose -> LEye
        (16, 18), # LEye -> LEar
        (11, 19), # RAnkle -> RBigToe (approximate)
        (19, 20), # RBigToe -> RSmallToe
        (11, 21), # RAnkle -> RHeel (approximate, using 14->21 sometimes)
        (14, 22), # LAnkle -> LBigToe
        (22, 23), # LBigToe -> LSmallToe
        (14, 24), # LAnkle -> LHeel
    ]
    num_joints = 25
    A = np.zeros((num_joints, num_joints), dtype=np.float32)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    # Add self-loops
    A += np.eye(num_joints, dtype=np.float32)
    return A


def normalize_adjacency(A):
    """
    Symmetric normalization of adjacency matrix: D^{-1/2} A D^{-1/2}
    """
    D = np.sum(A, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D + 1e-8))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return A_norm


class STGCNBlock(nn.Module):
    """
    A single Spatial-Temporal Graph Convolution block.

    Spatial: Graph convolution using adjacency matrix.
    Temporal: 1D convolution along the time axis.
    """

    def __init__(self, in_channels, out_channels, A, kernel_size=9, stride=1, dropout=0.3, residual=True):
        super().__init__()

        self.A = A  # Normalized adjacency (V, V)

        # Spatial Graph Convolution
        self.gcn = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.gcn_bn = nn.BatchNorm2d(out_channels)

        # Temporal Convolution
        padding = (kernel_size - 1) // 2
        self.tcn = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size, 1),
                      stride=(stride, 1), padding=(padding, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
        )

        # Residual connection
        self.use_residual = residual
        if residual:
            if in_channels != out_channels or stride != 1:
                self.residual = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                    nn.BatchNorm2d(out_channels),
                )
            else:
                self.residual = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Args:
            x: (N, C, T, V) where N=batch, C=channels, T=frames, V=joints
        Returns:
            (N, C_out, T, V)
        """
        # Residual
        res = self.residual(x) if self.use_residual else 0

        # Spatial GCN: multiply features by adjacency
        # x: (N, C, T, V) -> matmul with A: (V, V) -> (N, C, T, V)
        A = self.A.to(x.device)
        x = torch.einsum('nctv,vw->nctw', x, A)

        # 1x1 convolution for channel mixing
        x = self.gcn_bn(self.gcn(x))
        x = self.relu(x)

        # Temporal convolution
        x = self.tcn(x)

        # Residual + activation
        x = self.relu(x + res)
        return x


class STGCN(nn.Module):
    """
    Spatial-Temporal Graph Convolutional Network for skeleton-based
    action recognition.

    Designed to be lightweight and efficient for the MMASD dataset.
    """

    def __init__(self, in_channels, num_classes, num_joints=25,
                 hidden_channels=None, dropout=0.3):
        super().__init__()

        if hidden_channels is None:
            hidden_channels = [64, 64, 128, 128, 256, 256]

        # Build and normalize adjacency matrix
        A_raw = build_openpose_adjacency()
        A_norm = normalize_adjacency(A_raw)
        self.register_buffer('A', torch.tensor(A_norm, dtype=torch.float32))

        # Input batch normalization
        self.input_bn = nn.BatchNorm1d(in_channels * num_joints)

        # Build ST-GCN blocks
        layers = []
        channels = [in_channels] + hidden_channels
        for i in range(len(hidden_channels)):
            stride = 2 if i in [2, 4] else 1  # Downsample temporally at layers 3 and 5
            layers.append(
                STGCNBlock(channels[i], channels[i + 1], self.A,
                           kernel_size=9, stride=stride, dropout=dropout)
            )
        self.stgcn_blocks = nn.ModuleList(layers)

        # Global average pooling + classifier
        self.fc = nn.Linear(hidden_channels[-1], num_classes)

    def forward(self, x):
        """
        Args:
            x: (N, C, T, V) — batch, channels, frames, joints
        Returns:
            (N, num_classes) — class logits
        """
        N, C, T, V = x.shape

        # Input normalization
        x_flat = x.permute(0, 2, 1, 3).reshape(N * T, C * V)  # (N*T, C*V)
        x_flat = self.input_bn(x_flat)
        x = x_flat.reshape(N, T, C, V).permute(0, 2, 1, 3)  # (N, C, T, V)

        # ST-GCN blocks
        for block in self.stgcn_blocks:
            x = block(x)

        # Global average pooling over time and joints
        x = x.mean(dim=[2, 3])  # (N, C_out)

        # Classifier
        x = self.fc(x)
        return x


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    # Quick sanity check
    model = STGCN(in_channels=2, num_classes=11, num_joints=25)
    total, trainable = count_parameters(model)
    print(f"Model Parameters: {total:,} total, {trainable:,} trainable")

    # Test forward pass with dummy data
    dummy = torch.randn(4, 2, 150, 25)  # batch=4, channels=2(x,y), frames=150, joints=25
    out = model(dummy)
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {out.shape}")  # Expected: (4, 11)
