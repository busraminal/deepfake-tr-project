"""
Ağız ROI dizisi → M_video (dudak hareketi embedding).
S_l = 1 - cos(E_text, M_video) için kullanılır.
"""
from typing import Optional

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = None


if TORCH_AVAILABLE:
    class MouthEncoder(nn.Module):
        """Temporal CNN: (B, T, C, H, W) → (B, embedding_dim)."""

        def __init__(self, in_channels: int = 3, embedding_dim: int = 256):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv3d(in_channels, 32, (3, 3, 3), padding=1),
                nn.ReLU(inplace=True),
                # Zaman boyutu (T) AVLips için çok kısa olabildiği için
                # sadece uzamsal (H, W) havuzlama yapıyoruz.
                nn.MaxPool3d((1, 2, 2)),
                nn.Conv3d(32, 64, (3, 3, 3), padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool3d((1, 2, 2)),
                nn.Conv3d(64, 128, (3, 3, 3), padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool3d(1),
            )
            self.fc = nn.Linear(128, embedding_dim)
            self._embedding_dim = embedding_dim

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (B, T, C, H, W) -> (B, C, T, H, W)
            x = x.permute(0, 2, 1, 3, 4)
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

else:
    MouthEncoder = None  # type: ignore
