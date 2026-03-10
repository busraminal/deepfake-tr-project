"""
Görsel model: yüz/frame dizisi → S_v (görsel sahtecilik skoru, 0–1).
Backbone (örn. ResNet) + temporal pooling + MLP.
"""
from typing import Optional

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = None  # type: ignore


if TORCH_AVAILABLE:
    class VisualModel(nn.Module):
        """Frame dizisi alır; her frame'i backbone'dan geçirip temporal ortalamayla S_v üretir."""

        def __init__(
            self,
            backbone: str = "resnet18",
            num_classes: int = 1,
            pretrained: bool = True,
            in_channels: int = 3,
        ):
            super().__init__()
            if backbone == "resnet18":
                from torchvision.models import resnet18
                try:
                    from torchvision.models import ResNet18_Weights
                    w = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
                    res = resnet18(weights=w)
                except (ImportError, TypeError):
                    res = resnet18(pretrained=pretrained)
                self.backbone = nn.Sequential(*list(res.children())[:-1])
                feat_dim = 512
            else:
                self.backbone = nn.Sequential(
                    nn.Conv2d(in_channels, 32, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d(1),
                )
                feat_dim = 32
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(feat_dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes),
                nn.Sigmoid() if num_classes == 1 else nn.Identity(),
            )
            self._feat_dim = feat_dim

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (B, T, C, H, W)
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            feats = self.backbone(x)
            feats = feats.view(B, T, -1).mean(dim=1)
            return self.fc(feats).squeeze(-1)

else:
    VisualModel = None  # type: ignore
