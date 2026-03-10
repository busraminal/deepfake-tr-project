"""
Füzyon: S_f = alpha * S_v + (1 - alpha) * S_l.
Görsel ve senkron modellerinden skor alıp birleştirir.
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
    class FusionModel(nn.Module):
        """
        S_v ve S_l skorlarını alıp S_f = alpha * S_v + (1-alpha) * S_l üretir.
        alpha öğrenilebilir veya sabit.
        """

        def __init__(self, learn_alpha: bool = False, init_alpha: float = 0.5):
            super().__init__()
            if learn_alpha:
                self.alpha = nn.Parameter(torch.tensor(init_alpha))
            else:
                self.register_buffer("alpha", torch.tensor(init_alpha))
            self.learn_alpha = learn_alpha

        def forward(
            self,
            s_v: "torch.Tensor",
            s_l: "torch.Tensor",
        ) -> "torch.Tensor":
            alpha = torch.sigmoid(self.alpha) if self.learn_alpha else self.alpha
            return alpha * s_v + (1 - alpha) * s_l

    def compute_sync_score_from_embeddings(
        e_text: "torch.Tensor",
        m_video: "torch.Tensor",
    ) -> "torch.Tensor":
        """S_l = 1 - cos(E_text, M_video). Uyumsuzluk skoru (yüksek = uyumsuz)."""
        e = e_text / (e_text.norm(dim=1, keepdim=True) + 1e-8)
        m = m_video / (m_video.norm(dim=1, keepdim=True) + 1e-8)
        cos = (e * m).sum(dim=1)
        return 1 - cos

else:
    FusionModel = None  # type: ignore
    compute_sync_score_from_embeddings = None  # type: ignore
