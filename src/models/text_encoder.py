"""
Transkript → E_text (metin embedding).
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
    class TextEncoder(nn.Module):
        """Basit embedding + LSTM veya sadece mean pool. Girdi: (B, seq_len) token ids."""

        def __init__(self, vocab_size: int = 1000, embedding_dim: int = 256, hidden_size: int = 128):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(hidden_size * 2, embedding_dim)
            self._embedding_dim = embedding_dim

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (B, L)
            e = self.embed(x)
            out, _ = self.lstm(e)
            out = out.mean(dim=1)
            return self.fc(out)

else:
    TextEncoder = None  # type: ignore
