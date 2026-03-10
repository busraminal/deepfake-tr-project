"""
Senkron model için veri seti: ağız ROI dizisi + transkript → S_l (konuşma–dudak uyumsuzluk).
Girdi: mouth ROI sequence, transcript (veya text embedding); çıktı: label_sync (1=uyumlu, 0=uyumsuz).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object


def _load_mouth_seq(mouths_dir: Path, size: tuple[int, int] = (96, 96), max_frames: int = 64):
    import cv2
    paths = sorted(mouths_dir.glob("*.jpg"))[:max_frames]
    frames = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, size)
            frames.append(img)
    if not frames:
        return np.zeros((1, size[0], size[1], 3), dtype=np.uint8)
    return np.stack(frames, axis=0)


class SyncDataset:
    """Mouth ROI dizisi + transcript metni; hedef: label_sync (1/0)."""

    def __init__(
        self,
        samples: list[dict],
        base_dir: Path,
        mouth_size: tuple[int, int] = (96, 96),
        max_frames: int = 64,
    ):
        self.samples = samples
        self.base_dir = Path(base_dir)
        self.mouth_size = mouth_size
        self.max_frames = max_frames

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> tuple[np.ndarray, str, float]:
        s = self.samples[i]
        mouths_dir = s.get("mouths_dir")
        if not mouths_dir:
            mouths_dir = self.base_dir / "data" / "processed" / "mouths" / s.get("sample_id", "")
        else:
            mouths_dir = self.base_dir / mouths_dir
        mouth_seq = _load_mouth_seq(mouths_dir, size=self.mouth_size, max_frames=self.max_frames)
        transcript = s.get("transcript_tr", "") or ""
        label = float(s.get("label_sync", 0))
        return mouth_seq.astype(np.float32) / 255.0, transcript, label


if TORCH_AVAILABLE:
    class SyncTorchDataset(Dataset):
        def __init__(self, sync_dataset: SyncDataset, tokenizer_max_length: int = 128):
            self.ds = sync_dataset
            self.tokenizer_max_length = tokenizer_max_length

        def __len__(self) -> int:
            return len(self.ds)

        def __getitem__(self, i: int):
            mouth, text, label = self.ds[i]
            # Metin için basit padding: karakter/kelime indeksleri (stub)
            text_encoded = np.zeros(self.tokenizer_max_length, dtype=np.int64)
            for j, c in enumerate(text[: self.tokenizer_max_length]):
                text_encoded[j] = min(ord(c) % 1000, 999)
            return (
                torch.from_numpy(mouth).permute(0, 3, 1, 2),
                torch.from_numpy(text_encoded),
                torch.tensor(label, dtype=torch.float32),
            )
