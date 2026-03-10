"""
Füzyon modeli için veri seti: görsel + senkron özellikleri veya ham görsel + ağız + metin.
S_v ve S_l skorlarını birleştiren modeli beslemek için kullanılır.
"""
from __future__ import annotations

from pathlib import Path

from src.datasets.visual_dataset import VisualDataset
from src.datasets.sync_dataset import SyncDataset


class FusionDataset:
    """
    Her örnek için hem görsel hem senkron girdileri sağlar.
    Hedef: binary (real/fake) veya label_main.
    """

    def __init__(
        self,
        samples: list[dict],
        base_dir: Path,
        face_size: tuple[int, int] = (224, 224),
        mouth_size: tuple[int, int] = (96, 96),
        max_frames: int = 64,
    ):
        self.visual_ds = VisualDataset(
            samples, base_dir, face_size=face_size, max_frames=max_frames, use_label_visual=False
        )
        self.sync_ds = SyncDataset(samples, base_dir, mouth_size=mouth_size, max_frames=max_frames)
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        face_seq, _ = self.visual_ds[i]
        mouth_seq, transcript, _ = self.sync_ds[i]
        s = self.samples[i]
        # Final hedef: real_sync = 0, diğerleri 1
        label = 0.0 if s.get("label_main") == "real_sync" else 1.0
        return face_seq, mouth_seq, transcript, label
