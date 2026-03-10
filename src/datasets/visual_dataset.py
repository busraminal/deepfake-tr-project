"""
Görsel model için veri seti: yüz crop dizisi → S_v (görsel sahtecilik).
Metadata'daki faces_dir veya frames_dir kullanılır.
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
    Dataset = object  # type: ignore


def _load_face_seq(faces_dir: Path, size: tuple[int, int] = (224, 224), max_frames: int = 64):
    import cv2
    paths = sorted(faces_dir.glob("*.jpg"))[:max_frames]
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


class VisualDataset:
    """Yüz crop dizisi ve label_visual_fake (veya binary real/fake)."""

    def __init__(
        self,
        samples: list[dict],
        base_dir: Path,
        face_size: tuple[int, int] = (224, 224),
        max_frames: int = 64,
        use_label_visual: bool = True,
    ):
        self.samples = samples
        self.base_dir = Path(base_dir)
        self.face_size = face_size
        self.max_frames = max_frames
        self.use_label_visual = use_label_visual

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> tuple[np.ndarray, float]:
        s = self.samples[i]
        # Önce faces_dir, yoksa metadata'da frames_dir + detect ile face crop beklenebilir
        faces_dir = s.get("faces_dir") or (Path(s.get("metadata_dir", "")) / "faces" / s.get("sample_id", ""))
        if not faces_dir:
            # Fallback: processed/faces / sample_id
            faces_dir = self.base_dir / "data" / "processed" / "faces" / s.get("sample_id", "")
        else:
            faces_dir = self.base_dir / faces_dir
        seq = _load_face_seq(faces_dir, size=self.face_size, max_frames=self.max_frames)
        if self.use_label_visual:
            label = float(s.get("label_visual_fake", 0))
        else:
            # Binary: real_sync = 0, diğerleri 1
            label = 0.0 if s.get("label_main") == "real_sync" else 1.0
        return seq.astype(np.float32) / 255.0, label


if TORCH_AVAILABLE:
    class VisualTorchDataset(Dataset):
        def __init__(self, visual_dataset: VisualDataset):
            self.ds = visual_dataset

        def __len__(self) -> int:
            return len(self.ds)

        def __getitem__(self, i: int):
            x, y = self.ds[i]
            return torch.from_numpy(x).permute(0, 3, 1, 2), torch.tensor(y, dtype=torch.float32)
