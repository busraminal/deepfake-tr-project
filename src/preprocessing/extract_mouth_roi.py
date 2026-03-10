"""
Yüz crop'ından ağız ROI'si çıkarır. Yüz bbox'ının alt kısmı (dudak bölgesi).
"""
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def mouth_roi_from_face(face_image: np.ndarray, size: Tuple[int, int] = (96, 96)) -> np.ndarray:
    """
    Yüz görüntüsünün alt ~%40'ını ağız bölgesi kabul edip kırpar.
    Landmark yoksa bu heuristik kullanılır.
    """
    h, w = face_image.shape[:2]
    # Alt 35–40% genelde ağız
    y_start = int(h * 0.50)
    y_end = h
    roi = face_image[y_start:y_end, :]
    if roi.size == 0:
        roi = face_image
    return cv2.resize(roi, size)
def extract_mouth_rois_from_face_dir(
    face_dir,
    out_dir,
    size: Tuple[int, int] = (96, 96),
    prefix: str = "mouth",
):
    """Face crop klasöründeki her görüntüden mouth ROI kaydeder."""
    face_dir = Path(face_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, fp in enumerate(sorted(face_dir.glob("*.jpg"))):
        img = cv2.imread(str(fp))
        if img is None:
            continue
        mouth = mouth_roi_from_face(img, size)
        out_path = out_dir / f"{prefix}_{i:06d}.jpg"
        cv2.imwrite(str(out_path), mouth)
        paths.append(out_path)
    return paths
