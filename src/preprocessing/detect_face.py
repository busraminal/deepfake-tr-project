"""
Frame üzerinde yüz tespiti. OpenCV Haar veya DNN kullanılabilir.
"""
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def _get_haar_detector():
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(path)


def detect_face_haar(image: np.ndarray):
    """(x, y, w, h) bbox listesi döner."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    detector = _get_haar_detector()
    boxes = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(64, 64))
    return [tuple(int(v) for v in b) for b in boxes]


def detect_face_one(image: np.ndarray, padding: float = 0.1):
    """
    En büyük yüz bbox'ını döner. Yoksa None.
    padding: bbox genişletme oranı (0.1 = %10).
    """
    boxes = detect_face_haar(image)
    if not boxes:
        return None
    # En büyük alan
    best = max(boxes, key=lambda b: b[2] * b[3])
    x, y, w, h = best
    H, W = image.shape[:2]
    pad_w = int(w * padding)
    pad_h = int(h * padding)
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(W, x + w + pad_w)
    y2 = min(H, y + h + pad_h)
    return (x1, y1, x2 - x1, y2 - y1)


def crop_face(image: np.ndarray, size: Tuple[int, int] = (224, 224)):
    """Tek yüz crop (size). Yüz yoksa None."""
    box = detect_face_one(image)
    if box is None:
        return None
    x, y, w, h = box
    crop = image[y : y + h, x : x + w]
    return cv2.resize(crop, size)


def crop_faces_from_frame_paths(
    frame_dir,
    out_dir,
    size: Tuple[int, int] = (224, 224),
    prefix: str = "face",
):
    """Bir videonun frame klasöründen yüz crop'ları kaydeder."""
    frame_dir = Path(frame_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, fp in enumerate(sorted(frame_dir.glob("*.jpg"))):
        img = cv2.imread(str(fp))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face = crop_face(img_rgb, size)
        if face is not None:
            out_path = out_dir / f"{prefix}_{i:06d}.jpg"
            cv2.imwrite(str(out_path), cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
            paths.append(out_path)
    return paths
