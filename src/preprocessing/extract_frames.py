"""
Videodan frame çıkarır. Belirtilen FPS'e göre örnekler.
"""
from pathlib import Path

import cv2


def extract_frames(
    video_path,
    out_dir,
    fps=25,
    prefix="frame",
):
    """
    Videoyu açıp `fps` hızında frame kaydeder.
    Dönen liste kaydedilen dosya yollarıdır.
    """
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    video_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    interval = max(1, round(video_fps / fps))
    paths = []
    idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            name = f"{prefix}_{saved:06d}.jpg"
            p = out_dir / name
            cv2.imwrite(str(p), frame)
            paths.append(p)
            saved += 1
        idx += 1

    cap.release()
    return paths


def load_frames(frame_paths, size=None):
    """Frame listesini (H, W, 3) dizisi olarak yükler. size verilirse resize."""
    import numpy as np
    frames = []
    for p in frame_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        if size:
            img = cv2.resize(img, size)
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return np.array(frames) if frames else np.zeros((0, 224, 224, 3), dtype=np.uint8)
