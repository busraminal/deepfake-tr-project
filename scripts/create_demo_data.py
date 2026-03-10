"""
Demo için minimal veri üretir: yüz/ağız görüntüleri, sessiz WAV, metadata.
Video indirmeden pipeline'ı (preprocess atlanır, split + train) çalıştırmak için.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.schema import sample_schema, LABEL_REAL_SYNC, MISMATCH_NONE
from src.utils.io import load_config, save_metadata, save_split, project_root


def write_dummy_image(path: Path, w: int, h: int, color_bgr: tuple = (200, 180, 160)) -> None:
    """Tek renk placeholder uretir; gercek yuz degildir. Gorsel model bu veriyle anlamli ogrenemez."""
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((h, w, 3), color_bgr, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def write_dummy_wav(path: Path, duration_sec: float = 1.0, sr: int = 16000) -> None:
    try:
        import soundfile as sf
    except ImportError:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(duration_sec * sr)
    silent = np.zeros(n, dtype=np.float32)
    sf.write(str(path), silent, sr)


def create_demo_data(
    base_dir: Path | None = None,
    num_speakers: int = 3,
    samples_per_speaker: int = 2,
    frames_per_sample: int = 8,
    face_size: int = 224,
    mouth_size: int = 96,
) -> list[dict]:
    base_dir = base_dir or project_root()
    config = load_config(base_dir / "configs" / "data.yaml")
    data = config.get("data", {})
    processed = base_dir / data.get("processed_dir", "data/processed")
    raw_audio = base_dir / data.get("raw_audio_dir", "data/raw_audio")
    metadata_dir = processed / "metadata"
    faces_base = processed / "faces"
    mouths_base = processed / "mouths"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    raw_audio.mkdir(parents=True, exist_ok=True)

    samples = []
    idx = 0
    for s in range(num_speakers):
        speaker_id = f"spk_d{s+1}"
        for _ in range(samples_per_speaker):
            sample_id = f"demo_{idx:03d}"
            idx += 1
            faces_dir = faces_base / sample_id
            mouths_dir = mouths_base / sample_id
            faces_dir.mkdir(parents=True, exist_ok=True)
            mouths_dir.mkdir(parents=True, exist_ok=True)
            for f in range(frames_per_sample):
                write_dummy_image(faces_dir / f"face_{f:06d}.jpg", face_size, face_size)
                write_dummy_image(mouths_dir / f"mouth_{f:06d}.jpg", mouth_size, mouth_size, (180, 120, 100))
            audio_path = raw_audio / f"{sample_id}.wav"
            write_dummy_wav(audio_path, duration_sec=2.0)
            rel_faces = str(faces_dir.relative_to(base_dir))
            rel_mouths = str(mouths_dir.relative_to(base_dir))
            rel_audio = str(audio_path.relative_to(base_dir))
            meta = sample_schema(
                sample_id=sample_id,
                speaker_id=speaker_id,
                video_path="",
                audio_path=rel_audio,
                transcript_tr="Demo metin.",
                label_main=LABEL_REAL_SYNC,
                label_visual_fake=0,
                label_audio_fake=0,
                label_sync=1,
                sync_shift_ms=None,
                mismatch_type=MISMATCH_NONE,
                domain="demo",
                duration_sec=2.0,
                faces_dir=rel_faces,
                mouths_dir=rel_mouths,
            )
            save_metadata(meta, metadata_dir / f"{sample_id}.json")
            samples.append(meta)

    # Fakes metadata klasoru (generate_fakes metadata_dir'deki real'lari okur)
    print(f"Demo: {len(samples)} real sample yazildi -> {metadata_dir}")
    print("  UYARI: Gorseller tek renk placeholder (gercek yuz degil). Pipeline calisir ama Visual model anlamli ogrenemez.")
    print("  Gercek sonuc icin: gercek videolardan preprocess veya add_real_videos.py (AVLips/URL) kullanin.")
    return samples


def build_demo_splits(base_dir: Path | None = None) -> None:
    from src.preprocessing.build_splits import run_build_splits
    base_dir = base_dir or project_root()
    train, val, test = run_build_splits(base_dir / "configs" / "data.yaml")
    print(f"Splits: train={len(train)}, val={len(val)}, test={len(test)}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--speakers", type=int, default=3)
    p.add_argument("--per-speaker", type=int, default=2)
    p.add_argument("--build-splits", action="store_true", help="create_demo_data sonrası build_splits çalıştır")
    args = p.parse_args()
    samples = create_demo_data(
        num_speakers=args.speakers,
        samples_per_speaker=args.per_speaker,
    )
    if args.build_splits:
        build_demo_splits()
