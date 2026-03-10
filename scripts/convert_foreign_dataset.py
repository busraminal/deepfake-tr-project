"""
Dış veri setini (AVLips vb.) proje şemasına çevirir: data/processed/metadata ve splits.
AVLips klasör yapısı: genelde real/ ve fake/ (veya benzeri) alt klasörler + videolar.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.schema import (
    sample_schema,
    LABEL_REAL_SYNC,
    LABEL_FAKE_VISUAL,
    LABEL_FAKE_SYNC_SHIFT,
    MISMATCH_NONE,
)
from src.utils.io import save_metadata, save_split, load_config, project_root


def find_videos(dir_path: Path, exts: tuple = (".mp4", ".avi", ".mov", ".mkv")) -> list[Path]:
    videos = []
    for ext in exts:
        videos.extend(dir_path.rglob(f"*{ext}"))
    return sorted(videos)


def convert_avlips(input_dir: Path, output_metadata_dir: Path, output_splits_dir: Path, seed: int = 42) -> list[dict]:
    """
    AVLips indirilip açıldıktan sonra klasör yapısına göre tarar.
    Beklenen: input_dir altında real/ ve fake/ (veya real_videos/, fake_videos/) ve içinde videolar.
    """
    input_dir = Path(input_dir)
    output_metadata_dir = Path(output_metadata_dir)
    output_splits_dir = Path(output_splits_dir)
    output_metadata_dir.mkdir(parents=True, exist_ok=True)
    output_splits_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    sample_id = 0

    # Olası klasör isimleri
    try:
        input_dir_resolved = Path(input_dir).resolve()
        root_resolved = ROOT.resolve()
        use_relative = root_resolved in input_dir_resolved.parents or input_dir_resolved == root_resolved
    except Exception:
        use_relative = False

    for real_name in ("real", "real_videos", "Real", "original"):
        real_dir = input_dir / real_name
        if real_dir.exists():
            for vp in find_videos(real_dir):
                try:
                    rel_video = str(vp.relative_to(ROOT)) if use_relative else str(vp)
                except ValueError:
                    rel_video = str(vp)
                meta = sample_schema(
                    sample_id=f"AVLips_{sample_id:05d}",
                    speaker_id="external",
                    video_path=rel_video,
                    audio_path="",
                    transcript_tr="",
                    label_main=LABEL_REAL_SYNC,
                    label_visual_fake=0,
                    label_audio_fake=0,
                    label_sync=1,
                    sync_shift_ms=None,
                    mismatch_type=MISMATCH_NONE,
                    domain="external",
                    duration_sec=0,
                    source_path=str(vp),
                )
                samples.append(meta)
                save_metadata(meta, output_metadata_dir / f"{meta['sample_id']}.json")
                sample_id += 1
            break

    for fake_name in ("fake", "fake_videos", "Fake", "manipulated"):
        fake_dir = input_dir / fake_name
        if fake_dir.exists():
            for vp in find_videos(fake_dir):
                try:
                    rel_video = str(vp.relative_to(ROOT)) if use_relative else str(vp)
                except ValueError:
                    rel_video = str(vp)
                meta = sample_schema(
                    sample_id=f"AVLips_{sample_id:05d}",
                    speaker_id="external",
                    video_path=rel_video,
                    audio_path="",
                    transcript_tr="",
                    label_main=LABEL_FAKE_SYNC_SHIFT,
                    label_visual_fake=1,
                    label_audio_fake=0,
                    label_sync=0,
                    sync_shift_ms=None,
                    mismatch_type="lip_sync_fake",
                    domain="external",
                    duration_sec=0,
                    source_path=str(vp),
                )
                samples.append(meta)
                save_metadata(meta, output_metadata_dir / f"{meta['sample_id']}.json")
                sample_id += 1
            break

    if not samples:
        # Tek seviye: tüm videoları listele, isimde real/fake geçiyorsa ayır
        for vp in find_videos(input_dir):
            name = vp.stem.lower()
            if "fake" in name or "synced" in name or "manip" in name:
                label_main = LABEL_FAKE_SYNC_SHIFT
                label_sync = 0
            else:
                label_main = LABEL_REAL_SYNC
                label_sync = 1
            try:
                rel_video = str(vp.relative_to(ROOT)) if use_relative else str(vp)
            except ValueError:
                rel_video = str(vp)
            meta = sample_schema(
                sample_id=f"AVLips_{sample_id:05d}",
                speaker_id="external",
                video_path=rel_video,
                audio_path="",
                transcript_tr="",
                label_main=label_main,
                label_visual_fake=0 if label_sync else 1,
                label_audio_fake=0,
                label_sync=label_sync,
                sync_shift_ms=None,
                mismatch_type=MISMATCH_NONE if label_sync else "lip_sync_fake",
                domain="external",
                duration_sec=0,
                source_path=str(vp),
            )
            samples.append(meta)
            save_metadata(meta, output_metadata_dir / f"{meta['sample_id']}.json")
            sample_id += 1

    # Basit split: %70 train, %15 val, %15 test (speaker hep "external" olduğu için rastgele)
    random.seed(seed)
    random.shuffle(samples)
    n = len(samples)
    n_test = max(1, n // 7)
    n_val = max(1, n // 7)
    n_train = n - n_test - n_val
    train = samples[:n_train]
    val = samples[n_train : n_train + n_val]
    test = samples[n_train + n_val :]
    save_split(train, output_splits_dir / "train.json")
    save_split(val, output_splits_dir / "val.json")
    save_split(test, output_splits_dir / "test.json")
    print(f"Örnek sayısı: {len(samples)}, train: {len(train)}, val: {len(val)}, test: {len(test)}")
    return samples


def main():
    parser = argparse.ArgumentParser(description="Dış veri setini proje şemasına çevir")
    parser.add_argument("format", choices=["avlips"], help="Veri seti formatı")
    parser.add_argument("--input", type=str, required=True, help="AVLips açılmış klasör yolu")
    parser.add_argument("--output", type=str, default=None, help="Çıktı base (metadata + splits); varsayılan data/processed")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base = project_root()
    input_dir = Path(args.input)
    if not input_dir.is_absolute():
        input_dir = base / input_dir
    if not input_dir.exists():
        print(f"Klasör bulunamadı: {input_dir}")
        sys.exit(1)

    output_base = Path(args.output) if args.output else base / "data" / "processed"
    if not output_base.is_absolute():
        output_base = base / output_base
    metadata_dir = output_base / "metadata"
    splits_dir = base / "data" / "splits"

    if args.format == "avlips":
        convert_avlips(input_dir, metadata_dir, splits_dir, seed=args.seed)


if __name__ == "__main__":
    main()
