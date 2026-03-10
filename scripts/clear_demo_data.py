"""
Demo (placeholder) veriyi tamamen kaldirir. Sonra gercek veri add_real_videos veya preprocess ile eklenir.
Kullanim: python scripts/clear_demo_data.py [--dry-run]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description="Demo veriyi sil; gercek veri icin alan ac")
    p.add_argument("--dry-run", action="store_true", help="Silmeden sadece ne silinecegini yazdir")
    args = p.parse_args()

    from src.utils.io import load_config, project_root

    base = project_root()
    config = load_config(base / "configs" / "data.yaml")
    data = config.get("data", {})

    metadata_dir = base / data.get("metadata_dir", "data/processed/metadata")
    fakes_dir = base / data.get("fakes_dir", "data/processed/fakes")
    fakes_meta = fakes_dir / "metadata"
    fakes_audio = fakes_dir / "audio"
    faces_base = base / "data" / "processed" / "faces"
    mouths_base = base / "data" / "processed" / "mouths"
    raw_audio_dir = base / data.get("raw_audio_dir", "data/raw_audio")

    removed = []

    # 1) Real metadata: demo_*.json
    if metadata_dir.exists():
        for f in metadata_dir.glob("demo_*.json"):
            removed.append(str(f.relative_to(base)))
            if not args.dry_run:
                f.unlink()

    # 2) Fakes metadata: sample_id "demo_" ile baslayan (demo_000_shift_100 vb.)
    if fakes_meta.exists():
        for f in fakes_meta.glob("*.json"):
            if f.stem.startswith("demo_"):
                removed.append(str(f.relative_to(base)))
                if not args.dry_run:
                    f.unlink()

    # 3) Fakes audio: demo_ ile baslayan .wav
    if fakes_audio.exists():
        for f in fakes_audio.glob("*.wav"):
            if f.stem.startswith("demo_"):
                removed.append(str(f.relative_to(base)))
                if not args.dry_run:
                    f.unlink()

    # 4) Faces: demo_* klasorleri
    if faces_base.exists():
        for d in faces_base.iterdir():
            if d.is_dir() and d.name.startswith("demo_"):
                removed.append(str(d.relative_to(base)))
                if not args.dry_run:
                    shutil.rmtree(d, ignore_errors=True)

    # 5) Mouths: demo_* klasorleri
    if mouths_base.exists():
        for d in mouths_base.iterdir():
            if d.is_dir() and d.name.startswith("demo_"):
                removed.append(str(d.relative_to(base)))
                if not args.dry_run:
                    shutil.rmtree(d, ignore_errors=True)

    # 6) Raw audio: demo_*.wav
    if raw_audio_dir.exists():
        for f in raw_audio_dir.glob("demo_*.wav"):
            removed.append(str(f.relative_to(base)))
            if not args.dry_run:
                f.unlink()

    print(f"Toplam {len(removed)} oge {'(dry-run, silinmedi)' if args.dry_run else 'silindi'}.")
    for r in removed[:30]:
        print(f"  {r}")
    if len(removed) > 30:
        print(f"  ... ve {len(removed) - 30} tane daha.")

    if not args.dry_run and removed:
        print("\nSonraki adim: gercek veri ekle (add_real_videos.py avlips veya urls), sonra generate_fakes, build_splits.")


if __name__ == "__main__":
    main()
