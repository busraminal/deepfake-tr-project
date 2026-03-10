"""
Dataset genişletme planı — gerçek veri hedefi: 20 konuşmacı, 100 video, 300 fake, ~400 toplam.

Kullanım:
  python scripts/dataset_expansion.py protocol   -> video_list.csv şablonu + cümle ataması
  python scripts/dataset_expansion.py validate   -> mevcut metadata/splits ile hedef karşılaştırması
  python scripts/dataset_expansion.py targets    -> hedef sayıları yazdır
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Hedef (makale minimum)
TARGET_SPEAKERS = 20
TARGET_VIDEOS = 100
TARGET_FAKES = 300
TARGET_TOTAL = 400
VIDEOS_PER_SPEAKER = 5


def load_sentence_pool() -> list[str]:
    pool_path = ROOT / "data" / "sentence_pool_tr.txt"
    if not pool_path.exists():
        return []
    lines = []
    for line in pool_path.read_text(encoding="utf-8").strip().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    return lines


def cmd_protocol(args):
    """video_list.csv şablonu üretir: video_path, sample_id, speaker_id, sentence"""
    sentences = load_sentence_pool()
    if not sentences:
        print("data/sentence_pool_tr.txt bos veya yok; ornek cumleler kullaniliyor.")
        sentences = [
            "Bugun model sonuclarini yeniden degerlendirecegiz.",
            "Bu proje Turkce konusma ile dudak hareketlerini analiz eder.",
        ]
    out = ROOT / "data" / "video_list_protocol.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["video_path,sample_id,speaker_id,sentence"]
    for spk in range(1, TARGET_SPEAKERS + 1):
        speaker_id = f"spk_{spk:02d}"
        for v in range(1, VIDEOS_PER_SPEAKER + 1):
            idx = (spk - 1) * VIDEOS_PER_SPEAKER + v
            sample_id = f"TRDF_{idx:05d}"
            sentence = sentences[(idx - 1) % len(sentences)]
            video_path = f"data/raw_videos/{speaker_id}/{sample_id}.mp4"
            lines.append(f"{video_path},{sample_id},{speaker_id},{sentence}")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Yazildi: {out}  ({len(lines)-1} satir)")
    print("Videolari bu yollara koyup run_pipeline.py preprocess --video-list data/video_list_protocol.csv calistirin.")


def cmd_validate(args):
    """Mevcut metadata ve splits ile hedefi karsilastirir."""
    from src.utils.io import load_config, load_json_splits, project_root
    base = project_root()
    config = load_config(base / "configs" / "data.yaml")
    data_cfg = config.get("data", {})
    meta_dir = base / data_cfg.get("metadata_dir", "data/processed/metadata")
    fakes_meta = base / data_cfg.get("fakes_dir", "data/processed/fakes") / "metadata"
    splits_dir = base / data_cfg.get("splits_dir", "data/splits")

    n_real = len(list(meta_dir.glob("*.json"))) if meta_dir.exists() else 0
    n_fake = len(list(fakes_meta.glob("*.json"))) if fakes_meta.exists() else 0
    train, val, test = load_json_splits(splits_dir)
    n_train, n_val, n_test = len(train), len(val), len(test)
    n_total = n_train + n_val + n_test
    speakers = set()
    for s in train + val + test:
        speakers.add(s.get("speaker_id", ""))
    n_speakers = len(speakers)

    print("Hedef vs mevcut:")
    print(f"  Konusmaci:  hedef {TARGET_SPEAKERS}  mevcut {n_speakers}")
    print(f"  Real:       hedef {TARGET_VIDEOS}  mevcut {n_real}")
    print(f"  Fake:       hedef {TARGET_FAKES}  mevcut {n_fake}")
    print(f"  Toplam:     hedef {TARGET_TOTAL}  mevcut {n_total}  (train+val+test: {n_train}+{n_val}+{n_test})")
    if n_total >= TARGET_TOTAL and n_speakers >= TARGET_SPEAKERS:
        print("  -> Hedef karsilandi.")
    else:
        print("  -> Eksik: daha fazla real video + generate_fakes gerekli.")


def cmd_targets(args):
    print("Dataset genisletme hedefleri (makale minimum):")
    print(f"  Konusmaci:     {TARGET_SPEAKERS}")
    print(f"  Video/konus.:  {VIDEOS_PER_SPEAKER}  => real video: {TARGET_SPEAKERS * VIDEOS_PER_SPEAKER}")
    print(f"  Real:         {TARGET_VIDEOS}")
    print(f"  Fake:         {TARGET_FAKES}  (sync_shift + content_mismatch + synthetic_audio)")
    print(f"  Toplam:       N = N_real * (1 + N_fake_per_real) ~ {TARGET_TOTAL}")


def main():
    p = argparse.ArgumentParser(description="Dataset genisletme plani")
    p.add_argument("cmd", choices=["protocol", "validate", "targets"], help="protocol | validate | targets")
    args = p.parse_args()
    if args.cmd == "protocol":
        cmd_protocol(args)
    elif args.cmd == "validate":
        cmd_validate(args)
    else:
        cmd_targets(args)


if __name__ == "__main__":
    main()
