"""
Gerçek videoları veri setine ekler: AVLips indir+dönüştür+ön işleme veya URL listesinden indir+ön işleme.

Kullanım:
  # AVLips (İngilizce lip-sync real+fake, ~9 GB):
  python scripts/add_real_videos.py avlips [--download] [--skip-preprocess]

  # URL listesi (CSV: url,sample_id,speaker_id) ile indir + ön işleme:
  python scripts/add_real_videos.py urls --csv data/video_urls.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import load_config, project_root, save_metadata


def _ensure_faces_mouths(metadata_dir: Path, base_dir: Path) -> int:
    """video_path olan ama faces_dir olmayan tüm metadata için preprocess çalıştırır."""
    from src.preprocessing.run_preprocess import run_preprocess_one

    config = load_config(base_dir / "configs" / "data.yaml")
    data_cfg = config.get("data", {})
    raw_videos = base_dir / data_cfg.get("raw_videos_dir", "data/raw_videos")

    count = 0
    for jpath in sorted(metadata_dir.glob("*.json")):
        meta = json.loads(jpath.read_text(encoding="utf-8"))
        if not meta.get("video_path"):
            continue
        if meta.get("faces_dir") and (base_dir / meta["faces_dir"]).exists():
            continue
        vpath = meta["video_path"]
        if not Path(vpath).is_absolute():
            vpath = base_dir / vpath
        else:
            vpath = Path(vpath)
        if not vpath.exists():
            vpath = raw_videos / Path(meta["video_path"]).name
        if not vpath.exists():
            print(f"  Atla (dosya yok): {meta['sample_id']} -> {meta['video_path']}")
            continue
        print(f"  On isleniyor: {meta['sample_id']}")
        out = run_preprocess_one(
            str(vpath),
            meta["sample_id"],
            meta.get("speaker_id", "external"),
            config=config,
            base_dir=base_dir,
            use_whisper=False,
        )
        if out:
            save_metadata(out, jpath)
            count += 1
    return count


def cmd_avlips(download: bool, skip_preprocess: bool) -> None:
    base = project_root()
    out_dir = base / "data" / "raw_videos" / "avlips"
    metadata_dir = base / "data" / "processed" / "metadata"
    splits_dir = base / "data" / "splits"

    if download:
        try:
            import gdown
        except ImportError:
            print("pip install gdown")
            sys.exit(1)
        out_dir.mkdir(parents=True, exist_ok=True)
        zip_path = out_dir / "AVLips.zip"
        url = "https://drive.google.com/uc?id=1fEiUo22GBSnWD7nfEwDW86Eiza-pOEJm"
        print("AVLips indiriliyor (~9 GB)...")
        gdown.download(url, str(zip_path), quiet=False, fuzzy=True)
        print("Indirme tamam. Zip'i acin: 7z x AVLips.zip veya unzip (data/raw_videos/avlips icinde).")
        if skip_preprocess:
            return
        # Zip açılmış mı kontrol et
        avlips_unzip = out_dir / "AVLips"
        if not avlips_unzip.exists():
            for d in out_dir.iterdir():
                if d.is_dir() and (d / "real").exists() or (d / "fake").exists():
                    avlips_unzip = d
                    break
        if not avlips_unzip.exists():
            print("AVLips klasoru bulunamadi. Zip'i acip tekrar calistirin.")
            return
        input_dir = avlips_unzip
    else:
        # AVLips zaten var; convert + preprocess
        input_dir = None
        for d in [out_dir, out_dir / "AVLips", out_dir / "avlips"]:
            if not d.exists():
                continue
            if (d / "real").exists() or (d / "fake").exists() or list(d.glob("*.mp4")):
                input_dir = d
                break
        if input_dir is None:
            print("AVLips verisi bulunamadi. Once: python scripts/add_real_videos.py avlips --download")
            sys.exit(1)

    # Dönüştür (metadata + split)
    from scripts.convert_foreign_dataset import convert_avlips

    metadata_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    convert_avlips(input_dir, metadata_dir, splits_dir, seed=42)
    print("Donusum tamam. Split'ler data/splits/ altinda (mevcut demo ile birlestirmek icin build_splits tekrar calistirilabilir).")

    if not skip_preprocess:
        print("Preprocess (yuz/agiz cikarma) calistiriliyor...")
        n = _ensure_faces_mouths(metadata_dir, base)
        print(f"  {n} ornek on islendi.")
    else:
        print("Preprocess atlandi (--skip-preprocess). Videolar icin yuz/agiz cikarmak isterseniz --skip-preprocess olmadan tekrar calistirin.")


def cmd_urls(csv_path: Path, out_video_dir: Path) -> None:
    """CSV'den (url, sample_id, speaker_id) okuyup yt-dlp ile indirir, preprocess çalıştırır."""
    base = project_root()
    if not csv_path.is_absolute():
        csv_path = base / csv_path
    if not csv_path.exists():
        print(f"CSV bulunamadi: {csv_path}")
        sys.exit(1)

    out_video_dir = out_video_dir or (base / "data" / "raw_videos" / "from_urls")
    if not out_video_dir.is_absolute():
        out_video_dir = base / out_video_dir
    out_video_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = base / "data" / "processed" / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    try:
        import yt_dlp
    except ImportError:
        print("URL indirmek icin: pip install yt-dlp")
        sys.exit(1)

    from src.preprocessing.run_preprocess import run_preprocess_one

    config = load_config(base / "configs" / "data.yaml")
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("url") or not row.get("sample_id"):
                continue
            rows.append((row["url"].strip(), row["sample_id"].strip(), row.get("speaker_id", "spk_web").strip()))

    for url, sample_id, speaker_id in rows:
        out_file = out_video_dir / f"{sample_id}.mp4"
        if out_file.exists():
            print(f"  Zaten var: {sample_id}")
        else:
            print(f"  Indiriliyor: {sample_id} <- {url[:60]}...")
            opts = {"outtmpl": str(out_file), "format": "best[ext=mp4]/best", "quiet": False}
            with yt_dlp.YoutubeDL(opts) as ydl:
                try:
                    ydl.download([url])
                except Exception as e:
                    print(f"  Hata: {e}")
                    continue
        if not out_file.exists():
            continue
        print(f"  On isleniyor: {sample_id}")
        out = run_preprocess_one(str(out_file), sample_id, speaker_id, config=config, base_dir=base, use_whisper=True)
        if out:
            save_metadata(out, metadata_dir / f"{sample_id}.json")
    print("URL listesi tamam. Sonra: python run_pipeline.py generate_fakes && python run_pipeline.py build_splits")


def main():
    p = argparse.ArgumentParser(description="Gerçek videolari veri setine ekle")
    sub = p.add_subparsers(dest="source", required=True)

    a = sub.add_parser("avlips", help="AVLips (LipFD) indir + donustur + istege bagli preprocess")
    a.add_argument("--download", action="store_true", help="Google Drive'dan indir")
    a.add_argument("--skip-preprocess", action="store_true", help="Yuz/agiz cikarmayi atla")
    a.set_defaults(run=lambda a: cmd_avlips(a.download, a.skip_preprocess))

    u = sub.add_parser("urls", help="CSV'deki URL'lerden indir (yt-dlp) + preprocess")
    u.add_argument("--csv", type=str, default="data/video_urls.csv", help="url,sample_id,speaker_id CSV")
    u.add_argument("--out-dir", type=str, default="data/raw_videos/from_urls", help="Indirilen videolarin klasoru")
    u.set_defaults(run=lambda a: cmd_urls(Path(a.csv), Path(a.out_dir)))

    args = p.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
