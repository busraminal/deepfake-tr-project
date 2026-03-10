"""
Hazır veri setlerini indirir; proje data/ yapısına uygun hedefe koyar.

Kullanım:
  python scripts/download_dataset.py avlips --out-dir data/raw_videos/avlips
  python scripts/download_dataset.py hf --dataset ControlNet/AV-Deepfake1M --out-dir data/external/AV-Deepfake1M
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Proje kökü
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# AVLips (LipFD) – Google Drive ID
AVLIPS_GDRIVE_ID = "1fEiUo22GBSnWD7nfEwDW86Eiza-pOEJm"


def download_avlips(out_dir: Path) -> None:
    """AVLips veri setini gdown ile indirir."""
    try:
        import gdown
    except ImportError:
        print("Kurulum: pip install gdown")
        sys.exit(1)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "AVLips.zip"
    url = f"https://drive.google.com/uc?id={AVLIPS_GDRIVE_ID}"
    print("AVLips indiriliyor (~9 GB), bu biraz sürebilir...")
    gdown.download(url, str(zip_path), quiet=False, fuzzy=True)
    if zip_path.exists():
        print(f"İndirildi: {zip_path}")
        print("Açmak için: unzip veya 7z ile AVLips.zip dosyasını açın.")
    else:
        print("İndirme başarısız. Google Drive linkini tarayıcıdan manuel indirebilirsiniz:")
        print("https://drive.google.com/file/d/1fEiUo22GBSnWD7nfEwDW86Eiza-pOEJm/view?usp=share_link")


def download_hf(dataset_id: str, out_dir: Path) -> None:
    """Hugging Face dataset indirir (metadata + mümkünse parça parça örnek)."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Kurulum: pip install huggingface_hub")
        sys.exit(1)
    out_dir = Path(out_dir)
    print(f"Hugging Face: {dataset_id} -> {out_dir}")
    snapshot_download(repo_id=dataset_id, repo_type="dataset", local_dir=str(out_dir))
    print(f"İndirildi: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Hazır deepfake/lip-sync veri seti indir")
    parser.add_argument("source", choices=["avlips", "hf"], help="avlips (LipFD) veya hf (Hugging Face)")
    parser.add_argument("--out-dir", type=str, default=None, help="Hedef klasör")
    parser.add_argument("--dataset", type=str, default="ControlNet/AV-Deepfake1M", help="HF dataset id (source=hf için)")
    args = parser.parse_args()

    if args.out_dir is None:
        if args.source == "avlips":
            args.out_dir = ROOT / "data" / "raw_videos" / "avlips"
        else:
            args.out_dir = ROOT / "data" / "external" / args.dataset.split("/")[-1]
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    if args.source == "avlips":
        download_avlips(out_dir)
    elif args.source == "hf":
        download_hf(args.dataset, out_dir)


if __name__ == "__main__":
    main()
