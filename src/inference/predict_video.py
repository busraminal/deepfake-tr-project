"""
Tek bir video veya sample_id için deepfake skoru (S_v) üretir.
Kullanım: python -m src.inference.predict_video --sample-id demo_001
         python -m src.inference.predict_video --video path/to/video.mp4 --sample-id out1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def predict_sample_id(sample_id: str, checkpoint_path: Path | None = None) -> float:
    """Metadata ve faces_dir ile yüklü örnek için S_v (0-1) döner."""
    from src.utils.io import load_config, load_metadata, project_root
    base = project_root()
    config = load_config(base / "configs" / "data.yaml")
    metadata_dir = base / config.get("data", {}).get("metadata_dir", "data/processed/metadata")
    meta_path = metadata_dir / f"{sample_id}.json"
    if not meta_path.exists():
        fakes_meta = base / config.get("data", {}).get("fakes_dir", "data/processed/fakes") / "metadata"
        meta_path = fakes_meta / f"{sample_id}.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata yok: {sample_id}")
    meta = load_metadata(meta_path)
    samples = [meta]
    face_size = config.get("preprocess", {}).get("face_size", 224)
    from src.datasets.visual_dataset import VisualDataset, VisualTorchDataset
    from src.models.visual_model import VisualModel
    import torch
    ds = VisualDataset(samples, base, face_size=(face_size, face_size), use_label_visual=False)
    torch_ds = VisualTorchDataset(ds)
    loader = torch.utils.data.DataLoader(torch_ds, batch_size=1, shuffle=False)
    model_cfg = (load_config(base / "configs" / "train_visual.yaml") or {}).get("model", {})
    model = VisualModel(backbone=model_cfg.get("backbone", "resnet18"), num_classes=1, pretrained=False)
    ckpt = checkpoint_path or base / "checkpoints" / "visual_model.pt"
    try:
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True), strict=True)
    except TypeError:
        model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
    model.eval()
    with torch.no_grad():
        x, _ = next(iter(loader))
        out = model(x)
    return float(out.item())


def main():
    p = argparse.ArgumentParser(description="Deepfake skoru (S_v) hesapla")
    p.add_argument("--sample-id", type=str, help="Örnek ID (metadata + faces mevcut)")
    p.add_argument("--video", type=str, help="Video yolu (ön işleme gerekir; şimdilik sample-id kullan)")
    p.add_argument("--checkpoint", type=str, default=None, help="visual_model.pt yolu")
    args = p.parse_args()
    if not args.sample_id and not args.video:
        p.print_help()
        return
    sample_id = args.sample_id or Path(args.video).stem
    ckpt = Path(args.checkpoint) if args.checkpoint else None
    try:
        score = predict_sample_id(sample_id, ckpt)
        print(f"S_v (görsel sahte olasılığı): {score:.4f}  (0=gerçek, 1=sahte)")
    except Exception as e:
        print("Hata:", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
