"""
Görsel model eğitimi: S_v. Config: configs/train_visual.yaml
"""
from __future__ import annotations

from pathlib import Path

from src.utils.io import load_config, load_json_splits, project_root
from src.utils.seed import set_seed


def main(config_path: str | Path | None = None) -> None:
    base = project_root()
    config = load_config(config_path or base / "configs" / "train_visual.yaml")
    set_seed(42)
    data_cfg = config.get("data", {})
    split_name = data_cfg.get("split", "train")
    data_config_path = base / data_cfg.get("config", "configs/data.yaml")
    data_config = load_config(data_config_path)
    splits_dir = base / data_config.get("data", {}).get("splits_dir", "data/splits")
    train, val, test = load_json_splits(splits_dir)
    samples = train if split_name == "train" else val
    if not samples:
        print("No samples in split. Run preprocess and build_splits first.")
        return
    try:
        from src.datasets.visual_dataset import VisualDataset, VisualTorchDataset
        from src.models.visual_model import VisualModel
        import torch
        from torch.utils.data import DataLoader
    except ImportError as e:
        print("PyTorch required for training:", e)
        return
    face_size = data_config.get("preprocess", {}).get("face_size", 224)
    ds = VisualDataset(samples, base, face_size=(face_size, face_size))
    torch_ds = VisualTorchDataset(ds)
    loader = DataLoader(torch_ds, batch_size=config.get("train", {}).get("batch_size", 16), shuffle=True, num_workers=0)
    model_cfg = config.get("model", {})
    model = VisualModel(
        backbone=model_cfg.get("backbone", "resnet18"),
        num_classes=model_cfg.get("num_classes", 1),
        pretrained=model_cfg.get("pretrained", True),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=config.get("train", {}).get("lr", 1e-3))
    criterion = torch.nn.BCELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    epochs = config.get("train", {}).get("epochs", 30)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.unsqueeze(1).to(device)
            opt.zero_grad()
            out = model(batch_x)
            loss = criterion(out.unsqueeze(1), batch_y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        n_batches = len(loader)
        if (epoch + 1) % 5 == 0 or epochs <= 5:
            avg = total_loss / n_batches if n_batches else 0.0
            print(f"Visual epoch {epoch+1}/{epochs} loss={avg:.4f}")
    out_dir = base / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "visual_model.pt")
    print("Saved checkpoints/visual_model.pt")


if __name__ == "__main__":
    main()
