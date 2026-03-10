"""
Senkron model eğitimi: E_text ve M_video → S_l. Config: configs/train_sync.yaml
"""
from __future__ import annotations

from pathlib import Path

from src.utils.io import load_config, load_json_splits, project_root
from src.utils.seed import set_seed


def main(config_path: str | Path | None = None) -> None:
    base = project_root()
    config = load_config(config_path or base / "configs" / "train_sync.yaml")
    set_seed(42)
    data_cfg = config.get("data", {})
    data_config = load_config(base / data_cfg.get("config", "configs/data.yaml"))
    splits_dir = base / data_config.get("data", {}).get("splits_dir", "data/splits")
    train, val, test = load_json_splits(splits_dir)
    if not train:
        print("No train samples. Run preprocess and build_splits first.")
        return
    try:
        from src.datasets.sync_dataset import SyncDataset, SyncTorchDataset
        from src.models.mouth_model import MouthEncoder
        from src.models.text_encoder import TextEncoder
        from src.models.fusion_model import compute_sync_score_from_embeddings
        import torch
        from torch.utils.data import DataLoader
    except ImportError as e:
        print("PyTorch required:", e)
        return
    mouth_size = data_config.get("preprocess", {}).get("mouth_size", 96)
    ds = SyncDataset(train, base, mouth_size=(mouth_size, mouth_size))
    torch_ds = SyncTorchDataset(ds)
    loader = DataLoader(torch_ds, batch_size=config.get("train", {}).get("batch_size", 16), shuffle=True, num_workers=0)
    emb_dim = config.get("model", {}).get("embedding_dim", 256)
    mouth_enc = MouthEncoder(embedding_dim=emb_dim)
    text_enc = TextEncoder(vocab_size=1000, embedding_dim=emb_dim)
    opt = torch.optim.AdamW(
        list(mouth_enc.parameters()) + list(text_enc.parameters()),
        lr=config.get("train", {}).get("lr", 5e-4),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mouth_enc, text_enc = mouth_enc.to(device), text_enc.to(device)
    epochs = config.get("train", {}).get("epochs", 40)
    for epoch in range(epochs):
        mouth_enc.train()
        text_enc.train()
        total_loss = 0.0
        for mouth, text, label in loader:
            mouth, text, label = mouth.to(device), text.to(device), label.to(device)
            opt.zero_grad()
            m_emb = mouth_enc(mouth)
            e_emb = text_enc(text)
            s_l = compute_sync_score_from_embeddings(e_emb, m_emb)
            # label=1 uyumlu, s_l düşük olmalı; label=0 uyumsuz, s_l yüksek olmalı
            loss = torch.nn.functional.mse_loss(s_l, 1 - label)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"Sync epoch {epoch+1}/{epochs} loss={total_loss/len(loader):.4f}")
    out_dir = base / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"mouth": mouth_enc.state_dict(), "text": text_enc.state_dict()}, out_dir / "sync_model.pt")
    print("Saved checkpoints/sync_model.pt")


if __name__ == "__main__":
    main()
