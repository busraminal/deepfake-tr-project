"""
Füzyon model eğitimi: S_f = alpha * S_v + (1-alpha) * S_l. Config: configs/fusion.yaml
"""
from __future__ import annotations

from pathlib import Path

from src.utils.io import load_config, load_json_splits, project_root
from src.utils.seed import set_seed


def main(config_path: str | Path | None = None) -> None:
    base = project_root()
    config = load_config(config_path or base / "configs" / "fusion.yaml")
    set_seed(42)
    data_config = load_config(base / config.get("data", {}).get("config", "configs/data.yaml"))
    splits_dir = base / data_config.get("data", {}).get("splits_dir", "data/splits")
    train, val, test = load_json_splits(splits_dir)
    if not train:
        print("No train samples.")
        return
    try:
        import torch
        from src.models.visual_model import VisualModel
        from src.models.fusion_model import FusionModel
    except ImportError as e:
        print("PyTorch required:", e)
        return
    # S_v ve S_l'yi mevcut modellerden üretmek gerekir; basit örnek için sabit skorlarla fusion ağırlığını eğitelim
    fusion_cfg = config.get("fusion", {})
    model = FusionModel(
        learn_alpha=config.get("model", {}).get("learn_alpha", False),
        init_alpha=fusion_cfg.get("alpha", 0.5),
    )
    # Bu script fusion katmanını (alpha) eğitmek için S_v, S_l önceden hesaplanmış kabul edebilir
    # veya visual/sync checkpoint'leri yükleyip uçtan uca eğitebilir. MVP: sadece alpha'yı sabit kullan
    out_dir = base / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "fusion_model.pt")
    print("Saved checkpoints/fusion_model.pt (alpha fixed or learned)")


if __name__ == "__main__":
    main()
