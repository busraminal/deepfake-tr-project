"""
Füzyon: S_v ve S_l (veya sadece mevcut checkpoint'ler) ile S_f hesaplanıp test setinde değerlendirilir.
Şu an visual + sync skorlarını birleştirmek için ayrı fusion checkpoint gerekebilir; basit versiyonda
sadece visual sonucu raporlanabilir veya S_v ile S_l ağırlıklı ortalama yapılır.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.io import load_config, load_json_splits, project_root
from src.evaluation.metrics import compute_all
from src.evaluation.evaluate_visual import run_evaluate_visual
from src.evaluation.evaluate_sync import run_evaluate_sync


def run_evaluate_fusion(
    alpha: float = 0.5,
    split: str = "test",
    return_scores: bool = False,
) -> dict[str, Any]:
    """
    S_v ve S_l skorlarını örnek bazında birleştirip S_f = alpha*S_v + (1-alpha)*S_l ile değerlendirir.
    Bunun için her iki modelin test seti üzerinde skor üretmesi gerekir.
    """
    base = project_root()
    data_config = load_config(base / "configs" / "data.yaml")
    splits_dir = base / data_config.get("data", {}).get("splits_dir", "data/splits")
    train, val, test = load_json_splits(splits_dir)
    samples = {"train": train, "val": val, "test": test}.get(split, test)
    if not samples:
        return {"error": "no_samples", "split": split}

    # Skorları örnek sırasıyla almak için evaluation'ı batch yerine tek tek veya aynı sırada yapmalıyız.
    # Basit yol: visual ve sync evaluate'dan y_true aynı; skorları dosyadan veya tekrar hesaplayıp birleştir.
    # En temiz: aynı loader ile her iki modeli çalıştırıp S_v ve S_l listelerini al, sonra S_f = a*S_v + (1-a)*S_l.
    try:
        import torch
        from src.datasets.visual_dataset import VisualDataset, VisualTorchDataset
        from src.datasets.sync_dataset import SyncDataset, SyncTorchDataset
        from src.models.visual_model import VisualModel
        from src.models.mouth_model import MouthEncoder
        from src.models.text_encoder import TextEncoder
        from src.models.fusion_model import compute_sync_score_from_embeddings
    except ImportError as e:
        return {"error": "import_failed", "message": str(e)}

    face_size = data_config.get("preprocess", {}).get("face_size", 224)
    mouth_size = data_config.get("preprocess", {}).get("mouth_size", 96)
    visual_ds = VisualDataset(samples, base, face_size=(face_size, face_size), use_label_visual=False)
    sync_ds = SyncDataset(samples, base, mouth_size=(mouth_size, mouth_size))
    visual_loader = torch.utils.data.DataLoader(VisualTorchDataset(visual_ds), batch_size=8, shuffle=False, num_workers=0)
    sync_loader = torch.utils.data.DataLoader(SyncTorchDataset(sync_ds), batch_size=8, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    s_v_list, s_l_list, y_true_list = [], [], []

    # Visual
    cp_visual = base / "checkpoints" / "visual_model.pt"
    if not cp_visual.exists():
        return {"error": "visual_checkpoint_not_found"}
    model_cfg = (load_config(base / "configs" / "train_visual.yaml") or {}).get("model", {})
    vis_model = VisualModel(backbone=model_cfg.get("backbone", "resnet18"), num_classes=1, pretrained=False)
    try:
        vis_model.load_state_dict(torch.load(cp_visual, map_location="cpu", weights_only=True), strict=True)
    except TypeError:
        vis_model.load_state_dict(torch.load(cp_visual, map_location="cpu"), strict=True)
    vis_model.eval().to(device)
    with torch.no_grad():
        for batch_x, batch_y in visual_loader:
            out = vis_model(batch_x.to(device))
            if out.dim() == 0:
                out = out.unsqueeze(0)
            s_v_list.extend(out.cpu().numpy().tolist())
            y_true_list.extend(batch_y.numpy().tolist())

    # Sync
    cp_sync = base / "checkpoints" / "sync_model.pt"
    if not cp_sync.exists():
        return {"error": "sync_checkpoint_not_found", "fusion_skipped": True}
    sync_cfg = (load_config(base / "configs" / "train_sync.yaml") or {}).get("model", {})
    emb_dim = sync_cfg.get("embedding_dim", 256)
    mouth_enc = MouthEncoder(embedding_dim=emb_dim)
    text_enc = TextEncoder(vocab_size=1000, embedding_dim=emb_dim)
    try:
        st = torch.load(cp_sync, map_location="cpu", weights_only=True)
    except TypeError:
        st = torch.load(cp_sync, map_location="cpu")
    mouth_enc.load_state_dict(st["mouth"], strict=True)
    text_enc.load_state_dict(st["text"], strict=True)
    mouth_enc.eval().to(device)
    text_enc.eval().to(device)
    s_l_list = []
    with torch.no_grad():
        for mouth, text, _ in sync_loader:
            mouth, text = mouth.to(device), text.to(device)
            s_l = compute_sync_score_from_embeddings(text_enc(text), mouth_enc(mouth))
            s_l_list.extend(s_l.cpu().numpy().tolist())

    y_true = [float(x) for x in y_true_list]
    s_v = [float(x) for x in s_v_list]
    s_l = s_l_list if len(s_l_list) == len(y_true) else [0.5] * len(y_true)
    s_f = [alpha * v + (1 - alpha) * l for v, l in zip(s_v, s_l)]
    metrics = compute_all(y_true, s_f)  # threshold=None -> Youden optimal
    metrics["split"] = split
    metrics["n_samples"] = len(y_true)
    metrics["alpha"] = alpha
    if return_scores:
        metrics["y_true"] = y_true
        metrics["y_score"] = s_f
    return metrics


def main() -> None:
    result = run_evaluate_fusion(alpha=0.5, split="test")
    if "error" in result:
        print("Fusion:", result.get("error"), result.get("message", ""))
        return
    print("Fusion (test set):")
    print(f"  alpha: {result['alpha']}  n_samples: {result['n_samples']}")
    print(f"  accuracy: {result['accuracy']:.4f}  AUC: {result['auc']:.4f}  EER: {result['eer']:.4f}")


if __name__ == "__main__":
    main()
