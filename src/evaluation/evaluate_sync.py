"""
Senkron modeli test seti üzerinde değerlendirir: S_l (uyumsuzluk skoru) -> binary label.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.io import load_config, load_json_splits, project_root
from src.evaluation.metrics import compute_all


def run_evaluate_sync(
    checkpoint_path: str | Path | None = None,
    config_path: str | Path | None = None,
    split: str = "test",
    return_scores: bool = False,
) -> dict[str, Any]:
    base = project_root()
    config_path = config_path or base / "configs" / "data.yaml"
    data_config = load_config(config_path)
    splits_dir = base / data_config.get("data", {}).get("splits_dir", "data/splits")
    train, val, test = load_json_splits(splits_dir)
    samples = {"train": train, "val": val, "test": test}.get(split, test)
    if not samples:
        return {"error": "no_samples", "split": split}

    checkpoint_path = Path(checkpoint_path or base / "checkpoints" / "sync_model.pt")
    if not checkpoint_path.exists():
        return {"error": "checkpoint_not_found", "path": str(checkpoint_path)}

    try:
        import torch
        from src.datasets.sync_dataset import SyncDataset, SyncTorchDataset
        from src.models.mouth_model import MouthEncoder
        from src.models.text_encoder import TextEncoder
        from src.models.fusion_model import compute_sync_score_from_embeddings
    except ImportError as e:
        return {"error": "import_failed", "message": str(e)}

    mouth_size = data_config.get("preprocess", {}).get("mouth_size", 96)
    ds = SyncDataset(samples, base, mouth_size=(mouth_size, mouth_size))
    torch_ds = SyncTorchDataset(ds)
    loader = torch.utils.data.DataLoader(torch_ds, batch_size=8, shuffle=False, num_workers=0)

    model_cfg = (load_config(base / "configs" / "train_sync.yaml") or {}).get("model", {})
    emb_dim = model_cfg.get("embedding_dim", 256)
    mouth_enc = MouthEncoder(embedding_dim=emb_dim)
    text_enc = TextEncoder(vocab_size=1000, embedding_dim=emb_dim)
    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(checkpoint_path, map_location="cpu")
    mouth_enc.load_state_dict(state["mouth"], strict=True)
    text_enc.load_state_dict(state["text"], strict=True)
    mouth_enc.eval()
    text_enc.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mouth_enc, text_enc = mouth_enc.to(device), text_enc.to(device)

    y_true, y_score = [], []
    with torch.no_grad():
        for mouth, text, label in loader:
            mouth, text = mouth.to(device), text.to(device)
            m_emb = mouth_enc(mouth)
            e_emb = text_enc(text)
            s_l = compute_sync_score_from_embeddings(e_emb, m_emb)
            y_score.extend(s_l.cpu().numpy().tolist())
            y_true.extend(label.numpy().tolist())

    y_true = [float(x) for x in y_true]
    y_score = [float(x) for x in y_score]
    metrics = compute_all(y_true, y_score)  # threshold=None -> Youden optimal
    metrics["split"] = split
    metrics["n_samples"] = len(y_true)
    if return_scores:
        metrics["y_true"] = y_true
        metrics["y_score"] = y_score
    return metrics


def main() -> None:
    result = run_evaluate_sync(split="test")
    if "error" in result:
        print("Sync:", result.get("error"), result.get("path", result.get("message", "")))
        return
    print("Sync model (test set):")
    print(f"  n_samples: {result['n_samples']}  accuracy: {result['accuracy']:.4f}  AUC: {result['auc']:.4f}  EER: {result['eer']:.4f}")


if __name__ == "__main__":
    main()
