"""
Görsel modeli test seti üzerinde değerlendirir; accuracy, AUC, EER raporlar.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.io import load_config, load_json_splits, project_root
from src.evaluation.metrics import compute_all


def run_evaluate_visual(
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

    checkpoint_path = Path(checkpoint_path or base / "checkpoints" / "visual_model.pt")
    if not checkpoint_path.exists():
        return {"error": "checkpoint_not_found", "path": str(checkpoint_path)}

    try:
        import torch
        from src.datasets.visual_dataset import VisualDataset, VisualTorchDataset
        from src.models.visual_model import VisualModel
    except ImportError as e:
        return {"error": "import_failed", "message": str(e)}

    face_size = data_config.get("preprocess", {}).get("face_size", 224)
    ds = VisualDataset(samples, base, face_size=(face_size, face_size), use_label_visual=False)
    torch_ds = VisualTorchDataset(ds)
    loader = torch.utils.data.DataLoader(torch_ds, batch_size=8, shuffle=False, num_workers=0)

    model_cfg = (load_config(base / "configs" / "train_visual.yaml") or {}).get("model", {})
    model = VisualModel(
        backbone=model_cfg.get("backbone", "resnet18"),
        num_classes=model_cfg.get("num_classes", 1),
        pretrained=False,
    )
    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    y_true, y_score = [], []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            out = model(batch_x)
            if out.dim() == 0:
                out = out.unsqueeze(0)
            y_score.extend(out.cpu().numpy().tolist())
            y_true.extend(batch_y.numpy().tolist())

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
    result = run_evaluate_visual(split="test")
    if "error" in result:
        print("Error:", result)
        return
    print("Visual model (test set):")
    print(f"  n_samples: {result['n_samples']}")
    print(f"  accuracy:  {result['accuracy']:.4f}")
    print(f"  AUC:       {result['auc']:.4f}")
    print(f"  EER:       {result['eer']:.4f}")
    print(f"  precision: {result['precision']:.4f}  recall: {result['recall']:.4f}  F1: {result['f1']:.4f}")


if __name__ == "__main__":
    main()
