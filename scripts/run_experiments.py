"""
Makale deney pipeline'i: tum modelleri test setinde degerlendirir, metrikleri sozluk ve tablo uretir.
Cikti: accuracy, precision, recall, f1, auc, eer (sklearn uyumlu isimler).
Kullanim: python scripts/run_experiments.py [--split test] [--out results.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(split: str | None = None, out: str | None = None):
    p = argparse.ArgumentParser(description="Deney pipeline: visual, sync, fusion metrikleri")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--out", type=str, default=None, help="JSON cikti (opsiyonel)")
    parsed = p.parse_args()
    split = split if split is not None else parsed.split
    out = out if out is not None else parsed.out

    from src.evaluation.evaluate_visual import run_evaluate_visual
    from src.evaluation.evaluate_sync import run_evaluate_sync
    from src.evaluation.evaluate_fusion import run_evaluate_fusion

    results = {}
    for name, fn in [("visual", run_evaluate_visual), ("sync", run_evaluate_sync), ("fusion", run_evaluate_fusion)]:
        res = fn(split=split)
        if "error" in res:
            results[name] = {"error": res["error"]}
            continue
        results[name] = {
            "accuracy": res.get("accuracy"),
            "precision": res.get("precision"),
            "recall": res.get("recall"),
            "f1": res.get("f1"),
            "auc": res.get("auc"),
            "eer": res.get("eer"),
            "n_samples": res.get("n_samples"),
        }

    print("Results (paper table source):")
    print(json.dumps(results, indent=2, ensure_ascii=False))

    if out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Written: {out_path}")

    # Tablo da uret
    from scripts.export_results_table import main as export_main
    sys.argv = ["export_results_table.py", "--split", split, "--out", str(ROOT / "paper" / "results_table.md")]
    export_main()


if __name__ == "__main__":
    main()
