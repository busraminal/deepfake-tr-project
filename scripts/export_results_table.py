"""
Değerlendirme sonuçlarını makale tablosu formatında çıkarır.
Çalıştırma: python scripts/export_results_table.py [--split test] [--out results_table.md]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description="Export evaluation table for paper")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--out", type=str, default=None, help="Output path (.md or .csv); default: stdout")
    args = p.parse_args()

    from src.evaluation.evaluate_visual import run_evaluate_visual
    from src.evaluation.evaluate_sync import run_evaluate_sync
    from src.evaluation.evaluate_fusion import run_evaluate_fusion

    split = args.split
    rows = []

    res_vis = run_evaluate_visual(split=split)
    if "error" not in res_vis:
        rows.append(("Visual only (S_v)", res_vis.get("accuracy"), res_vis.get("precision"), res_vis.get("recall"), res_vis.get("f1"), res_vis.get("auc"), res_vis.get("eer")))
    else:
        rows.append(("Visual only (S_v)", None, None, None, None, None, None))

    res_sync = run_evaluate_sync(split=split)
    if "error" not in res_sync:
        rows.append(("Sync only (S_l)", res_sync.get("accuracy"), res_sync.get("precision"), res_sync.get("recall"), res_sync.get("f1"), res_sync.get("auc"), res_sync.get("eer")))
    else:
        rows.append(("Sync only (S_l)", None, None, None, None, None, None))

    res_fus = run_evaluate_fusion(split=split)
    if "error" not in res_fus:
        rows.append(("Fusion (S_f)", res_fus.get("accuracy"), res_fus.get("precision"), res_fus.get("recall"), res_fus.get("f1"), res_fus.get("auc"), res_fus.get("eer")))
    else:
        rows.append(("Fusion (S_f)", None, None, None, None, None, None))

    def fmt(x):
        return f"{x:.4f}" if x is not None else "—"

    lines = [
        "| Model | Accuracy | Precision | Recall | F1 | AUC | EER |",
        "|-------|----------|-----------|--------|-----|-----|-----|",
    ]
    for row in rows:
        name = row[0]
        vals = row[1:]
        lines.append(f"| {name} | {fmt(vals[0])} | {fmt(vals[1])} | {fmt(vals[2])} | {fmt(vals[3])} | {fmt(vals[4])} | {fmt(vals[5])} |")

    table = "\n".join(lines)
    n = res_vis.get("n_samples") or res_sync.get("n_samples") or res_fus.get("n_samples") or "?"
    header = f"# Results ({args.split} set, n={n})\n\n"
    out_text = header + table + "\n"

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(out_text, encoding="utf-8")
        print(f"Written: {path}")
    else:
        print(out_text)


if __name__ == "__main__":
    main()
