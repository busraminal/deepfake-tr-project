"""
Fusion alpha ablasyonu: alpha in {0.25, 0.5, 0.75} icin Fusion metrikleri.
Sync + Visual checkpoint gerekir. Cikti: konsol tablosu + istege bagli paper/ablation_alpha.tex
Kullanim: python scripts/run_ablation_alpha.py [--split test] [--out paper/ablation_alpha.tex]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description="Fusion alpha ablation")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--out", type=str, default=None, help="LaTeX table output path")
    p.add_argument("--alphas", type=str, default="0.25,0.5,0.75", help="Comma-separated alpha values")
    args = p.parse_args()
    alphas = [float(x.strip()) for x in args.alphas.split(",")]

    from src.evaluation.evaluate_fusion import run_evaluate_fusion

    rows = []
    for alpha in alphas:
        res = run_evaluate_fusion(alpha=alpha, split=args.split)
        if "error" in res:
            rows.append((alpha, None, None, None))
            continue
        rows.append((alpha, res.get("accuracy"), res.get("auc"), res.get("eer")))

    # Markdown table
    print("Alpha ablation (Fusion):")
    print("| Alpha | Accuracy | AUC | EER |")
    print("|-------|----------|-----|-----|")
    for alpha, acc, auc, eer in rows:
        a = f"{acc:.4f}" if acc is not None else "--"
        b = f"{auc:.4f}" if auc is not None else "--"
        c = f"{eer:.4f}" if eer is not None else "--"
        print(f"| {alpha} | {a} | {b} | {c} |")

    # LaTeX
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "% Fusion alpha ablation. Run: python scripts/run_ablation_alpha.py --out paper/ablation_alpha.tex",
            "\\begin{table}[htbp]",
            "  \\centering",
            "  \\caption{Fusion ablation: effect of $\\alpha$ in $S_f = \\alpha S_v + (1-\\alpha) S_l$.}",
            "  \\label{tab:ablation_alpha}",
            "  \\begin{tabular}{lccc}",
            "    \\toprule",
            "    $\\alpha$ & Acc. & AUC & EER \\\\",
            "    \\midrule",
        ]
        for alpha, acc, auc, eer in rows:
            if acc is None:
                lines.append(f"    {alpha} & \\multicolumn{{3}}{{c}}{{--}} \\\\")
            else:
                lines.append(f"    {alpha} & {acc:.3f} & {auc:.3f} & {eer:.3f} \\\\")
        lines.append("    \\bottomrule")
        lines.append("  \\end{tabular}")
        lines.append("\\end{table}")
        out_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
