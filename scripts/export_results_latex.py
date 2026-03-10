"""
Evaluation sonuclarini LaTeX tablo olarak yazar (paper/results_table.tex).
Kullanim: python scripts/export_results_latex.py [--split test] [--out paper/results_table.tex]
main.tex icinde: \input{results_table} veya tabloyu kopyala.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _fmt(x, ndec=3):
    if x is None:
        return "\\multicolumn{1}{c}{--}"
    return f"{x:.{ndec}f}"


def main():
    p = argparse.ArgumentParser(description="Export results as LaTeX table")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--out", type=str, default=None, help="Output .tex path; default paper/results_table.tex")
    args = p.parse_args()

    from src.evaluation.evaluate_visual import run_evaluate_visual
    from src.evaluation.evaluate_sync import run_evaluate_sync
    from src.evaluation.evaluate_fusion import run_evaluate_fusion

    split = args.split
    res_vis = run_evaluate_visual(split=split)
    res_sync = run_evaluate_sync(split=split)
    res_fus = run_evaluate_fusion(split=split)

    n = res_vis.get("n_samples") or res_sync.get("n_samples") or res_fus.get("n_samples") or "?"

    def row(name, res):
        if res and "error" not in res:
            return f"    {name} & {_fmt(res.get('accuracy'))} & {_fmt(res.get('precision'))} & {_fmt(res.get('recall'))} & {_fmt(res.get('f1'))} & {_fmt(res.get('auc'))} & {_fmt(res.get('eer'))} \\\\"
        return f"    {name} & \\multicolumn{{6}}{{c}}{{--}} \\\\"

    lines = [
        "% Auto-generated. Run: python scripts/export_results_latex.py --split test",
        "\\begin{table}[htbp]",
        "  \\centering",
        f"  \\caption{{Test set results ($n={n}$).}}",
        "  \\label{tab:main}",
        "  \\begin{tabular}{lcccccc}",
        "    \\toprule",
        "    Model & Acc. & Prec. & Rec. & F1 & AUC & EER \\\\",
        "    \\midrule",
        row("Visual only ($S_v$)", res_vis),
        row("Sync only ($S_l$)", res_sync),
        row("Fusion ($S_f$)", res_fus),
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
    ]
    out_path = Path(args.out) if args.out else ROOT / "paper" / "results_table.tex"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
