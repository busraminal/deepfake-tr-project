"""
Skor dağılımı teşhisi: real vs fake skorlarının histogramı.
Threshold 0.5 uygun değilse ROC Youden optimal eşik kullanılır (metrics.py).
Bu script dağılımı görselleştirir: plt.hist(scores_real), plt.hist(scores_fake).
Kullanım: python scripts/plot_score_histograms.py [--split test] [--out paper/figures/score_hist.png]
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description="Plot score histograms (real vs fake) for Visual, Sync, Fusion")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--out", type=str, default=None, help="Save figure path (default: paper/figures/score_hist.png)")
    args = p.parse_args()

    import numpy as np
    from src.evaluation.evaluate_visual import run_evaluate_visual
    from src.evaluation.evaluate_sync import run_evaluate_sync
    from src.evaluation.evaluate_fusion import run_evaluate_fusion
    from src.evaluation.metrics import optimal_threshold_youden

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("pip install matplotlib")
        return

    out_path = Path(args.out) if args.out else ROOT / "paper" / "figures" / "score_hist.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    configs = [
        ("Visual ($S_v$)", run_evaluate_visual),
        ("Sync ($S_l$)", run_evaluate_sync),
        ("Fusion ($S_f$)", lambda **kw: run_evaluate_fusion(alpha=0.5, **kw)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, (title, fn) in zip(axes, configs):
        res = fn(split=args.split, return_scores=True)
        if "error" in res or "y_true" not in res:
            ax.text(0.5, 0.5, f"{title}\n(res unavailable)", ha="center", va="center")
            ax.set_xlim(0, 1)
            continue
        y_true = np.asarray(res["y_true"])
        y_score = np.asarray(res["y_score"])
        scores_real = y_score[y_true == 0]
        scores_fake = y_score[y_true == 1]
        opt_t = optimal_threshold_youden(y_true, y_score)
        ax.hist(scores_real, bins=15, alpha=0.6, label="real", color="green", density=True)
        ax.hist(scores_fake, bins=15, alpha=0.6, label="fake", color="red", density=True)
        ax.axvline(0.5, color="gray", linestyle="--", label="0.5")
        ax.axvline(opt_t, color="black", linestyle="-", label=f"Youden={opt_t:.2f}")
        ax.set_xlabel("Score")
        ax.set_legend(fontsize=8)
        ax.set_title(title)
        ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
