"""
Mimari figürü PNG olarak uretir (matplotlib).
Makale icin: paper/figures/architecture.png
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    print("matplotlib gerekli: pip install matplotlib")
    sys.exit(1)


def main():
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")

    def box(x, y, w, h, text, fontsize=9):
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", facecolor="white", edgecolor="black", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.2))

    # Video
    box(4, 11, 2, 0.5, "Video")
    arrow(5, 11, 5, 10.2)
    box(3.5, 9.5, 3, 0.5, "Frame extraction")

    # Face -> CNN -> S_v
    box(1.5, 8.2, 1.8, 0.45, "Face detection")
    arrow(4, 9.75, 2.4, 8.42)
    box(1.5, 7.2, 1.8, 0.45, "CNN")
    arrow(2.4, 8.2, 2.4, 7.65)
    box(1.5, 6.2, 1.8, 0.45, "S_v")
    arrow(2.4, 7.2, 2.4, 6.65)

    # Mouth -> Temporal -> M_video
    box(4.2, 8.2, 1.6, 0.45, "Mouth ROI")
    arrow(5, 9.5, 5, 8.65)
    box(4.2, 7.2, 1.6, 0.45, "Temporal model")
    arrow(5, 8.2, 5, 7.65)
    box(4.2, 6.2, 1.6, 0.45, "M_video")
    arrow(5, 7.2, 5, 6.65)

    # Audio -> ASR -> Transcript -> E_text
    box(6.8, 8.2, 1.5, 0.45, "Audio")
    arrow(5.5, 9.5, 7.55, 8.42)
    box(6.8, 7.2, 1.5, 0.45, "Whisper ASR")
    arrow(7.55, 8.2, 7.55, 7.65)
    box(6.8, 6.2, 1.5, 0.45, "Transcript")
    arrow(7.55, 7.2, 7.55, 6.65)
    box(6.8, 5.2, 1.5, 0.45, "E_text")
    arrow(7.55, 6.2, 7.55, 5.65)

    # S_l
    box(3.2, 3.8, 3.6, 0.5, "S_l = 1 - cos(E_text, M_video)", fontsize=8)
    arrow(5, 6.2, 5, 4.5)
    arrow(7.55, 5.2, 5.8, 4.05)

    # S_f
    box(3.5, 2.5, 3, 0.5, "S_f = alpha*S_v + (1-alpha)*S_l", fontsize=8)
    arrow(2.4, 6.2, 2.4, 3.5)
    arrow(5, 3.8, 5, 3.05)

    out = ROOT / "paper" / "figures" / "architecture.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Yazildi: {out}")


if __name__ == "__main__":
    main()
