"""Regenerate the README figures (needs matplotlib). Numbers are aggregate results only."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).parent / "figures"
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983", "#e6e5e0"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2})


def score_chart():
    labels = ["From-scratch\n3D CNN", "DINOv2-small\n3 slices, 1 model", "DINOv2-small\n12 slices, 5 folds",
              "Blend with public\nmodels (final)"]
    public, private = [0.661, 0.820, 0.895, 0.943], [0.633, 0.807, 0.879, 0.927]
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=200, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    x = range(len(labels))
    ax.plot(x, public, color=BLUE, lw=2, marker="o", ms=8, mec=SURFACE, mew=2, label="Public leaderboard", zorder=3)
    ax.plot(x, private, color=ORANGE, lw=2, marker="s", ms=7.5, mec=SURFACE, mew=2, label="Private leaderboard (final ranking)", zorder=3)
    for i in x:
        if i == len(labels) - 1:  # right of the marker: clear of the winner's reference line
            ax.annotate(f"{public[i]:.3f}", (i, public[i]), xytext=(11, 1), textcoords="offset points", va="center", fontsize=10, color=INK)
            ax.annotate(f"{private[i]:.3f}", (i, private[i]), xytext=(11, -3), textcoords="offset points", va="center", fontsize=10, color=INK)
            continue
        ax.annotate(f"{public[i]:.3f}", (i, public[i]), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=10, color=INK)
        ax.annotate(f"{private[i]:.3f}", (i, private[i]), xytext=(0, -18), textcoords="offset points", ha="center", fontsize=10, color=INK)
    ax.axhline(0.960, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(-0.3, 0.963, "Winning private score, 0.960", fontsize=9, color=INK2, va="bottom")
    ax.set_xticks(list(x), labels, fontsize=10)
    ax.set_xlim(-0.35, 3.5); ax.set_ylim(0.58, 1.0)
    ax.set_ylabel("Macro ROC AUC", fontsize=10)
    ax.yaxis.grid(True, color=GRID, lw=1); ax.set_axisbelow(True)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID); ax.tick_params(length=0)
    ax.set_title("Score across the four main submissions", loc="left", fontsize=13, fontweight="bold", pad=26)
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "score_progression.png", facecolor=SURFACE); plt.close(fig)


def box(ax, x, y, w, h, title, sub="", fill="#f1f0ec", edge="#c9c8c1"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12", fc=fill, ec=edge, lw=1.2))
    ty = y + h / 2 + (0.17 if sub else 0)
    ax.text(x + w / 2, ty, title, ha="center", va="center", fontsize=10.5, fontweight="bold", color=INK)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.2, sub, ha="center", va="center", fontsize=8.6, color=INK2, linespacing=1.25)


def arrow(ax, a, b):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.3, shrinkA=0, shrinkB=0))


def pipeline():
    fig, ax = plt.subplots(figsize=(12, 5.6), dpi=200, facecolor=SURFACE)
    ax.set_xlim(0, 12.6); ax.set_ylim(0, 6.2); ax.axis("off")
    H = 1.25; W = 2.6; TOP = 4.5; MID = 2.5; LOW = 0.45
    ax.text(0.1, 6.0, "Our model, trained five times on different folds", fontsize=11, color=INK2, fontweight="bold")
    box(ax, 0.1, TOP, W, H, "Knee MRI study", "DICOM series, 3 to 14 per study")
    box(ax, 3.3, TOP, W, H, "Sequence slots", "up to 6, chosen from headers\nleft-knee convention, 130 mm crop")
    box(ax, 6.5, TOP, W, H, "Depth zones", "4 zones x 3 adjacent slices\n= 12 slices per slot, 336 px")
    box(ax, 9.7, TOP, W, H, "DINOv2-small", "one RGB image per (slot, zone)\nlast 6 of 12 blocks fine-tuned")
    for a in (2.7, 5.9, 9.1): arrow(ax, (a, TOP + H / 2), (a + 0.6, TOP + H / 2))
    box(ax, 9.7, MID, W, H, "Per-finding attention", "12 queries over (slot, zone) tokens\none score per finding")
    arrow(ax, (11, TOP), (11, MID + H))
    box(ax, 3.3, MID, W, H, "Report labels", "public LLM labels, weighted by\nconfidence; expert labels at 3x", fill="#e8f0fb", edge="#a9c6ec")
    ax.annotate("", xy=(9.7, MID + H / 2), xytext=(5.9, MID + H / 2),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.3, ls=(0, (4, 3))))
    ax.text(7.8, MID + H / 2 + 0.14, "training targets only", ha="center", fontsize=8.8, color=INK2)
    box(ax, 9.7, LOW, W, H, "Five-fold ensemble", "rank mean of 5 models\npublic LB 0.895, private 0.879")
    arrow(ax, (11, MID), (11, LOW + H))
    box(ax, 5.2, LOW, W + 0.2, H, "Final rank blend", "75% public, 25% ours\npublic LB 0.943, private 0.927", fill="#fdeee7", edge="#f0b89f")
    arrow(ax, (9.7, LOW + H / 2), (8.0, LOW + H / 2))
    box(ax, 0.9, LOW, W + 0.2, H, "Public ensemble", "others' published models:\nDINOv2, RadImageNet, CoAtNet")
    arrow(ax, (3.7, LOW + H / 2), (5.2, LOW + H / 2))
    fig.subplots_adjust(0.005, 0.005, 0.995, 0.995)
    fig.savefig(OUT / "pipeline.png", facecolor=SURFACE); plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    score_chart(); pipeline()
