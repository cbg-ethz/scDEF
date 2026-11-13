"""Schematic of the simulated data (Supplementary Figure fig:simulation_procedure), redrawn 2026-11-12 to match
simulate.R: the 2024 drawing showed 1,000 cells for the single batch (4,000 in the benchmark) and two of the four
non-shared groups removed per batch (simulate.R keeps each of them in exactly one batch, and the shared groups are
always the four under one top-level group, the first four of the sorted group list).

    python simulation_procedure.py   -> figures/simulation_procedure.{pdf,png}
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

import simcommon as S  # noqa: E402

plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42})
TOP = ["#2f5fb3", "#9db4e0"]                                  # 2 groups
MID = ["#f6c6a8", "#e8792b", "#c0561a", "#f2a77a"]            # 4 groups
LOW = ["#e3f1dd", "#bfe0b0", "#8cc479", "#4f8f3f", "#2f5f27", "#5f9e4c", "#93d07c", "#b8f08f"]   # 8 groups
GREY = "0.55"


def tree(ax, x0, top, mids, lows, labels):
    """One top-level group with its two middle and four bottom groups."""
    pos_top = (x0 + 1.5, 2.0)
    pos_mid = [(x0 + 0.5, 1.0), (x0 + 2.5, 1.0)]
    pos_low = [(x0 + i, 0.0) for i in range(4)]
    for k, pm in enumerate(pos_mid):
        ax.plot([pos_top[0], pm[0]], [pos_top[1], pm[1]], color="black", lw=1.2, zorder=1)
        for pl in pos_low[2 * k:2 * k + 2]:
            ax.plot([pm[0], pl[0]], [pm[1], pl[1]], color="black", lw=1.2, zorder=1)
    for (x, y), c, t in [(pos_top, top, labels[0])] + list(zip(pos_mid, mids, labels[1])) + list(zip(pos_low, lows, labels[2])):
        ax.scatter([x], [y], s=150, color=c, edgecolor="0.25", lw=0.6, zorder=2)
        ax.text(x, y, t, ha="center", va="center", fontsize=6.5, zorder=3,
                color="white" if c in (TOP[0], MID[2], LOW[4], LOW[3]) else "black")


def matrix(ax, keep, title=None, ylabel="", xlabel="1,750 genes"):
    """Rows: the eight bottom groups in order; columns: genes 1-1000 (top level), 1001-1500 (middle), 1501-1750
    (bottom), coloured by the group of each row at that level; removed groups in grey."""
    widths = [1000, 500, 250]
    x_edges = [0, 1000, 1500, 1750]
    for g in range(8):
        y = 7 - g
        if not keep[g]:
            ax.add_patch(Rectangle((0, y), 1750, 1, color=GREY, lw=0))
            continue
        for b, (x, w) in enumerate(zip(x_edges[:-1], widths)):
            c = [TOP[g // 4], MID[g // 2], LOW[g]][b]
            ax.add_patch(Rectangle((x, y), w, 1, color=c, lw=0))
    ax.set_xlim(0, 1750)
    ax.set_ylim(0, 8)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel(xlabel, fontsize=6)
    ax.set_ylabel(ylabel, fontsize=6)
    if title:
        ax.set_title(title, fontsize=8, pad=3)


def main():
    fig = plt.figure(figsize=(7.2, 3.9))
    axA = fig.add_axes([0.03, 0.56, 0.52, 0.38])
    tree(axA, 0, TOP[0], MID[:2], LOW[:4], ("1", ("1", "2"), ("1", "2", "3", "4")))
    tree(axA, 4.6, TOP[1], MID[2:], LOW[4:], ("2", ("3", "4"), ("5", "6", "7", "8")))
    axA.text(-0.7, 2.0, "2 groups", ha="right", va="center", fontsize=6)
    axA.text(-0.7, 1.0, "4 groups", ha="right", va="center", fontsize=6)
    axA.text(-0.7, 0.0, "8 groups", ha="right", va="center", fontsize=6)
    axA.set_xlim(-2.2, 8.4)
    axA.set_ylim(-0.4, 2.4)
    axA.axis("off")

    axB = fig.add_axes([0.66, 0.56, 0.17, 0.33])
    matrix(axB, [True] * 8, title="Gene expression matrix", ylabel="4,000 cells")

    fig.text(0.03, 0.40, "Four distinct batches", fontsize=7, fontweight="bold")
    fig.text(0.03, 0.32, "Shared groups, kept in every batch:\nthe four under top-level group 1", fontsize=6.3, va="top")
    fig.text(0.03, 0.17, "Each of the other four groups is\nkept in one batch only", fontsize=6.3, va="top")
    specific = [4, 5, 6, 7]                                   # one per batch, drawn at random in each data set
    for b in range(4):
        ax = fig.add_axes([0.31 + b * 0.172, 0.07, 0.14, 0.3])
        keep = [g < 4 or g == specific[b] for g in range(8)]
        matrix(ax, keep, title=f"Batch {b + 1}", ylabel="1,000 cells" if b == 0 else "")
    for x, y, t in ((0.01, 0.95, "A"), (0.60, 0.95, "B"), (0.01, 0.42, "C")):
        fig.text(x, y, t, fontsize=10, fontweight="bold", va="bottom")
    out = os.path.join(S.OUT_ROOT, "figures")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"simulation_procedure.{ext}"), dpi=200)
    print("wrote", os.path.join(out, "simulation_procedure.pdf"))


if __name__ == "__main__":
    main()
