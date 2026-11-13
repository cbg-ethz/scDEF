"""Supplementary figure: the hierarchical-clustering baselines on the four real data sets (computed by hc.py,
scored by score.py).

    python plot_ward.py          (any env with pandas + matplotlib)

Rows: hierarchy accuracy, signature precision and AUPRC against the marker lists (2026-11-11), signature sparsity
and hierarchical signature consistency; the one figure with signature metrics for the real data. Columns: 3k PBMCs, two donors, interferon, planaria. In each panel, one row per method: its own hierarchy (the 2024 Leiden sweep on its embedding,
filled) joined to Ward linkage on its embedding cut at the scDEF layer sizes (open); Leiden + Ward (the Leiden
clusters of the PCA run, joined by Ward on their mean PCA coordinates) has only the open marker. scDEF (with the
batch annotations on the two-batch data sets) is the green line. Method colours as plot.py. Written to
out/realdata_benchmark/realdata_ward.{pdf,png}.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import common as C
import plot as P

DATA = [("pbmcs3k", "3k PBMCs"), ("pbmcs2b", "Two donors"), ("pbmcsifn", "Interferon"), ("planaria", "Planaria")]
METRICS = [("Hierarchy accuracy", "Hierarchy accuracy"), ("Signature precision", "Signature precision (top 20)"),
           ("Signature AUPRC", "Signature AUPRC"), ("Signature sparsity", "Signature sparsity"),
           ("Hierarchical signature consistency", "Hierarchical signature\nconsistency")]
ROWS = ["PCA", "NMF", "scHPF", "fscLVM", "scVI", "Harmony", "Scanorama", "Leiden"]
LEIDEN_COLOUR = "0.15"

plt.rcParams.update({"font.size": 6.5, "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6.3,
                     "axes.titlesize": 7, "axes.linewidth": 0.5, "xtick.major.width": 0.5,
                     "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 0, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False})


def scores(data):
    s = pd.read_csv(os.path.join(C.out_dir(data), "scores.csv"))
    return s.pivot_table(index="method", columns="metric", values="value")


def panel(ax, s, metric, names):
    y = np.arange(len(ROWS))[::-1]
    for m, yy in zip(ROWS, y):
        col = LEIDEN_COLOUR if m == "Leiden" else P.COLORS[m]
        own = s.at[m, metric] if m != "Leiden" and m in s.index else np.nan
        ward = s.at[f"{m}_Ward", metric] if f"{m}_Ward" in s.index else np.nan
        if np.isfinite(own) and np.isfinite(ward):
            ax.plot([own, ward], [yy, yy], color=col, lw=0.8, zorder=1)
        if np.isfinite(own):
            ax.plot(own, yy, "o", ms=3.6, mfc=col, mec=col, zorder=2)
        if np.isfinite(ward):
            ax.plot(ward, yy, "o", ms=3.6, mfc="white", mec=col, mew=0.9, zorder=3)
    ax.axvline(s.at["scDEF", metric], color=P.COLORS["scDEF"], lw=1.0, zorder=0)
    ax.set_yticks(y, [P.LABELS.get(m, m) if m != "Leiden" else "Leiden" for m in ROWS] if names else [])
    ax.set_ylim(-0.7, len(ROWS) - 0.3)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax.grid(axis="x", color="0.9", lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def main():
    S = {d: scores(d) for d, _ in DATA}
    fig, axs = plt.subplots(len(METRICS), len(DATA), figsize=(7.2, 8.0), squeeze=False)
    fig.subplots_adjust(left=0.1, right=0.99, top=0.96, bottom=0.09, wspace=0.16, hspace=0.45)
    for r, (metric, ylab) in enumerate(METRICS):
        for c, (d, title) in enumerate(DATA):
            ax = axs[r, c]
            panel(ax, S[d], metric, names=c == 0)
            if r == 0:
                ax.set_title(title, pad=4)
            ax.set_xlabel(ylab.replace("\n", " "), labelpad=1.5)
        fig.text(0.008, axs[r, 0].get_position().y1 + 0.01, "ABCDE"[r], fontsize=10, fontweight="bold", va="bottom")
    handles = [Line2D([], [], ls="none", marker="o", ms=3.6, mfc="0.4", mec="0.4", label="hierarchy of the method"),
               Line2D([], [], ls="none", marker="o", ms=3.6, mfc="white", mec="0.4", mew=0.9,
                      label="Ward linkage, cut at the scDEF layer sizes"),
               Line2D([], [], color=P.COLORS["scDEF"], lw=1.0, label="scDEF")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=6.3, bbox_to_anchor=(0.55, 0.0))
    out = os.path.join(C.OUT_ROOT if hasattr(C, "OUT_ROOT") else os.path.dirname(C.out_dir("pbmcs3k")))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"realdata_ward.{ext}"), dpi=250)
    print("wrote", os.path.join(out, "realdata_ward.pdf"))


if __name__ == "__main__":
    main()
