"""Draw one data set's benchmark figure in the layout of the 2024 figures
(scDEF-overleaf/figures/{planaria,pbmcs3k,pbmcsifn}_benchmark.pdf).

    python plot.py <data>          (any env with pandas + matplotlib)

Single batch: A cell-type ARI and ASW, B hierarchy accuracy and runtime. Two batches: A cell-type ARI and ASW,
B batch ARI and ASW, C hierarchy accuracy and runtime. The 2024 signature metrics (signature accuracy and sparsity,
hierarchical signature consistency) are in plot_ward.py's supplementary figure only (2026-11-10, pre-declared in
exp-simulation-benchmark-rerun, item 5).
Method colours: matplotlib's tab10 (2026-11-10; NMF and scHPF, Harmony and Scanorama were too alike in the 2024
colours). scDEF comes first (on the two-batch sets, the model with the batch annotations), then the other methods
and, after a gap, the integration methods. scDEF without the batch annotations is scored but not drawn (one workflow;
Pedro, 2026-11-10); the decomposition is read only for batch-specific groups, which the real data have no measure
for, so it is not drawn either (it stays in Figure 4). Runtime: scDEF's bar is the kept
seed; the outline behind it is the five-seed total. Writes <data>_benchmark.{pdf,png} next to scores.csv.
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

import common as C

COLORS = {"scDEF": "#2ca02c", "scDEF_decomposed": "#98df8a", "scDEF_un": "#c7e9c0", "PCA": "#e377c2",
          "NMF": "#1f77b4", "scHPF": "#9467bd", "fscLVM": "#17becf", "nSBM": "#7f7f7f", "scVI": "#ff7f0e",
          "Harmony": "#d62728", "Scanorama": "#8c564b"}
LABELS = {"scDEF_decomposed": "scDEF (decomposed)", "scDEF_un": "scDEF (no batch)", "fscLVM": "f-scLVM"}
ORDER_SINGLE = ["scDEF", "PCA", "NMF", "scHPF", "fscLVM", "nSBM"]
ORDER_MULTI = ["scDEF", "PCA", "NMF", "scHPF", "fscLVM", "nSBM", "scVI", "Harmony", "Scanorama"]
GAP_BEFORE = {"PCA", "scVI"}       # scDEF | the other methods | the integration methods
PANELS_SINGLE = [("Cell Type ARI", "Cell Type ASW"), ("Hierarchy accuracy", "Runtime")]
PANELS_MULTI = [("Cell Type ARI", "Cell Type ASW"), ("Batch ARI", "Batch ASW"), ("Hierarchy accuracy", "Runtime")]
YLABEL = {"Runtime": "Runtime (s)", "Cell Type ARI": "Cell-type ARI", "Cell Type ASW": "Cell-type ASW"}

plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})


def bar_panel(ax, wide, metric, methods):
    xs, labels = [], []
    pos = 0.0
    for j, m in enumerate(methods):
        if j and m in GAP_BEFORE:
            pos += 0.5
        v = wide.at[metric, m] if (metric in wide.index and m in wide.columns) else np.nan
        hatch = None
        if metric == "Runtime" and m.startswith("scDEF") and "Runtime (five seeds)" in wide.index:
            tot = wide.at["Runtime (five seeds)", m] if m in wide.columns else np.nan
            if np.isfinite(tot):
                ax.bar(pos, tot, width=0.8, facecolor="none", edgecolor=COLORS[m], lw=0.8, ls="--")
        if np.isfinite(v):
            col = COLORS.get(m, "0.5")
            ax.bar(pos, v, width=0.8, facecolor="white" if hatch else col, edgecolor=col if hatch else "white",
                   lw=0.6 if hatch else 0.4, hatch=hatch)
        else:
            ax.text(pos, 0, "n/a", ha="center", va="bottom", fontsize=5.5, color="0.4", rotation=90)
        xs.append(pos)
        labels.append(LABELS.get(m, m))
        pos += 1.0
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=90, fontsize=6.5 if len(xs) <= 6 else 5.5)
    ax.set_ylabel(YLABEL.get(metric, metric))
    ax.axhline(0, color="0.2", lw=0.6)
    ax.tick_params(axis="x", length=0)


def main(data):
    data = C.canonical(data)
    spec = C.DATASETS[data]
    d = C.out_dir(data)
    long = pd.read_csv(os.path.join(d, "scores.csv"))
    wide = long.pivot_table(index="metric", columns="method", values="value", aggfunc="first")
    multi = spec["batch"] is not None
    methods = ORDER_MULTI if multi else ORDER_SINGLE
    panels = PANELS_MULTI if multi else PANELS_SINGLE
    nrows = (len(panels) + 1) // 2
    fig = plt.figure(figsize=(7.2, 2.35 * nrows))
    gs = GridSpec(nrows, 2, figure=fig, hspace=0.75, wspace=0.3)
    for i, metrics in enumerate(panels):
        sub = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[i // 2, i % 2], wspace=0.55)
        for j, metric in enumerate(metrics):
            ax = fig.add_subplot(sub[0, j])
            bar_panel(ax, wide, metric, methods)
            if j == 0:
                ax.text(-0.42, 1.08, "ABCDE"[i], transform=ax.transAxes, fontsize=11, fontweight="bold")
    # No title or in-panel notes: this is the paper's supplementary figure, and the caption says that every
    # method was re-run, that scDEF is the saved best-of-five fit, and what the dashed runtime outline is.
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(d, f"{data}_benchmark.{ext}"), dpi=200, bbox_inches="tight")
    print("wrote", os.path.join(d, f"{data}_benchmark.pdf"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("data")
    main(p.parse_args().data)
