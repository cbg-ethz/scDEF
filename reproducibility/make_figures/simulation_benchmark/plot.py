"""Draw the simulation benchmark figures from the gathered tables (run gather.py first).

    python plot.py          (any env with pandas + matplotlib)

- ``sims_levels.{pdf,png}`` (Figure 2 candidate): A level recovery and B level signatures for each true level
  (8, 4 and 2 groups; rows), per scenario (columns: single batch, four similar batches, four distinct batches),
  low and high group similarity on x, one box per method over the five replicates, including the two hierarchical
  baselines (Leiden + Ward, one-layer scDEF + Ward) and, with four batches, the decomposition; C the layer sizes of
  the kept scDEF models (every replicate, marker per layer) against the true 8, 4 and 2 groups (dashed); D shared and
  batch-specific groups with four batches (specific groups kept, of four, with distinct batches; shared groups split
  by batch, with similar and with distinct batches).
- ``sims_2024_metrics.{pdf,png}`` (supplementary figure): the 2024 metrics in the layout of the 2024 figures
  (scDEF-overleaf/figures/sims_main.pdf, sims_supp.pdf): metric rows, scenario columns, Low/High on x, boxes per
  method; naming fix on. ``sims_2024_metrics_verbatim.{pdf,png}``: the same with the 2024 code verbatim (for review).
- ``sims_ward.{pdf,png}`` (supplementary figure): Ward on each method's embedding against scDEF: level recovery,
  level signatures, hierarchy accuracy and hierarchical signature consistency.

Method colours and labels as realdata_benchmark/plot.py; the decomposition and the Ward-on-embedding variants are
hatched in their method's colour. No figure titles; 7.2 in wide, under a page tall. Written to
out/simulation_benchmark/figures/.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import simcommon as S

rdp = S._load("rd_plot", "plot.py")
COLORS = dict(rdp.COLORS)
COLORS.update({"Leiden_Ward": "0.15", "scDEF_1L_Ward": "#bcbd22", S.DECOMPOSED: rdp.COLORS["scDEF_decomposed"]})
LABELS = dict(rdp.LABELS)                      # scDEF_decomposed -> "scDEF (decomposed)", fscLVM -> "f-scLVM"
LABELS.update({"Leiden_Ward": "Leiden + Ward", "scDEF_1L_Ward": "1-layer scDEF + Ward",
               S.DECOMPOSED: "scDEF (decomposed)"})
HC = ["scDEF_1L_Ward", "Leiden_Ward"]
ORDER = {"single": list(rdp.ORDER_SINGLE) + HC, "similar": list(rdp.ORDER_MULTI) + HC}   # scDEF, then the rest
ORDER["distinct"] = ORDER["similar"]
WARD = {sc: ["scDEF"] + [f"{m}_Ward" for m in S.WARD_EMBEDDINGS["multi" if sc != "single" else "single"]] + HC
        for sc in S.SCENARIOS}
GAP_BEFORE = {"scDEF_1L_Ward", "PCA", "scVI"}   # scDEF | the other methods | the integration methods | baselines
LEGEND_ORDER = list(rdp.ORDER_MULTI) + HC     # scDEF without the batch annotations is scored but not drawn
SIMS = ["Low", "High"]
SCENARIOS = list(S.SCENARIOS)
LEVEL_LABEL = {"GroupC": "8 groups", "GroupB": "4 groups", "GroupA": "2 groups"}
METRICS_2024 = ["Cell Type ARI", "Cell Type ASW", "Batch ARI", "Batch ASW",
                "Signature sparsity", "Hierarchy accuracy", "Hierarchical signature consistency", "Runtime"]
# (signature accuracy left out, 2026-11-11: the signatures are scored by level signatures and their AUPRC)
YLABEL = {"Hierarchical signature consistency": "Hierarchical sig.\nconsistency", "Runtime": "Runtime (s)",
          "Signature accuracy": "Signature\naccuracy", "Signature sparsity": "Signature\nsparsity",
          "Hierarchy accuracy": "Hierarchy\naccuracy"}
LAYER_MARKERS = ["o", "s", "^", "D", "v", "P"]
BOUNDED = {"Cell Type ARI", "Batch ARI", "Signature accuracy", "Signature sparsity", "Hierarchy accuracy",
           "Hierarchical signature consistency"}

plt.rcParams.update({"font.size": 6.5, "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6,
                     "axes.titlesize": 7, "axes.linewidth": 0.5, "xtick.major.width": 0.5,
                     "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
                     "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
                     "hatch.linewidth": 0.5})


def base(m):
    return m[:-5] if m.endswith("_Ward") and m not in HC else m


def label(m):
    if m.endswith("_Ward") and m not in HC:
        return LABELS.get(base(m), base(m)) + " + Ward"
    return LABELS.get(m, m)


def style(m):
    """Box style of the 2024 figures: outline, whiskers, median and outliers in the method's colour, the box filled
    with the same colour at half opacity; the decomposition and the Ward-on-embedding variants hatched."""
    col = COLORS.get(base(m), COLORS.get(m, "0.5"))
    if m.endswith("_Ward") and m not in HC:
        return dict(facecolor="white", edgecolor=col, hatch="//////", lw=0.7), col
    return dict(facecolor=matplotlib.colors.to_rgba(col, 0.45), edgecolor=col, lw=0.7), col


def slots(methods, gaps=None):
    """x offsets (in box widths) of the methods: a gap of 0.6 box after scDEF, before the integration methods and
    before the hierarchical-clustering baselines."""
    gaps = set(GAP_BEFORE if gaps is None else gaps)
    x, out = 0.0, []
    for j, m in enumerate(methods):
        if j and m in gaps:
            x += 0.6
        out.append(x)
        x += 1.0
    out = np.array(out)
    return out - (out[-1] if len(out) else 0) / 2, (out[-1] + 1.0 if len(out) else 1.0)


def boxes(ax, df, methods, ylim=None, log=False, gaps=None, group_width=0.9):
    off, span = slots(methods, gaps)
    w = group_width / span
    for i, sim in enumerate(SIMS):
        for j, m in enumerate(methods):
            v = df.loc[(df["similarity"] == sim) & (df["method"] == m), "value"].astype(float)
            v = v[np.isfinite(v)]
            if not len(v):
                continue
            pos = i + off[j] * w
            bp, col = style(m)
            ax.boxplot(v, positions=[pos], widths=w * 0.82, patch_artist=True, manage_ticks=False,
                       boxprops=bp, medianprops=dict(color=col, lw=0.8), whiskerprops=dict(color=col, lw=0.7),
                       capprops=dict(color=col, lw=0.7),
                       flierprops=dict(marker="d", ms=2.2, mfc=col, mec=col, mew=0))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(SIMS)
    ax.set_xlim(-0.5, 1.5)
    if log:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    if ylim is not None:
        ax.set_ylim(*ylim)


def legend(fig, methods, y, extra=(), ncol=7):
    handles = []
    for m in methods:
        bp, _ = style(m)
        handles.append(Patch(label=label(m), **bp))
    handles += list(extra)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, y), ncol=min(len(handles), ncol),
               frameon=False, fontsize=6.3, handlelength=1.3, columnspacing=1.0)


def letter(ax, s):
    ax.text(-0.36, 1.1, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


def present(df, order=LEGEND_ORDER):
    have = set(df["method"].unique())
    return [m for m in order if m in have]


def read(name):
    path = os.path.join(S.OUT_ROOT, name)
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(
        columns=["scenario", "similarity", "method", "metric", "value", "true_level", "sizes", "rep"])


def layer_panel(ax, sizes, scenario, ymax):
    sub = sizes[sizes["scenario"] == scenario]
    methods = ["scDEF"]
    for t in (8, 4, 2):
        ax.axhline(t, color="0.55", lw=0.5, ls="--", zorder=0)
    n = len(methods)
    for i, sim in enumerate(SIMS):
        for j, m in enumerate(methods):
            rows = sub[(sub["similarity"] == sim) & (sub["method"] == m)].sort_values("rep")
            x0 = i + (j - (n - 1) / 2) * 0.36
            for r, (_, row) in enumerate(rows.iterrows()):
                x = x0 + (r - 2) * 0.05
                for k, s in enumerate(json.loads(row["sizes"])):
                    ax.plot(x, s, LAYER_MARKERS[k % len(LAYER_MARKERS)], ms=2.3, mfc=COLORS[m], mec="0.2",
                            mew=0.3, zorder=3)
    ax.set_yscale("log", base=2)
    ax.set_yticks([1, 2, 4, 8, 16, 32, 64])
    ax.set_yticklabels(["1", "2", "4", "8", "16", "32", "64"])
    ax.set_ylim(0.8, max(ymax * 1.25, 12))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(SIMS)
    ax.set_xlim(-0.5, 1.5)


def fig_levels(out):
    lv = read("level_scores_all.csv")
    sizes = read("layer_sizes_all.csv")
    sizes = sizes[sizes["method"].astype(str).isin(["scDEF", "scDEF_un"])]
    sc_ = read("scores_all.csv")
    rows = ([("Level recovery", tl) for tl in S.HIERARCHY_OBS] + [("Level signatures", tl) for tl in S.HIERARCHY_OBS]
            + [("Level signatures AUPRC", tl) for tl in S.HIERARCHY_OBS])
    fig = plt.figure(figsize=(7.2, 10.4))
    gs = GridSpec(len(rows) + 2, 3, figure=fig, hspace=0.62, wspace=0.28,
                  height_ratios=[1] * len(rows) + [1.1, 1.0], top=0.975, bottom=0.115, left=0.095, right=0.99)
    ymax = max([max(json.loads(s)) for s in sizes["sizes"]] + [8]) if len(sizes) else 8
    for r, (metric, tl) in enumerate(rows):
        for c, scen in enumerate(SCENARIOS):
            ax = fig.add_subplot(gs[r, c])
            sub = lv[(lv["metric"] == metric) & (lv["true_level"] == tl) & (lv["scenario"] == scen)]
            boxes(ax, sub, ORDER[scen], ylim=(-0.03, 1.03))
            if r == 0:
                ax.set_title(S.SCENARIOS[scen]["label"], pad=4)
            if c == 0:
                short = {"Level recovery": "Recovery", "Level signatures": "Precision, top 20",
                         "Level signatures AUPRC": "AUPRC"}[metric]
                ax.set_ylabel(f"{short}\n({LEVEL_LABEL[tl]})")
                if r in (0, 3, 6):
                    letter(ax, "ABC"[r // 3])
            ax.set_xticklabels([])
    for c, scen in enumerate(SCENARIOS):
        ax = fig.add_subplot(gs[len(rows), c])
        layer_panel(ax, sizes, scen, ymax)
        ax.set_xticklabels([])
        if c == 0:
            ax.set_ylabel("scDEF factors\nper layer")
            letter(ax, "D")
    panels = [("Specific groups kept", "distinct", (-0.2, 4.2), "Specific groups kept\n(of 4; distinct batches)"),
              ("Shared groups split by batch", "similar", (-0.03, 1.03), "Shared groups split\n(similar batches)"),
              ("Shared groups split by batch", "distinct", (-0.03, 1.03), "Shared groups split\n(distinct batches)")]
    for c, (metric, scen, ylim, ylab) in enumerate(panels):
        ax = fig.add_subplot(gs[len(rows) + 1, c])
        sub = sc_[(sc_["metric"] == metric) & (sc_["scenario"] == scen)]
        boxes(ax, sub, ORDER[scen], ylim=ylim)
        ax.set_ylabel(ylab)
        ax.set_xlabel("Group similarity")
        if c == 0:
            letter(ax, "E")
    nl = int(max([len(json.loads(s)) for s in sizes["sizes"]] + [1])) if len(sizes) else 1
    extra = [Line2D([], [], ls="none", marker=LAYER_MARKERS[k], ms=3, mfc="white", mec="0.2", mew=0.5,
                    label=f"layer {k}") for k in range(nl)]
    legend(fig, present(lv), 0.0, extra, ncol=8)
    save(fig, out, "sims_levels")


def row_limits(v, metric):
    """One y range per metric row, so that the three scenarios read on the same scale."""
    v = v.astype(float)
    v = v[np.isfinite(v)]
    if not len(v):
        return None
    if metric == "Runtime":
        return (10 ** np.floor(np.log10(max(v.min(), 1e-3))), 10 ** np.ceil(np.log10(v.max())))
    lo, hi = float(v.min()), float(v.max())
    if metric in BOUNDED:
        return (min(-0.03, lo - 0.03), 1.03)
    pad = 0.05 * max(hi - lo, 0.1)
    return (min(lo, 0) - pad, max(hi, 0) + pad)


def fig_2024(out, table, name):
    sc_ = read(table)
    fig = plt.figure(figsize=(7.2, 8.5))
    gs = GridSpec(len(METRICS_2024), 3, figure=fig, hspace=0.5, wspace=0.28, top=0.975, bottom=0.085,
                  left=0.1, right=0.99)
    for r, metric in enumerate(METRICS_2024):
        ylim = row_limits(sc_.loc[sc_["metric"] == metric, "value"], metric)
        for c, scen in enumerate(SCENARIOS):
            ax = fig.add_subplot(gs[r, c])
            if scen == "single" and metric.startswith("Batch"):
                ax.axis("off")
                continue
            sub = sc_[(sc_["metric"] == metric) & (sc_["scenario"] == scen)]
            boxes(ax, sub, ORDER[scen], ylim=ylim, log=metric == "Runtime")
            if r == 0:
                ax.set_title(S.SCENARIOS[scen]["label"], pad=4)
            if c == 0 or (metric.startswith("Batch") and c == 1):
                ax.set_ylabel(YLABEL.get(metric, metric))
            if r < len(METRICS_2024) - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("Group similarity")
    legend(fig, present(sc_), 0.0)
    save(fig, out, name)


def fig_ward(out):
    lv = read("level_scores_all.csv")
    sc_ = read("scores_all.csv")
    rows = [("Level recovery", tl) for tl in S.HIERARCHY_OBS] + [("Hierarchy accuracy", None)]
    fig = plt.figure(figsize=(7.2, 8.0))
    gs = GridSpec(len(rows) + 1, 3, figure=fig, hspace=0.62, wspace=0.28, top=0.97, bottom=0.11, left=0.1,
                  right=0.99, height_ratios=[1] * len(rows) + [2.4])
    for r, (metric, tl) in enumerate(rows):
        for c, scen in enumerate(SCENARIOS):
            ax = fig.add_subplot(gs[r, c])
            if tl is None:
                sub = sc_[(sc_["metric"] == metric) & (sc_["scenario"] == scen)]
                ylab = YLABEL.get(metric, metric)
            else:
                sub = lv[(lv["metric"] == metric) & (lv["true_level"] == tl) & (lv["scenario"] == scen)]
                ylab = f"{metric}\n({LEVEL_LABEL[tl]})"
            boxes(ax, sub, WARD[scen], ylim=(-0.03, 1.03))
            if r == 0:
                ax.set_title(S.SCENARIOS[scen]["label"], pad=4)
            if c == 0:
                ax.set_ylabel(ylab)
                if r in (0, len(rows) - 1):
                    fig.text(0.008, ax.get_position().y1 + 0.006, "A" if r == 0 else "B", fontsize=10,
                             fontweight="bold", va="bottom")
            if r < len(rows) - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("Group similarity")
    # C: the scatter of Figure 2 D for the Ward baselines (four distinct batches; scDEF read as there), with the
    # batch-specific groups recovered (left pair) and nested (right pair)
    axs = tradeoff_row(fig, gs[len(rows), :], sc_, methods=WARD["distinct"])
    fig.text(0.008, axs[0].get_position().y1 + 0.012, "C", fontsize=10, fontweight="bold", va="bottom")
    legend(fig, [m for m in WARD["similar"] if m in set(lv["method"])], 0.0, ncol=5)
    save(fig, out, "sims_ward")


ONE_LAYER = "scDEF_1L_Ward"           # the one-layer scDEF: its Ward tree in A, its own layer 0 in B-D
MAIN_GAPS = {"PCA", "scVI"}           # scDEF and its one-layer version | the other methods | the integration methods
# The one-layer comparison moved to its own supplementary figure (Pedro, 2026-11-11; fig_one_layer).
MAIN_METHODS = {sc: [m for m in ORDER[sc] if m not in HC] for sc in SCENARIOS}
MAIN_LABELS = {ONE_LAYER: "1-layer scDEF"}
GROUP_METHODS = list(rdp.ORDER_MULTI)
# Panel D reads scDEF as one model with two views (pre-declared 2026-11-10, item 1): batch-specific groups and their
# nesting from the decomposition, shared groups from the model with the batch annotations.
VIEW = {("scDEF", "Specific groups kept"): S.DECOMPOSED_T1, ("scDEF", "Specific groups nested"): S.DECOMPOSED_T1}
# (top_layer=1, the package default; pre-declared 2026-11-10 before its results. The top_layer=2 decomposition,
#  S.DECOMPOSED, is reported in the text.)


def main_label(m):
    return MAIN_LABELS.get(m, label(m))


KEPT_LABEL = "Batch-specific groups recovered\nas batch-specific clusters (of 4)"
NESTED_LABEL = "Batch-specific groups nested\nunder their parent (of 4)"
SHARED_LABEL = "Shared groups recovered\nas batch-mixing clusters (of 4)"


def tradeoff(ax, sc_, sim, title, y_metric="Specific groups kept", methods=None, view=None, step=None, ms=5,
             short=False):
    """Four distinct batches, one similarity level: each method at (shared groups recovered, batch-specific groups
    recovered or nested), means over the five replicates; scDEF read as one model with two views (VIEW) unless
    ``view`` says otherwise. Coincident means are drawn side by side, ``step`` apart (in data units; by default
    one marker width at the size of the axes)."""
    view = VIEW if view is None else view
    sub = sc_[(sc_["scenario"] == "distinct") & (sc_["similarity"] == sim)]
    pts = {}
    for m in (methods or GROUP_METHODS):
        y = sub.loc[(sub["method"] == view.get((m, y_metric), m)) & (sub["metric"] == y_metric), "value"].astype(float)
        x = sub.loc[(sub["method"] == m) & (sub["metric"] == "Shared groups kept"), "value"].astype(float)
        if len(x) and len(y):
            pts[m] = (float(x.mean()), float(y.mean()))
    if step is None:
        fig, bb = ax.figure, ax.get_position()
        side = min(bb.width * fig.get_figwidth(), bb.height * fig.get_figheight()) * 72
        step = 1.1 * ms * 4.75 / side
    groups = {}
    for m, (x, y) in pts.items():
        groups.setdefault((round(x, 1), round(y, 1)), []).append(m)
    reach = max([(len(v) - 1) / 2 * step for v in groups.values()] + [0])
    for (x, y), ms_ in groups.items():
        offs = (np.arange(len(ms_)) - (len(ms_) - 1) / 2) * step       # coincident means side by side
        for m, dx in zip(ms_, offs):
            _, col = style(m)
            big = m == "scDEF"
            ax.plot(x + dx, y, "o", ms=ms * 1.4 if big else ms, mfc=col, mec="0.15" if big else col,
                    mew=0.8 if big else 0.5, zorder=4)
    lo = -max(0.25, reach + 0.2)                                   # room for the means offset left of 0
    ax.set_xlim(lo, 4.5)
    ax.set_ylim(lo, 4.5)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_yticks([0, 1, 2, 3, 4])
    kept = y_metric == "Specific groups kept"
    if short:                                                      # four panels in a row
        ax.set_xlabel("Shared groups\nrecovered (of 4)")
        ax.set_ylabel("Batch-specific groups\n" + ("recovered (of 4)" if kept else "nested (of 4)"))
    else:
        ax.set_xlabel(SHARED_LABEL)
        ax.set_ylabel(KEPT_LABEL if kept else NESTED_LABEL)
    ax.set_title(title, pad=4)
    ax.set_aspect("equal")


def tradeoff_row(fig, spec, sc_, methods=None, view=None, metrics=("Specific groups kept", "Specific groups nested")):
    """A row of the scatter of Figure 2 D: for each metric in ``metrics``, low and high similarity side by side."""
    n = len(metrics)
    ratios = sum(([1, 1, 0.32] for _ in range(n)), [])[:-1]
    g = spec.subgridspec(1, len(ratios), wspace=0.08, width_ratios=ratios)
    axs = []
    for k, metric in enumerate(metrics):
        for j, sim in enumerate(SIMS):
            ax = fig.add_subplot(g[3 * k + j]); axs.append(ax)
            tradeoff(ax, sc_, sim, f"{sim} similarity", y_metric=metric, methods=methods, view=view,
                     ms=4 if n > 1 else 5, short=n > 1)
            if j:
                ax.set_ylabel(""); ax.set_yticklabels([])
    same_limits(axs)
    return axs


def same_limits(axs):
    """One range for every scatter of a row, so that the unlabelled panels read on the axes of their neighbour."""
    lo = min(ax.get_xlim()[0] for ax in axs)
    for ax in axs:
        ax.set_xlim(lo, 4.5); ax.set_ylim(lo, 4.5)


def fig_main(out):
    sc_ = read("scores_all.csv")
    rows = ["Hierarchy accuracy", "Cell Type ARI", "Cell Type ASW"]
    fig = plt.figure(figsize=(7.2, 6.5))
    gs = GridSpec(len(rows) + 2, 3, figure=fig, hspace=0.45, wspace=0.22,
                  height_ratios=[1, 1, 1, 0.3, 1.75], top=0.965, bottom=0.08, left=0.095, right=0.99)
    for r, metric in enumerate(rows):
        lim = row_limits(sc_.loc[sc_["metric"] == metric, "value"], metric) if metric == "Cell Type ASW" else (-0.03, 1.03)
        for c, scen in enumerate(SCENARIOS):
            ax = fig.add_subplot(gs[r, c])
            sub = sc_[(sc_["metric"] == metric) & (sc_["scenario"] == scen)]
            boxes(ax, sub, MAIN_METHODS[scen], ylim=lim, gaps=MAIN_GAPS, group_width=0.74)
            if r == 0:
                ax.set_title(S.SCENARIOS[scen]["label"], pad=4)
            if c == 0:
                ax.set_ylabel(YLABEL.get(metric, metric).replace("Cell Type", "Cell-type"))
                fig.text(0.008, ax.get_position().y1 + 0.008, "ABC"[r], fontsize=10, fontweight="bold", va="bottom")
            if r < len(rows) - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("Group similarity")
    methods = [m for m in GROUP_METHODS if m in set(sc_["method"])]
    handles = [Patch(label=main_label(m), **style(m)[0]) for m in methods]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.53, gs[3, :].get_position(fig).y0 + 0.012),
               ncol=10, frameon=False, fontsize=6, handlelength=1.1, columnspacing=0.7, handletextpad=0.4)
    g = gs[4, :].subgridspec(1, 2, wspace=0.05)
    axs = [fig.add_subplot(g[0]), fig.add_subplot(g[1])]
    tradeoff(axs[0], sc_, "Low", "Four distinct batches, low similarity", step=0.17)
    tradeoff(axs[1], sc_, "High", "Four distinct batches, high similarity", step=0.17)
    axs[1].set_ylabel("")
    same_limits(axs)
    fig.text(0.008, axs[0].get_position().y1 + 0.012, "D", fontsize=10, fontweight="bold", va="bottom")
    save(fig, out, "sims_main")


def fig_one_layer(out):
    """Supplementary figure: scDEF against 1-layer scDEF (the same workflow with one layer of factors). A hierarchy
    accuracy (1-layer scDEF: Ward linkage of its factors, cut at the layer sizes of scDEF), B cell-type ARI, C
    cell-type ASW, D four distinct batches: the scatter of Figure 2 D, with the batch-specific groups recovered and
    nested. Both are the models with the batch annotations, without decomposition, so that only the number of
    layers differs."""
    sc_ = read("scores_all.csv")
    methods = ["scDEF", ONE_LAYER]
    rows = ["Hierarchy accuracy", "Cell Type ARI", "Cell Type ASW"]
    fig = plt.figure(figsize=(7.2, 7.4))
    gs = GridSpec(len(rows) + 1, 3, figure=fig, hspace=0.62, wspace=0.25, height_ratios=[1, 1, 1, 1.55],
                  top=0.965, bottom=0.12, left=0.1, right=0.99)
    for r, metric in enumerate(rows):
        lim = row_limits(sc_.loc[sc_["metric"] == metric, "value"], metric) if metric == "Cell Type ASW" else (-0.03, 1.03)
        for c, scen in enumerate(SCENARIOS):
            ax = fig.add_subplot(gs[r, c])
            sub = sc_[(sc_["metric"] == metric) & (sc_["scenario"] == scen)]
            boxes(ax, sub, methods, ylim=lim, gaps=set(), group_width=0.5)
            if r == 0:
                ax.set_title(S.SCENARIOS[scen]["label"], pad=4)
            if c == 0:
                ax.set_ylabel(YLABEL.get(metric, metric).replace("Cell Type", "Cell-type"))
                fig.text(0.008, ax.get_position().y1 + 0.008, "ABC"[r], fontsize=10, fontweight="bold", va="bottom")
            ax.set_xticklabels([] if r < len(rows) - 1 else SIMS)
            if r == len(rows) - 1:
                ax.set_xlabel("Group similarity")
    axs = tradeoff_row(fig, gs[3, :], sc_, methods=methods, view={})
    fig.text(0.008, axs[0].get_position().y1 + 0.012, "D", fontsize=10, fontweight="bold", va="bottom")
    handles = [Line2D([], [], ls="none", marker="o", ms=4, mfc=style(m)[1], mec=style(m)[1], label=main_label(m))
               for m in methods]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=6.3, bbox_to_anchor=(0.55, 0.0))
    save(fig, out, "sims_one_layer")


def fig_batch_groups(out):
    """Supplementary figure: the scatter of Figure 2 D with the batch-specific groups nested under their parent
    instead of recovered (four distinct batches, low and high similarity; every method of Figure 2, scDEF read as
    there). The recovered pair is Figure 2 D itself."""
    sc_ = read("scores_all.csv")
    fig = plt.figure(figsize=(7.2, 3.4))
    g = GridSpec(1, 1, figure=fig, top=0.92, bottom=0.27, left=0.16, right=0.84)
    axs = tradeoff_row(fig, g[0], sc_, metrics=("Specific groups nested",))
    for ax, sim in zip(axs, ("low", "high")):
        ax.set_title(f"Four distinct batches, {sim} similarity", pad=4)
    methods = [m for m in GROUP_METHODS if m in set(sc_["method"])]
    handles = [Line2D([], [], ls="none", marker="o", ms=4.5, mfc=style(m)[1], mec="0.15" if m == "scDEF" else style(m)[1],
                      label=main_label(m)) for m in methods]
    fig.legend(handles=handles, loc="lower center", ncol=len(methods), frameon=False, fontsize=6.3,
               bbox_to_anchor=(0.5, 0.0), handletextpad=0.2, columnspacing=0.9)
    save(fig, out, "sims_batch_groups")


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=200)
    plt.close(fig)
    print("wrote", os.path.join(out, f"{name}.pdf"))


def main():
    out = os.path.join(S.OUT_ROOT, "figures")
    os.makedirs(out, exist_ok=True)
    fig_main(out)
    fig_batch_groups(out)
    fig_one_layer(out)
    fig_levels(out)
    fig_2024(out, "scores_all.csv", "sims_all_metrics")
    fig_2024(out, "scores_all_2024_verbatim.csv", "sims_2024_metrics_verbatim")
    fig_ward(out)


if __name__ == "__main__":
    main()
