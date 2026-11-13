"""Figure 4 — batch integration, built on the recommended workflow.

Correct with the batch key, then audit with ``decompose_batch_effects(top_layer=2)``
(scDEF-markdown/decisions/decision-batch-workflow.md), on two PBMC data sets:

- two healthy donors, Drop-seq (pbmcsca, ``Method == 'Drop-seq'``, pbmc1 / pbmc2);
- Kang et al. IFN-beta: eight lupus patients, cells pooled into a control and a stimulated batch.

Uses the models saved by ``scDEF-notebooks/scdef-{pbmcs2b,pbmcsifn}-tutorial.ipynb``
(``*-ref`` = corrected fit, ``*-dec-top2`` = audit fit); nothing is refitted.

    python reproducibility/make_figures/fig4_batch.py [--out DIR]

Writes ``fig4_batch_integration.{pdf,png}`` (the two data sets as parallel columns) and
``fig4_numbers.json`` with every number the text and caption quote.
"""
import argparse
import json
import os
import warnings

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
warnings.filterwarnings("ignore")
import logging

logging.disable(logging.WARNING)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scdef as scd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from sklearn.metrics import adjusted_rand_score

HERE = os.path.dirname(os.path.abspath(__file__))
NB = os.path.expanduser("~/projects/scDEF/scdef_notebooks_revision")
DATA = os.path.expanduser("~/projects/scDEF/scdef_data/pbmcs_data")

# The ISG list of the IFN tutorial's Harmony appendix, so the numbers match it.
ISG = ["ISG15", "IFI6", "IFIT1", "IFIT3", "ISG20", "MX1", "OAS1", "IFITM3",
       "CXCL10", "RSAD2", "IFI44L", "LY6E", "EPSTI1", "IFIT2", "APOBEC3A"]
LAYERS = ["L0", "L1", "L2"]
MIN_CELLS = 60                         # within-type mixing is not read below this
TECHNICAL_SHAPES = ("branch_split", "overlaid")
# How the three fits are named in every figure, matching the manuscript and the package.
FIT = {"nobatch": "without batch annotations", "ref": "with batch annotations",
       "dec": "decomposed"}

plt.rcParams.update({
    "font.size": 6.5, "axes.titlesize": 7, "axes.labelsize": 6.5, "xtick.labelsize": 6,
    "ytick.labelsize": 6, "legend.fontsize": 5.8, "axes.linewidth": 0.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "pdf.fonttype": 42,
})
INK, MUTED = "#222222", "#6b6b6b"
BATCH_COLORS = ["#3b6fb6", "#e08a2c"]
SHAPE_COLOR = {"branch_split": "#b0532a", "overlaid": "#b0532a", "branch_skewed": "#2f6f9e"}


# ----------------------------------------------------------------------------- data


def prep_ifn():
    a = sc.read_h5ad(f"{DATA}/ifnb/ifnb.h5ad")
    a.obs = pd.read_csv(f"{DATA}/ifnb/ifnb_metadata.csv", index_col=0)
    a.raw = None
    a.layers["counts"] = a.X.copy()
    sc.pp.filter_cells(a, min_genes=200)
    sc.pp.filter_genes(a, min_cells=10)
    a.var["mt"] = a.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(a, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    a = a[a.obs.n_genes_by_counts < 5000, :]
    a = a[a.obs.pct_counts_mt < 5, :]
    sc.pp.highly_variable_genes(a, flavor="seurat_v3", n_top_genes=4000, layer="counts",
                                batch_key="stim")
    a = a[:, a.var.highly_variable].copy()
    a.obs["cell_type"] = a.obs["seurat_annotations"].astype(str)
    a.obs["batch"] = a.obs["stim"].map({"CTRL": "control", "STIM": "IFN-β"}).astype(str)
    return categorical(a)


def prep_2b():
    a = sc.read_h5ad(f"{DATA}/pbmcsca.h5ad")
    a = a[a.obs["Method"] == "Drop-seq"].copy()
    sc.pp.filter_cells(a, min_genes=200)
    sc.pp.filter_genes(a, min_cells=10)
    a.var["mt"] = a.var_names.str.startswith("MT-")
    a.var["rb"] = a.var_names.str.startswith("RPS") | a.var_names.str.startswith("RPL")
    sc.pp.calculate_qc_metrics(a, qc_vars=["mt", "rb"], percent_top=None, log1p=False, inplace=True)
    a = a[a.obs.n_genes_by_counts < 2000, :]
    a = a[a.obs.pct_counts_mt < 8, :]
    a.layers["counts"] = a.X.toarray()
    nb = json.load(open(f"{NB}/scdef-pbmcs2b-tutorial.ipynb"))
    ns = {}
    exec("".join(nb["cells"][2]["source"]), ns)          # the tutorial's marker list
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, flavor="seurat_v3", n_top_genes=4000, layer="counts",
                                batch_key="Experiment")
    for g in {x for gs in ns["markers"].values() for x in gs}:
        if g in a.var_names:
            a.var.loc[g, "highly_variable"] = True
    a = a[:, a.var.highly_variable].copy()
    a.obs["cell_type"] = a.obs["CellType"].astype(str)
    a.obs["batch"] = a.obs["Experiment"].map({"pbmc1": "donor 1", "pbmc2": "donor 2"}).astype(str)
    return categorical(a)


def categorical(a):
    """make_graph's wedges need categorical columns; batch colours match the other panels."""
    for k in ("cell_type", "batch"):
        a.obs[k] = pd.Categorical(a.obs[k], categories=sorted(a.obs[k].unique()))
    a.uns["batch_colors"] = BATCH_COLORS[: len(a.obs["batch"].cat.categories)]
    return a


def load(name, adata):
    m = scd.scDEF.load(f"{NB}/{name}", adata=adata.copy())
    if "-dec-top2" in name:
        # Saved before decompositions recorded their anchor on the model; every decomposed model
        # here used top_layer=2, whose frozen z is the corrected view.
        m.decomposition_top_layer = 2
        m.annotate_adata()
    elif "L0" not in m.adata.obs:
        m.annotate_adata()
    return m


# ----------------------------------------------------------------------------- numbers


def placement(m):
    obs = m.adata.obs
    out = {}
    for c in sorted(obs["cell_type"].unique()):
        mk = (obs["cell_type"] == c).to_numpy()
        if mk.sum() < MIN_CELLS or obs.loc[mk, "batch"].value_counts().min() < 10:
            continue
        out[c] = {l: float(adjusted_rand_score(obs.loc[mk, "batch"], obs.loc[mk, l].astype(str)))
                  for l in LAYERS if l in obs and obs[l].nunique() > 1}
    return pd.DataFrame(out).T[LAYERS]


def l0_weights(m):
    W = np.asarray(m.pmeans["L0W"])
    if W.shape[0] != len(m.factor_names[0]):
        W = W[np.asarray(m.factor_lists[0])]
    return W


def batch_factors(dec, ref):
    """The audit's batch-specific L0 factors: shape label, batch skew, cell type, top genes."""
    rep = scd.tl.batch_structure_report(dec, batch_key="batch", reference=ref)
    rep = rep[rep["shape"] != "balanced"]
    obs = dec.adata.obs
    W = l0_weights(dec)
    genes = np.asarray(dec.adata.var_names)
    idx = {f: i for i, f in enumerate(dec.factor_names[0])}
    rows = []
    for f, r in rep.iterrows():
        mk = (obs["L0"].astype(str) == f).to_numpy()
        if mk.sum() < 20:                                   # too small to read
            continue
        t = obs.loc[mk, "cell_type"].value_counts(normalize=True)
        rows.append(dict(factor=f, n=int(mk.sum()), dom_batch=r["dom_batch"],
                         frac_dom=float(r["frac_dom_batch"]), type=t.index[0],
                         type_frac=float(t.iloc[0]), shape=r["shape"],
                         genes=list(genes[np.argsort(W[idx[f]])[::-1][:5]])))
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def isg_axes(ref, dec, adata):
    """Where the interferon programme lives: best |r| of any single dimension with the ISG score."""
    a = adata.copy()
    a.X = a.layers["counts"].copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.tl.score_genes(a, [g for g in ISG if g in a.var_names], score_name="isg", random_state=0)
    isg = a.obs["isg"].to_numpy()

    def best(rep):
        r = np.array([abs(np.corrcoef(rep[:, j], isg)[0, 1]) for j in range(rep.shape[1])])
        return float(np.nanmax(r)), int(np.nanargmax(r))

    out = {}
    for key, m in (("corrected", ref), ("audit", dec)):
        r, j = best(np.asarray(m.adata.obsm["X_L0"]))
        out[key] = dict(r=r, factor=m.factor_names[0][j])
    # Harmony on the tutorial's PCA (normalise, log, regress out depth and mt, scale).
    import harmonypy
    sc.pp.regress_out(a, ["total_counts", "pct_counts_mt"])
    sc.pp.scale(a, max_value=10)
    sc.tl.pca(a, n_comps=50, svd_solver="arpack", random_state=0)
    ho = harmonypy.run_harmony(a.obsm["X_pca"], a.obs, ["batch"], verbose=False)
    z = np.asarray(ho.Z_corr)
    z = z if z.shape[0] == a.n_obs else z.T
    out["harmony"] = dict(r=best(z)[0], factor=None)
    gs = scd.tl.get_batch_specific_genes_from_gene_scale(ref)
    col = [c for c in gs.columns if "STIM" in str(c) or "IFN" in str(c)][0]
    pct = gs[col].rank(pct=True)
    out["gene_scale_isg_median_percentile"] = float(np.median(pct[[g for g in ISG if g in pct.index]]))
    return out


# ----------------------------------------------------------------------------- drawing


def palette(cats):
    cmap = plt.get_cmap("tab20")
    return {c: cmap(i % 20) for i, c in enumerate(cats)}


def umap_grid(fig, spec, m, title, batch_label):
    scd.tl.umap(m, layers=[2, 1, 0])
    obs = m.adata.obs
    types = sorted(obs["cell_type"].unique())
    tcol = palette(types)
    batches = sorted(obs["batch"].unique())
    bcol = dict(zip(batches, BATCH_COLORS))
    g = GridSpecFromSubplotSpec(2, 3, subplot_spec=spec, wspace=0.05, hspace=0.08)
    rng = np.random.default_rng(0)
    order = rng.permutation(m.adata.n_obs)            # no batch drawn on top of the other
    for j, layer in enumerate(LAYERS):
        xy = np.asarray(m.adata.obsm[f"X_umap_{layer}"])[order]
        for i, (key, col) in enumerate((("cell_type", tcol), ("batch", bcol))):
            ax = fig.add_subplot(g[i, j])
            ax.scatter(xy[:, 0], xy[:, 1], s=0.15, lw=0, rasterized=True,
                       c=[col[v] for v in obs[key].to_numpy()[order]])
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#cccccc")
            if i == 0:
                ax.set_title(f"Layer {layer[1]}", pad=2)
            if j == 0:
                ax.set_ylabel("cell type" if i == 0 else batch_label, labelpad=2)
    return tcol, bcol


def legend_row(ax, colors, title, ncol, loc="upper left", fontsize=None):
    ax.axis("off")
    h = [plt.Line2D([], [], marker="o", ls="", ms=3.2, mfc=c, mec="none") for c in colors.values()]
    ax.legend(h, list(colors), title=title, ncol=ncol, frameon=False, loc=loc, fontsize=fontsize,
              handletextpad=0.1, columnspacing=0.6 if fontsize is None else 0.35, title_fontsize=6,
              borderaxespad=0)


def placement_panel(ax, corr, audit):
    rows = corr.index.union(audit.index)
    M = np.hstack([corr.reindex(rows).to_numpy(), np.full((len(rows), 1), np.nan),
                   audit.reindex(rows).to_numpy()])
    im = ax.imshow(M, cmap="magma_r", vmin=0, vmax=0.9, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{max(M[i, j], 0):.2f}", ha="center", va="center", fontsize=5.4,
                        color="white" if M[i, j] > 0.5 else INK)
    ax.set_yticks(range(len(rows)), rows)
    ax.set_xticks([0, 1, 2, 4, 5, 6], LAYERS + LAYERS)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(1, -0.75, "with batch\nannotations", ha="center", va="bottom", fontsize=6)
    ax.text(5, -0.75, FIT["dec"], ha="center", va="bottom", fontsize=6)
    cb = plt.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.ax.set_title("batch\nARI", fontsize=5.6, pad=3)   # on top: a side label overflows the right column
    cb.ax.tick_params(labelsize=5.4, width=0.4, length=2)
    cb.outline.set_linewidth(0.4)


def factor_table(ax, df, bcol, max_rows=9, highlight=(), xs=(0.0, 0.075, 0.145, 0.36, 0.57, 0.715),
                 n_genes=5, compact=False):
    """One row per batch-specific factor: skew bar, cell type, diagnostic shape, top genes."""
    df = df.head(max_rows)
    ax.set_xlim(0, 1)
    ax.set_ylim(len(df) + 0.3, -0.9)
    ax.axis("off")
    heads = ["factor", "cells", "dominant batch", "cell type", "diagnostic shape", "top genes"]
    if compact:
        heads[0], heads[1], heads[4] = "factor (cells)", "", "shape"
    cols = list(zip(xs, heads))
    x_f, x_n, x_b, x_t, x_s, x_g = xs
    bar_w = (x_t - x_b) * 0.38
    for x, h in cols:
        ax.text(x, -0.55, h, fontsize=5.8, color=MUTED, va="center")
    ax.plot([0, 1], [-0.2, -0.2], color="#cccccc", lw=0.5)
    for i, r in df.iterrows():
        y = i + 0.35
        weight = "bold" if r["factor"] in highlight else "normal"
        if r["factor"] in highlight:
            ax.add_patch(plt.Rectangle((-0.005, y - 0.48), 1.0, 0.96, color="#f3ead7", zorder=0, lw=0))
        if compact:
            ax.text(x_f, y, f"{r['factor']} ({r['n']:,})", va="center", fontweight=weight)
        else:
            ax.text(x_f, y, r["factor"], va="center", fontweight=weight)
            ax.text(x_n, y, f"{r['n']:,}", va="center")
        ax.add_patch(plt.Rectangle((x_b, y - 0.22), bar_w * r["frac_dom"], 0.44,
                                   color=bcol.get(r["dom_batch"], MUTED), lw=0))
        ax.text(x_b + bar_w + 0.008, y, f"{100 * r['frac_dom']:.0f}% {r['dom_batch']}", va="center")
        ax.text(x_t, y, f"{r['type']} ({100 * r['type_frac']:.0f}%)", va="center")
        ax.text(x_s, y, r["shape"].replace("branch_", "").replace("_", " "), va="center",
                color=SHAPE_COLOR.get(r["shape"], INK))
        ax.text(x_g, y, " ".join(r["genes"][:n_genes]), va="center", style="italic")


def isg_panel(ax, isg):
    labels = ["with batch\nannotations", FIT["dec"]]
    vals = [isg["corrected"]["r"], isg["audit"]["r"]]
    ax.barh(range(2), vals, color=["#9a9a9a", "#2f6f9e"], height=0.55)
    ax.set_yticks(range(2), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_ylim(1.6, -1.25)
    ax.set_xlabel("best factor, |r| with the cells' interferon score", labelpad=2)
    for i, (v, k) in enumerate(zip(vals, ("corrected", "audit"))):
        ax.text(v + 0.02, i, f"{v:.2f}  ({isg[k]['factor']})", va="center", fontsize=6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.text(0.02, -0.75, "With batch annotations the interferon genes sit at the "
                      f"{100 * isg['gene_scale_isg_median_percentile']:.1f}th\npercentile of the "
                      "per-batch gene scaling, not in factors.", fontsize=5.8, color=MUTED, va="center")


def tree_png(dec, path):
    g = scd.pl.make_graph(dec, wedged="batch", show_signatures=False, n_cells_label=False)
    g.attr(dpi="300", rankdir="LR", nodesep="0.04", ranksep="0.9")
    g.format = "png"
    return g.render(path, cleanup=True)


def gene_scale_panel(ax, ref, ratio_label, annotate_top=0, step=8, n_isg=6):
    """What the per-batch gene scaling absorbed: genes ranked by the log2 ratio of the second batch's
    scaling to the first's; interferon-stimulated genes in orange."""
    gs = scd.tl.get_batch_specific_genes_from_gene_scale(ref)
    cols = [c for c in gs.columns if c in set(map(str, ref.batches))]
    v = gs[cols[-1]].sort_values(ascending=False)
    x = np.arange(len(v))
    isg = np.isin(v.index, ISG)
    ax.scatter(x[~isg], v.to_numpy()[~isg], s=1.0, lw=0, color="#b5b5b5", rasterized=True)
    ax.scatter(x[isg], v.to_numpy()[isg], s=6, lw=0, color="#e08a2c", zorder=3)
    labelled = [g for g in v.index[isg] if v.index.get_loc(g) < 300][:n_isg]
    labelled += [g for g in v.index[:annotate_top] if g not in labelled]
    for k, g in enumerate(labelled):
        ax.annotate(g, (x[v.index.get_loc(g)], v[g]), xytext=(14, -step * k + 1), textcoords="offset points",
                    fontsize=4.8, style="italic", color=INK if g in ISG else MUTED,
                    arrowprops=dict(arrowstyle="-", lw=0.3, color=MUTED))
    ax.axhline(0, color="#cccccc", lw=0.5)
    ax.text(0.01, 0.03, "genes, ranked", transform=ax.transAxes, ha="left", va="bottom", fontsize=5.2,
            color=MUTED)
    ax.set_ylabel(f"log2 ratio\n{ratio_label}", labelpad=1)
    ax.set_xticks([])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    pct = v.rank(pct=True)[[g for g in ISG if g in v.index]]
    ax.text(0.98, 0.95, f"interferon genes (orange):\nmedian percentile {100 * np.median(pct):.1f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=5.4, color=MUTED)
    return float(np.median(pct))


def top_genes_fitting(genes, n=3, width=19):
    """Up to ``n`` top genes, as many as fit in ``width`` characters, so the column never runs into the next."""
    out = list(genes[:1])
    for g in genes[1:n]:
        if len(" ".join(out + [g])) > width:
            break
        out.append(g)
    return " ".join(out)


def factor_table_one_line(ax, df, bcol, max_rows=9, highlight=()):
    """Half-width table, one line per factor: factor (cells), batch skew, cell type, top genes, shape."""
    df = df.head(max_rows).reset_index(drop=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(len(df) - 0.4, -1.1)
    ax.axis("off")
    cols = ((0.0, "factor (cells)", "left"), (0.195, "batch", "left"), (0.40, "cell type", "left"),
            (0.635, "top genes", "left"), (1.0, "shape", "right"))
    for x, h, ha in cols:
        ax.text(x, -0.85, h, fontsize=5.2, color=MUTED, va="center", ha=ha)
    ax.plot([0, 1], [-0.5, -0.5], color="#cccccc", lw=0.5)
    for i, r in df.iterrows():
        if r["factor"] in highlight:
            ax.add_patch(plt.Rectangle((-0.01, i - 0.48), 1.02, 0.96, color="#f3ead7", zorder=0, lw=0))
        weight = "bold" if r["factor"] in highlight else "normal"
        ax.text(0.0, i, f"{r['factor']} ({r['n']:,})", va="center", fontsize=5.0, fontweight=weight)
        ax.text(0.195, i, f"{100 * r['frac_dom']:.0f}% {r['dom_batch']}", va="center", fontsize=5.0,
                color=bcol.get(r["dom_batch"], INK), fontweight="bold")
        ax.text(0.40, i, r["type"], va="center", fontsize=5.0)
        ax.text(0.635, i, top_genes_fitting(r["genes"]), va="center", fontsize=4.9, style="italic", color=MUTED)
        ax.text(1.0, i, r["shape"].replace("branch_", "").replace("_", " "), va="center", ha="right",
                fontsize=5.0, color=SHAPE_COLOR.get(r["shape"], INK))


# ----------------------------------------------------------------------------- main


def main(out):
    os.makedirs(out, exist_ok=True)
    a2b, aif = prep_2b(), prep_ifn()
    ref2b, dec2b = load("scdef-pbmcs2b-ref-best5", a2b), load("scdef-pbmcs2b-dec-top2-best5", a2b)
    refif, decif = load("scdef-ifn-ref-best5", aif), load("scdef-ifn-dec-top2-best5", aif)

    p2b = (placement(ref2b), placement(dec2b))
    pif = (placement(refif), placement(decif))
    f2b = batch_factors(dec2b, ref2b)
    fif = batch_factors(decif, refif)
    isg = isg_axes(refif, decif, aif)
    # Shade the factors specific to the stimulated pool (>= 90% of their cells) whose diagnostic
    # shape is a technical pattern: the label says technical; their genes are the interferon response.
    misread = tuple(r["factor"] for _, r in fif.iterrows()
                    if r["shape"] in TECHNICAL_SHAPES and r["dom_batch"] == "IFN-β"
                    and r["frac_dom"] >= 0.9)

    # Two parallel columns, one per data set; every row shows the same kind of panel for both.
    # Sized so that, with its caption, the figure leaves room for text on the page (Pedro, 2026-10-20).
    fig = plt.figure(figsize=(7.2, 5.6))
    G = GridSpec(5, 2, figure=fig, height_ratios=[0.08, 1.75, 0.62, 1.25, 1.0], width_ratios=[1, 1],
                 hspace=0.34, wspace=0.22, left=0.06, right=0.955, top=0.99, bottom=0.01)
    titles = ("Two healthy donors: batches are biological replicates",
              "Control vs IFN-β: batch is confounded with condition")
    columns = ((a2b, ref2b, p2b, f2b, "donor", "donor 2 / donor 1", (), 4, 3),
               (aif, refif, pif, fif, "condition", "IFN-β / control", misread, 5, 0))
    letters = []
    for c, (a, ref, plc, fac, blabel, rlabel, shade, ncol, top) in enumerate(columns):
        axt = fig.add_subplot(G[0, c]); axt.axis("off")
        axt.text(0.0, 0.5, titles[c], fontsize=7.0, fontweight="bold", va="center")
        g1 = GridSpecFromSubplotSpec(2, 1, subplot_spec=G[1, c], height_ratios=[1, 0.3], hspace=0.04)
        tcol, bcol = umap_grid(fig, g1[0], ref, "", blabel)
        legend_row(fig.add_subplot(g1[1]), {**tcol, **bcol}, None, ncol, fontsize=5.4)
        gB = GridSpecFromSubplotSpec(1, 2, subplot_spec=G[2, c], width_ratios=[0.12, 1], wspace=0)
        axB = fig.add_subplot(gB[1])
        gene_scale_panel(axB, ref, rlabel, annotate_top=top, step=5.5, n_isg=5)
        gC = GridSpecFromSubplotSpec(1, 2, subplot_spec=G[3, c], width_ratios=[0.2, 1], wspace=0)
        axC = fig.add_subplot(gC[1])
        placement_panel(axC, *plc)
        axD = fig.add_subplot(G[4, c])
        factor_table_one_line(axD, fac, bcol, max_rows=9, highlight=shade)
        base = "ABCD" if c == 0 else "EFGH"
        letters += list(zip((g1[0], gB[0], gC[0], G[4, c]), base))
    for spec, letter in letters:
        bb = spec.get_position(fig)
        fig.text(bb.x0 - 0.04, bb.y1 + 0.003, letter, fontsize=10, fontweight="bold", va="bottom")

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"fig4_batch_integration.{ext}"), dpi=400)

    numbers = dict(
        pbmcs2b=dict(placement_corrected=p2b[0].round(3).to_dict("index"),
                     placement_audit=p2b[1].round(3).to_dict("index"),
                     batch_factors=f2b.to_dict("records"),
                     layer_sizes=dict(corrected=[len(f) for f in ref2b.factor_lists],
                                      audit=[len(f) for f in dec2b.factor_lists])),
        ifn=dict(placement_corrected=pif[0].round(3).to_dict("index"),
                 placement_audit=pif[1].round(3).to_dict("index"),
                 batch_factors=fif.to_dict("records"), isg=isg, misread_factors=list(misread),
                 layer_sizes=dict(corrected=[len(f) for f in refif.factor_lists],
                                  audit=[len(f) for f in decif.factor_lists])),
    )
    with open(os.path.join(out, "fig4_numbers.json"), "w") as f:
        json.dump(numbers, f, indent=1, default=float)
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    main(ap.parse_args().out)
