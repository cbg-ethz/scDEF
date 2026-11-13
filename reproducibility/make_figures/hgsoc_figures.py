"""Figures 5 and 6 (HGSOC) and the supplementary HGSOC tree, in the layout of the submitted figures.

Data: GSE165897 cancer cells (Zhang et al. 2022), 11 patients before and after neoadjuvant
chemotherapy, as prepared in scdef_data/data/hgsoc-pre.ipynb (refit_best_of_seeds.HGSOC).
Models: ``scdef-hgsoc-{ref,dec-top2}-best5`` (``refit_best_of_seeds.py hgsoc``): patient IDs as batch
annotations, and the decomposition of that fit with layer 2 kept fixed.

Figure 5 (patients): A hierarchy, nodes split by patient; B cells per patient per factor, layers 0-2;
C the decomposition's patient-specific layer-0 factors; D PCA UMAP; E Harmony UMAP; F patients per cluster.
Figure 6 (treatment): A cells per factor by treatment and by patient, sorted by post-NACT share;
B PROGENy pathway scores per factor; C radial hierarchy by treatment and by patient; D radial
hierarchy coloured by the scores of four genes.

    python reproducibility/make_figures/hgsoc_figures.py [--fig6-model ref|dec-top2] [--genes JUN,CD74,HIST1H4C,ISG15]
"""
import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig4_batch as f4  # noqa: E402  (style, paths)
import refit_best_of_seeds as rb  # noqa: E402  (data path)

import harmonypy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scdef as scd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scdef.tools.factor import _kept_layer_w
from scdef.utils import data_utils

HERE = os.path.dirname(os.path.abspath(__file__))
TREAT = ["treatment-naive", "post-NACT"]
TCOL = dict(zip(TREAT, f4.BATCH_COLORS))
LAYERS = [0, 1, 2]
MIN_SHARE = 0.05          # a patient counts towards a cluster when it holds >= 5% of the cluster's cells
PROGENY = os.path.join(f4.NB, "progeny.csv")
MIN_PHASE = 20            # a patient enters the per-condition comparison with this many cells in each phase
# Layer-2 factors read as likely technical (fit after the initialisation fix, 2026-11-04, seed 1; within-patient
# Spearman correlations of a cell's layer-2 weight share). L2_11 is made of the most highly expressed genes (FTH1, FTL,
# S100A6) and follows the share of a cell's reads in its top 50 genes (0.33); L2_5 is made of mitochondrial transcripts
# and follows the mitochondrial read share (0.89); L2_2 is led by nuclear lncRNAs (MALAT1, NEAT1) and is higher in
# cells with fewer counts and genes (-0.21, -0.22). The same three kinds were found in the earlier fits (L2_0, L2_3
# and L2_10 of the pre-fix fit).
TECHNICAL = ("L2_11", "L2_5", "L2_2")
# Names from the over-representation analysis (factor_enrichment), each resting on the library in NAME_SOURCE,
# or on the state of the original study (Zhang et al. 2022) whose markers the factor carries: L2_3 is its
# stress-associated state (C7; JUN, FOS), L2_7 its proliferative state (C5). A factor without a significant term
# that describes its top genes keeps its identifier and two genes (L2_1, L2_4, L2_9, L2_10).
NAMES = {"L2_3": "Stress-associated (AP-1)", "L2_7": "Proliferative", "L2_0": "Inflammatory chemokines",
         "L2_8": "IFN-\u03b3 response (MHC class II)", "L2_6": "Interferon response (S100A8/9, LCN2)",
         "L2_11": "Likely technical", "L2_5": "Likely technical", "L2_2": "Likely technical"}
NAME_SOURCE = {"L2_3": "Zhang et al. 2022, C7", "L2_7": "MSigDB_Hallmark_2020 and Zhang et al. 2022, C5",
               "L2_0": "GO_Biological_Process_2023", "L2_8": "MSigDB_Hallmark_2020 and GO_Biological_Process_2023",
               "L2_6": "MSigDB_Hallmark_2020"}
ENRICHMENT = os.path.join(HERE, "out", "hgsoc_factor_enrichment.csv")


def data():
    a = sc.read_h5ad(rb.HGSOC)
    order = a.obs["patient_id"].value_counts().index.tolist()
    a.obs["patient_id"] = pd.Categorical(a.obs["patient_id"].astype(str), categories=order)
    a.obs["treatment_phase"] = pd.Categorical(a.obs["treatment_phase"].astype(str), categories=TREAT)
    return a


def patient_colors(patients):
    cmap = plt.get_cmap("tab20")
    return {p: plt.matplotlib.colors.to_hex(cmap((2 * i) % 20 + (i // 10))) for i, p in enumerate(patients)}


def load(name, a, pcol):
    m = scd.scDEF.load(os.path.join(f4.NB, name))
    if "dec-top2" in name:
        m.decomposition_top_layer = 2
    m.annotate_adata()                                   # tree rule above layer 0 (package default)
    obs = m.adata.obs
    for k in ("patient_id", "treatment_phase"):
        obs[k] = pd.Categorical(a.obs.loc[obs.index, k].astype(str), categories=a.obs[k].cat.categories)
    m.adata.uns["patient_id_colors"] = [pcol[p] for p in obs["patient_id"].cat.categories]
    m.adata.uns["treatment_phase_colors"] = [TCOL[t] for t in TREAT]
    scd.tl.set_confident_signatures(m)
    return m


def radial_tree(m, **kw):
    """The extracted tree (strongest parent, single-child chains merged; Methods), drawn radially."""
    g = scd.pl.make_graph(m, hierarchy=scd.tl.get_hierarchy(m, simplified=True), show_signatures=False,
                          n_cells=True, show_label=False, **kw)
    return radial(g, m)


def radial(g, m):
    g.engine = "twopi"
    g.graph_attr.update(root=str(m.factor_names[-1][0]), ranksep="0.9", overlap="false")
    return g


def graph_png(ax, g, size, dpi="300"):
    g.format = "png"
    g.graph_attr.update(dpi=dpi, ratio="fill", size=size)
    with tempfile.TemporaryDirectory() as d:
        ax.imshow(plt.imread(g.render(os.path.join(d, "g"), cleanup=True)))
    ax.set_anchor("N")
    ax.axis("off")


def factor_order(m, k):
    """Layer-k factors grouped by their parent in the layer above, then by size."""
    names = list(m.factor_names[k])
    sizes = m.adata.obs[m.layer_names[k]].astype(str).value_counts()
    keep = [i for i in range(len(names)) if sizes.get(names[i], 0) > 0]
    if k + 1 < m.n_layers:
        parent = np.argmax(_kept_layer_w(m, k + 1), axis=0)
        return [names[i] for i in sorted(keep, key=lambda i: (parent[i], -sizes.get(names[i], 0)))]
    return sorted([names[i] for i in keep], key=lambda f: -sizes.get(f, 0))


def stacked_bars(ax, m, k, key, colors, order, legend=False):
    col = m.adata.obs[m.layer_names[k]].astype(str)
    ct = pd.crosstab(col, m.adata.obs[key]).reindex(order).fillna(0)
    bottom = np.zeros(len(ct))
    for c in ct.columns:
        ax.bar(range(len(ct)), ct[c], bottom=bottom, color=colors[c], width=0.85, lw=0, label=c)
        bottom += ct[c].to_numpy()
    ax.set_xticks(range(len(ct)), [f.split("_")[-1] for f in order], rotation=90, fontsize=4.4)
    ax.tick_params(axis="x", length=1.5, pad=1)
    ax.set_xlim(-0.6, len(ct) - 0.4)
    ax.tick_params(axis="y", labelsize=5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return ct


def scatter(ax, xy, values, colors, title=None):
    order = np.random.default_rng(0).permutation(len(values))
    ax.scatter(xy[order, 0], xy[order, 1], s=0.3, lw=0, rasterized=True, c=[colors[v] for v in values[order]])
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#cccccc")
    if title:
        ax.set_title(title, fontsize=6.5, pad=2)


def embeddings(a):
    """Uncorrected UMAP (the PCA UMAP stored with the data) and a Harmony UMAP; Leiden on both."""
    out = {}
    x = a.copy()
    sc.pp.neighbors(x, use_rep="X_pca", n_neighbors=15, random_state=0)
    sc.tl.leiden(x, resolution=1.0, flavor="igraph", n_iterations=2, directed=False, random_state=0)
    out["PCA"] = (np.asarray(a.obsm["X_umap"]), x.obs["leiden"].astype(str).to_numpy())
    z = np.asarray(harmonypy.run_harmony(a.obsm["X_pca"], a.obs, ["patient_id"], verbose=False).Z_corr)
    z = z if z.shape[0] == a.n_obs else z.T
    y = sc.AnnData(np.zeros((a.n_obs, 1))); y.obsm["H"] = z
    sc.pp.neighbors(y, use_rep="H", n_neighbors=15, random_state=0)
    sc.tl.umap(y, random_state=0)
    sc.tl.leiden(y, resolution=1.0, flavor="igraph", n_iterations=2, directed=False, random_state=0)
    out["Harmony"] = (np.asarray(y.obsm["X_umap"]), y.obs["leiden"].astype(str).to_numpy())
    return out


def patients_per_cluster(labels, patients):
    df = pd.crosstab(pd.Series(labels, name="c"), pd.Series(patients, name="p"))
    share = df.div(df.sum(axis=1), axis=0)
    return (share >= MIN_SHARE).sum(axis=1).to_numpy()


def progeny():
    net = pd.read_csv(PROGENY, index_col=0)
    return net.sort_values("padj").groupby("source").head(500)       # top 500 genes per pathway


def patient_factors(dec, ref, min_cells=20):
    """The decomposition's patient-specific layer-0 factors (``batch_structure_report``, as in Figure 4)."""
    rep = scd.tl.batch_structure_report(dec, batch_key="patient_id", reference=ref)
    rep = rep[(rep["shape"] != "balanced") & (rep["n_cells"] >= min_cells)]
    obs = dec.adata.obs
    W, genes = f4.l0_weights(dec), np.asarray(dec.adata.var_names)
    idx = {f: i for i, f in enumerate(dec.factor_names[0])}
    rows = []
    for f, r in rep.iterrows():
        mk = (obs["L0"].astype(str) == f).to_numpy()
        rows.append(dict(factor=f, n=int(mk.sum()), patient=r["dom_batch"], share=float(r["frac_dom_batch"]),
                         post=float((obs.loc[mk, "treatment_phase"] == "post-NACT").mean()), shape=r["shape"],
                         genes=list(genes[np.argsort(W[idx[f]])[::-1][:3]])))
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def patient_factor_table(ax, df, pcol):
    """One line per factor: factor (cells), patient (share), post-NACT share, top genes, shape."""
    ax.set_xlim(0, 1)
    ax.set_ylim(len(df) - 0.4, -1.1)
    ax.axis("off")
    cols = ((0.0, "factor (cells)", "left"), (0.205, "patient", "left"), (0.535, "post-NACT", "right"),
            (0.565, "top genes", "left"), (1.0, "shape", "right"))
    for x, h, ha in cols:
        ax.text(x, -0.85, h, fontsize=4.8, color=f4.MUTED, va="center", ha=ha)
    ax.plot([0, 1], [-0.5, -0.5], color="#cccccc", lw=0.5)
    for i, r in df.iterrows():
        genes = " ".join(r["genes"])
        if len(genes) > 20:                     # keep clear of the shape column
            genes = " ".join(r["genes"][:2])
        ax.text(0.0, i, f"{r['factor']} ({r['n']:,})", va="center", fontsize=4.6)
        ax.text(0.205, i, f"{r['patient']} {100 * r['share']:.0f}%", va="center", fontsize=4.6,
                color=pcol[r["patient"]], fontweight="bold")
        ax.text(0.535, i, f"{100 * r['post']:.0f}%", va="center", ha="right", fontsize=4.6)
        ax.text(0.565, i, genes, va="center", fontsize=4.4, style="italic", color=f4.MUTED)
        ax.text(1.0, i, r["shape"].replace("branch_", ""), va="center", ha="right", fontsize=4.6,
                color=f4.SHAPE_COLOR.get(r["shape"], f4.INK))


def fig5(ref, dec, a, pcol, emb, out):
    patients = list(a.obs["patient_id"].cat.categories)
    fig = plt.figure(figsize=(7.2, 4.4))
    G = GridSpec(3, 2, figure=fig, width_ratios=[3.2, 1], height_ratios=[1.0, 1.15, 1.05],
                 hspace=0.45, wspace=0.14, left=0.05, right=0.99, top=0.955, bottom=0.1)
    letters = []
    ax = fig.add_subplot(G[0, 0])
    g = scd.pl.make_graph(ref, wedged="patient_id", show_signatures=False, n_cells=True, show_label=False)
    graph_png(ax, g, "16,3.6!")
    letters.append((G[0, 0], "A"))
    widths = [len(factor_order(ref, k)) + 3 for k in LAYERS]
    gB = GridSpecFromSubplotSpec(1, len(LAYERS), subplot_spec=G[1, 0], width_ratios=widths, wspace=0.3)
    for j, k in enumerate(LAYERS):
        axb = fig.add_subplot(gB[j])
        stacked_bars(axb, ref, k, "patient_id", pcol, factor_order(ref, k))
        axb.set_title(f"Layer {k}", fontsize=6.5, pad=2)
        if j == 0:
            axb.set_ylabel("cells", fontsize=6, labelpad=1)
            letters.append((G[1, 0], "B"))
    pf = patient_factors(dec, ref)
    half = (len(pf) + 1) // 2
    gC = GridSpecFromSubplotSpec(1, 2, subplot_spec=G[2, 0], wspace=0.1)
    for j, part in enumerate((pf.iloc[:half], pf.iloc[half:])):
        axc = fig.add_subplot(gC[j])
        patient_factor_table(axc, part.reset_index(drop=True), pcol)
        if j == 0:
            letters.append((G[2, 0], "C"))
    pv = ref.adata.obs["patient_id"].astype(str).to_numpy()
    av = a.obs["patient_id"].astype(str).to_numpy()
    for i, name in enumerate(("PCA", "Harmony")):
        axr = fig.add_subplot(G[i, 1])
        scatter(axr, emb[name][0], av, pcol, name)
        letters.append((G[i, 1], "DE"[i]))
    axf = fig.add_subplot(G[2, 1])
    groups = {"PCA": patients_per_cluster(emb["PCA"][1], av),
              "Harmony": patients_per_cluster(emb["Harmony"][1], av),
              "scDEF": patients_per_cluster(ref.adata.obs["L0"].astype(str).to_numpy(), pv),
              "scDEF\ndecomp.": patients_per_cluster(dec.adata.obs["L0"].astype(str).to_numpy(),
                                                         dec.adata.obs["patient_id"].astype(str).to_numpy())}
    for i, (name, v) in enumerate(groups.items()):
        axf.boxplot(v, positions=[i], widths=0.55, showfliers=False, medianprops=dict(color=f4.INK, lw=0.8),
                    boxprops=dict(lw=0.6), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
        jit = np.random.default_rng(i).uniform(-0.15, 0.15, len(v))
        axf.scatter(i + jit, v, s=4, lw=0, color="#2f6f9e" if name.startswith("scDEF") else "#9a9a9a", zorder=3)
    axf.set_xticks(range(len(groups)), list(groups), fontsize=5.2)
    axf.set_ylabel("patients per cluster", fontsize=6, labelpad=1)
    axf.set_ylim(0, len(patients) + 0.5)
    axf.tick_params(axis="y", labelsize=5)
    for side in ("top", "right"):
        axf.spines[side].set_visible(False)
    letters.append((G[2, 1], "F"))
    h = [plt.Line2D([], [], marker="o", ls="", ms=3, mfc=pcol[p], mec="none") for p in patients]
    fig.legend(h, patients, loc="lower left", ncol=len(patients), frameon=False, fontsize=5.2,
               bbox_to_anchor=(0.04, 0.0), handletextpad=0.1, columnspacing=0.6)
    for spec, letter in letters:            # from the grid cell, so letters align by column and row
        bb = spec.get_position(fig)
        fig.text(max(bb.x0 - 0.035, 0.002), bb.y1 + 0.004, letter, fontsize=10, fontweight="bold", va="bottom")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"fig5_hgsoc.{ext}"), dpi=400)
    plt.close(fig)
    ppc = {k: dict(n_clusters=int(len(v)), median=float(np.median(v)), quartiles=[float(q) for q in np.percentile(v, [25, 75])],
                   single_patient=int((v == 1).sum()), all_patients=int((v == len(patients)).sum()))
           for k, v in groups.items()}
    return ppc, pf


def factor_enrichment(m):
    """Enrichr (MSigDB Hallmark 2020, GO Biological Process 2023) on each factor's top 50 genes, layers 0-2.

    Needs network access to fetch the libraries, so the result is cached in ``ENRICHMENT``."""
    if os.path.exists(ENRICHMENT):
        return pd.read_csv(ENRICHMENT, index_col=0)
    enr = scd.tl.gsea(m, libs=("MSigDB_Hallmark_2020", "GO_Biological_Process_2023"), layers=LAYERS, top_genes=50,
                      cutoff=0.05)
    enr.to_csv(ENRICHMENT)
    return enr


def factor_label(m, f, sig, sep=" "):
    return NAMES.get(f) or f"{f}{sep}({', '.join(sig[f][:2])})"


def tex_p(p):
    mant, exp = f"{p:.0e}".split("e")
    return f"${mant} \\times 10^{{{int(exp)}}}$"


def names_table(m, enr, path):
    """Supplementary table rows: each layer-2 factor's name, top genes and the best term of each library."""
    sig = m.get_signatures_dict(top_genes=4)
    lines = []
    for f in m.factor_names[2]:
        sub = enr[enr["factor"] == f]
        cells = []
        for lib in ("MSigDB_Hallmark_2020", "GO_Biological_Process_2023"):
            t = sub[sub["Gene_set"] == lib].sort_values("Adjusted P-value").head(1)
            cells.append("--" if not len(t) else f"{t['Term'].iloc[0].split(' (GO:')[0]} ({tex_p(t['Adjusted P-value'].iloc[0])})")
        name = NAMES.get(f, "--").replace("\u03b3", "$\\gamma$").replace("\u03b1", "$\\alpha$")
        genes = ", ".join("\\textit{" + g + "}" for g in sig[f][:4])
        row = [f.replace("_", "\\_"), name, genes] + cells
        lines.append(" & ".join(row) + " \\\\")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return lines


def condition_shares(m, k):
    """Mean share of each layer-k factor in the cells' layer-k weights, per patient and treatment phase."""
    obs = m.adata.obs
    pat = obs["patient_id"].astype(str).to_numpy()
    ph = obs["treatment_phase"].astype(str).to_numpy()
    z = np.asarray(m.pmeans[f"{m.layer_names[k]}z"])[:, m.factor_lists[k]]
    share = z / z.sum(1, keepdims=True)
    keep = [p for p in obs["patient_id"].cat.categories
            if min(((pat == p) & (ph == t)).sum() for t in TREAT) >= MIN_PHASE]
    rows = {(p, t): share[(pat == p) & (ph == t)].mean(0) for p in keep for t in TREAT}
    return pd.DataFrame(rows, index=list(m.factor_names[k])).T


def change_order(m, k):
    """Layer-k factors ordered by their mean within-patient change in weight share, rising after NACT first."""
    S = condition_shares(m, k)
    d = (S.xs("post-NACT", level=1) - S.xs("treatment-naive", level=1)).mean()
    return d.sort_values(ascending=False).index.tolist()


def wrap_name(name, width=13):
    """A factor name on at most three lines of ``width`` characters, padded to three so the titles align."""
    import textwrap
    lines = textwrap.wrap(name, width=width, break_on_hyphens=True)
    return "\n".join([""] * (3 - len(lines)) + lines)


def paired_panel(fig, spec, m, pcol):
    """Each non-technical layer-2 state: each patient's mean weight share before and after NACT."""
    S = condition_shares(m, 2)
    sig = m.get_signatures_dict(top_genes=3)
    order = [f for f in change_order(m, 2) if f not in TECHNICAL]
    patients = S.index.get_level_values(0).unique().tolist()
    g = GridSpecFromSubplotSpec(2, len(order), subplot_spec=spec, wspace=0.6, hspace=0, height_ratios=[0.42, 1])
    for j, f in enumerate(order):
        ax = fig.add_subplot(g[1, j])
        for p in patients:
            v = [S.loc[(p, t), f] for t in TREAT]
            ax.plot([0, 1], v, "-o", color=pcol[p], lw=0.7, ms=1.8)
        ax.set_xticks([0, 1], ["naive", "post"], fontsize=5)
        ax.set_xlim(-0.35, 1.35)
        ax.tick_params(axis="y", labelsize=4.6, length=1.5, pad=1)
        ax.tick_params(axis="x", length=1.5, pad=1)
        name = wrap_name(NAMES[f] if f in NAMES else f"({', '.join(sig[f][:2])})")
        ax.set_title(f"{name}\n{f}", fontsize=4.9, pad=2, linespacing=1.15)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if j == 0:
            ax.set_ylabel("mean weight share", fontsize=5.6, labelpad=1)
    return order, len(patients)


def fig6(m, a, pcol, genes, out):
    patients = list(a.obs["patient_id"].cat.categories)
    fig = plt.figure(figsize=(7.2, 4.1))
    G = GridSpec(2, 2, figure=fig, width_ratios=[2.6, 1.05], height_ratios=[0.9, 1.0],
                 hspace=0.26, wspace=0.12, left=0.06, right=0.99, top=0.955, bottom=0.085)
    letters = []
    widths = [len(factor_order(m, k)) + 3 for k in LAYERS]
    order_a, n_pat = paired_panel(fig, G[0, 0], m, pcol)
    letters.append((G[0, 0], "A"))
    orders = {k: [f for f in change_order(m, k) if f in set(factor_order(m, k))] for k in LAYERS}
    # ORA of each factor's top 50 genes against PROGENy, z-scored across pathways (Methods)
    res = data_utils.prepare_pathway_factor_scores(m, progeny(), top_genes=50, method="ora", z_score=True)
    mats, net_paths = res[0]["Pathway"], res[2]["Pathway"]
    gB = GridSpecFromSubplotSpec(1, len(LAYERS), subplot_spec=G[1, 0], width_ratios=widths, wspace=0.3)
    vmax = max(float(np.nanmax(np.abs(mats[k]))) for k in LAYERS)
    scores = {}
    for j, k in enumerate(LAYERS):
        names = list(m.factor_names[k])
        M = np.asarray(mats[k])[:, [names.index(f) for f in orders[k]]]
        scores[m.layer_names[k]] = pd.DataFrame(M, index=net_paths, columns=orders[k]).round(2).to_dict()
        ax = fig.add_subplot(gB[j])
        im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(orders[k])), [f.split("_")[-1] for f in orders[k]], rotation=90, fontsize=4.4)
        for t, f in zip(ax.get_xticklabels(), orders[k]):
            if f in TECHNICAL:
                t.set_color("#a0a0a0")
        ax.set_yticks(range(len(net_paths)), net_paths if j == 0 else [], fontsize=5)
        ax.tick_params(length=0)
        ax.set_title(f"Layer {k}", fontsize=6.5, pad=2)
        if j == 0:
            letters.append((G[1, 0], "B"))
    cb = fig.colorbar(im, ax=fig.axes[-len(LAYERS):], fraction=0.02, pad=0.01)
    cb.set_label("PROGENy z-score", fontsize=5.4, labelpad=2); cb.ax.tick_params(labelsize=5, length=2)
    gC = GridSpecFromSubplotSpec(1, 2, subplot_spec=G[0, 1], wspace=0.02)
    for i, (key, title) in enumerate((("treatment_phase", "Treatment"), ("patient_id", "Patient"))):
        ax = fig.add_subplot(gC[i])
        g = radial_tree(m, wedged=key)
        graph_png(ax, g, "4,4")
        ax.set_title(title, fontsize=6.5, pad=1)
        if i == 0:
            letters.append((G[0, 1], "C"))
    gD = GridSpecFromSubplotSpec(2, 2, subplot_spec=G[1, 1], wspace=0.02, hspace=0.18)
    for i, gene in enumerate(genes[:4]):
        ax = fig.add_subplot(gD[i // 2, i % 2])
        g = radial_tree(m, gene_score=gene)
        graph_png(ax, g, "4,4")
        ax.set_title(gene, fontsize=6.2, pad=1, style="italic")
        if i == 0:
            letters.append((G[1, 1], "D"))
    h = [plt.Line2D([], [], marker="s", ls="", ms=4, mfc=TCOL[t], mec="none") for t in TREAT]
    h += [plt.Line2D([], [], marker="o", ls="", ms=3, mfc=pcol[p], mec="none") for p in patients]
    fig.legend(h, TREAT + patients, loc="lower left", ncol=len(h), frameon=False, fontsize=5.0,
               bbox_to_anchor=(0.03, -0.01), handletextpad=0.1, columnspacing=0.6)
    for spec, letter in letters:
        bb = spec.get_position(fig)
        fig.text(max(bb.x0 - 0.04, 0.002), bb.y1 + 0.004, letter, fontsize=10, fontweight="bold", va="bottom")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"fig6_hgsoc.{ext}"), dpi=400)
    plt.close(fig)
    S = condition_shares(m, 2)
    return dict(panel_a_order=order_a, panel_a_patients=n_pat, technical=list(TECHNICAL), names=NAMES,
                condition_shares_L2={f"{p} {t}": v for (p, t), v in S.round(4).to_dict("index").items()},
                progeny_ora=scores)


def effective_factors(m):
    """Per-cell effective number of layer-0 factors, by treatment within each patient."""
    scd.tl.set_cell_entropies(m)
    obs = m.adata.obs
    col = [c for c in obs.columns if c.endswith("_effective_n_factors")][0]
    by = obs.groupby(["patient_id", "treatment_phase"], observed=True)[col].median().unstack()
    return dict(column=col, median_by_patient=by.round(3).to_dict("index"),
                patients_higher_post=int((by["post-NACT"] > by["treatment-naive"]).sum()),
                n_patients=int(by.dropna().shape[0]))


def tree_pdf(m, path):
    g = scd.pl.make_graph(m, wedged="treatment_phase", show_signatures=True, top_genes=10, n_cells_label=True)
    g.format = "pdf"
    return g.render(path, cleanup=True)


def main(fig6_model, genes, out, ref_name="scdef-hgsoc-ref-best5", dec_name="scdef-hgsoc-dec-top2-best5"):
    os.makedirs(out, exist_ok=True)
    a = data()
    pcol = patient_colors(list(a.obs["patient_id"].cat.categories))
    ref = load(ref_name, a, pcol)
    dec = load(dec_name, a, pcol)
    emb = embeddings(a)
    numbers = dict(layer_sizes=dict(ref=[len(f) for f in ref.factor_lists], dec=[len(f) for f in dec.factor_lists]))
    numbers["patients_per_cluster"], pf = fig5(ref, dec, a, pcol, emb, out)
    numbers["patient_specific_factors"] = dict(n=int(len(pf)), patients=sorted(pf["patient"].unique().tolist()),
                                               min_share=float(pf["share"].min()), shapes=pf["shape"].value_counts().to_dict(),
                                               rows=pf.round(3).to_dict("records"))
    m6 = ref if fig6_model == "ref" else dec
    numbers["fig6_model"] = fig6_model
    numbers["fig6"] = fig6(m6, a, pcol, genes, out)
    names_table(m6, factor_enrichment(m6), os.path.join(out, "supp_hgsoc_names_table.tex"))
    numbers["effective_factors"] = {name: effective_factors(m) for name, m in (("ref", ref), ("dec", dec))}
    tree_pdf(m6, os.path.join(out, "supp_hgsoc_tree"))
    with open(os.path.join(out, "hgsoc_numbers.json"), "w") as f:
        json.dump(numbers, f, indent=1, default=float)
    print(json.dumps(dict(layer_sizes=numbers["layer_sizes"], patients_per_cluster=numbers["patients_per_cluster"],
                          effective_factors={k: {kk: v[kk] for kk in ("patients_higher_post", "n_patients")}
                                             for k, v in numbers["effective_factors"].items()}), indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig6-model", default="ref", choices=["ref", "dec-top2"])
    ap.add_argument("--genes", default="JUN,CD74,HIST1H4C,ISG15")
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--ref", default="scdef-hgsoc-ref-best5")
    ap.add_argument("--dec", default="scdef-hgsoc-dec-top2-best5")
    a_ = ap.parse_args()
    main(a_.fig6_model, a_.genes.split(","), a_.out, a_.ref, a_.dec)
