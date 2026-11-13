"""Supplementary figures for Figure 4: the two PBMC batch data sets.

Four figures, on the same saved tutorial models and data as ``fig4_batch.py`` (nothing refitted):

- ``supp_pbmcs2b.{pdf,png}`` — two donors: the uncorrected embedding (cell type, donor, Leiden),
  scDEF without and with batch annotations at layers 0 and 2, and batch mixing per layer for the
  three fits (no batch key, corrected, audit);
- ``supp_ifn.{pdf,png}`` — IFN-beta: the same, plus the corrected fit's per-batch gene scaling,
  where the interferon genes end up;
- ``supp_pbmcs2b_tree.pdf`` — hierarchy of the corrected fit, top 10 genes per factor;
- ``supp_ifn_tree.pdf`` — hierarchy of the audit fit, top 10 genes per factor.

    python reproducibility/make_figures/supp_pbmc_batch.py [--out DIR]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig4_batch as f4  # noqa: E402  (sets the environment, rcParams and data paths)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scdef as scd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

INK, MUTED = f4.INK, f4.MUTED


def uncorrected_embedding(adata):
    """The tutorials' scanpy embedding: normalise, log, regress out depth and mt, scale, PCA,
    10-neighbour graph on 40 PCs, UMAP, Leiden at resolution 1."""
    a = adata.copy()
    a.X = a.layers["counts"].copy()
    a.uns.pop("log1p", None)              # X is raw counts again
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.regress_out(a, ["total_counts", "pct_counts_mt"])
    sc.pp.scale(a, max_value=10)
    sc.tl.pca(a, svd_solver="arpack", random_state=0)
    sc.pp.neighbors(a, n_neighbors=10, n_pcs=40, random_state=0)
    sc.tl.umap(a, random_state=0)
    sc.tl.leiden(a, resolution=1.0, key_added="leiden", flavor="igraph", n_iterations=2,
                 directed=False, random_state=0)
    return np.asarray(a.obsm["X_umap"]), a.obs["leiden"].astype(str).to_numpy()


def big_palette(cats):
    pal = sc.pl.palettes.default_102
    return {c: pal[i % len(pal)] for i, c in enumerate(cats)}


def scatter(ax, xy, values, colors, title, labels_on_plot=False, order=None):
    order = np.arange(len(values)) if order is None else order
    ax.scatter(xy[order, 0], xy[order, 1], s=0.15, lw=0, rasterized=True,
               c=[colors[v] for v in values[order]])
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#cccccc")
    ax.set_title(title, pad=2, fontsize=6.5)
    if labels_on_plot:
        for v in pd.unique(values):
            m = values == v
            if m.sum() < 80:
                continue
            x, y = np.median(xy[m], axis=0)
            ax.text(x, y, v.replace("L0_", "").replace("L2_", ""), fontsize=4.6, ha="center",
                    va="center", color=INK,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.7))


def placement_multi(ax, frames):
    """Batch ARI within cell type per layer, one block of columns per fit."""
    rows = sorted(set().union(*[f.index for f in frames.values()]))
    blocks, ticks, labels, centers = [], [], [], []
    col = 0
    for name, f in frames.items():
        blocks.append(f.reindex(rows)[f4.LAYERS].to_numpy())
        ticks += [col, col + 1, col + 2]
        labels += f4.LAYERS
        centers.append((col + 1, name))
        col += 4
        blocks.append(np.full((len(rows), 1), np.nan))
    M = np.hstack(blocks[:-1])
    im = ax.imshow(M, cmap="magma_r", vmin=0, vmax=0.9, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{max(M[i, j], 0):.2f}", ha="center", va="center", fontsize=5,
                        color="white" if M[i, j] > 0.5 else INK)
    ax.set_yticks(range(len(rows)), rows)
    ax.set_xticks(ticks, labels)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for x, name in centers:
        ax.text(x, -0.9, name, ha="center", fontsize=6.2)
    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("batch ARI within cell type", fontsize=5.8)
    cb.ax.tick_params(labelsize=5.4, width=0.4, length=2)
    cb.outline.set_linewidth(0.4)


def gene_scale_panel(ax, ref):
    gs = scd.tl.get_batch_specific_genes_from_gene_scale(ref)
    col = [c for c in gs.columns if "STIM" in str(c) or "IFN" in str(c)][0]
    v = gs[col].sort_values(ascending=False)
    x = np.arange(len(v))
    isg = np.isin(v.index, f4.ISG)
    ax.scatter(x[~isg], v.to_numpy()[~isg], s=1.2, lw=0, color="#b5b5b5", rasterized=True)
    ax.scatter(x[isg], v.to_numpy()[isg], s=7, lw=0, color="#e08a2c", zorder=3)
    for k, (g, val) in enumerate(v[isg].items()):
        if k < 8:
            ax.annotate(g, (x[v.index.get_loc(g)], val), xytext=(10 + 4 * (k % 2), -9 * k + 2),
                        textcoords="offset points", fontsize=5, style="italic",
                        arrowprops=dict(arrowstyle="-", lw=0.3, color=MUTED))
    ax.axhline(0, color="#cccccc", lw=0.5)
    ax.set_xlabel("genes ranked by stimulated / control gene scaling")
    ax.set_ylabel("log2 ratio")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    pct = v.rank(ascending=True, pct=True)[f4.ISG].dropna()
    ax.text(0.98, 0.95, f"interferon genes (orange):\nmedian percentile {100 * np.median(pct):.1f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=5.6, color=MUTED)


def factor_labels(m):
    """'L0_3: CD14 Mono' for factors where at least half the cells are one annotated type."""
    obs = m.adata.obs
    out = {}
    for layer_idx, names in enumerate(m.factor_names):
        key = f"L{layer_idx}"
        if key not in obs:
            continue
        for f in names:
            mk = (obs[key].astype(str) == f).to_numpy()
            if not mk.any():
                continue
            t = obs.loc[mk, "cell_type"].value_counts(normalize=True)
            out[f] = f"{f}: {t.index[0]}" if t.iloc[0] >= 0.5 else f
    return out


def tree_pdf(m, path):
    scd.tl.set_confident_signatures(m)      # filtering clears them; make_graph needs them
    g = scd.pl.make_graph(m, wedged="batch", show_signatures=True, top_genes=10,
                          factor_annotations=factor_labels(m), n_cells_label=True)
    g.format = "pdf"
    return g.render(path, cleanup=True)


def supp_figure(path, title_batch, a, nob, ref, dec, extra=None):
    xy, leiden = uncorrected_embedding(a)
    obs = a.obs
    order = np.random.default_rng(0).permutation(a.n_obs)
    types = sorted(obs["cell_type"].astype(str).unique())
    batches = sorted(obs["batch"].astype(str).unique())
    has_extra = extra is not None

    fig = plt.figure(figsize=(7.2, 7.4 if has_extra else 6.4))
    G = GridSpec(3, 1, figure=fig, height_ratios=[1.15, 1.0, 1.05 if has_extra else 0.9],
                 hspace=0.32, left=0.07, right=0.97, top=0.965, bottom=0.06)

    # A–C: uncorrected
    gA = GridSpecFromSubplotSpec(1, 3, subplot_spec=G[0], wspace=0.62)
    tcol = f4.palette(types)
    bcol = dict(zip(batches, f4.BATCH_COLORS))
    lcol = big_palette(sorted(set(leiden), key=int))
    axA = fig.add_subplot(gA[0]); scatter(axA, xy, obs["cell_type"].astype(str).to_numpy(), tcol, "cell type", order=order)
    axB = fig.add_subplot(gA[1]); scatter(axB, xy, obs["batch"].astype(str).to_numpy(), bcol, title_batch, order=order)
    axC = fig.add_subplot(gA[2]); scatter(axC, xy, leiden, lcol, "Leiden clusters", labels_on_plot=False, order=order)
    h = [plt.Line2D([], [], marker="o", ls="", ms=3, mfc=c, mec="none") for c in tcol.values()]
    axA.legend(h, types, ncol=1, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0),
               fontsize=4.4, handletextpad=0.05, borderaxespad=0.1, labelspacing=0.25)
    h = [plt.Line2D([], [], marker="o", ls="", ms=3, mfc=c, mec="none") for c in bcol.values()]
    axB.legend(h, batches, ncol=1, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0),
               fontsize=5.2, handletextpad=0.05, borderaxespad=0.1)

    # D–E: scDEF factors on the uncorrected embedding, layers 0 and 2
    gD = GridSpecFromSubplotSpec(1, 4, subplot_spec=G[1], wspace=0.08)
    for k, (m, lab) in enumerate(((nob, f4.FIT["nobatch"]), (ref, f4.FIT["ref"]))):
        for j, layer in enumerate(("L0", "L2")):
            vals = m.adata.obs[layer].astype(str).reindex(obs.index).to_numpy()
            ax = fig.add_subplot(gD[2 * k + j])
            scatter(ax, xy, vals, big_palette(sorted(set(vals))), f"{lab}\nlayer {layer[1]}",
                    labels_on_plot=True, order=order)

    # F (and G): batch mixing per layer; gene scaling for IFN
    if has_extra:
        gF = GridSpecFromSubplotSpec(1, 2, subplot_spec=G[2], width_ratios=[0.8, 1.6], wspace=0.55)
        axF = fig.add_subplot(gF[0])
        gene_scale_panel(axF, ref)
        axG = fig.add_subplot(gF[1])
        placement_multi(axG, {f4.FIT["nobatch"]: f4.placement(nob), f4.FIT["ref"]: f4.placement(ref),
                              f4.FIT["dec"]: f4.placement(dec)})
        letters = [(axA, "A"), (axB, "B"), (axC, "C"), (gD[0], "D"), (gD[2], "E"), (axF, "F"), (axG, "G")]
    else:
        gF = GridSpecFromSubplotSpec(1, 2, subplot_spec=G[2], width_ratios=[0.13, 1], wspace=0)
        axF = fig.add_subplot(gF[1])                 # the spacer column holds the row labels
        placement_multi(axF, {f4.FIT["nobatch"]: f4.placement(nob), f4.FIT["ref"]: f4.placement(ref),
                              f4.FIT["dec"]: f4.placement(dec)})
        letters = [(axA, "A"), (axB, "B"), (axC, "C"), (gD[0], "D"), (gD[2], "E"), (axF, "F")]
    for obj, letter in letters:
        bb = obj.get_position(fig)
        lift = 0.032 if letter in ("D", "E") else 0.012      # above the two-line titles
        fig.text(bb.x0 - 0.03, bb.y1 + lift, letter, fontsize=10, fontweight="bold", va="bottom")
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", dpi=400)
    plt.close(fig)
    return dict(n_leiden=int(len(set(leiden))))


def main(out):
    os.makedirs(out, exist_ok=True)
    numbers = {}
    a2b, aif = f4.prep_2b(), f4.prep_ifn()

    nob = f4.load("scdef-pbmcs2b-nobatch-best5", a2b)
    ref = f4.load("scdef-pbmcs2b-ref-best5", a2b)
    dec = f4.load("scdef-pbmcs2b-dec-top2-best5", a2b)
    numbers["pbmcs2b"] = supp_figure(os.path.join(out, "supp_pbmcs2b"), "donor", a2b, nob, ref, dec)
    numbers["pbmcs2b"]["placement_no_batch_key"] = f4.placement(nob).round(3).to_dict("index")
    tree_pdf(ref, os.path.join(out, "supp_pbmcs2b_tree"))

    nob = f4.load("scdef-ifn-nobatch-best5", aif)
    ref = f4.load("scdef-ifn-ref-best5", aif)
    dec = f4.load("scdef-ifn-dec-top2-best5", aif)
    numbers["ifn"] = supp_figure(os.path.join(out, "supp_ifn"), "condition", aif, nob, ref, dec, extra=True)
    numbers["ifn"]["placement_no_batch_key"] = f4.placement(nob).round(3).to_dict("index")
    tree_pdf(dec, os.path.join(out, "supp_ifn_tree"))

    with open(os.path.join(out, "supp_pbmc_numbers.json"), "w") as f:
        json.dump(numbers, f, indent=1, default=float)
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
    main(ap.parse_args().out)
