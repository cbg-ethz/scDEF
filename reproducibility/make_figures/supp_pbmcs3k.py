"""Supplementary figures for the 3k PBMCs: the unsupervised hierarchy, the learned scales, and the tree.

Model: ``scdef-pbmcs3k-best5`` (``refit_best_of_seeds.py pbmcs3k``: the tutorial's fit, filter, re-fit
and filter steps from five seeds, lowest final loss kept). Upper-layer assignments follow the tree
(the package default). Replaces the original submission's ``pbmcs3k``, ``pbmcs3k_scales`` and
``pbmcs3k_tree_sigs_obs``, which came from an older model.

    python reproducibility/make_figures/supp_pbmcs3k.py [--model NAME] [--out DIR]

Writes ``out/supp_pbmcs3k.{pdf,png}``, ``out/supp_pbmcs3k_scales.{pdf,png}``, ``out/supp_pbmcs3k_tree.pdf``
and ``out/supp_pbmcs3k_numbers.json``.
"""
import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig4_batch as f4  # noqa: E402  (style, palette, paths)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scdef as scd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scdef.utils import data_utils

TYPES = ["Naive CD4 T", "Memory CD4 T", "CD8 T", "NK", "B", "CD14+ Mono", "FCGR3A+ Mono", "DC", "Platelet"]


def tutorial_markers():
    nb = json.load(open(f"{f4.NB}/scdef-pbmcs3k-tutorial.ipynb"))
    ns = {}
    exec("".join(nb["cells"][2]["source"]), ns)          # the tutorial's marker list
    return ns["markers"]


def load(name):
    m = scd.scDEF.load(os.path.join(f4.NB, name))
    m.annotate_adata()                                   # tree rule above layer 0 (package default)
    obs = m.adata.obs
    obs["cell_type"] = obs["Cell types"].astype(str)
    obs["Cell types"] = pd.Categorical(obs["Cell types"].astype(str), categories=TYPES)
    m.adata.uns["Cell types_colors"] = [plt.matplotlib.colors.to_hex(c) for c in f4.palette(TYPES).values()]
    scd.tl.set_confident_signatures(m)                   # filtering clears them; graphs and scores need them
    return m


def layer_title(m, k):
    """'Layer 1', or for iscDEF's named layers 'Layer 2 (markers)'."""
    name = m.layer_names[k]
    if getattr(m, "markers_layer", None) == k:
        return f"Layer {k} (markers)"
    return f"Layer {k}" if name == f"L{k}" else f"Layer {k} ({name})"


def shown_layers(m):
    return [k for k in range(m.n_layers) if len(m.factor_names[k]) > 1]


def factor_labels(m):
    """'L0_3: CD14+ Mono' where at least half of a factor's cells share one annotation."""
    obs, out = m.adata.obs, {}
    for k, names in enumerate(m.factor_names):
        col = m.layer_names[k]
        for f in names:
            mk = (obs[col].astype(str) == f).to_numpy()
            if mk.any():
                t = obs.loc[mk, "cell_type"].value_counts(normalize=True)
                out[f] = f"{f}: {t.index[0]}" if t.iloc[0] >= 0.5 else f
    return out


def graph_image(m, signatures):
    g = scd.pl.make_graph(m, wedged="Cell types", show_signatures=signatures, top_genes=10,
                          factor_annotations=factor_labels(m), n_cells_label=True)
    return g


def hierarchy_panel(ax, m, size="14,4.2!", fontsize="14"):
    """The hierarchy at page width: factor names only, node size by number of cells, wedges by type;
    the labelled tree with signatures is its own figure."""
    g = scd.pl.make_graph(m, wedged="Cell types", show_signatures=False, n_cells=True)
    g.format = "png"
    g.graph_attr.update(dpi="300", ratio="fill", size=size, nodesep="0.08")
    g.node_attr.update(fontsize=fontsize)
    with tempfile.TemporaryDirectory() as d:
        ax.imshow(plt.imread(g.render(os.path.join(d, "hierarchy"), cleanup=True)))
    ax.axis("off")


def umap_panels(fig, spec, m, tcol):
    layers = [k for k in shown_layers(m) if len(m.factor_names[k]) > 2]   # two factors embed as a line
    scd.tl.umap(m, layers=layers)
    obs = m.adata.obs
    order = np.random.default_rng(0).permutation(m.adata.n_obs)
    g = GridSpecFromSubplotSpec(1, len(layers), subplot_spec=spec, wspace=0.05)
    for i, k in enumerate(layers):
        ax = fig.add_subplot(g[i])
        xy = np.asarray(m.adata.obsm[f"X_umap_{m.layer_names[k]}"])
        ax.scatter(xy[order, 0], xy[order, 1], s=0.4, lw=0, rasterized=True,
                   c=[tcol[t] for t in obs["cell_type"].to_numpy()[order]])
        ax.set_title(f"Layer {k}", fontsize=6.5, pad=2)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#cccccc")
    return fig.axes[-len(layers)]


def score_blocks(fig, spec, m, mats, title, cmap, vmax=1.0):
    """One heatmap per layer: rows are the annotated types, columns that layer's factors ordered by
    the type they score highest for."""
    layers = shown_layers(m)
    widths = [len(m.factor_names[k]) for k in layers]
    g = GridSpecFromSubplotSpec(1, len(layers), subplot_spec=spec, width_ratios=widths, wspace=0.06)
    first = None
    for i, k in enumerate(layers):
        M = np.nan_to_num(mats[k])
        order = sorted(range(M.shape[1]), key=lambda j: (int(np.argmax(M[:, j])), -M[:, j].max()))
        ax = fig.add_subplot(g[i])
        im = ax.imshow(M[:, order], cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(order)), [m.factor_names[k][j] for j in order], rotation=90, fontsize=4.6)
        ax.set_yticks(range(len(TYPES)), TYPES if i == 0 else [], fontsize=5.6)
        ax.tick_params(length=0)
        marker = getattr(m, "markers_layer", None) == k
        ax.set_title(layer_title(m, k), fontsize=6.5, pad=2, color="#2e7d32" if marker else f4.INK,
                     fontweight="bold" if marker else "normal")
        first = first or ax
    cb = fig.colorbar(im, ax=fig.axes[-len(layers):], fraction=0.02, pad=0.01)
    cb.ax.set_title(title, fontsize=5.4, pad=3, loc="left")
    cb.ax.tick_params(labelsize=5.2, width=0.4, length=2)
    cb.outline.set_linewidth(0.4)
    return first


def relevance_panels(ax_c, ax_d, m):
    """BRD of every layer-0 factor of the final fit, and against the sparsity (Gini) of its loadings;
    kept and removed factors marked (the package's QC plots deliberately do not)."""
    brd = np.asarray(m.pmeans["brd"]).ravel()
    W = np.asarray(m.pmeans[f"{m.layer_names[0]}W"])
    gini = np.array([scd.utils.score_utils.gini(W[k]) for k in range(W.shape[0])])
    kept = np.isin(np.arange(brd.size), np.asarray(m.factor_lists[0]))
    col = np.where(kept, "#2f6f9e", "#b5b5b5")
    ax_c.bar(np.arange(brd.size), brd, color=col, width=0.8)
    ax_c.set_xlabel("layer-0 factor, before the last filter", labelpad=1)
    ax_c.set_ylabel("BRD posterior mean", labelpad=1)
    ax_c.set_xticks([])
    ax_d.scatter(gini[~kept], brd[~kept], s=10, color="#b5b5b5", lw=0, label=f"removed ({(~kept).sum()})")
    ax_d.scatter(gini[kept], brd[kept], s=10, color="#2f6f9e", lw=0, label=f"kept ({kept.sum()})")
    ax_d.set_xlabel("Gini index of the gene loadings", labelpad=1)
    ax_d.set_ylabel("BRD posterior mean", labelpad=1)
    ax_d.set_xlim(0, 1)
    ax_d.legend(frameon=False, fontsize=5.6, loc="upper left", handletextpad=0.1)
    for a in (ax_c, ax_d):
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    return dict(n_l0_before_last_filter=int(brd.size), n_kept=int(kept.sum()),
                spearman_gini_brd=float(pd.Series(gini).corr(pd.Series(brd), method="spearman")))


def main(name, out):
    os.makedirs(out, exist_ok=True)
    m = load(name)
    markers = tutorial_markers()
    tcol = f4.palette(TYPES)
    f1 = data_utils.get_assignment_scores(m, "Cell types", TYPES)
    sig = data_utils.get_signature_scores(m, "Cell types", TYPES, markers=markers, top_genes=20)

    fig = plt.figure(figsize=(7.2, 9.6))
    G = GridSpec(5, 2, figure=fig, height_ratios=[2.3, 1.05, 1.15, 1.25, 1.25], width_ratios=[1.6, 1],
                 hspace=0.45, wspace=0.25, left=0.1, right=0.95, top=0.99, bottom=0.04)
    letters = []
    ax = fig.add_subplot(G[0, :]); hierarchy_panel(ax, m); letters.append((ax, "A"))
    letters.append((umap_panels(fig, G[1, :], m, tcol), "B"))
    handles = [plt.Line2D([], [], marker="o", lw=0, ms=3, color=tcol[t], label=t) for t in TYPES]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=5.6,
               bbox_to_anchor=(0.52, G[1, :].get_position(fig).y0 - 0.002), handletextpad=0.1, columnspacing=0.8)
    ax_c, ax_d = fig.add_subplot(G[2, 0]), fig.add_subplot(G[2, 1])
    brd_numbers = relevance_panels(ax_c, ax_d, m)
    letters += [(ax_c, "C"), (ax_d, "D")]
    letters.append((score_blocks(fig, G[3, :], m, f1, "F1", "viridis"), "E"))
    letters.append((score_blocks(fig, G[4, :], m, sig, "score", "viridis"), "F"))
    for a, letter in letters:
        bb = a.get_position()
        fig.text(max(bb.x0 - 0.075, 0.005), bb.y1 + 0.004, letter, fontsize=10, fontweight="bold", va="bottom")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"supp_pbmcs3k.{ext}"), dpi=400)
    plt.close(fig)

    fig = scd.pl.scales(m, figsize=(7.2, 3.2), fontsize=8, legend_fontsize=7, show=False)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"supp_pbmcs3k_scales.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    g = graph_image(m, signatures=True)
    g.format = "pdf"
    g.render(os.path.join(out, "supp_pbmcs3k_tree"), cleanup=True)

    obs = m.adata.obs
    numbers = dict(model=name, n_cells=int(m.adata.n_obs), n_genes=int(m.adata.n_vars),
                   layer_sizes=[len(f) for f in m.factor_lists], brd=brd_numbers, composition={})
    for k in shown_layers(m):
        col = m.layer_names[k]
        ct = pd.crosstab(obs[col].astype(str), obs["cell_type"])
        numbers["composition"][col] = {f: {t: round(v, 2) for t, v in (r / r.sum()).sort_values(ascending=False).items() if v >= 0.1}
                                       | {"n": int(r.sum())} for f, r in ct.iterrows()}
    numbers["best_f1_per_type"] = {m.layer_names[k]: dict(zip(TYPES, np.round(np.nanmax(f1[k], axis=1), 2).tolist()))
                                   for k in shown_layers(m)}
    numbers["best_signature_score_per_type"] = {m.layer_names[k]: dict(zip(TYPES, np.round(np.nanmax(sig[k], axis=1), 2).tolist()))
                                                for k in shown_layers(m)}
    with open(os.path.join(out, "supp_pbmcs3k_numbers.json"), "w") as f:
        json.dump(numbers, f, indent=1, default=float)
    print(json.dumps(dict(layer_sizes=numbers["layer_sizes"], best_f1_per_type=numbers["best_f1_per_type"]), indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="scdef-pbmcs3k-best5")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
    a = ap.parse_args()
    main(a.model, a.out)
