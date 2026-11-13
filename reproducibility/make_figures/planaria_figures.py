"""Figure 3 (whole adult planarian) and its supplementary hierarchy, in the layout of the submitted figure.

Data: Plass et al. 2018 (``refit_best_of_seeds.PLANARIA``). Model: ``scdef-planaria-best5``
(``refit_best_of_seeds.py planaria``).

Figure 3: A UMAP of the cell weights at layers 0-3, by cell type (legend grouped by tissue); B hierarchy with
nodes split by cell type (colours of A); C association (F1) between tissues and the factors of each layer; D hierarchy coloured by
the scores of mhc (muscle) and chat (neurons); E the finest layer at which each cell is confidently assigned, by
differentiation stage; F where neoblasts, progenitors and differentiated cells sit along the differentiation path
of each lineage.

    python reproducibility/make_figures/planaria_figures.py [--model NAME] [--out DIR]
"""
import argparse
import json
import re
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig4_batch as f4  # noqa: E402  (style, paths)
import hgsoc_figures as hf  # noqa: E402  (graph rendering helpers)
import refit_best_of_seeds as rb  # noqa: E402  (data)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scdef as scd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scdef.utils import data_utils

HERE = os.path.dirname(os.path.abspath(__file__))
STAGES = ["neoblast", "progenitor", "differentiated"]
STAGE_COLORS = dict(zip(STAGES, ["#8c8c8c", "#e0a33a", "#2f6f9e"]))
LINEAGES = ["epidermis", "neurons", "muscle", "gut", "parenchyma", "pharynx"]


def load(name):
    a = rb.tutorial_adata("planaria")
    m = scd.scDEF.load(os.path.join(f4.NB, name))
    if m.adata is None or "clusters" not in m.adata.obs:
        m.adata = a[m.adata.obs_names].copy() if m.adata is not None else a
    m.annotate_adata()
    obs = m.adata.obs
    for k in ("clusters", "clusters_coarse"):
        obs[k] = pd.Categorical(a.obs.loc[obs.index, k].astype(str), categories=a.obs[k].cat.categories)
        m.adata.uns[f"{k}_colors"] = list(a.uns[f"{k}_colors"])
    cl, coarse = obs["clusters"].astype(str), obs["clusters_coarse"].astype(str)
    # tissue neoblasts ("epidermal neoblasts", "epidermis DVb neoblast") are neoblasts too, not differentiated cells
    stage = np.where((coarse == "stem cells") | cl.str.contains("neoblast"), "neoblast",
                     np.where(cl.str.contains("progenitor"), "progenitor", "differentiated"))
    obs["stage"] = pd.Categorical(stage, STAGES)
    return m


def shown_layers(m):
    """Layers drawn in panels A and C: all but a top layer made of a single root factor, which carries no grouping."""
    return list(range(m.n_layers - (1 if len(m.factor_names[-1]) == 1 else 0)))


def umap_layers(m):
    """Layers drawn in panel A: the first four, as in the submitted figure. Above them the layers hold a handful of
    factors (five and three in the kept fit), and a UMAP of so few dimensions draws threads, not groups; panels B and
    C show every layer."""
    return shown_layers(m)[:4]


def colors(m, key):
    return dict(zip(m.adata.obs[key].cat.categories, m.adata.uns[f"{key}_colors"]))


def umap_row(fig, spec, m):
    layers = umap_layers(m)
    g = GridSpecFromSubplotSpec(1, len(layers), subplot_spec=spec, wspace=0.04)
    scd.tl.umap(m, layers=layers)
    col = colors(m, "clusters")
    v = m.adata.obs["clusters"].astype(str).to_numpy()
    for k in layers:
        ax = fig.add_subplot(g[k])
        hf.scatter(ax, np.asarray(m.adata.obsm[f"X_umap_{m.layer_names[k]}"]), v, col, f"Layer {k}")


def type_legend(fig, spec, m, ncols=6):
    """Cell types grouped by tissue, in the colours of panel A: whole tissue blocks flowed into ``ncols`` columns.
    The 13 neoblast clusters are one entry, coloured as their middle cluster."""
    obs = m.adata.obs
    col = colors(m, "clusters")
    tissue_of = obs.groupby("clusters", observed=True)["clusters_coarse"].agg(lambda x: x.astype(str).mode().iloc[0])
    ax = fig.add_subplot(spec)
    ax.axis("off")
    blocks = []
    for t in [t for t in obs["clusters_coarse"].cat.categories if t in set(tissue_of)]:
        types = [c for c in tissue_of.index if tissue_of[c] == t]
        neo = [c for c in types if re.fullmatch(r"neoblast \d+", c)]
        if len(neo) > 2:
            nums = sorted(int(c.split()[-1]) for c in neo)
            mid = f"neoblast {nums[len(nums) // 2]}"
            types = [c for c in types if c not in neo] + [(f"neoblast {nums[0]}\u2013{nums[-1]}", col[mid])]
        blocks.append((t, [(c, col[c]) if isinstance(c, str) else c for c in types]))
    rows = max(max(len(b) + 1 for _, b in blocks), int(np.ceil(sum(len(b) + 1 for _, b in blocks) / ncols)))
    while True:                                   # the fewest rows that fit every block into ncols columns
        cols, used = 1, 0
        for _, b in blocks:
            if used and used + len(b) + 1 > rows:
                cols, used = cols + 1, 0
            used += len(b) + 1
        if cols <= ncols:
            break
        rows += 1
    # assign blocks to columns, then size each column by its longest label
    colof, c, used = [], 0, 0
    for t, b in blocks:
        if used and used + len(b) + 1 > rows:
            c, used = c + 1, 0
        colof.append((c, used)); used += len(b) + 1
    longest = [max(max((len(n) + 3 for _, bb in [blk] for n, _ in bb), default=0), len(blk[0]))
               for blk in blocks]
    widths = [max([longest[i] for i, (cc, _) in enumerate(colof) if cc == k] or [1]) for k in range(colof[-1][0] + 1)]
    xs = np.concatenate([[0], np.cumsum(widths)[:-1]]) / sum(widths)
    dy = 1.0 / rows
    for (t, b), (c, used) in zip(blocks, colof):
        x = xs[c]
        y = 1.0 - used * dy
        ax.text(x, y, t, fontsize=4.4, fontweight="bold", va="top", transform=ax.transAxes)
        for i, (c, color) in enumerate(b, start=1):
            yy = y - i * dy
            ax.scatter([x + 0.005], [yy - 0.35 * dy], s=5, color=color, transform=ax.transAxes, clip_on=False, lw=0)
            ax.text(x + 0.012, yy, c, fontsize=3.9, va="top", transform=ax.transAxes)


def association_panel(fig, spec, m):
    """F1 score between each tissue and each factor of each layer (hard assignments), as in the submitted figure."""
    mats, clusters, vals = data_utils.prepare_obs_factor_scores(m, ["clusters_coarse"], data_utils.get_assignment_scores)
    mats, tissues = mats["clusters_coarse"], list(vals["clusters_coarse"])
    layers = shown_layers(m)
    widths = [max(len(m.factor_names[k]) + 1, 7) for k in layers]     # room for the title of a narrow layer
    g = GridSpecFromSubplotSpec(1, len(layers), subplot_spec=spec, width_ratios=widths, wspace=0.08)
    out = {}
    for k in layers:
        M = np.asarray(mats[k])
        names = list(m.factor_names[k])
        order = sorted(range(len(names)), key=lambda j: (int(np.argmax(M[:, j])), -float(M[:, j].max())))
        ax = fig.add_subplot(g[k])
        im = ax.imshow(M[:, order], cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(order)), [names[j].split("_")[-1] for j in order], fontsize=4.2, rotation=90)
        ax.set_yticks(range(len(tissues)), tissues if k == 0 else [], fontsize=4.6)
        ax.tick_params(length=0, pad=1)
        ax.set_title(f"Layer {k}", fontsize=6, pad=2)
        out[m.layer_names[k]] = pd.DataFrame(M, index=tissues, columns=names).round(3).to_dict()
    cb = fig.colorbar(im, ax=fig.axes[-len(layers):], fraction=0.025, pad=0.01)
    cb.set_label("F1 score", fontsize=5.2, labelpad=1)
    cb.ax.tick_params(labelsize=4.6, length=1.5)
    return out


def confident_panel(ax, m):
    """Finest layer at which each cell is confidently assigned (root when none), by differentiation stage."""
    scd.tl.assign_confident(m)
    obs = m.adata.obs
    ct = pd.crosstab(obs["stage"], obs["confident_best_layer"], normalize="index").reindex(STAGES)
    ct = ct[[c for c in m.layer_names if c in ct.columns]]
    cmap = plt.get_cmap("Blues_r", len(ct.columns) + 2)
    left = np.zeros(len(ct))
    for j, layer in enumerate(ct.columns):
        ax.barh(range(len(ct)), ct[layer], left=left, color=cmap(j), lw=0, height=0.7,
                label="root" if j == len(ct.columns) - 1 and len(m.factor_names[-1]) == 1 else f"layer {layer[1:]}")
        left += ct[layer].to_numpy()
    ax.set_yticks(range(len(ct)), ct.index, fontsize=5.4)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of cells", fontsize=5.6, labelpad=1)
    ax.tick_params(axis="x", labelsize=5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=4.8, frameon=False, ncol=len(ct.columns), loc="lower left", bbox_to_anchor=(0, 1.0),
              handlelength=1, columnspacing=0.6, handletextpad=0.3)
    return ct.round(3).to_dict("index")


def lineage_paths(m):
    """For each lineage, the differentiation path to the layer-0 factor holding most of its differentiated cells
    (highest path weight product), and the median position of neoblasts, progenitors and differentiated cells of that
    tissue along it. The leaf is chosen by the differentiated cells, not all the tissue's cells: where progenitors
    outnumber them (epidermis, pharynx) and have a factor of their own, a leaf chosen by all cells is a progenitor
    factor, and the path no longer runs from neoblasts to a resolved cell type."""
    scd.tl.build_differentiation_paths(m)
    scd.tl.score_paths(m, paths_key="differentiation_paths")
    obs = m.adata.obs
    paths = list(m.adata.uns["differentiation_paths"]["paths"])
    pos = m.adata.obsm["differentiation_paths_positions"]
    coarse = obs["clusters_coarse"].astype(str).to_numpy()
    stage = obs["stage"].astype(str).to_numpy()
    l0 = obs["L0"].astype(str).to_numpy()
    out = {}
    for t in LINEAGES:
        leaf = pd.Series(l0[(coarse == t) & (stage == "differentiated")]).value_counts()
        if leaf.empty:
            continue
        leaf = leaf.index[0]
        cand = [(i, p) for i, p in enumerate(paths) if p["nodes"][-1] == leaf]
        if not cand:
            continue
        i, p = max(cand, key=lambda ip: ip[1].get("path_weight_product", 0.0))
        sel = (coarse == t) | (stage == "neoblast")
        med = {s: float(np.nanmedian(pos[sel & (stage == s), i])) for s in STAGES if (sel & (stage == s)).sum() >= 10}
        out[t] = dict(path=list(p["nodes"]), median_position=med, n_cells={s: int((sel & (stage == s)).sum()) for s in STAGES})
    return out


def lineage_panel(ax, lin):
    rows = [t for t in LINEAGES if t in lin]
    for r, t in enumerate(rows):
        med = lin[t]["median_position"]
        xs = [med.get(s, np.nan) for s in STAGES]
        ax.plot([x for x in xs if np.isfinite(x)], [r] * sum(np.isfinite(xs)), color="#cccccc", lw=0.8, zorder=1)
        for s, x in zip(STAGES, xs):
            if np.isfinite(x):
                ax.scatter([x], [r], s=14, color=STAGE_COLORS[s], zorder=2, lw=0)
    ax.set_yticks(range(len(rows)), rows, fontsize=5.4)
    ax.invert_yaxis()
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("median position on the lineage's path\n(top of the hierarchy = 0, leaf = 1)", fontsize=5.4, labelpad=1)
    ax.tick_params(axis="x", labelsize=5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    h = [plt.Line2D([], [], marker="o", ls="", ms=3.5, mfc=STAGE_COLORS[s], mec="none") for s in STAGES]
    ax.legend(h, STAGES, fontsize=4.8, frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0),
              handletextpad=0.2, columnspacing=0.6)


def _step(name, t0=[None]):
    import time
    now = time.time()
    if t0[0] is not None:
        print(f"  {name}: {now - t0[0]:.0f} s since the last step", flush=True)
    t0[0] = now


def fig3(m, out):
    _step("start")
    fig = plt.figure(figsize=(7.2, 6.2))
    G = GridSpec(3, 3, figure=fig, height_ratios=[1.95, 1.55, 1.05], width_ratios=[1.0, 1.25, 1.0],
                 hspace=0.3, wspace=0.32, left=0.06, right=0.965, top=0.965, bottom=0.07)
    gA = GridSpecFromSubplotSpec(2, 1, subplot_spec=G[0, :], height_ratios=[1.25, 0.7], hspace=0.06)
    letters = []
    umap_row(fig, gA[0], m)
    letters.append((gA[0], "A"))
    _step('umap (A)')
    type_legend(fig, gA[1], m)
    ax = fig.add_subplot(G[1, 0])
    hf.graph_png(ax, hf.radial(scd.pl.make_graph(m, hierarchy=scd.tl.get_hierarchy(m, simplified=True),
                                                 wedged="clusters", show_signatures=False, n_cells=True,
                                                 show_label=False), m), "4,4")
    letters.append((G[1, 0], "B"))
    _step('radial tree (B)')
    numbers = dict(layer_sizes=[len(f) for f in m.factor_lists])
    numbers["association_f1"] = association_panel(fig, G[1, 1:], m)
    letters.append((G[1, 1:], "C"))
    _step('association (C)')
    gD = GridSpecFromSubplotSpec(1, 2, subplot_spec=G[2, 0], wspace=0.05)
    for i, gene in enumerate(("mhc", "chat")):
        ax = fig.add_subplot(gD[i])
        hf.graph_png(ax, hf.radial_tree(m, gene_score=gene), "4,4")
        ax.set_title(gene, fontsize=6, pad=1, style="italic")
    letters.append((G[2, 0], "D"))
    _step('gene trees (D)')
    ax = fig.add_subplot(G[2, 1])
    numbers["confident_layer_by_stage"] = confident_panel(ax, m)
    letters.append((G[2, 1], "E"))
    _step('confident (E)')
    ax = fig.add_subplot(G[2, 2])
    lin = lineage_paths(m)
    lineage_panel(ax, lin)
    numbers["lineage_paths"] = lin
    letters.append((G[2, 2], "F"))
    _step('paths (F)')
    for spec, letter in letters:
        bb = spec.get_position(fig)
        fig.text(max(bb.x0 - 0.04, 0.002), bb.y1 + 0.006, letter, fontsize=10, fontweight="bold", va="bottom")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"fig3_planaria.{ext}"), dpi=400)
    plt.close(fig)
    return numbers


def tree_pdf(m, path):
    g = scd.pl.make_graph(m, wedged="clusters_coarse", show_signatures=True, top_genes=10, n_cells_label=True)
    g.format = "pdf"
    return g.render(path, cleanup=True)


def main(name, out):
    os.makedirs(out, exist_ok=True)
    m = load(name)
    numbers = fig3(m, out)
    tree_pdf(m, os.path.join(out, "supp_planaria_tree"))
    with open(os.path.join(out, "planaria_numbers.json"), "w") as f:
        json.dump(numbers, f, indent=1, default=float)
    print(json.dumps(dict(layer_sizes=numbers["layer_sizes"], confident=numbers["confident_layer_by_stage"],
                          lineages={t: v["median_position"] for t, v in numbers["lineage_paths"].items()}), indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="scdef-planaria-best5")
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    a_ = ap.parse_args()
    main(a_.model, a_.out)
