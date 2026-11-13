"""Supplementary figure: the corrected view from step 4 of the batch workflow (the roll-up).

The analyst reads the batch-specific factors of the decomposed fit (Figure 4D, H) and marks those
that are biology from their genes: a factor is kept only if it is specific to one cell type and its
genes are a response (Pedro, 2026-10-19); a programme shared across cell types is rolled up like any
other batch difference. Each batch-specific factor not kept is then replaced by its node in the
corrected layer 2, and each other factor in a branch whose clusters are split by batch by the node
above it — its layer-1 parent, then layer 2 if the split persists.

Three choices, each forced by a failure on the interferon data:
- Batch-specific factors are rolled up explicitly, because the shift test reads cluster labels: a
  programme that is mostly a secondary loading (the lymphocyte interferon programme L0_1, the top
  factor for few of the cells it marks) never splits the labels.
- They go straight to the corrected layer 2, because their parent can itself be batch-specific
  (L0_1's parent L1_7 is the lymphocyte interferon node of the re-learned layer 1).
- The representation is the raw cell weights (`X_L{k}`), not the per-cell normalised probabilities:
  in the probabilities a removed factor's share stays behind as a shrinkage of every other column,
  which still separates the stimulated lymphocytes. Dimensions are replaced for every cell,
so the representation itself, not only the cluster labels, no longer contains the batch-split
dimensions. Every dimension of the result is a named node.

An intermediate version replaced only the cells' labels; its embedding still separated the batches.
Mixing is scored the same way for both methods: Leiden clusters (resolution 1) of each embedding. Scoring
scDEF by its node labels instead gave flattering numbers that its own UMAP did not support (the kept
lymphocyte interferon programme separates the T and B cells by condition, which the node labels hid);
those numbers are kept in the JSON as `mixing_node_labels`.

The split is measured as the batch ARI of the clusters within a branch (> 0.05), the same shift test
as the earlier roll-up check. A first version chose the factors to replace by batch dominance (>= 70%
of a factor's cells from one batch); it missed moderate shifts and left the two-donor CD14+
monocytes split (batch ARI 0.19).

The marking below was written down from the genes in Figure 4 before this script was run; it is a
demonstration of the workflow, not a benchmark.

    python reproducibility/make_figures/supp_rollup.py [--out DIR]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig4_batch as f4  # noqa: E402

import harmonypy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scdef as scd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from sklearn.metrics import adjusted_rand_score

# Batch-specific factors kept as biology (decided 2026-10-19 from Figure 4H). Principle (Pedro): keep
# a factor only if it is specific to one cell type and its genes are a response; a programme shared
# across cell types is rolled up like any other batch difference. Two healthy donors: nothing is kept.
# Interferon (factor names of the fit after the initialisation fix, 2026-11-05): the four monocyte response
# programmes of the stimulated pool, three in CD14 and one in CD16 monocytes; the lymphocyte programme L0_0
# (ISG15 ISG20 IFI6) spans lineages and is rolled up, as are the control-dominated monocyte factors (IL8 S100A8,
# VMO1 FCGR3A, LGALS3), which are no response. (A first version also kept the lymphocyte programme.)
KEEP = {
    "pbmcs2b": [],
    "ifn": ["L0_23",   # CXCL10 CXCL11 IFIT3, CD14 monocytes
            "L0_10",   # ISG15 APOBEC3B APOBEC3A, CD14 monocytes
            "L0_5",    # CCL2 CCL8 CCL7, CD14 monocytes
            "L0_1"],   # IFI27 CTSC IFITM3, CD16 monocytes
}
TAU = 0.05   # a branch is split by batch when the batch ARI of its clusters exceeds this


def rolled_up(dec, ref, keep, audited):
    """Replace dimensions, not just labels: a technical factor's column is removed for every cell and
    its parent's column used instead, so no batch-split dimension remains in the representation."""
    obs = dec.adata.obs
    batch = obs["batch"].astype(str).to_numpy()
    a0, a1, a2 = (obs[f"L{k}"].astype(str).to_numpy() for k in range(3))
    names = {k: list(dec.factor_names[k]) for k in range(3)}
    weights = {k: np.asarray(dec.adata.obsm[f"X_L{k}"]) for k in range(3)}   # raw, not normalised

    def majority(values, mask):
        return pd.Series(values[mask]).value_counts().index[0]

    present = [f for f in names[0] if (a0 == f).any()]
    l1_of = {f: majority(a1, a0 == f) for f in present}       # where each factor's cells sit
    split1 = {p: (a1 == p).sum() >= 30 and len(set(a0[a1 == p])) > 1
              and adjusted_rand_score(batch[a1 == p], a0[a1 == p]) > TAU for p in set(a1)}
    node = {f: (1, l1_of[f]) if f not in keep and split1[l1_of[f]] else (0, f) for f in present}
    # batch-specific factors go to the corrected anchor: their parent can itself be batch-specific
    node.update({f: (2, majority(a2, a0 == f)) for f in present if f in audited and f not in keep})
    # The decomposition re-learns layer 1, so a layer-1 node above a batch-specific factor that is rolled up
    # carries that factor's batch difference in its weights: factors that would roll up to it go to layer 2
    # instead (2026-11-04; in the interferon fit, L1_6 above the lymphocyte programme L0_1).
    tainted = {l1_of[f] for f in present if f in audited and f not in keep}
    node.update({f: (2, majority(a2, a0 == f)) for f in present if node[f][0] == 1 and node[f][1] in tainted})
    cur = np.array([node[f][1] for f in a0])
    for q in sorted(set(a2)):                        # then the corrected layer 2
        mk = a2 == q
        if mk.sum() >= 30 and len(set(cur[mk])) > 1 and adjusted_rand_score(batch[mk], cur[mk]) > TAU:
            for f in present:
                if f not in keep and majority(a2, a0 == f) == q:
                    node[f] = (2, q)
    columns = sorted(set(node.values()))
    emb = np.column_stack([weights[l][:, names[l].index(n)] for l, n in columns])
    labels = np.array([node[f][1] for f in a0])
    W = f4.l0_weights(dec)
    genes = np.asarray(dec.adata.var_names)

    def top(f):
        return " ".join(genes[np.argsort(W[names[0].index(f)])[::-1][:4]])
    replaced = sorted(f for f in present if node[f][0] > 0)
    decision = dict(kept=[(f, top(f)) for f in keep],
                    rolled_up=[f"{f}\u2192{node[f][1]}" for f in replaced],
                    cells_rolled=float(np.mean([node[f][0] > 0 for f in a0])),
                    columns=[n for _, n in columns])
    return emb, labels, decision


def harmony(adata):
    x = adata.copy()
    x.X = x.layers["counts"].copy()
    x.uns.pop("log1p", None)
    sc.pp.normalize_total(x, target_sum=1e4)
    sc.pp.log1p(x)
    sc.pp.regress_out(x, ["total_counts", "pct_counts_mt"])
    sc.pp.scale(x, max_value=10)
    sc.tl.pca(x, n_comps=50, svd_solver="arpack", random_state=0)
    z = np.asarray(harmonypy.run_harmony(x.obsm["X_pca"], x.obs, ["batch"], verbose=False).Z_corr)
    return z if z.shape[0] == x.n_obs else z.T


def umap_and_clusters(emb, n_obs):
    y = sc.AnnData(np.zeros((n_obs, 1)))
    y.obsm["E"] = np.asarray(emb, dtype=float)
    sc.pp.neighbors(y, use_rep="E", n_neighbors=15, random_state=0)
    sc.tl.umap(y, random_state=0)
    sc.tl.leiden(y, resolution=1.0, flavor="igraph", n_iterations=2, directed=False, random_state=0)
    return np.asarray(y.obsm["X_umap"]), y.obs["leiden"].astype(str).to_numpy()


def mixing(ct, batch, labels):
    out = {}
    for t in sorted(set(ct)):
        mk = ct == t
        if mk.sum() < f4.MIN_CELLS or pd.Series(batch[mk]).value_counts().min() < 10:
            continue
        out[t] = adjusted_rand_score(batch[mk], labels[mk])
    return out


def scatter(ax, xy, values, colors, order):
    ax.scatter(xy[order, 0], xy[order, 1], s=0.15, lw=0, rasterized=True,
               c=[colors[v] for v in values[order]])
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#cccccc")


def main(out):
    os.makedirs(out, exist_ok=True)
    fig = plt.figure(figsize=(7.2, 8.4))
    G = GridSpec(5, 2, figure=fig, height_ratios=[0.08, 1.0, 1.0, 1.25, 0.42], width_ratios=[1, 1],
                 hspace=0.36, wspace=0.22, left=0.06, right=0.97, top=0.99, bottom=0.01)
    titles = ("Two healthy donors: nothing kept as biology",
              "Control vs IFN-β: monocyte response kept as biology")
    numbers, letters = {}, []
    for c, (data, prep, blabel) in enumerate((("pbmcs2b", f4.prep_2b, "donor"),
                                              ("ifn", f4.prep_ifn, "condition"))):
        a = prep()
        ref = f4.load(f"scdef-{data}-ref-best5", a)
        dec = f4.load(f"scdef-{data}-dec-top2-best5", a)
        audited = set(f4.batch_factors(dec, ref)["factor"])     # the factors listed in Figure 4D, H
        emb, labels, decision = rolled_up(dec, ref, KEEP[data], audited)
        xy_s, leiden_s = umap_and_clusters(emb, a.n_obs)
        xy_h, leiden_h = umap_and_clusters(harmony(a), a.n_obs)
        ct = a.obs["cell_type"].astype(str).to_numpy()
        b = a.obs["batch"].astype(str).to_numpy()
        mix = pd.DataFrame({"scDEF, rolled up": mixing(ct, b, leiden_s),
                            "Harmony": mixing(ct, b, leiden_h)})
        numbers[data] = dict(decision=decision, n_dimensions=int(emb.shape[1]),
                             n_clusters=dict(scdef=len(set(leiden_s)), harmony=len(set(leiden_h))),
                             mixing=mix.round(3).to_dict("index"),
                             n_node_labels=int(len(set(labels))),
                             mixing_node_labels={k: round(v, 3) for k, v in mixing(ct, b, labels).items()})

        tcol = f4.palette(sorted(set(ct)))
        bcol = dict(zip(sorted(set(b)), f4.BATCH_COLORS))
        order = np.random.default_rng(0).permutation(a.n_obs)
        axt = fig.add_subplot(G[0, c]); axt.axis("off")
        axt.text(0, 0.5, titles[c], fontsize=7.0, fontweight="bold", va="center")
        for r, (xy, name) in enumerate(((xy_s, "scDEF, decomposed and rolled up"), (xy_h, "Harmony"))):
            g = GridSpecFromSubplotSpec(1, 2, subplot_spec=G[1 + r, c], wspace=0.05)
            ax1 = fig.add_subplot(g[0]); scatter(ax1, xy, ct, tcol, order)
            ax2 = fig.add_subplot(g[1]); scatter(ax2, xy, b, bcol, order)
            ax1.set_ylabel(name, fontsize=6.2)
            if r == 0:
                ax1.set_title("cell type", fontsize=6.5, pad=2); ax2.set_title(blabel, fontsize=6.5, pad=2)
            letters.append((g[0], "ABCD"[r] if c == 0 else "EFGH"[r]))
        # within-type batch ARI of the clusters, scDEF rolled-up view vs Harmony
        gb = GridSpecFromSubplotSpec(1, 2, subplot_spec=G[3, c], width_ratios=[0.3, 1], wspace=0)
        ax = fig.add_subplot(gb[1])
        y = np.arange(len(mix))
        ax.barh(y - 0.2, mix["scDEF, rolled up"], height=0.38, color="#2f6f9e", label="scDEF, rolled up")
        ax.barh(y + 0.2, mix["Harmony"], height=0.38, color="#9a9a9a", label="Harmony")
        ax.set_yticks(y, mix.index); ax.invert_yaxis()
        ax.axvline(0, color="#999999", lw=0.5)
        ax.set_xlim(-0.1, 0.9)                       # shared scale, so the two columns compare
        ax.set_xlabel("batch ARI of Leiden clusters within cell type", labelpad=1)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(frameon=False, fontsize=5.6, loc="lower right")
        letters.append((gb[0], "C" if c == 0 else "G"))
        # the decision
        axd = fig.add_subplot(G[4, c]); axd.axis("off")
        import textwrap
        kept = "; ".join(f"{f} ({g})" for f, g in decision["kept"]) or "none"
        rolled = ", ".join(decision["rolled_up"])
        wrap = lambda t: "\n".join(textwrap.wrap(t, 66))
        k = wrap(f"Kept as biology: {kept}")
        axd.text(0, 1.0, k, fontsize=5.5, va="top")
        axd.text(0, 1.0 - 0.2 * (k.count("\n") + 1.6), wrap(
            f"Replaced by the node above ({100 * decision['cells_rolled']:.0f}% of cells): {rolled}"),
            fontsize=5.5, va="top", color=f4.MUTED)
        letters.append((G[4, c], "D" if c == 0 else "H"))
    for spec, letter in letters:
        bb = spec.get_position(fig)
        fig.text(bb.x0 - 0.045, bb.y1 + 0.003, letter, fontsize=10, fontweight="bold", va="bottom")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"supp_rollup.{ext}"), dpi=400)
    with open(os.path.join(out, "supp_rollup_numbers.json"), "w") as f:
        json.dump(numbers, f, indent=1, default=float)
    print(json.dumps({d: dict(n_dimensions=v["n_dimensions"], n_clusters=v["n_clusters"],
                              rolled=len(v["decision"]["rolled_up"]), mixing=v["mixing"])
                      for d, v in numbers.items()}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
    main(ap.parse_args().out)
