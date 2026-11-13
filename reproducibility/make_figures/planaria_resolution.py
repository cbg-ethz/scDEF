"""Why does the fixed workflow give so few layer-0 factors on the planaria? One seed per variant, scored at every step.

Pedro, 2026-11-04: the method's best solution should resolve the planaria subtypes, as the submitted fit did
(about 45 layer-0 factors). The kept fixed-workflow fit has 15, and even its 36 refit factors before the second
filter resolve only 15 of the 51 cell types (F1 >= 0.5). Each variant changes one setting of the model or the
workflow; for each, the cell types resolved at layer 0 are counted after the first fit (all factors, and those with
BRD >= 1), after the first filter, after the refit and after the second filter.

    python reproducibility/make_figures/planaria_resolution.py [--data NAME] [variant ...]

``--data`` (default planaria) runs the same variants on another data set of ``refit_best_of_seeds`` without
batches (for example pbmcs3k), to check that a change made for the planaria does not hurt elsewhere; cell types
are ``obs["cell_type"]``, and the numbers go to ``out/<data>_resolution.json``.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refit_best_of_seeds as rb  # noqa: E402

import numpy as np
import pandas as pd
import scdef as scd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "planaria_resolution.json")
SEED = 42
# model kwargs, fit kwargs, second-filter overrides
VARIANTS = {
    "base": ({}, {}, {}),
    "hw0.5": (dict(hierarchy_weight=0.5), {}, {}),        # alpha ~0.29: the cell-weight shape of scdef 0.0.22 (0.3)
    "hw0.3": (dict(hierarchy_weight=0.3), {}, {}),        # weaker inheritance of cell weights across layers
    "hw0.1": (dict(hierarchy_weight=0.1), {}, {}),
    "hw3": (dict(hierarchy_weight=3.0), {}, {}),          # stronger
    "k200": (dict(n_factors=200), {}, {}),               # more starting factors
    # The ARD prior (Exponential(1) by default) lets a few factors take most of the loading: in the base refit the
    # sum of W per factor runs from 0.002 (lower quartile) to 19.5. Tighter priors around the same mean of 1.
    "ard10": (dict(shrinkage_shape=10.0, shrinkage_rate=10.0), {}, {}),
    "ard100": (dict(shrinkage_shape=100.0, shrinkage_rate=100.0), {}, {}),
    # the initial z drawn tightly around the k-means labels instead of Gamma(0.058, .)
    "zconc10": ({}, dict(z_init_concentration=10.0), {}),
    "zconc100": ({}, dict(z_init_concentration=100.0), {}),
    "zconc10_ard100": (dict(shrinkage_shape=100.0, shrinkage_rate=100.0), dict(z_init_concentration=10.0), {}),
    # the package defaults after the initialisation fix of 2026-11-04 (concentration 10 with the hierarchical
    # init, per-layer concentrations used), alone and with the tight ARD prior
    "initfix": ({}, {}, {}),
    "initfix_ard100": (dict(shrinkage_shape=100.0, shrinkage_rate=100.0), {}, {}),
}


def resolved(m, a, idx=None):
    """Cell types (and tissues) with a layer-0 factor of F1 >= 0.5, assigning each cell to its largest weight among
    the factors ``idx`` (default: the kept ones)."""
    z = np.asarray(m.pmeans["L0z"])
    idx = np.asarray(m.factor_lists[0] if idx is None else idx)
    lab = idx[np.argmax(z[:, idx], 1)]
    types = a.obs["cell_type"].astype(str).to_numpy()
    tissue = a.obs["clusters_coarse" if "clusters_coarse" in a.obs else "cell_type"].astype(str).to_numpy()
    out = {}
    for key, y in (("types", types), ("tissues", tissue)):
        ct = pd.crosstab(y, lab)
        tp = ct.to_numpy()
        prec = tp / np.maximum(tp.sum(0, keepdims=True), 1)
        rec = tp / np.maximum(tp.sum(1, keepdims=True), 1)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0)
        best = pd.Series(f1.max(1), index=ct.index)
        out[key] = int((best >= 0.5).sum())
        if key == "types":
            t_of = pd.Series(tissue).groupby(types).agg(lambda s: s.mode().iloc[0])
            out["types_not_neoblast"] = int((best[[c for c in best.index if t_of[c] != "stem cells"]] >= 0.5).sum())
            out["by_tissue"] = {t: f"{int((best[t_of[best.index] == t] >= 0.5).sum())}/{int((t_of == t).sum())}"
                                for t in sorted(set(tissue))}
    out["n_factors"] = int(len(idx))
    from sklearn.metrics import adjusted_rand_score
    out["ari_types"] = round(float(adjusted_rand_score(types, lab)), 3)
    return out


def brd_kept(m):
    scd.tl.factor_diagnostics(m)
    fo = m.adata.uns["factor_obs"]
    l0 = fo[fo.index.astype(str).str.startswith("L0")]
    brd = l0["BRD"].astype(float).to_numpy()
    return np.asarray(m.factor_lists[0])[brd >= 1.0]


def run(name, a, data="planaria"):
    mk, fk, second = VARIANTS[name]
    t0 = time.time()
    m = scd.scDEF(a.copy(), counts_layer="counts", seed=SEED, **mk)
    m.fit(hierarchical_init=True, **fk)
    rec = dict(alpha=float(m.alpha), first_fit=dict(all=resolved(m, a, range(np.asarray(m.pmeans["L0z"]).shape[1])),
                                                   brd_ge_1=resolved(m, a, brd_kept(m))))
    scd.tl.filter(m, **rb.FIXED_FIRST)
    rec["first_filter"] = resolved(m, a)
    m.fit(hierarchical_init=True, **fk)
    rec["refit"] = resolved(m, a, range(np.asarray(m.pmeans["L0z"]).shape[1]))
    scd.tl.filter(m, **{**rb.FIXED_SECOND, **second})
    rec["second_filter"] = resolved(m, a)
    rec["layer_sizes"] = [len(f) for f in m.factor_lists]
    rec["final_loss"] = rb.final_loss(m)
    rec["minutes"] = round((time.time() - t0) / 60, 1)
    rb.save(m, f"{data}-resolution-{name}")
    return rec


if __name__ == "__main__":
    args = sys.argv[1:]
    data = "planaria"
    if args[:1] == ["--data"]:
        data, args = args[1], args[2:]
    out = OUT if data == "planaria" else os.path.join(HERE, "out", f"{data}_resolution.json")
    log = json.load(open(out)) if os.path.exists(out) else {}
    a = rb.tutorial_adata(data)
    for name in args or list(VARIANTS):
        if name in log:
            continue
        log[name] = run(name, a, data)
        json.dump(log, open(out, "w"), indent=1)
        r = log[name]
        print(data, name, "alpha", round(r["alpha"], 3), "| types resolved: first fit", r["first_fit"]["all"]["types"],
              "(BRD>=1:", r["first_fit"]["brd_ge_1"]["types"], ") first filter", r["first_filter"]["types"],
              "refit", r["refit"]["types"], "second filter", r["second_filter"]["types"],
              "| ARI", r["second_filter"]["ari_types"], "| layers", r["layer_sizes"],
              "| loss", r["final_loss"], "|", r["minutes"], "min", flush=True)
    print("done", flush=True)
