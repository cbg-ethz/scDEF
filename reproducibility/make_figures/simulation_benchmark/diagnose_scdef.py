"""Why does the kept scDEF model score low on the simulations with similar groups? (2026-11-06)

    JAX_PLATFORMS=cpu python diagnose_scdef.py <id> [<id> ...]

Reads the saved kept model of each data set (no fitting) and prints:
- the layer sizes and the ARI of every layer against the 8 true groups;
- for layer 0 and layer 1, how the factors cover the true groups: per factor its cells, its main group and the
  fraction of its cells in that group (purity); per group the number of factors that hold at least 10% of its cells;
- whether the information is in the data: ARI of KMeans with 8 clusters on the PCA that scDEF is initialised from,
  and on the layer-0 weights of scDEF (row-normalised);
- whether a split of a group follows library size: the fraction of the variance of log total counts within the group
  explained by the layer-0 factor (eta squared), for groups split over at least two factors.
"""
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari

import simcommon as S


def eta2(y, g):
    y = np.asarray(y, float)
    tot = ((y - y.mean()) ** 2).sum()
    if tot == 0:
        return np.nan
    between = sum(((y[g == k].mean() - y.mean()) ** 2) * (g == k).sum() for k in np.unique(g))
    return between / tot


def main(ds):
    import os
    import scdef as scd
    a = S.scdef_adata(ds)
    m = scd.scDEF.load(os.path.join(S.ds_dir(ds, "scdef", "scDEF"), "model"), adata=a)
    if "L0" not in m.adata.obs:
        m.annotate_adata()
    obs = m.adata.obs
    true = obs["GroupC"].astype(str).to_numpy()
    print(f"\n===== {ds}: layer sizes {[len(f) for f in m.factor_names]}")
    for i, name in enumerate(m.layer_names):
        if name in obs:
            print(f"  ARI {name} vs 8 groups: {ari(true, obs[name].astype(str)):.2f}   vs 4: "
                  f"{ari(obs['GroupB'].astype(str), obs[name].astype(str)):.2f}   vs 2: "
                  f"{ari(obs['GroupA'].astype(str), obs[name].astype(str)):.2f}")
    for name in m.layer_names[:2]:
        lab = obs[name].astype(str).to_numpy()
        ct = pd.crosstab(lab, true)
        fac = pd.DataFrame({"cells": ct.sum(1), "main": ct.idxmax(1), "purity": (ct.max(1) / ct.sum(1)).round(2)})
        print(f"  {name} factors (cells, main group, purity):")
        print("   " + "; ".join(f"{k}:{r.cells},{r.main},{r.purity}" for k, r in fac.sort_values("main").iterrows()))
        frac = ct / ct.sum(0)
        print(f"  {name} factors holding >=10% of each group: "
              + ", ".join(f"{g}:{int((frac[g] >= 0.1).sum())}" for g in sorted(ct.columns)))
        print(f"  {name} median purity {fac.purity.median():.2f}; factors with purity >= 0.8: "
              f"{int((fac.purity >= 0.8).sum())}/{len(fac)}")
    pca = np.asarray(a.obsm["X_pca"])
    km = KMeans(8, n_init=10, random_state=0).fit_predict(pca)
    xl = np.asarray(m.adata.obsm["X_L0"], float)
    xl = xl / np.maximum(xl.sum(1, keepdims=True), 1e-12)
    km2 = KMeans(8, n_init=10, random_state=0).fit_predict(xl)
    print(f"  KMeans(8) ARI: on the PCA {ari(true, km):.2f}; on row-normalised layer-0 weights {ari(true, km2):.2f}")
    lib = np.log(np.asarray(a.layers["counts"].sum(1)).ravel())
    l0 = obs[m.layer_names[0]].astype(str).to_numpy()
    out = []
    for g in sorted(np.unique(true)):
        sel = true == g
        if len(np.unique(l0[sel])) >= 2:
            out.append(f"{g}:{eta2(lib[sel], l0[sel]):.2f}")
    print("  eta^2 of log library size by L0 factor within each split group: " + ", ".join(out))
    print(f"  eta^2 of log library size by the 8 true groups: {eta2(lib, true):.2f}")


if __name__ == "__main__":
    for ds in sys.argv[1:]:
        main(ds)
