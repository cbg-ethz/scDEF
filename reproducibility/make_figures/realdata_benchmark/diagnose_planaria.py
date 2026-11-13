"""Why scDEF scores low on two planaria benchmark metrics (2026-11-05). Run in the scdef env, after score.py.

    JAX_PLATFORMS=cpu python diagnose_planaria.py

1. Cell-type ARI: the benchmark labels merge the 13 neoblast clusters of Plass et al. into one class (37% of
   cells). Prints ARI on the 39 labels, on the 51 original clusters, and without neoblasts; how each method's
   finest clustering spreads the neoblasts; and scDEF's ARI with its mostly-neoblast factors merged.
2. Cell-type ASW: the 2024 rule uses each method's raw latent space; for scDEF that is the unnormalised
   posterior mean of the layer-0 cell weights. Prints ASW raw and with each cell's weights scaled to sum to one,
   for every method, with and without neoblasts (8,000 random cells, seed 0).
3. The two scDEF neoblast factors against the 13 original clusters, and their sequencing depth.
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import roc_auc_score, silhouette_score as sil

import common as C
import score as S

inp, _ = S.load_input("planaria")
y = inp.obs["Cell types"].astype(str).to_numpy()
neo = y == "neoblast"
fine = C.refit_module().tutorial_adata("planaria").obs["clusters"].astype(str).to_numpy()
R = C.out_dir("planaria")
model = S.load_scdef("planaria", "scdef-planaria-best5", inp)

assign, latent = {}, {}
for m in ["PCA", "NMF", "scHPF", "fscLVM", "nSBM"]:
    out = json.load(open(f"{R}/methods/{m}/outputs.json"))
    assign[m] = [np.asarray(x).astype(str) for x in out["assignments"]]
    latent[m] = np.load(f"{R}/methods/{m}/latent0.npy")
assign["scDEF"] = [model.adata.obs[l].astype(str).to_numpy() for l in model.layer_names if l in model.adata.obs]
latent["scDEF"] = np.asarray(model.adata.obsm["X_L0"])

print("1. Cell-type ARI (finest level)")
rows = []
for m, levels in assign.items():
    l0 = levels[0]
    vc = pd.Series(l0[neo]).value_counts(normalize=True)
    rows.append(dict(method=m, clusters=len(set(l0)), ARI_39=ari(y, l0), ARI_51=ari(fine, l0),
                     ARI_without_neoblasts=ari(y[~neo], l0[~neo]), neoblasts_in_largest=vc.iloc[0],
                     clusters_with_5pct_of_neoblasts=int((vc > 0.05).sum())))
print(pd.DataFrame(rows).round(3).to_string(index=False))
l0 = assign["scDEF"][0]
ct = pd.crosstab(l0, neo)
neo_f = list(ct.index[(ct[True] / ct.sum(1)) > 0.5])
print(f"scDEF factors with mostly neoblasts {neo_f}; ARI_39 with them merged "
      f"{ari(y, np.where(np.isin(l0, neo_f), 'neoblast', l0)):.3f}")

print("\n2. Cell-type ASW, raw and row-normalised (8,000 cells)")
idx = np.random.default_rng(0).choice(len(y), 8000, replace=False)
i2 = idx[~neo[idx]]


def rn(X):
    X = np.clip(X, 0, None)
    return X / np.maximum(X.sum(1, keepdims=True), 1e-12)


FACTOR_MODELS = {"NMF", "scHPF", "fscLVM", "scDEF"}     # non-negative weights; scaling to one only means something here
for m, X in latent.items():
    if m not in FACTOR_MODELS:
        print(f"  {m:7s} raw {sil(X[idx], y[idx]):.3f}  (not a factor model: no normalised version)  | without neoblasts: "
              f"raw {sil(X[i2], y[i2]):.3f}")
        continue
    print(f"  {m:7s} raw {sil(X[idx], y[idx]):.3f}  normalised {sil(rn(X)[idx], y[idx]):.3f}  | without neoblasts: "
          f"raw {sil(X[i2], y[i2]):.3f}  normalised {sil(rn(X)[i2], y[i2]):.3f}")

print("\n3. The two large scDEF neoblast factors")
big = list(pd.Series(l0[neo]).value_counts().index[:2])
nf = np.char.startswith(fine.astype(str), "neoblast")
tab = pd.crosstab(fine[nf], l0[nf], normalize="index")[big]
tab["n"] = pd.Series(fine[nf]).value_counts()
print(tab.loc[sorted(tab.index, key=lambda s: int(s.split()[-1]))].round(2).to_string())
tc = inp.obs["total_counts"].to_numpy()
s = neo & np.isin(l0, big)
print(f"median UMIs {big[0]} {np.median(tc[neo & (l0 == big[0])]):.0f}, {big[1]} "
      f"{np.median(tc[neo & (l0 == big[1])]):.0f}; AUC of UMIs between them {roc_auc_score(l0[s] == big[1], tc[s]):.2f}")
