"""Transition paths across the hierarchy on the PBMC fits of the paper (pre-declared 2026-11-12 in
scDEF-markdown/experiments/exp-transition-showcase.md before computing).

    JAX_PLATFORMS=cpu python transition_showcase.py score [CASE ...]   -> out/transition_showcase.json
    JAX_PLATFORMS=cpu python transition_showcase.py heatmaps   -> out/transition_<case>.{pdf,png} (passing cases)

For each case: the layer-0 factor with the most cells of the source (target) label, the shortest transition path
between them (max_path_len=9), score_paths, and the cells at affinity >= 0.5. Pass criteria: >= 50 cells of each
label; source median position < 0.3 and target > 0.7; Spearman >= 0.6 with diffusion pseudotime on the same cells;
case C also needs an AUC >= 0.9 of position for stimulated against control CD14 monocytes. No model is re-fitted.
"""
import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scanpy as sc  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

import scdef as scd  # noqa: E402
from scdef.tools.trajectory import _ancestor_slot, _strongest_parent_maps  # noqa: E402

import fig4_batch as f4  # noqa: E402
import supp_pbmcs3k as s3k  # noqa: E402

OUT = os.path.join(f4.HERE, "out")
MIN_AFF, MIN_CELLS = 0.5, 50

# case: (model, source label, target label, extra criterion); labels are values of obs["label"]
CASES = {
    "A_3k_monocytes": ("pbmcs3k", "CD14+ Mono", "FCGR3A+ Mono", None),
    "Arep_ifn_monocytes": ("ifn", "CD14 Mono", "CD16 Mono", None),
    "Arep_2b_monocytes": ("pbmcs2b", "CD14+ monocyte", "CD16+ monocyte", None),
    "B_3k_cd4": ("pbmcs3k", "Naive CD4 T", "Memory CD4 T", None),
    "C_ifn_response": ("ifn", "CD14 Mono | control", "CD14 Mono | IFN-β", "auc"),
    # added after the results above, at Pedro's suggestion (declared before computing): the decomposed model
    "Cdec_ifn_response": ("ifn_dec", "CD14 Mono | control", "CD14 Mono | IFN-β", "auc"),
}

_MODELS = {}


def model(key):
    if key not in _MODELS:
        if key == "pbmcs3k":
            m = s3k.load("scdef-pbmcs3k-best5")
            m.adata.obs["label"] = m.adata.obs["cell_type"].astype(str)
        elif key in ("ifn", "ifn_dec"):
            m = f4.load("scdef-ifn-ref-best5" if key == "ifn" else "scdef-ifn-dec-top2-best5", f4.prep_ifn())
            obs = m.adata.obs
            obs["label"] = obs["cell_type"].astype(str)
            mono = obs["cell_type"].astype(str) == "CD14 Mono"
            obs["label_resp"] = np.where(mono, "CD14 Mono | " + obs["batch"].astype(str), "other")
        elif key == "pbmcs2b":
            m = f4.load("scdef-pbmcs2b-ref-best5", f4.prep_2b())
            m.adata.obs["label"] = m.adata.obs["cell_type"].astype(str)
        _MODELS[key] = m
    return _MODELS[key]


def labels_of(m, src):
    return m.adata.obs["label_resp" if "|" in src else "label"].astype(str).to_numpy()


def factor_of(m, lab, label):
    l0 = m.adata.obs["L0"].astype(str).to_numpy()
    counts = pd.Series(l0[lab == label]).value_counts()
    return (counts.index[0], int(counts.iloc[0]), int(counts.sum())) if len(counts) else (None, 0, 0)


def apex_covers_all(m, apex):
    layer = int(next(k for k in range(m.n_layers) if apex in m.factor_names[k]))
    slot = m.factor_names[layer].index(apex)
    pm = _strongest_parent_maps(m)
    below = [_ancestor_slot(pm, 0, i, layer) == slot for i in range(len(m.factor_names[0]))]
    return bool(all(below)), int(sum(below))


def dpt(m, mask, src_mask):
    a = m.adata[mask].copy()
    if "counts" in a.layers:
        a.X = a.layers["counts"].copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=min(2000, a.n_vars))
    a = a[:, a.var["highly_variable"]].copy()
    sc.pp.pca(a, n_comps=30)
    sc.pp.neighbors(a, n_neighbors=15, n_pcs=30)
    sc.tl.diffmap(a)
    dc1 = a.obsm["X_diffmap"][:, 1]
    s = src_mask[mask]
    src_low = np.median(dc1[s]) < np.median(dc1[~s])
    idx = np.where(s)[0]
    a.uns["iroot"] = int(idx[np.argmin(dc1[idx])] if src_low else idx[np.argmax(dc1[idx])])
    sc.tl.dpt(a)
    return a.obs["dpt_pseudotime"].to_numpy()


def run_case(name):
    mkey, src, tgt, extra = CASES[name]
    m = model(mkey)
    lab = labels_of(m, src)
    res = dict(case=name, model=mkey, source_label=src, target_label=tgt,
               n_source_label=int((lab == src).sum()), n_target_label=int((lab == tgt).sum()))
    if res["n_source_label"] == 0 or res["n_target_label"] == 0:
        return dict(res, status="not evaluable", reason="label missing from the data set")
    fs, ns, _ = factor_of(m, lab, src)
    ft, nt, _ = factor_of(m, lab, tgt)
    res.update(source_factor=fs, source_factor_cells=ns, target_factor=ft, target_factor_cells=nt)
    if fs == ft:
        return dict(res, status="not evaluable", reason=f"both labels map to {fs}")
    key = f"tp_{name}"
    paths = scd.tl.build_transition_paths(m, sources=[fs], targets=[ft], max_path_len=9, key_added=key)
    if not paths:
        return dict(res, status="not evaluable", reason="no path within 9 nodes")
    path = paths[0]
    layer_of = {f: k for k in range(m.n_layers) for f in m.factor_names[k]}
    apex = max(path, key=lambda f: layer_of[f])
    covers, n_below = apex_covers_all(m, apex)
    res.update(path=path, apex=apex, l0_factors_below_apex=n_below, n_l0_factors=len(m.factor_names[0]))
    if covers:
        return dict(res, status="not evaluable", reason=f"apex {apex} is an ancestor of every layer-0 factor")
    scd.tl.score_paths(m, paths_key=key)
    aff = m.adata.obsm[f"{key}_affinities"][:, 0]
    pos = m.adata.obsm[f"{key}_positions"][:, 0]
    on = aff >= MIN_AFF
    s_on, t_on = on & (lab == src), on & (lab == tgt)
    res.update(cells_on_path=int(on.sum()), n_source_on=int(s_on.sum()), n_target_on=int(t_on.sum()),
               median_pos_source=float(np.median(pos[s_on])) if s_on.any() else None,
               median_pos_target=float(np.median(pos[t_on])) if t_on.any() else None)
    # composition of the cells on the path
    res["on_path_labels"] = pd.Series(m.adata.obs["label"].astype(str).to_numpy()[on]).value_counts().head(8).to_dict()
    c1 = res["n_source_on"] >= MIN_CELLS and res["n_target_on"] >= MIN_CELLS
    c2 = c1 and res["median_pos_source"] < 0.3 and res["median_pos_target"] > 0.7
    both = s_on | t_on
    rho = None
    if c1:
        pt = dpt(m, both, lab == src)
        rho = float(spearmanr(pos[both], pt).correlation)
    c3 = rho is not None and rho >= 0.6
    res.update(spearman_dpt=rho, crit1=c1, crit2=c2, crit3=c3)
    ok = c1 and c2 and c3
    if extra == "auc":
        auc = float(roc_auc_score((lab[both] == tgt).astype(int), pos[both])) if c1 else None
        c4 = auc is not None and auc >= 0.9
        res.update(auc=auc, crit4=c4)
        ok = ok and c4
    res["status"] = "pass" if ok else "fail"
    return res


def score(names=None):
    path = os.path.join(OUT, "transition_showcase.json")
    old = {r["case"]: r for r in json.load(open(path))} if names and os.path.exists(path) else {}
    for name in names or CASES:
        r = run_case(name)
        print(json.dumps(r, indent=1, default=str), flush=True)
        old[name] = r
    with open(path, "w") as f:
        json.dump([old[n] for n in CASES if n in old], f, indent=1, default=str)


def heatmaps():
    import matplotlib

    matplotlib.use("Agg")
    results = json.load(open(os.path.join(OUT, "transition_showcase.json")))
    for r in results:
        if r["status"] != "pass":
            continue
        name = r["case"]
        m = model(r["model"])
        key = f"tp_{name}"
        scd.tl.build_transition_paths(m, sources=[r["source_factor"]], targets=[r["target_factor"]],
                                      max_path_len=9, key_added=key)
        scd.tl.score_paths(m, paths_key=key)
        fig = scd.pl.path_trajectory_heatmap(m, path_id=0, paths_key=key, min_sort_affinity=MIN_AFF,
                                             annotation_obs_key="label", show=False)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(OUT, f"transition_{name}.{ext}"), dpi=200, bbox_inches="tight")
        print("wrote", name)


if __name__ == "__main__":
    {"score": score, "heatmaps": heatmaps}[sys.argv[1]](*([sys.argv[2:]] if len(sys.argv) > 2 else []))
