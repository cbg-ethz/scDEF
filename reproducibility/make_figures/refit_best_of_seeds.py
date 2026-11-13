"""Refit the PBMC figure models from several seeds and keep the lowest final loss.

The tutorial fits used one seed (42), and on the two-donor data that seed reaches a worse local
optimum (scDEF-markdown/experiments/exp-nb-scdef-pbmcs2b-tutorial.md, seed test). Here every figure
model is fitted from SEEDS with the fixed workflow (``fit_fixed``; the per-tutorial steps until 2026-11-03); the fit with the lowest final loss
(mean of the last 100 iterations of the last fit, lower is better) is kept, and the decomposition is
redone from the chosen fit with batch annotations. The choice never looks at cell types.

    python reproducibility/make_figures/refit_best_of_seeds.py {pbmcs2b,ifn,pbmcs3k,hgsoc,planaria,visium}

Saves ``scdef-<data>-{nobatch,ref,dec-top2}-best5`` next to the tutorial models (for the single-batch
3k PBMCs, the planaria and the Visium section only ``scdef-<data>-best5``), ``refit_best_of_seeds_<data>.json`` (every seed: loss, layer
sizes, agreement with cell types) and ``refit_best_of_seeds_<data>_assignments.csv`` (every seed's
per-cell factor at each layer, for comparing seeds).
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig4_batch as f4  # noqa: E402  (environment, data preparation, paths)

import numpy as np
import pandas as pd
import scanpy as sc
import scdef as scd
from sklearn.metrics import adjusted_rand_score

SEEDS = [42, 1, 2, 3, 4]
BATCH_KEY = {"pbmcs2b": "Experiment", "ifn": "stim", "hgsoc": "patient_id"}
# GSE165897 cancer cells (Zhang et al. 2022), as prepared by scdef_data/data/hgsoc-pre.ipynb: counts,
# log-normalised X, 4000 genes chosen per patient, and the PCA hierarchical_init reads.
HGSOC = os.path.expanduser("~/projects/scDEF/scdef_data/data/hgsoc_cancer_preprocessed.h5ad")
# Whole adult planarian (Plass et al. 2018), as the planaria tutorial reads it: QC'd, HVG-filtered, counts layer,
# PCA; transcript ids renamed to the gene symbols the tutorial uses.
PLANARIA = os.path.expanduser("~/projects/scDEF/scdef_data/planaria/planaria_pp.h5ad")
# 10x CytAssist FFPE human breast cancer Visium section (Janesick et al. 2023), from the 10x Genomics site.
VISIUM = os.path.expanduser("~/projects/scDEF/spatial_data/naturecomms/visium")
KNOWN_GENES = {
    "dd_Smed_v6_659_0": "piwi", "dd_Smed_v6_332_0": "prog-1", "dd_Smed_v6_363_0": "prog-2-2",
    "dd_Smed_v6_920_0": "agat1", "dd_Smed_v6_432_0": "mhc", "dd_Smed_v6_12634_0": "myoD",
    "dd_Smed_v6_6208_0": "chat", "dd_Smed_v6_3977_0": "snap25", "dd_Smed_v6_1117_0": "npp-18",
    "dd_Smed_v6_8548_0": "ncam-2", "dd_Smed_v6_19336_0": "gbrb-1", "dd_Smed_v6_9893_0": "smed-coe",
    "dd_Smed_v6_8494_0": "roboA", "dd_Smed_v6_14852_0": "netrin", "dd_Smed_v6_13817_0": "musashi",
    "dd_Smed_v6_1013_0": "pmp-7", "dd_Smed_v6_364_0": "vim-1", "dd_Smed_v6_11310_0": "egfr",
}
# The tutorials' filter settings, unchanged.
PLAIN_FIRST = dict(brd_min=1.0, ard_min=0.001, n_eff_parents_max=10.1, filter_up=True)
PLAIN_SECOND = dict(brd_min=1.0, ard_min=0.001, n_eff_parents_max=1.5, filter_up=True,
                    min_cells_upper=1, brd_exceptional=4)
RELAXED = dict(n_eff_parents_max=10.0, brd_min=0.5, ard_min=0.0, clarity_min=0.0)
# The workflow applied to every data set (Pedro, 2026-11-03), replacing the per-tutorial settings above: fit;
# keep the relevant layer-0 factors (BRD posterior mean >= 1) without judging their place in the hierarchy, which
# the refit re-learns; refit; keep the relevant factors with at most 1.5 effective parents unless exceptionally
# relevant (BRD >= 4), and upper-layer factors with any cells. Package defaults otherwise. No ARD threshold
# (Pedro, 2026-11-03): a share of the total ARD removed rare but real cell types on the planaria
# (diagnose_factor_count.py). With batches the batch key goes to the model and to both filters, and the
# decomposition follows (DECOMPOSED).
FIXED_FIRST = dict(brd_min=1.0, ard_min=0.0, n_eff_parents_max=10.1, filter_up=True)
FIXED_SECOND = dict(brd_min=1.0, ard_min=0.0, n_eff_parents_max=1.5, filter_up=True, min_cells_upper=1,
                    brd_exceptional=4)
DECOMPOSED = dict(brd_min=1.0, ard_min=0.0, n_eff_parents_max=10.0, clarity_min=0.0)   # ard_min 0.001 until 2026-11-03


def prep_3k(markers=None):
    """The 3k tutorial's data preparation (its cells 5-8), unchanged, including the PCA that
    hierarchical_init reads; only the plotting is left out. ``markers`` (default: the tutorial's)
    are kept among the genes; the iscDEF notebook prepares the data the same way with its own."""
    if markers is None:
        nb = json.load(open(f"{f4.NB}/scdef-pbmcs3k-tutorial.ipynb"))
        ns = {}
        exec("".join(nb["cells"][2]["source"]), ns)      # the tutorial's marker list
        markers = ns["markers"]
    d = f"{f4.DATA}/pbmcs3k"
    a = sc.read_10x_mtx(f"{d}/data/filtered_gene_bc_matrices/hg19/", var_names="gene_symbols")
    a.obs.index = a.obs.index.str.replace("-1", "")
    sc.pp.filter_cells(a, min_genes=200)
    sc.pp.filter_genes(a, min_cells=3)
    a.var["mt"] = a.var_names.str.startswith("MT-")
    a.var["rb"] = a.var_names.str.startswith("RPS") | a.var_names.str.startswith("RPL")
    sc.pp.calculate_qc_metrics(a, qc_vars=["mt", "rb"], percent_top=None, log1p=False, inplace=True)
    a = a[a.obs.n_genes_by_counts < 2500, :]
    a = a[a.obs.pct_counts_mt < 5, :]
    a.layers["counts"] = a.X.toarray()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, flavor="seurat_v3", layer="counts", n_top_genes=4000)
    for g in {x for gs in markers.values() for x in gs}:
        if g in a.var_names:
            a.var.loc[g, "highly_variable"] = True
    a = a[:, a.var.highly_variable].copy()
    sc.pp.regress_out(a, ["total_counts", "pct_counts_mt"])
    sc.pp.scale(a, max_value=10)
    sc.tl.pca(a, svd_solver="arpack")
    sc.pp.neighbors(a, n_neighbors=10, n_pcs=40)
    sc.tl.umap(a)
    ann = pd.read_csv(f"{d}/annotations.csv", index_col=0)
    a.obs["Cell types"] = ann["seurat_annotations"]
    a.obs["Coarse"] = a.obs["Cell types"].map(
        {c: ("T" if c.endswith(" T") else "Mono" if (c.endswith("Mono") or c == "DC") else c)
         for c in ann["seurat_annotations"].astype("category").cat.categories}).astype("category")
    a.obs["cell_type"] = a.obs["Cell types"].astype(str)
    return a


def tutorial_adata(data):
    """The tutorial's data, including the PCA that hierarchical_init reads.

    One difference from the two-donor tutorial: the PCA input is normalised after restricting to the
    selected genes, whereas that tutorial normalised all genes first. The IFN tutorial already used
    this order, and there seed 42 reproduces the tutorial fit exactly; on the two-donor data it does
    not. All seeds share the same preprocessing.
    """
    if data == "pbmcs3k":
        return prep_3k()
    if data == "visium":
        # the spatial notebook's preprocessing (scDEF-notebooks/scdef-spatial.ipynb, cells 24-28)
        a = sc.read_visium(VISIUM, count_file="CytAssist_FFPE_Human_Breast_Cancer_filtered_feature_bc_matrix.h5")
        a.var_names_make_unique()
        a.var["mt"] = a.var_names.str.startswith("MT-")
        sc.pp.calculate_qc_metrics(a, qc_vars=["mt"], inplace=True)
        a = a[a.obs["pct_counts_mt"] < 20].copy()
        sc.pp.filter_genes(a, min_cells=10)
        a.layers["counts"] = a.X.copy()
        sc.pp.normalize_total(a)
        sc.pp.log1p(a)
        sc.pp.highly_variable_genes(a, n_top_genes=4000)
        a = a[:, a.var["highly_variable"]].copy()
        sc.pp.pca(a)
        a.obs["cell_type"] = "spot"                     # no annotation; ARI in describe() is not used
        return a
    if data == "planaria":
        a = sc.read_h5ad(PLANARIA)
        a.var["transcript_id"] = a.var_names.astype(str)
        a.var_names = [KNOWN_GENES.get(g, g) for g in a.var_names]
        a.var_names_make_unique()
        a.obs["cell_type"] = a.obs["clusters"].astype(str)
        return a
    if data == "hgsoc":
        a = sc.read_h5ad(HGSOC)
        a.obs["cell_type"] = a.obs["cell_subtype"].astype(str)     # the study's tumour subtypes
        return a
    a = f4.prep_2b() if data == "pbmcs2b" else f4.prep_ifn()
    x = a.copy()
    x.X = x.layers["counts"].copy()
    x.uns.pop("log1p", None)
    sc.pp.normalize_total(x, target_sum=1e4)
    sc.pp.log1p(x)
    sc.pp.regress_out(x, ["total_counts", "pct_counts_mt"])
    sc.pp.scale(x, max_value=10)
    sc.tl.pca(x, svd_solver="arpack")
    a.obsm["X_pca"] = x.obsm["X_pca"]
    return a


def fit_plain(a, seed):
    m = scd.scDEF(a.copy(), counts_layer="counts", seed=seed)
    m.fit(hierarchical_init=True)
    scd.tl.filter(m, **PLAIN_FIRST)
    m.fit(hierarchical_init=True, collapse_l1_fraction=0.9)
    scd.tl.filter(m, **PLAIN_SECOND)
    return m


def fit_fixed(a, seed, batch_key=None):
    """The fixed workflow (FIXED_FIRST, refit, FIXED_SECOND), with or without a batch key."""
    m = scd.scDEF(a.copy(), counts_layer="counts", batch_key=batch_key, seed=seed)
    m.fit(hierarchical_init=True)
    scd.tl.filter(m, batch_key=batch_key, **FIXED_FIRST)
    m.fit(hierarchical_init=True)
    scd.tl.filter(m, batch_key=batch_key, **FIXED_SECOND)
    return m


def fit_planaria(a, seed):
    """The planaria tutorial's steps: fit, filter with the default settings, refit (also used for Visium)."""
    m = scd.scDEF(a.copy(), counts_layer="counts", seed=seed)
    m.fit(hierarchical_init=True)
    scd.tl.filter(m)
    m.fit(hierarchical_init=True)
    return m


def fit_ref(a, seed, batch_key):
    m = scd.scDEF(a.copy(), counts_layer="counts", batch_key=batch_key, seed=seed)
    m.fit(hierarchical_init=True)
    scd.tl.filter(m, batch_key=batch_key, **RELAXED)
    m.fit(hierarchical_init=True)
    scd.tl.filter(m, batch_key=batch_key, **RELAXED)
    return m


def final_loss(m):
    return float(np.asarray(m.elbos[-1])[-100:].mean())


def describe(m, seed, minutes):
    obs = m.adata.obs
    ct = obs["cell_type"].astype(str)
    agree = {f"ARI_L{k}": float(adjusted_rand_score(ct, obs[f"L{k}"].astype(str)))
             for k in range(min(3, m.n_layers)) if f"L{k}" in obs}
    return dict(seed=seed, final_loss=final_loss(m), minutes=round(minutes, 1),
                layer_sizes=[len(f) for f in m.factor_lists], **agree)


def save(m, name):
    # save() on this code base does not clear the fused-step cache, which holds unpicklable
    # optimizer closures (fixed on branch batch-factors, not on dev): clear it here instead.
    for attr in ("_learn_step_cache", "_learn_jit_cache"):
        if hasattr(m, attr):
            setattr(m, attr, None)
    m.save(os.path.join(f4.NB, name), overwrite=True)


SINGLE_BATCH = ("pbmcs3k", "planaria", "visium")


def model_name(data, kind):
    return f"scdef-{data}-best5" if data in SINGLE_BATCH else f"scdef-{data}-{kind}-best5"


def main(data):
    """Fit every seed, keeping the lowest final loss. Resumable: the best fit so far is saved after
    each seed and recorded as ``<kind>_best_so_far``, so a restart skips the seeds already logged."""
    a = tutorial_adata(data)
    out_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out",
                            f"refit_best_of_seeds_{data}.json")
    out_csv = out_json.replace(".json", "_assignments.csv")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    log = dict(data=data, seeds=SEEDS, rule="lowest final loss", nobatch=[], ref=[])
    best, assignments = {}, {}
    # 3k PBMCs have one batch; for the HGSOC figures only the patient-annotated fit and its
    # decomposition are used
    kinds = {"pbmcs3k": ("nobatch",), "planaria": ("nobatch",), "visium": ("nobatch",), "hgsoc": ("ref",)}.get(data, ("nobatch", "ref"))
    if os.path.exists(out_json):
        prev = json.load(open(out_json))
        for kind in kinds:
            so_far = prev.get(f"{kind}_best_so_far")
            if so_far is not None and os.path.isdir(os.path.join(f4.NB, model_name(data, kind))):
                log[kind] = prev[kind]
                log[f"{kind}_best_so_far"] = so_far
                best[kind] = (so_far["final_loss"], so_far["seed"],
                              scd.scDEF.load(os.path.join(f4.NB, model_name(data, kind))))
                print("resuming", kind, "after seeds", [r["seed"] for r in log[kind]], flush=True)
        if best and os.path.exists(out_csv):
            assignments = pd.read_csv(out_csv, index_col=0).astype(str).to_dict("series")
    for kind in kinds:
        done = {r["seed"] for r in log[kind]}
        for seed in SEEDS:
            if seed in done:
                continue
            t0 = time.time()
            # the fixed workflow for every data set (decision-fixed-workflow, 2026-11-03); fit_plain, fit_ref and
            # fit_planaria are the per-tutorial workflows the figures used until then (models archived as *-tutorialwf)
            m = fit_fixed(a, seed, batch_key=None if kind == "nobatch" else BATCH_KEY[data])
            d = describe(m, seed, (time.time() - t0) / 60)
            log[kind].append(d)
            for ln in m.layer_names:
                if ln in m.adata.obs:
                    assignments[f"{kind}_seed{seed}_{ln}"] = m.adata.obs[ln].astype(str).to_numpy()
            pd.DataFrame(assignments, index=a.obs_names).to_csv(out_csv)
            print(kind, json.dumps(d), flush=True)
            if kind not in best or d["final_loss"] < best[kind][0]:
                best[kind] = (d["final_loss"], seed, m)
                save(m, model_name(data, kind))
                log[f"{kind}_best_so_far"] = dict(seed=seed, final_loss=d["final_loss"])
            json.dump(log, open(out_json, "w"), indent=1)
        log[f"{kind}_chosen_seed"] = best[kind][1]
        json.dump(log, open(out_json, "w"), indent=1)
    if data in SINGLE_BATCH:                    # one batch: nothing to decompose
        print("done", data, flush=True)
        return
    t0 = time.time()
    dec = scd.scDEF.decompose_batch_effects(best["ref"][2], top_layer=2)
    scd.tl.filter(dec, batch_key=BATCH_KEY[data], **DECOMPOSED)
    log["decomposed"] = dict(from_seed=best["ref"][1], minutes=round((time.time() - t0) / 60, 1),
                             layer_sizes=[len(f) for f in dec.factor_lists])
    save(dec, f"scdef-{data}-dec-top2-best5")
    json.dump(log, open(out_json, "w"), indent=1)
    print("done", data, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
