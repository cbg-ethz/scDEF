"""Methods run in the `scdef` environment: uncorrected PCA, Harmony, Harmony + DE, and scDEF.

Each method returns an embedding (for metric 1) and, where it has one, per-cell-type gene scores
for "biological response here" and a global per-gene "technical" score (for metric 2). The exact
procedures are fixed in the pre-registration (exp-confounded-batch-attribution); thresholds here
are copied from it and must not be tuned.
"""
import json
import os
import time
from typing import Dict, Optional

import numpy as np
import pandas as pd
import scanpy as sc

from .data import BATCH, CTYPE

# Pre-registered constants.
RELAXED = dict(n_eff_parents_max=10.0, brd_min=0.5, ard_min=0.0, clarity_min=0.0)
DECOMPOSED = dict(brd_min=1.0, ard_min=0.001, n_eff_parents_max=10.0, clarity_min=0.0)
# Amendment 2 (2026-11-12): the paper's workflow, scDEF.learn() and the decomposition filter of
# make_figures/refit_best_of_seeds.DECOMPOSED (ard_min 0 since 2026-11-03).
PAPER_DECOMPOSED = dict(brd_min=1.0, ard_min=0.0, n_eff_parents_max=10.0, clarity_min=0.0)
SKEWED_FRAC = 0.7          # a factor is batch-skewed at frac_dom_batch >= 0.7
# Amendment 1 (2026-10-18): the benchmark's stand-in for the analyst. Technical = duplicated per
# batch (branch_split) or spread across lineages (overlaid); branch_skewed and balanced are kept.
TECHNICAL_SHAPES = ("branch_split", "overlaid")
LEIDEN_RES = 1.0
RULE_FC = 0.5              # Harmony + DE + rule: |log2FC| > 0.5 ...
RULE_FRAC = 0.8            # ... with the same sign in >= 80% of clusters => technical
MIN_CLUSTER_BATCH = 10     # a cluster needs this many cells in each batch to give a fold change


def batches_of(adata):
    """(reference batch, other batch), sorted, so every fold change has one direction."""
    b = sorted(adata.obs[BATCH].astype(str).unique())
    assert len(b) == 2, b
    return b[0], b[1]


def preprocess(adata):
    """Counts in layers['counts']; log-normalised, scaled X; 50-component PCA in obsm['X_pca']."""
    adata = adata.copy()
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.layers["lognorm"] = adata.X.copy()
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=50, svd_solver="arpack")
    return adata


# --------------------------------------------------------------------------- Harmony


def run_harmony(adata):
    import harmonypy

    ho = harmonypy.run_harmony(adata.obsm["X_pca"], adata.obs, [BATCH], verbose=False)
    z = np.asarray(ho.Z_corr)
    if z.shape[0] != adata.n_obs:     # older harmonypy returns PCs x cells
        z = z.T
    return z


def _pseudobulk_fc(counts, mask_a, mask_b):
    ca = np.asarray(counts[mask_a].sum(0)).ravel()
    cb = np.asarray(counts[mask_b].sum(0)).ravel()
    return np.log2(cb / max(cb.sum(), 1) * 1e6 + 1) - np.log2(ca / max(ca.sum(), 1) * 1e6 + 1)


def harmony_de(adata, z_harmony):
    """Harmony -> Leiden -> clusters mapped to their plurality cell type -> per-cluster
    pseudobulk log2FC between batches. Returns the plain and the rule-based gene scores."""
    a, b = batches_of(adata)
    tmp = sc.AnnData(np.zeros((adata.n_obs, 1)), obs=adata.obs.copy())
    tmp.obsm["X_harmony"] = z_harmony
    sc.pp.neighbors(tmp, use_rep="X_harmony", n_neighbors=15)
    sc.tl.leiden(tmp, resolution=LEIDEN_RES, key_added="cl", flavor="igraph", n_iterations=2,
                 directed=False)
    cl = tmp.obs["cl"].astype(str).to_numpy()
    batch = adata.obs[BATCH].astype(str).to_numpy()
    ctype = adata.obs[CTYPE].astype(str).to_numpy()
    counts = adata.layers["counts"].tocsr()

    fcs, cl_type, cl_n = {}, {}, {}
    for k in np.unique(cl):
        m = cl == k
        ma, mb = m & (batch == a), m & (batch == b)
        if ma.sum() < MIN_CLUSTER_BATCH or mb.sum() < MIN_CLUSTER_BATCH:
            continue
        fcs[k] = _pseudobulk_fc(counts, ma, mb)
        cl_type[k] = pd.Series(ctype[m]).value_counts().index[0]
        cl_n[k] = int(m.sum())

    n_genes = adata.n_vars
    if fcs:
        F = np.vstack([fcs[k] for k in fcs])                       # clusters x genes
        sign = np.sign(np.median(F, axis=0))
        consistency = ((np.abs(F) > RULE_FC) & (np.sign(F) == sign)).mean(0)
    else:
        consistency = np.zeros(n_genes)

    bio_plain, bio_rule = {}, {}
    for t in np.unique(ctype):
        ks = [k for k in fcs if cl_type[k] == t]
        if not ks:
            bio_plain[t] = np.zeros(n_genes)
        else:
            w = np.array([cl_n[k] for k in ks], dtype=float)
            bio_plain[t] = (np.abs(np.vstack([fcs[k] for k in ks])) * w[:, None]).sum(0) / w.sum()
        bio_rule[t] = bio_plain[t] * (1.0 - consistency)
    info = dict(n_clusters=int(len(np.unique(cl))), n_usable_clusters=len(fcs),
                n_technical_genes=int((consistency >= RULE_FRAC).sum()))
    return dict(bio=bio_plain, tech=None, info=info), \
        dict(bio=bio_rule, tech=consistency, info=info)


# --------------------------------------------------------------------------- scDEF


def _l0_weights(model):
    """Signature weights of the kept L0 factors, rows aligned to model.factor_names[0],
    each row scaled to a maximum of 1."""
    W = np.asarray(model.pmeans["L0W"])
    kept = np.asarray(model.factor_lists[0])
    if W.shape[0] != len(model.factor_names[0]):
        W = W[kept]
    return W / np.maximum(W.max(1, keepdims=True), 1e-12)


def _plurality_types(model):
    obs = model.adata.obs
    out = {}
    for f in model.factor_names[0]:
        m = obs["L0"].astype(str) == f
        if m.sum():
            out[f] = obs.loc[m, CTYPE].astype(str).value_counts().index[0]
    return out


def _bio_scores(model, factors, all_types):
    """Per cell type: max normalised weight over `factors` whose plurality type is that type."""
    Wn = _l0_weights(model)
    idx = {f: i for i, f in enumerate(model.factor_names[0])}
    ptype = _plurality_types(model)
    out = {t: np.zeros(Wn.shape[1]) for t in all_types}
    for f in factors:
        t = ptype.get(f)
        if t is not None:
            out[t] = np.maximum(out[t], Wn[idx[f]])
    return out


def _fit_info(model, t0):
    return dict(seconds=round(time.time() - t0, 1),
                layer_sizes=[len(f) for f in model.factor_lists])


def _composition(model):
    """Layer-0 diagnostics (Amendment 2): the cells of each factor by cell type and by batch (hard
    assignment), and the mean cell scale against the mean library size in each batch."""
    obs = model.adata.obs
    l0 = obs["L0"].astype(str)
    lib = np.asarray(model.adata.layers["counts"].sum(1)).ravel() if "counts" in model.adata.layers \
        else np.asarray(model.adata.X.sum(1)).ravel()
    scale = np.asarray(model.pmeans["cell_scale"]).ravel()
    b = obs[BATCH].astype(str)
    return dict(
        by_type=pd.crosstab(l0, obs[CTYPE].astype(str)).to_dict("index"),
        by_batch=pd.crosstab(l0, b).to_dict("index"),
        cell_scale_by_batch={k: float(scale[(b == k).to_numpy()].mean()) for k in b.unique()},
        library_by_batch={k: float(lib[(b == k).to_numpy()].mean()) for k in b.unique()},
    )


def run_scdef(adata, seed, workflow="relaxed"):
    """The default fit, and the decomposed fit raw and with the shape rule (Amendment 1).

    ``workflow="learn"`` (Amendment 2): the default fit is ``scDEF.learn()``, the paper's workflow, and
    the decomposition is filtered with PAPER_DECOMPOSED; both models' layer-0 composition is recorded.
    """
    import scdef as scd

    all_types = sorted(adata.obs[CTYPE].astype(str).unique())
    out = {}

    # --- default: the tutorial's reference workflow (October) or the paper's (Amendment 2)
    t0 = time.time()
    # extra scDEF settings for pilots (2026-11-13: CONFOUNDED_SCDEF_KWARGS='{"cell_alpha_by_batch": true}');
    # recorded in the results, empty for the benchmark itself
    extra = json.loads(os.environ.get("CONFOUNDED_SCDEF_KWARGS", "{}"))
    ref = scd.scDEF(adata, counts_layer="counts", batch_key=BATCH, seed=seed, **extra)
    if workflow == "learn":
        ref.learn()
    else:
        ref.fit(hierarchical_init=True)
        scd.tl.filter(ref, batch_key=BATCH, **RELAXED)
        ref.fit(hierarchical_init=True)
        scd.tl.filter(ref, batch_key=BATCH, **RELAXED)
    info = _fit_info(ref, t0)
    info["scdef_kwargs"] = extra
    if workflow == "learn":
        info["composition"] = _composition(ref)
    rep = scd.tl.batch_structure_report(ref, batch_key=BATCH)
    skewed = rep.index[rep["frac_dom_batch"] >= SKEWED_FRAC].tolist()
    gs = scd.tl.get_batch_specific_genes_from_gene_scale(ref)
    batch_cols = [c for c in gs.columns if c in set(map(str, ref.batches))]
    tech = np.abs(gs[batch_cols].to_numpy()).max(1)
    out["scDEF_default"] = dict(
        emb=np.asarray(ref.adata.obsm["X_L0_probs"]),
        bio=_bio_scores(ref, skewed, all_types), tech=tech,
        info=dict(info, n_skewed=len(skewed), report=rep.reset_index().to_dict("list")),
    )

    # --- decomposed: decompose and filter, then report it raw and with the shape rule
    t0 = time.time()
    dec = scd.scDEF.decompose_batch_effects(ref, top_layer=2)
    scd.tl.filter(dec, batch_key=BATCH, **(PAPER_DECOMPOSED if workflow == "learn" else DECOMPOSED))
    dinfo = _fit_info(dec, t0)
    if workflow == "learn":
        dinfo["composition"] = _composition(dec)
    out.update(decomposed_variants(ref, dec, all_types, dinfo))
    return out


def decomposed_variants(ref, dec, all_types, info=None):
    """Amendment 1: the filtered decomposed fit with nothing removed, and with the shape rule.

    Separate from `run_scdef` so it can be checked on saved models without refitting.
    """
    import scdef as scd

    info = info or {}
    rep = scd.tl.batch_structure_report(dec, batch_key=BATCH, reference=ref)
    report = rep.reset_index().to_dict("list")
    skewed = rep.index[rep["frac_dom_batch"] >= SKEWED_FRAC].tolist()
    flagged = rep.index[rep["shape"].isin(TECHNICAL_SHAPES)].tolist()
    Wn = _l0_weights(dec)
    idx = {f: i for i, f in enumerate(dec.factor_names[0])}

    raw = dict(
        emb=np.array(dec.adata.obsm["X_L0_probs"], copy=True),
        bio=_bio_scores(dec, skewed, all_types), tech=None,
        info=dict(info, n_skewed=len(skewed), report=report),
    )
    bio = _bio_scores(dec, [f for f in skewed if f not in flagged], all_types)
    if flagged:
        scd.tl.set_batch_technical_factors(dec, flagged)
        scd.tl.factor_batch_correction(dec)
        emb = np.asarray(dec.adata.obsm["X_L0_batch_corrected"])
    else:
        emb = np.asarray(dec.adata.obsm["X_L0_probs"])
    shape = dict(
        emb=emb, bio=bio,
        tech=Wn[[idx[f] for f in flagged]].max(0) if flagged else np.zeros(Wn.shape[1]),
        info=dict(info, n_flagged=len(flagged), flagged=flagged, report=report),
    )
    return {"scDEF_decomposed_raw": raw, "scDEF_decomposed_shape": shape}
