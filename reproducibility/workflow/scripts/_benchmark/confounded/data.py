"""Datasets with a known split between biological and technical batch differences.

Pre-registered in scDEF-markdown/experiments/exp-confounded-batch-attribution.md. Every dataset
is a two-batch design in which batch is confounded with a condition, and comes with a ground
truth of which differences are biology and which are technical:

- D1: real Kang CTRL cells split at random into pseudo-batches A and B; a response programme is
  injected into B in chosen cell types, then a technical effect is injected into B.
- D2: real Kang CTRL vs STIM; a technical effect is injected into STIM.
- D2b: Kang donor pairs (donor i CTRL vs donor j STIM), no injection.
- D3: pbmcs2b, two healthy donors; technical only.

All injections act on counts. Upward changes are applied on the mean scale,
``X' = X + Poisson((f - 1) * mu[type, g] * s_c)``, so a gene can switch on from zero; downward
changes are binomial thinning.
"""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

DATA_ROOT = os.path.expanduser("~/projects/scDEF/scdef_data/pbmcs_data")
KANG_DIR = os.path.join(DATA_ROOT, "ifnb")
PBMCSCA = os.path.join(DATA_ROOT, "pbmcsca.h5ad")
KANG_DONORS = os.path.join(KANG_DIR, "gse96583", "ifnb_donor.csv")   # GSE96583 genotype calls
N_D2B_PAIRS = 4

BATCH, CTYPE = "batch", "cell_type"
N_TOP_GENES = 4000
DETECTED_FRAC = 0.10        # a gene counts as expressed in a type above this detection rate
D2_TRUTH_LOG2FC = 1.0       # D2/D2b biological truth: |pseudobulk log2FC| above this ...
MIN_TRUTH_CELLS = 20        # ... in a type with at least this many cells in each batch

D1_LOCALISATION = {
    "1": ["CD14 Mono"],
    "3": ["CD14 Mono", "CD16 Mono", "DC"],
    "all": None,             # every cell type
}


@dataclass
class Dataset:
    name: str
    adata: anndata.AnnData            # counts in .X (sparse int), obs[BATCH], obs[CTYPE]
    clean: Optional[anndata.AnnData]  # same cells and genes, before the technical injection
                                      # (None for D2b, whose reference is `clean_sets`)
    bio_genes: Dict[str, List[str]]   # cell type -> true biological response genes
    tech_genes: Optional[List[str]]   # true technical genes, None where undefined
    meta: dict = field(default_factory=dict)
    clean_sets: Optional[List[anndata.AnnData]] = None   # D2b: one within-donor set per donor


# --------------------------------------------------------------------------- loading


def _qc(adata):
    """The filters of prepare_pbmcsifn.py, so the cells match the paper's IFN analysis."""
    adata = adata[:, adata.var_names != "MALAT1"].copy()
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=10)
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False,
                               inplace=True)
    adata = adata[(adata.obs.n_genes_by_counts < 5000) & (adata.obs.pct_counts_mt < 5)].copy()
    return adata


def _hvg(adata, batch_key):
    """seurat_v3 HVGs on counts, per batch, as in the paper's preprocessing."""
    tmp = adata.copy()
    tmp.layers["counts"] = tmp.X.copy()
    sc.pp.highly_variable_genes(tmp, flavor="seurat_v3", layer="counts",
                                n_top_genes=N_TOP_GENES, batch_key=batch_key)
    return tmp.var_names[tmp.var.highly_variable].tolist()


def _as_counts(adata):
    x = adata.X
    x = sp.csr_matrix(x) if not sp.issparse(x) else x.tocsr()
    adata.X = x.astype(np.float32)
    return adata


_KANG_CACHE = {}


def load_kang():
    """Kang IFN-beta, QC'd, restricted to the HVGs chosen on the clean CTRL+STIM data.

    The gene set is fixed here once and shared by D1, D2 and D2b, so it never depends on an
    injection.
    """
    if "kang" in _KANG_CACHE:
        return _KANG_CACHE["kang"].copy()
    adata = sc.read_h5ad(os.path.join(KANG_DIR, "ifnb.h5ad"))
    meta = pd.read_csv(os.path.join(KANG_DIR, "ifnb_metadata.csv"), index_col=0)
    adata = adata[meta.index].copy()
    adata.obs = meta.loc[adata.obs_names].copy()
    adata = _as_counts(_qc(adata))
    genes = _hvg(adata, "stim")
    adata = adata[:, genes].copy()
    adata.obs[CTYPE] = adata.obs["seurat_annotations"].astype(str)
    adata.obs["stim"] = adata.obs["stim"].astype(str)
    donors = pd.read_csv(KANG_DONORS, index_col=0)["donor"]
    adata.obs["donor"] = donors.loc[adata.obs_names].astype(int).astype(str).to_numpy()
    adata.obs = adata.obs[[CTYPE, "stim", "donor"]]
    adata.raw = None
    adata.var = pd.DataFrame(index=adata.var_names.astype(str))
    adata.uns = {}
    _KANG_CACHE["kang"] = adata
    return adata.copy()


def load_pbmcs2b():
    adata = sc.read_h5ad(PBMCSCA)
    adata = adata[adata.obs["Method"] == "Drop-seq"].copy()
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata = _as_counts(adata)
    adata.obs[BATCH] = adata.obs["Experiment"].astype(str)
    adata = adata[:, _hvg(adata, BATCH)].copy()
    adata.obs[CTYPE] = adata.obs["CellType"].astype(str)
    adata.obs = adata.obs[[BATCH, CTYPE]]
    adata.raw = None
    adata.var = pd.DataFrame(index=adata.var_names.astype(str))
    adata.uns = {}
    return adata


# --------------------------------------------------------------------------- injections


def _size_factors(x):
    lib = np.asarray(x.sum(1)).ravel()
    return lib / np.median(lib), lib


def _type_means(x, s, types):
    """mu[type, g]: mean depth-normalised count per type."""
    xn = sp.diags(1.0 / s) @ x
    out = {}
    for t in np.unique(types):
        m = types == t
        out[t] = np.asarray(xn[m].mean(0)).ravel()
    return out


def scale_genes(x, rows, factors, types, rng):
    """Multiply the mean of every gene in `rows` cells by `factors` (one per gene).

    f > 1 adds Poisson((f - 1) * mu[type, g] * s_c); f < 1 thins binomially.
    """
    x = x.tocsr().astype(np.float64)
    s, _ = _size_factors(x)
    mu = _type_means(x, s, types)
    dense = x[rows].toarray()
    f = np.asarray(factors, dtype=np.float64)
    up, down = f > 1, f < 1
    for i, r in enumerate(np.flatnonzero(rows)):
        m = mu[types[r]]
        if up.any():
            dense[i, up] += rng.poisson((f[up] - 1.0) * m[up] * s[r])
        if down.any():
            dense[i, down] = rng.binomial(dense[i, down].astype(np.int64), f[down])
    x = x.tolil()
    x[np.flatnonzero(rows)] = dense
    return x.tocsr().astype(np.float32)


def inject_response(x, rows, types, responding, genes_idx, log2fc, rng):
    """Biology: scale `genes_idx` by 2**log2fc in `rows` cells of the responding types."""
    target = rows & np.isin(types, responding)
    f = np.ones(x.shape[1])
    f[genes_idx] = 2.0 ** log2fc
    return scale_genes(x, target, f, types, rng)


def tech_depth(x, rows, rng, p=0.5):
    """T1: binomial thinning of every count in `rows` cells."""
    x = x.tocsr().copy()
    sub = x[rows].tocoo()
    sub.data = rng.binomial(sub.data.astype(np.int64), p).astype(np.float32)
    x = x.tolil()
    x[np.flatnonzero(rows)] = sub.tocsr()
    return x.tocsr(), None


def tech_ambient(x, rows, rng, alpha=0.1):
    """T2: add Poisson(alpha * L_c * soup) with the batch's own mean profile as the soup."""
    x = x.tocsr()
    soup = np.asarray(x[rows].sum(0)).ravel()
    soup = soup / soup.sum()
    _, lib = _size_factors(x)
    dense = x[rows].toarray()
    dense += rng.poisson(alpha * lib[rows][:, None] * soup[None, :])
    x = x.tolil()
    x[np.flatnonzero(rows)] = dense
    top = np.argsort(soup)[::-1][: int(np.ceil(0.05 * soup.size))]
    return x.tocsr().astype(np.float32), top


def tech_capture(x, rows, types, rng, sigma=0.5):
    """T3: per-gene capture factor ~ LogNormal(0, sigma). What gene_scale models directly."""
    logf = rng.normal(0.0, sigma, size=x.shape[1])
    out = scale_genes(x, rows, np.exp(logf), types, rng)
    return out, np.flatnonzero(np.abs(logf) > 0.5)


def apply_tech(x, rows, types, tech, rng):
    if tech == "T1":
        return tech_depth(x, rows, rng)
    if tech == "T2":
        return tech_ambient(x, rows, rng)
    if tech == "T3":
        return tech_capture(x, rows, types, rng)
    raise ValueError(tech)


# --------------------------------------------------------------------------- truth


def pseudobulk_log2fc(adata, batch_key, a, b):
    """Per cell type, log2 CPM fold change b vs a (pseudocount 1), on adata's counts."""
    out = {}
    for t in adata.obs[CTYPE].unique():
        m = adata.obs[CTYPE] == t
        ma, mb = m & (adata.obs[batch_key] == a), m & (adata.obs[batch_key] == b)
        if ma.sum() < MIN_TRUTH_CELLS or mb.sum() < MIN_TRUTH_CELLS:
            continue
        ca = np.asarray(adata.X[ma.values].sum(0)).ravel()
        cb = np.asarray(adata.X[mb.values].sum(0)).ravel()
        out[t] = np.log2(cb / cb.sum() * 1e6 + 1) - np.log2(ca / ca.sum() * 1e6 + 1)
    return out


def _detected(adata, mask):
    return np.asarray((adata.X[mask] > 0).mean(0)).ravel()


def truth_from_fc(adata, batch_key, a, b):
    """D2/D2b biological truth: |log2FC| > 1 and detected in >= 10% of the type's cells."""
    fcs = pseudobulk_log2fc(adata, batch_key, a, b)
    genes = np.asarray(adata.var_names)
    out = {}
    for t, fc in fcs.items():
        det = _detected(adata, (adata.obs[CTYPE] == t).values)
        out[t] = genes[(np.abs(fc) > D2_TRUTH_LOG2FC) & (det >= DETECTED_FRAC)].tolist()
    return out


# --------------------------------------------------------------------------- builders


def build_d1(localisation, n_resp, tech, seed):
    rng = np.random.default_rng(1000 + seed)
    kang = load_kang()
    adata = kang[kang.obs["stim"] == "CTRL"].copy()
    adata.obs[BATCH] = np.where(rng.random(adata.n_obs) < 0.5, "A", "B")
    types = adata.obs[CTYPE].to_numpy()
    responding = D1_LOCALISATION[localisation] or sorted(np.unique(types))
    rows_b = (adata.obs[BATCH] == "B").to_numpy()

    resp_mask = np.isin(types, responding)
    det = _detected(adata, resp_mask)
    candidates = np.flatnonzero(det >= DETECTED_FRAC)
    genes_idx = rng.choice(candidates, size=min(n_resp, candidates.size), replace=False)
    log2fc = np.abs(rng.normal(0.0, 1.0, size=genes_idx.size)) + 1.0

    x = inject_response(adata.X, rows_b, types, responding, genes_idx, log2fc, rng)
    clean = adata.copy()
    clean.X = x
    x_t, tech_idx = apply_tech(x, rows_b, types, tech, rng)
    adata.X = x_t

    names = np.asarray(adata.var_names)
    bio = {t: (names[genes_idx].tolist() if t in responding else []) for t in np.unique(types)}
    return Dataset(
        name=f"D1_loc{localisation}_n{n_resp}_{tech}_s{seed}",
        adata=adata, clean=clean, bio_genes=bio,
        tech_genes=None if tech_idx is None else names[tech_idx].tolist(),
        meta=dict(design="D1", localisation=localisation, n_resp=n_resp, tech=tech,
                  seed=seed, responding=list(responding)),
    )


def build_d2(tech, seed):
    rng = np.random.default_rng(2000 + seed)
    adata = load_kang()
    adata.obs[BATCH] = adata.obs["stim"]
    types = adata.obs[CTYPE].to_numpy()
    rows = (adata.obs[BATCH] == "STIM").to_numpy()
    clean = adata.copy()
    x_t, tech_idx = apply_tech(adata.X, rows, types, tech, rng)
    adata.X = x_t
    names = np.asarray(adata.var_names)
    return Dataset(
        name=f"D2_{tech}_s{seed}", adata=adata, clean=clean,
        bio_genes=truth_from_fc(clean, BATCH, "CTRL", "STIM"),
        tech_genes=None if tech_idx is None else names[tech_idx].tolist(),
        meta=dict(design="D2", tech=tech, seed=seed),
    )


def d2b_pairs():
    """The four pre-registered confounded pairs (donor i CTRL, donor j STIM), drawn with seed 0."""
    donors = sorted(load_kang().obs["donor"].unique())
    rng = np.random.default_rng(0)
    pairs = []
    while len(pairs) < N_D2B_PAIRS:
        i, j = rng.choice(donors, size=2, replace=False)
        if (i, j) not in pairs:
            pairs.append((str(i), str(j)))
    return pairs


def build_d2b(pair_idx, seed):
    """Donor i CTRL against donor j STIM. Truth from the donors outside the pair, within donor."""
    kang = load_kang()
    i, j = d2b_pairs()[pair_idx]
    obs = kang.obs
    sel = ((obs["donor"] == i) & (obs["stim"] == "CTRL")) | \
          ((obs["donor"] == j) & (obs["stim"] == "STIM"))
    adata = kang[sel.values].copy()
    adata.obs[BATCH] = adata.obs["stim"]

    others = [k for k in sorted(obs["donor"].unique()) if k not in (i, j)]
    clean_sets, fcs = [], {}
    for k in others:
        ad = kang[(obs["donor"] == k).values].copy()
        ad.obs[BATCH] = ad.obs["stim"]
        clean_sets.append(ad)
        for t, fc in pseudobulk_log2fc(ad, BATCH, "CTRL", "STIM").items():
            fcs.setdefault(t, []).append(fc)
    rest = kang[np.isin(obs["donor"], others)].copy()
    names = np.asarray(kang.var_names)
    bio = {}
    for t, lst in fcs.items():
        fc = np.mean(lst, axis=0)
        det = _detected(rest, (rest.obs[CTYPE] == t).values)
        bio[t] = names[(np.abs(fc) > D2_TRUTH_LOG2FC) & (det >= DETECTED_FRAC)].tolist()
    return Dataset(
        name=f"D2b_p{pair_idx}_s{seed}", adata=adata, clean=None, clean_sets=clean_sets,
        bio_genes=bio, tech_genes=None,
        meta=dict(design="D2b", pair=[i, j], truth_donors=others, seed=seed),
    )


def build_d3(seed):
    adata = load_pbmcs2b()
    return Dataset(
        name=f"D3_s{seed}", adata=adata, clean=adata.copy(),
        bio_genes={t: [] for t in adata.obs[CTYPE].unique()}, tech_genes=None,
        meta=dict(design="D3", seed=seed),
    )


def build(spec):
    """`spec` is the dataset name, e.g. D1_loc3_n50_T2_s0, D2_T1_s1, D3_s0."""
    parts = spec.split("_")
    seed = int(parts[-1][1:])
    if parts[0] == "D1":
        return build_d1(parts[1][3:], int(parts[2][1:]), parts[3], seed)
    if parts[0] == "D2":
        return build_d2(parts[1], seed)
    if parts[0] == "D2b":
        return build_d2b(int(parts[1][1:]), seed)
    if parts[0] == "D3":
        return build_d3(seed)
    raise ValueError(spec)
