"""Score every finished method on one data set with the 2024 metric functions, and the saved scDEF models.

    python score.py <data> [--force]          (env: scdef)

Metrics come from ``workflow/scripts/_benchmark/evaluate.py`` (imported unchanged). Two shims:

- ``scDEF.get_hierarchy`` moved to ``scdef.tl.get_hierarchy(model)``; the model gets a forwarding method.
- Cluster naming (comparators only). The 2024 hierarchy helpers name a clustering's nodes by
  their position in the string-sorted list of labels (``get_hierarchy_from_clusters(use_names=False)``)
  or by ``range(number of clusters)`` (``get_cluster_obs_association_scores``, ``simplify_hierarchy``),
  while the signatures, sizes and cell assignments use the actual labels. With 11 or more clusters at a
  level ("10" sorts before "2"), or factors that win no cell, nodes are mismatched. Here the nodes are
  named by their actual labels. scDEF's own path names factors explicitly and is unaffected. The
  uncorrected numbers are written alongside (``scores_2024_verbatim.csv``).
- Hierarchy accuracy (every method, 2026-11-06): the 2024 ``compare_hierarchies`` counts a group's own
  children as false positives, capping a correctly nested group with two children at 0.67; fixed in
  ``_compare_hierarchies``. Off in ``scores_2024_verbatim.csv``.
- Clustering and embedding metrics (2026-11-06, decision-clustering-protocol, accepted with changes): see the
  "clustering protocol" section below. In short, proportions of the weights for the non-negative factor models;
  cell-type ARI (AMI, homogeneity, completeness) from KMeans at k = number of types; Leiden best of five as a
  supplementary ARI; batch ARI from Leiden at resolution 1.0; ASW on the same representation. The 2024 values
  (scDEF's layer-0 top factor and raw X_L0; the other methods' first level) stay in ``scores_2024_verbatim.csv``.

scDEF is not refitted: the best-of-five models are loaded on ``refit_best_of_seeds.tutorial_adata`` (they
were saved without their AnnData). Runtime is the kept seed's minutes from the refit log; the five-seed
total is reported as a separate row.

Writes scores.csv (long), scores_wide.csv, scores_2024_verbatim.csv, signature_accuracy_per_type.json,
runtimes.csv and settings.json to out/realdata_benchmark/<data>/; per-method results are cached in scores/.
"""
import argparse
import contextlib
import os
import re
import time
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")             # scoring loads the models on CPU
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
warnings.filterwarnings("ignore")
import logging

logging.disable(logging.WARNING)

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

import common as C

import scdef as scd
from scdef.utils import hierarchy_utils as hu

ev = C.bench_module("evaluate")

if not hasattr(scd.scDEF, "get_hierarchy"):
    scd.scDEF.get_hierarchy = lambda self, simplified=True, **kw: scd.tl.get_hierarchy(self, simplified=simplified, **kw)

# --------------------------------------------------------------------------- label-naming fix
_ORIG = {k: getattr(hu, k) for k in ("get_hierarchy_from_clusters", "get_cluster_obs_association_scores",
                                     "simplify_hierarchy")}


def _num(name):
    m = re.search(r"(\d+)$", str(name))
    return (int(m.group(1)) if m else -1, str(name))


def _get_hierarchy_from_clusters(clusters_levels, level_prefix="h", use_names=False):
    return _ORIG["get_hierarchy_from_clusters"](clusters_levels, level_prefix=level_prefix, use_names=True)


def _get_cluster_obs_association_scores(adata, clusters, layer_names, obs_key, obs_val):
    scores, factors, layers = [], [], []
    for layer_idx in range(len(clusters)):
        assignments = clusters[layer_idx]
        for name in sorted(set(map(str, assignments)), key=_num):
            scores.append(hu.compute_cluster_obs_association_score(adata, assignments, name, obs_key, obs_val))
            factors.append(name)
            layers.append(layer_idx)
    return scores, factors, layers


def _simplify_hierarchy(hierarchy, layer_names, layer_sizes, factor_names=None):
    if factor_names is None:
        nodes = set(hierarchy) | {c for cs in hierarchy.values() for c in cs}
        factor_names = [sorted([n for n in nodes if re.fullmatch(re.escape(ln) + r"\d+", str(n))], key=_num)
                        for ln in layer_names]
        layer_sizes = [len(f) for f in factor_names]
    return _ORIG["simplify_hierarchy"](hierarchy, layer_names, layer_sizes, factor_names=factor_names)


_ORIG["compare_hierarchies"] = hu.compare_hierarchies


def _compare_hierarchies(inf_complete, true_complete):
    """Hierarchy accuracy with the second 2024 bug fixed (found 2026-11-06). For a true group, the 2024 code
    counts as "not below" every group under any other true group, including the groups under its own
    ancestors, which include its own children; a correctly nested group with two children then scores at
    most 0.67. Here "not below" is every true group except the group itself and the groups truly below it.
    Everything else (matching, F1, leaves scored 1 when nothing is below them) is the 2024 code."""
    fi = hu.flatten_hierarchy(inf_complete)
    ft = hu.flatten_hierarchy(true_complete)
    levels = list(ft.keys())
    every = set(levels) | {g for below in ft.values() for g in below}
    scores = []
    for hrc in levels:
        true_below = ft[hrc]
        true_not_below = every - set(true_below) - {hrc}
        inf_below = fi[hrc]
        tp = sum(1 for g in inf_below if g in true_below)
        fp = sum(1 for g in inf_below if g in true_not_below)
        fn = sum(1 for g in true_below if g not in inf_below)
        scores.append(1 if fp == 0 and len(true_below) == 0 else hu.compute_fscore(tp, fp, fn))
    return np.mean(scores)


@contextlib.contextmanager
def label_fix(on=True):
    """The corrected 2024 metrics: the cluster-naming fix (comparators) and the hierarchy-accuracy fix (every
    method, scDEF included). Off: the 2024 code verbatim."""
    if on:
        hu.get_hierarchy_from_clusters = _get_hierarchy_from_clusters
        hu.get_cluster_obs_association_scores = _get_cluster_obs_association_scores
        hu.simplify_hierarchy = _simplify_hierarchy
        hu.compare_hierarchies = _compare_hierarchies
    try:
        yield
    finally:
        for k, f in _ORIG.items():
            setattr(hu, k, f)


# --------------------------------------------------------------------------- clustering protocol
# decision-clustering-protocol (proposed 2026-10-05, accepted with changes 2026-11-06, after the results):
# - Representation, for every metric of the cell embedding: the proportions of each cell's weights (rows
#   normalised to sum 1) for the non-negative factor models (scDEF and its variants, NMF, scHPF), since the total
#   weight of a cell mixes with its depth (planaria, scDEF: correlation -0.50 with library size, totals from 3.8 to
#   over 10,000); the embedding itself for PCA, f-scLVM, scVI, Harmony and Scanorama. A Ward baseline uses its
#   parent's embedding and rule (Leiden_Ward: PCA; scDEF_1L_Ward: the one-layer scDEF).
# - Cell-type ARI (and AMI, homogeneity, completeness): KMeans at k = number of annotated types (scib-metrics'
#   default; nothing tuned to the labels). nSBM has no embedding: its level whose size is closest to k.
# - Supplementary: Leiden at the five resolutions of the 2024 sweep, the best cell-type ARI (scIB's original).
# - Batch ARI and the batch-group metrics: Leiden at resolution 1.0, chosen without the labels (nSBM: its finest
#   level). The best-ARI clustering would also be the one that ignores the batches, for every method.
# - Cell-type and batch ASW: on the same representation.
PROTOCOL_RESOLUTIONS = [1.0, 0.6, 0.3, 0.1, 0.01]
PROPORTION_METHODS = {"scDEF", "scDEF_un", "scDEF_decomposed", "scDEF_1L", "NMF", "scHPF"}


def protocol_base(method):
    if method == "Leiden_Ward":
        return "PCA"
    if method == "scDEF_1L_Ward":
        return "scDEF_1L"
    return method[:-5] if method.endswith("_Ward") else method


def protocol_representation(method, latent):
    X = np.asarray(latent, dtype=np.float64)
    if protocol_base(method) in PROPORTION_METHODS:
        if X.min() < 0:
            raise ValueError(f"{method}: negative weights, cannot take proportions")
        X = X / np.maximum(X.sum(1, keepdims=True), 1e-12)
    return X


def _leiden(X, resolutions):
    import scanpy as sc
    b = ad.AnnData(np.asarray(X, dtype=np.float32))
    sc.pp.neighbors(b, use_rep="X", random_state=0)
    out = []
    for res in resolutions:
        sc.tl.leiden(b, resolution=res, random_state=0, flavor="igraph", n_iterations=2, key_added="p")
        out.append((res, b.obs["p"].astype(str).to_numpy()))
    return out


def protocol_values(obs, celltype_key, batch_key, method, latent=None, levels=None):
    """-> ({metric: value}, the label-free clustering for the batch metrics, description)."""
    from sklearn import metrics as skm
    from sklearn.cluster import KMeans
    y = obs[celltype_key].astype(str).to_numpy()
    k = int(len(np.unique(y)))
    if latent is not None:
        X = protocol_representation(method, latent)
        main = KMeans(k, n_init=10, random_state=0).fit_predict(X).astype(str)
        sweep = _leiden(X, PROTOCOL_RESOLUTIONS)
        batch_cl = sweep[0][1]                                   # resolution 1.0
        leiden_best = max(skm.adjusted_rand_score(y, c) for _, c in sweep)
        how = dict(representation="proportions" if protocol_base(method) in PROPORTION_METHODS else "embedding",
                   cell_type="KMeans, k = %d" % k, batch="Leiden, resolution 1.0")
    else:
        lv = [np.asarray(c).astype(str) for c in levels]
        sizes = [len(np.unique(c)) for c in lv]
        main = lv[int(np.argmin([abs(s - k) for s in sizes]))]   # ties: the finer level
        batch_cl = lv[0]
        leiden_best = max(skm.adjusted_rand_score(y, c) for c in lv)
        how = dict(representation="own levels", cell_type="level with size closest to k = %d (%d)" % (k, len(np.unique(main))),
                   batch="finest level")
    vals = {"Cell Type ARI": float(skm.adjusted_rand_score(y, main)),
            "Cell Type AMI": float(skm.adjusted_mutual_info_score(y, main)),
            "Cell Type homogeneity": float(skm.homogeneity_score(y, main)),
            "Cell Type completeness": float(skm.completeness_score(y, main)),
            "Cell Type ARI (Leiden, best of five)": float(leiden_best)}
    if latent is not None:
        vals["Cell Type ASW"] = float(skm.silhouette_score(X, y))
    if batch_key is not None and batch_key in obs:
        b = obs[batch_key].astype(str).to_numpy()
        vals["Batch ARI"] = float(skm.adjusted_rand_score(b, batch_cl))
        if latent is not None:
            vals["Batch ASW"] = float(skm.silhouette_score(X, b))
    how.update(k=k, batch_clusters=int(len(np.unique(batch_cl))), cell_type_clusters=int(len(np.unique(main))))
    return vals, batch_cl, how


# --------------------------------------------------------------------------- inputs


# --------------------------------------------------------------------------- signature readout of factor models
# Pre-declared 2026-11-11 (exp-simulation-benchmark-rerun, "Signature readout for factor models"; decided after the
# results): for the factor models, each annotated type is read through its most enriched factor at the finest level,
# not the factor holding the largest weight in its cells. The previous value is kept as
# "Signature accuracy (largest-weight readout)".
FACTOR_SIGNATURE_METHODS = {"scDEF", "scDEF_un", "scDEF_decomposed", "scDEF_decomposed_t1", "NMF", "scHPF", "fscLVM"}


def enriched_factors(weights, labels, groups):
    """For each group, the column with the largest mean normalised weight in the group over the mean outside it."""
    W = np.asarray(weights, dtype=float)
    W = W / np.maximum(W.sum(1, keepdims=True), 1e-12)
    labels = np.asarray(labels).astype(str)
    out = {}
    for g in groups:
        G = labels == str(g)
        if G.any() and not G.all():
            out[str(g)] = int(np.argmax(W[G].mean(0) / np.maximum(W[~G].mean(0), 1e-12)))
    return out


def signature_accuracy_enriched(weights, labels, markers, factor_signatures, top_genes=20):
    """The 2024 signature accuracy (scdef.utils.score_utils.score_signature of the top 20 genes, with the
    non-markers built as in evaluate.py, from the type names) with each type read through its most enriched factor;
    factor_signatures[k] is the ranked gene list of weight column k."""
    from scdef.utils import score_utils as su
    pick = enriched_factors(weights, labels, list(markers))
    scores = {}
    for ct, mk in markers.items():
        if ct in pick:
            sig = list(factor_signatures[pick[ct]])[:top_genes]
            non = [m for m in markers if m not in mk]          # as evaluate.py: iterates over the type names
            scores[ct] = float(su.score_signature(sig, mk, non))
    return (float(np.mean(list(scores.values()))) if scores else np.nan), scores, pick


def apply_signature_readout(res, weights, labels, markers, factor_signatures):
    v = res["fixed"]["values"]
    if "Signature accuracy" not in v:
        return
    sa, per, pick = signature_accuracy_enriched(weights, labels, markers, factor_signatures)
    v["Signature accuracy (largest-weight readout)"] = v["Signature accuracy"]
    v["Signature accuracy"] = sa
    res["signature_readout"] = dict(rule="most enriched factor at the finest level", per_type=per, picks=pick)


def signature_precision_auprc(labels, markers, picks, signatures, scores, var_names, top=20):
    """For each annotated type with markers: the cluster or factor picked for it (picks: type -> key), the fraction of
    the first n genes of its signature that are markers of the type (n = min(top, number of markers among the
    genes)), and the average precision of its scores over every gene against the markers. Returns the means over
    types and the per-type values (2026-11-11; replaces signature accuracy, which never counted false positives)."""
    from sklearn.metrics import average_precision_score
    genes = np.asarray(var_names).astype(str)
    prec, aps, per = [], [], {}
    for ct, mk in markers.items():
        mk = [g for g in mk if g in set(genes)]
        k = picks.get(str(ct))
        if not mk or k is None or k not in signatures:
            continue
        n = min(top, len(mk))
        p_ = len(set(list(signatures[k])[:n]) & set(mk)) / n
        ap = np.nan
        if scores is not None and k in scores:
            y = np.isin(genes, mk).astype(int)
            ap = float(average_precision_score(y, np.nan_to_num(np.asarray(scores[k], float), nan=-np.inf)))
        prec.append(p_); aps.append(ap); per[str(ct)] = dict(key=str(k), precision=p_, auprc=ap)
    mean = lambda v: float(np.nanmean(v)) if len(v) and np.isfinite(v).any() else np.nan
    return mean(prec), mean(aps), per


def best_f1_picks(labels, clusters, groups):
    """For each group, the cluster with the largest F1 score."""
    y = np.asarray(labels).astype(str)
    z = np.asarray(clusters).astype(str)
    tab = pd.crosstab(pd.Series(y, name="g"), pd.Series(z, name="k"))
    f1 = 2 * tab.to_numpy() / (tab.sum(1).to_numpy()[:, None] + tab.sum(0).to_numpy()[None, :])
    return {str(g): str(tab.columns[int(np.argmax(f1[i]))]) for i, g in enumerate(tab.index) if str(g) in
            set(map(str, groups))}


def factor_marker_signatures(inp, raw, latent):
    """Marker scores (scd.tl.marker_scores_from_usages, the cNMF gene spectra scores) of a factor model's
    finest-level factors, from the shared counts and the proportions of its weights (pre-declared 2026-11-11):
    (key -> ranked genes, key -> scores), keys str(k) for weight column k."""
    genes = [str(g) for g in raw["var_names"]]
    sub = inp if list(map(str, inp.var_names)) == genes else inp[:, genes]
    W = np.asarray(latent, dtype=float)
    P = W / np.maximum(W.sum(1, keepdims=True), 1e-12)
    B = scd.tl.marker_scores_from_usages(sub.layers["counts"], P)
    g = np.asarray(genes)
    return ({str(k): list(g[np.argsort(-B[k])]) for k in range(B.shape[0])},
            {str(k): B[k] for k in range(B.shape[0])})


def add_signature_precision(res, labels, markers, picks, signatures, scores, var_names):
    p_, ap, per = signature_precision_auprc(labels, markers, picks, signatures, scores, var_names)
    res["fixed"]["values"]["Signature precision"] = p_
    res["fixed"]["values"]["Signature AUPRC"] = ap
    res["signature_precision"] = per


def load_input(data):
    d = C.out_dir(data)
    a = ad.read_h5ad(os.path.join(d, "input.h5ad"))
    mk = C.read_json(os.path.join(d, "markers.json"))
    return a, mk


def method_outputs(d):
    out = C.read_json(os.path.join(d, "outputs.json"))
    npz = np.load(os.path.join(d, "scores.npz"))
    scores = {n: npz[f"s{i}"] for i, n in enumerate(out["score_names"])}
    assignments = out["assignments"]
    layer_names = ["h" * level for level in range(len(assignments))]
    layer_sizes = [len(np.unique(c)) for c in assignments]
    hierarchy = hu.get_hierarchy_from_clusters(assignments)     # as run_multiple_resolutions / run_nsbm
    simplified = hu.simplify_hierarchy(hierarchy, layer_names, layer_sizes)
    return dict(latents=[np.load(os.path.join(d, "latent0.npy"))], signatures=out["signatures"],
                assignments=assignments, scores=scores, sizes=out["sizes"], simplified_hierarchy=simplified), out


def scoring_adata(inp, var_names, obs_names):
    if list(obs_names) != list(inp.obs_names):
        raise ValueError("method cells differ from the shared input")
    return ad.AnnData(X=sp.csr_matrix((inp.n_obs, len(var_names)), dtype=np.float32), obs=inp.obs.copy(),
                      var=pd.DataFrame(index=pd.Index(var_names).astype(str)))


def flatten(df, method):
    """evaluate_methods' column for one method -> {metric: value}; list metrics (signature accuracy) averaged
    as gather_scores.py did, the per-type values returned separately."""
    vals, per_type = {}, {}
    for metric, v in df[method].items():
        if isinstance(v, (list, tuple, np.ndarray)):
            per_type[metric] = [float(x) for x in v]
            v = float(np.mean(v)) if len(v) else np.nan
        vals[metric] = float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else np.nan
    return vals, per_type


def evaluate(adata, outs, label, data, markers, metrics):
    spec = C.DATASETS[data]
    hobs = spec["hierarchy_obs"]
    true_h = hu.get_hierarchy_from_clusters([adata.obs[o].astype(str).values for o in hobs], use_names=True)
    df = ev.evaluate_methods(adata, metrics, {label: outs}, true_hierarchy=true_h, hierarchy_obs_keys=hobs,
                             markers=markers, celltype_obs_key=hobs,
                             batch_obs_key="Batch" if spec["batch"] else "batch")
    return flatten(df, label)


# --------------------------------------------------------------------------- scDEF


def scdef_runtime(data, label):
    spec = C.DATASETS[data]
    log = C.read_json(C.REFIT_JSON.format(spec["refit"]))
    kind = "nobatch" if spec["batch"] is None else {"scDEF": "ref", "scDEF_un": "nobatch",
                                                    "scDEF_decomposed": "ref"}[label]
    seeds = log[kind]
    chosen = log[f"{kind}_chosen_seed"]
    kept = [r["minutes"] for r in seeds if r["seed"] == chosen][0]
    total = sum(r["minutes"] for r in seeds)
    extra = log["decomposed"]["minutes"] if label == "scDEF_decomposed" else 0.0
    return dict(runtime_s=60 * (kept + extra), five_seed_total_s=60 * (total + extra), chosen_seed=chosen,
                kept_seed_minutes=kept, five_seed_minutes=[r["minutes"] for r in seeds],
                decomposition_minutes=extra or None, refit_kind=kind,
                source=os.path.relpath(C.REFIT_JSON.format(spec["refit"]), C.REPRO))


def load_scdef(data, name, inp, cache={}):
    spec = C.DATASETS[data]
    if data not in cache:                        # keyed by data set (a single key broke loops over data sets)
        rb = C.refit_module()
        cache[data] = rb.tutorial_adata(spec["refit"])
    a = cache[data].copy()
    if list(a.obs_names.astype(str)) != list(inp.obs_names) or list(a.var_names.astype(str)) != list(inp.var_names):
        raise ValueError("tutorial_adata no longer matches the shared input")
    for k in spec["hierarchy_obs"] + (["Batch"] if spec["batch"] else []):
        a.obs[k] = inp.obs[k].to_numpy()
    m = scd.scDEF.load(os.path.join(C.NB, name), adata=a)
    if "dec-top2" in name:
        m.decomposition_top_layer = 2            # saved before decompositions recorded their anchor
        m.annotate_adata()
    elif "L0" not in m.adata.obs:
        m.annotate_adata()
    return m


# --------------------------------------------------------------------------- main


def main(data, force=False):
    data = C.canonical(data)
    spec = C.DATASETS[data]
    d = C.out_dir(data)
    sdir = C.out_dir(data, "scores")
    inp, mk = load_input(data)
    markers = mk["markers"]
    metrics = list(spec["metrics"])
    if not markers:
        metrics.remove("Signature accuracy")
    rows, rows_verbatim, per_type, runtimes, notes = [], [], {}, [], {}

    def cached(label, mtime, fn):
        path = os.path.join(sdir, f"{label}.json")
        prev = C.read_json(path)
        if prev and not force and prev.get("source_mtime") == mtime:
            return prev
        res = fn()
        res["source_mtime"] = mtime
        C.write_json(res, path)
        return res

    # comparators, then the hierarchical-clustering baselines of hc.py (same format), when they have been run
    mdir = os.path.join(d, "methods")
    hc_done = sorted(m for m in os.listdir(mdir) if m.endswith("_Ward") and os.path.exists(os.path.join(mdir, m, "run.json")))
    for method in spec["methods"] + hc_done:
        md = os.path.join(d, "methods", method)
        run = C.read_json(os.path.join(md, "run.json"), {})
        if run.get("status") != "ok":
            runtimes.append(dict(method=method, status=run.get("status", "not run"),
                                 error=run.get("error"), budget_hours=run.get("budget_hours")))
            continue

        def compute(md=md, method=method):
            res = {}
            for fix in (True, False):
                with label_fix(fix):
                    outs, raw = method_outputs(md)
                    adata = scoring_adata(inp, raw["var_names"], raw["obs_names"])
                    vals, pt = evaluate(adata, outs, method, data, markers, metrics)
                res["fixed" if fix else "verbatim"] = dict(values=vals, per_type=pt)
            pv, _, info = protocol_values(inp.obs, spec["hierarchy_obs"][0], "Batch" if spec["batch"] else None, method,
                                          **({"levels": raw["assignments"]} if method == "nSBM"
                                             else {"latent": np.load(os.path.join(md, "latent0.npy"))}))
            res["fixed"]["values"].update(pv)
            res["protocol"] = info
            if method in FACTOR_SIGNATURE_METHODS and markers:
                lat = np.load(os.path.join(md, "latent0.npy"))
                apply_signature_readout(res, lat, adata.obs[spec["hierarchy_obs"][0]], markers,
                                        [raw["signatures"][str(k)] for k in range(lat.shape[1])])
            if markers:
                labels = adata.obs[spec["hierarchy_obs"][0]]
                z = np.load(os.path.join(md, "scores.npz"))
                gsc = {str(nm): z[f"s{i}"] for i, nm in enumerate(raw["score_names"])}
                if method in FACTOR_SIGNATURE_METHODS:
                    lat = np.load(os.path.join(md, "latent0.npy"))
                    picks = {g: str(k) for g, k in enriched_factors(lat, labels, list(markers)).items()}
                    msig, msc = factor_marker_signatures(inp, raw, lat)
                    add_signature_precision(res, labels, markers, picks, msig, msc, raw["var_names"])
                else:
                    picks = best_f1_picks(labels, raw["assignments"][0], list(markers))
                    add_signature_precision(res, labels, markers, picks, raw["signatures"], gsc, raw["var_names"])
            return res

        t = time.time()
        try:
            res = cached(method, os.path.getmtime(os.path.join(md, "run.json")), compute)
        except Exception as e:                       # record and go on with the other methods
            print(f"scoring {method} failed: {e!r}", flush=True)
            runtimes.append(dict(method=method, status="scoring failed", error=repr(e)))
            continue
        print(f"scored {method} ({time.time() - t:.0f} s)", flush=True)
        for kind, sink in (("fixed", rows), ("verbatim", rows_verbatim)):
            for metric, v in res[kind]["values"].items():
                sink.append(dict(data=data, method=method, metric=metric, value=v))
        per_type[method] = res["fixed"]["per_type"]
        rows.append(dict(data=data, method=method, metric="Clustering: batch-metric clusters",
                         value=res["protocol"]["batch_clusters"]))
        rows.append(dict(data=data, method=method, metric="Runtime", value=run["runtime_s"]))
        rows_verbatim.append(dict(data=data, method=method, metric="Runtime", value=run["runtime_s"]))
        runtimes.append(dict(method=method, status="ok", runtime_s=run["runtime_s"],
                             device="gpu" if run.get("cuda_visible_devices") not in (None, "") else "cpu",
                             torch_device=run.get("torch_device"), genes=run.get("genes"),
                             n_genes=run.get("n_genes"), levels=run.get("levels")))

    # scDEF (saved best-of-five models)
    for label, name in spec["scdef"].items():
        path = os.path.join(C.NB, name)
        if not os.path.isdir(path):
            runtimes.append(dict(method=label, status=f"model {name} not found"))
            continue

        def compute(label=label, name=name):
            m = load_scdef(data, name, inp)
            res = dict(layer_sizes=[len(f) for f in m.factor_names], model=name)
            for fix in (True, False):
                with label_fix(fix):
                    vals, pt = evaluate(m.adata, m, label, data, markers, metrics)
                res["fixed" if fix else "verbatim"] = dict(values=vals, per_type=pt)
            pv, _, info = protocol_values(inp.obs, spec["hierarchy_obs"][0], "Batch" if spec["batch"] else None, label,
                                          latent=m.adata.obsm["X_L0"])
            res["fixed"]["values"].update(pv)
            res["protocol"] = info
            if markers:
                sd, gsc = m.get_signatures_dict(scores=True)
                labels = m.adata.obs[spec["hierarchy_obs"][0]]
                apply_signature_readout(res, m.adata.obsm["X_L0"], labels, markers, [sd[f] for f in m.factor_names[0]])
                names0 = list(m.factor_names[0])
                picks = {g: names0[k] for g, k in enriched_factors(m.adata.obsm["X_L0"], labels, list(markers)).items()}
                msig, msc = scd.tl.marker_genes(m)
                add_signature_precision(res, labels, markers, picks, msig, msc, list(m.adata.var_names))
            return res

        t = time.time()
        try:
            res = cached(label, os.path.getmtime(path), compute)
        except Exception as e:
            print(f"scoring {label} failed: {e!r}", flush=True)
            runtimes.append(dict(method=label, status="scoring failed", error=repr(e)))
            continue
        print(f"scored {label} ({time.time() - t:.0f} s)", flush=True)
        rt = scdef_runtime(data, label)
        for kind, sink in (("fixed", rows), ("verbatim", rows_verbatim)):
            for metric, v in res[kind]["values"].items():
                sink.append(dict(data=data, method=label, metric=metric, value=v))
            sink.append(dict(data=data, method=label, metric="Runtime", value=rt["runtime_s"]))
            sink.append(dict(data=data, method=label, metric="Runtime (five seeds)", value=rt["five_seed_total_s"]))
        rows.append(dict(data=data, method=label, metric="Clustering: batch-metric clusters",
                         value=res["protocol"]["batch_clusters"]))
        per_type[label] = res["fixed"]["per_type"]
        runtimes.append(dict(method=label, status="ok", model=name, layer_sizes=res["layer_sizes"], **rt))

    long = pd.DataFrame(rows)
    long["group"] = np.where(long["method"] == "scDEF_decomposed", "decomposed (reported separately)", "main")
    long.to_csv(os.path.join(d, "scores.csv"), index=False)
    pd.DataFrame(rows_verbatim).to_csv(os.path.join(d, "scores_2024_verbatim.csv"), index=False)
    wide = long.pivot_table(index="metric", columns="method", values="value", aggfunc="first")
    order = [m for m in list(spec["scdef"]) + spec["methods"] + hc_done if m in wide.columns]
    wide[order].to_csv(os.path.join(d, "scores_wide.csv"))
    C.write_json(per_type, os.path.join(d, "signature_accuracy_per_type.json"))
    pd.DataFrame(runtimes).to_csv(os.path.join(d, "runtimes.csv"), index=False)

    settings = dict(
        data=data, input=C.read_json(os.path.join(d, "input_info.json")),
        markers=dict(source=mk["source"], missing_from_genes=mk["missing_from_genes"],
                     types_without_markers=mk["types_without_markers"]),
        metrics=metrics, metric_code="workflow/scripts/_benchmark/evaluate.py (unchanged)",
        label_naming_fix="on (scores.csv); off in scores_2024_verbatim.csv",
        hierarchy_accuracy_fix="on (scores.csv): a group's own children are not counted as false positives; "
                               "off in scores_2024_verbatim.csv",
        clustering_protocol=dict(rule="decision-clustering-protocol (2026-11-06): proportions for "
                                      f"{sorted(PROPORTION_METHODS)}; cell-type ARI/AMI/homogeneity/completeness by "
                                      "KMeans at k = number of types (nSBM: level closest to k); supplementary ARI by "
                                      f"Leiden best of {PROTOCOL_RESOLUTIONS}; batch ARI by Leiden at resolution 1.0 "
                                      "(nSBM: finest level); ASW on the same representation",
                                 kmeans="n_init 10, random_state 0",
                                 neighbors="scanpy defaults, random_state 0",
                                 leiden="flavor igraph, n_iterations 2, random_state 0"),
        scdef=dict(models=spec["scdef"], refitted=False, rule="best of five seeds by final loss "
                   "(make_figures/refit_best_of_seeds.py, fit_fixed)",
                   runtime="kept seed's minutes from the refit log; five-seed total as a separate row"),
        methods={m: dict(C.SETTINGS[m], **{k: v for k, v in (C.read_json(os.path.join(d, "methods", m, "run.json"), {}) or {}).items()
                                            if k in ("status", "genes", "n_genes", "runtime_s", "preprocess_s",
                                                     "cuda_visible_devices", "torch_device", "scvi_epochs",
                                                     "regressed_out", "batches_reordered_for_scanorama",
                                                     "versions", "levels", "error")})
                 for m in spec["methods"]},
        seed=C.SEED, budget_hours=C.BUDGET_HOURS, scored=time.strftime("%Y-%m-%d %H:%M:%S"))
    C.write_json(settings, os.path.join(d, "settings.json"))
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(wide[order].round(3))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("data")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    main(a.data, a.force)
