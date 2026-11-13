"""Score every finished method on one simulated data set.

    python score.py <id> [--force]          (env: scdef)

1. The 2024 metrics, from ``workflow/scripts/_benchmark/evaluate.py`` unchanged, through the real-data scorer
   (``realdata_benchmark/score.py``): Cell Type ARI and ASW against GroupC, Batch ARI and ASW (four batches),
   Signature accuracy (the 2024 Seurat top-10 markers of the GroupC groups), Signature sparsity, Hierarchy accuracy
   and Hierarchical signature consistency against the true GroupC -> GroupB -> GroupA hierarchy (built as 2024
   prepare_input.py did), Runtime. Comparators are scored with the cluster-naming fix (scores.csv) and with the 2024
   code verbatim (scores_2024_verbatim.csv); scDEF's own path is unaffected by the naming fix.
   Changed 2026-11-06 after the results (SCORE_VERSION 3; decision-benchmark-clustering-protocol): hierarchy
   accuracy no longer counts a group's own children as false positives (every method, scDEF included;
   realdata_benchmark/score.py _compare_hierarchies); Cell Type ARI, Batch ARI and the shared/specific batch groups
   follow decision-clustering-protocol (accepted with changes 2026-11-06; realdata_benchmark/score.py,
   "clustering protocol"): proportions of the weights for the non-negative factor models (scDEF, NMF, scHPF and
   their Ward baselines); Cell Type ARI, AMI, homogeneity and completeness from KMeans at k = 8 (nSBM: its level
   closest to 8); "Cell Type ARI (Leiden, best of five)" as a supplementary value; Batch ARI and the batch groups
   from Leiden at resolution 1.0 (nSBM: its finest level); Cell Type and Batch ASW on the same representation.
   scores_2024_verbatim.csv keeps the 2024 values.
2. The metrics added before any result (scDEF-markdown/experiments/exp-simulation-benchmark-rerun.md):
   - Level recovery: for each true level (GroupC 8, GroupB 4, GroupA 2 groups), the highest ARI between that level
     and any level of the method (scDEF: every layer's hard assignment, obs['L<k>']; others: every resolution of the
     sweep, or every nSBM level). Ties go to the first (finest) level.
   - Level signatures: at that level of the method, each true group is matched to the cluster or factor with the
     highest F1 (cells); the score is the fraction of the top n genes of its signature that are true up-regulated
     genes of the group (prepare.py's truth.json), n = min(20, number of such genes), averaged over the groups.
     Signatures: comparators, their 2024 signature at that resolution (Wilcoxon scores or factor loadings, as saved
     by run_method.py); scDEF, the factor's gene ranking (``get_signatures_dict``, as the 2024 signature accuracy).
   - Layer sizes: the number of factors in every layer of the kept scDEF model; summarised as the number of true
     levels (8, 4, 2) equal to some layer size, and the layer-0 size minus 8. For reference only, the other methods'
     cluster counts: Leiden at resolution 1.0 (PCA, scVI, Harmony, Scanorama: their default), every nSBM level, and
     the sweep sizes of NMF, scHPF and f-scLVM (no default).

3. Added later the same day (before any result): the hierarchical-clustering baselines of hc.py are scored like the
   comparators (all metrics; Runtime = upstream step + tree step, "Runtime (tree step)" alone); the decomposition
   (decompose.py, four batches) like scDEF; and, with four batches, for every method at its finest level:
   - Specific groups kept: the number of true batch-specific groups (data/groups.csv, from the generator) whose
     best-F1 cluster has F1 >= 0.5 and is called batch-specific (>= 70% of its cells from one batch, >= 20 cells);
     empty when the data set has no batch-specific group (similar batches);
   - Shared groups split by batch: the fraction of the cells of the true shared groups that lie in clusters called
     batch-specific (lower is better).
   Written to batch_groups.csv and as rows of scores.csv.

scDEF is the kept best-of-five model (fit_scdef.py), reloaded on ``simcommon.scdef_adata``; runtime is the kept seed's
minutes, with the five-seed total as a separate row. Per-method results are cached in scores/ (keyed by the source's
modification time and SCORE_VERSION). Writes scores.csv, scores_2024_verbatim.csv, level_scores.csv,
layer_sizes.csv, level_details.json, signature_accuracy_per_type.json, runtimes.csv and settings.json to <id>/.
"""
import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, average_precision_score

import simcommon as S

rs = S.rd_score()              # sets JAX_PLATFORMS=cpu (default), imports scdef, the 2024 evaluate, label_fix
scd, hu, ev = rs.scd, rs.hu, rs.ev
SCORE_VERSION = 9          # 3 (2026-11-06): hierarchy-accuracy fix and a Leiden protocol; 4 (same day): the agreed protocol;
                           # 5 (2026-11-10, after the results): shared groups kept, the counterpart of specific groups kept;
                           # 6 (same day, pre-declared, after the results): batch-specific groups nested under their parent;
                           # 7 (2026-11-11, pre-declared, after the results): factor-model signatures read through the
                           # most enriched factor at the finest level (realdata_benchmark/score.py, signature readout);
                           # 8 (same day, pre-declared, after the results): level signatures AUPRC;
                           # 9 (same day, pre-declared, after the results): level signatures of factor models from
                           # marker scores (cNMF gene spectra scores; scd.tl.marker_scores)
LEIDEN_DEFAULT = {"PCA", "scVI", "Harmony", "Scanorama"}       # resolution 1.0 is the first level of the sweep


# --------------------------------------------------------------------------- 2024 metrics
def evaluate(adata, outs, label, ds, markers):
    hobs = S.HIERARCHY_OBS
    true_h = hu.get_hierarchy_from_clusters([adata.obs[o].astype(str).values for o in hobs], use_names=True)
    df = ev.evaluate_methods(adata, S.metrics(ds), {label: outs}, true_hierarchy=true_h, hierarchy_obs_keys=hobs,
                             markers=markers, celltype_obs_key=hobs, batch_obs_key="Batch")
    return rs.flatten(df, label)


# --------------------------------------------------------------------------- added metrics
def _key(x):
    return rs._num(x)


def auprc(gene_scores, var_names, up):
    """Average precision of one cluster's or factor's scores over every gene against the group's true up-regulated
    genes (pre-declared 2026-11-11, "Level signatures AUPRC")."""
    if gene_scores is None or not len(up):
        return np.nan
    y = np.isin(np.asarray(var_names).astype(str), list(up)).astype(int)
    if y.sum() == 0:
        return np.nan
    return float(average_precision_score(y, np.nan_to_num(np.asarray(gene_scores, dtype=float), nan=-np.inf)))


def level_metrics(obs, levels, signatures, truth, top=S.TOP_GENES, gene_scores=None, var_names=None):
    """levels: [(name, labels)], finest first; signatures: cluster label -> ranked genes; truth: level -> group ->
    up-regulated genes; gene_scores: cluster label -> scores of every gene in var_names (for the AUPRC). Returns
    {true level: record}."""
    out = {}
    for tl in S.HIERARCHY_OBS:
        y = obs[tl].astype(str).to_numpy()
        aris = [float(adjusted_rand_score(y, np.asarray(z).astype(str))) for _, z in levels]
        j = int(np.argmax(aris))
        name, z = levels[j]
        z = np.asarray(z).astype(str)
        tab = pd.crosstab(pd.Series(y, name="true"), pd.Series(z, name="method"))
        tab = tab[sorted(tab.columns, key=_key)]
        size_g = tab.sum(1).to_numpy()[:, None]
        size_k = tab.sum(0).to_numpy()[None, :]
        f1 = 2 * tab.to_numpy() / (size_g + size_k)
        groups, scores, aps = {}, [], []
        for i, g in enumerate(tab.index):
            k = tab.columns[int(np.argmax(f1[i]))]
            up = truth[tl].get(g, [])
            n = min(top, len(up))
            sig = signatures.get(k)
            if n == 0 or sig is None:
                sc_g = np.nan
            else:
                sc_g = len(set(sig[:n]) & set(up)) / n
            ap = auprc((gene_scores or {}).get(k), var_names, up) if gene_scores is not None else np.nan
            groups[g] = dict(cluster=str(k), f1=float(f1[i].max()), cluster_cells=int(tab[k].sum()),
                             group_cells=int(size_g[i, 0]), n_true_up=len(up), n_top=n,
                             signature_missing=sig is None, score=sc_g, auprc=ap,
                             top_genes=list(sig[:n]) if sig else None)
            scores.append(sc_g)
            aps.append(ap)
        out[tl] = dict(n_groups=int(len(tab.index)), recovery=aris[j], method_level=name, method_level_index=j,
                       method_level_clusters=int(len(set(z))), aris=dict(zip([n for n, _ in levels], aris)),
                       signatures=float(np.nanmean(scores)) if np.isfinite(scores).any() else np.nan,
                       signatures_auprc=float(np.nanmean(aps)) if np.isfinite(aps).any() else np.nan,
                       groups=groups)
    return out


def nesting(obs, levels, groups, kept):
    """Batch-specific groups nested under their shared parent (pre-declared 2026-11-10, exp-simulation-benchmark-rerun,
    item 3; amended the same day after a test on one data set, see the note). levels: [(name, labels)] of one
    method's own hierarchy; kept: batch_groups(...)["specific"] of the same method (the panel D readout). A batch-
    specific group is nested when (i) it is kept (its best-F1 protocol cluster has F1 >= SPECIFIC_MIN_F1 and is
    batch-specific), and (ii) at the method's level with the best ARI to GroupB (as level_metrics), the cluster holding
    most of its cells also holds most of the cells of its sibling (the other child of its GroupB parent, from another
    batch) and has F1 >= SPECIFIC_MIN_F1 with the parent. As first declared, (ii) asked only for the F1 with the
    parent, which a cluster holding the group alone already reaches (about 0.67)."""
    zs = [np.asarray(z).astype(str) for _, z in levels]
    names = [n for n, _ in levels]
    y_c = obs["GroupC"].astype(str).to_numpy()
    y_b = obs["GroupB"].astype(str).to_numpy()
    aris_b = [float(adjusted_rand_score(y_b, z)) for z in zs]
    i_coarse = int(np.argmax(aris_b))
    coarse = zs[i_coarse]
    spec = sorted(set(groups.loc[groups["kind"] == "specific", "GroupC"].astype(str)))
    per, n_nested = {}, 0
    for g in spec:
        G = y_c == g
        parent = pd.Series(y_b[G]).mode().iloc[0]
        siblings = sorted(set(y_c[(y_b == parent) & ~G]))
        kc = pd.Series(coarse[G]).mode().iloc[0]
        joined = all(pd.Series(coarse[y_c == s]).mode().iloc[0] == kc for s in siblings) if siblings else False
        P, K = y_b == parent, coarse == kc
        f1p = 2 * float((P & K).sum()) / (P.sum() + K.sum())
        is_kept = bool(kept.get(g, {}).get("kept") in (True, "True"))
        ok = bool(is_kept and joined and f1p >= S.SPECIFIC_MIN_F1)
        n_nested += int(ok)
        per[g] = dict(kept=is_kept, parent=parent, siblings=siblings, coarse_cluster=kc, joined_with_siblings=joined,
                      parent_f1=f1p, nested=ok)
    return dict(nested=n_nested if spec else np.nan, coarse_level=names[i_coarse], coarse_level_ari=aris_b[i_coarse],
                groups=per)


def enriched_level_signatures(res, obs, weights, factor_signatures, truth, top=S.TOP_GENES, factor_scores=None,
                              var_names=None):
    """Level signatures at the 8-group level read through the most enriched factor, when the method's best level for
    it is its finest (the level whose weights are saved); the previous value is kept as signatures_largest_weight."""
    rec = res["levels"].get("GroupC")
    if rec is None or rec.get("method_level_index") != 0:
        return
    pick = rs.enriched_factors(weights, obs["GroupC"], list(truth["GroupC"]))
    scores, aps = [], []
    for g, up in truth["GroupC"].items():
        n = min(top, len(up))
        if n and g in pick:
            sig = list(factor_signatures[pick[g]])
            scores.append(len(set(sig[:n]) & set(up)) / n)
            if factor_scores is not None:
                aps.append(auprc(factor_scores[pick[g]], var_names, up))
    rec["signatures_largest_weight"] = rec["signatures"]
    rec["signatures"] = float(np.mean(scores)) if scores else np.nan
    if factor_scores is not None:
        rec["signatures_auprc_largest_weight"] = rec.get("signatures_auprc")
        rec["signatures_auprc"] = float(np.nanmean(aps)) if aps else np.nan
    rec["signature_readout"] = "most enriched factor at the finest level"


def comparator_levels(method, raw):
    a = raw["assignments"]
    sweep = S.rc.SETTINGS.get(method, {}).get("resolutions")
    if method == "nSBM" or sweep is None or len(sweep) != len(a):
        names = [f"level {i}" for i in range(len(a))]
    else:
        names = [("resolution %g" if method in LEIDEN_DEFAULT | {"PCA"} else "K %g") % r for r in sweep]
    return [(n, np.asarray(x)) for n, x in zip(names, a)]


def comparator_sizes(method, raw):
    sizes = [int(len(set(x))) for x in raw["assignments"]]
    if method not in S.rc.SETTINGS:
        return dict(what="clusters at each level of the tree cut (the kept scDEF layer sizes, capped)", sizes=sizes,
                    all_levels=sizes, default_minus_8=None)
    if method in LEIDEN_DEFAULT:
        return dict(what="Leiden clusters at resolution 1.0 (default)", sizes=[sizes[0]], all_levels=sizes,
                    default_minus_8=sizes[0] - 8)
    if method == "nSBM":
        return dict(what="clusters at each nSBM level", sizes=sizes, all_levels=sizes, default_minus_8=None)
    return dict(what="clusters at each K of the sweep (no default)", sizes=sizes, all_levels=sizes,
                default_minus_8=None)


def scdef_sizes(m):
    sizes = [int(len(f)) for f in m.factor_names]
    nonempty = [int(m.adata.obs[ln].astype(str).nunique()) for ln in m.layer_names]
    matched = sorted({8, 4, 2} & set(sizes), reverse=True)
    return dict(what="factors per layer of the kept model", sizes=sizes, nonempty_clusters=nonempty,
                true_levels_matched=len(matched), matched=matched, l0_minus_8=sizes[0] - 8)


def batch_groups(obs, finest, groups):
    """Shared and batch-specific groups at a method's finest level (four batches)."""
    z = np.asarray(finest).astype(str)
    batch = obs["Batch"].astype(str).to_numpy()
    y = obs["GroupC"].astype(str).to_numpy()
    tab = pd.crosstab(pd.Series(z, name="cluster"), pd.Series(batch, name="batch"))
    size = tab.sum(1)
    called = set(tab.index[(tab.max(1) / size >= S.SPECIFIC_MIN_FRACTION) & (size >= S.SPECIFIC_MIN_CELLS)])
    spec = sorted(set(groups.loc[groups["kind"] == "specific", "GroupC"].astype(str)))
    shared = sorted(set(groups.loc[groups["kind"] == "shared", "GroupC"].astype(str)))
    per = {}
    kept = 0
    for g in spec:
        G = y == g
        best, bf1 = None, -1.0
        for k in sorted(set(z), key=_key):
            K = z == k
            f1 = 2 * float((G & K).sum()) / (G.sum() + K.sum())
            if f1 > bf1:
                best, bf1 = k, f1
        ok = bf1 >= S.SPECIFIC_MIN_F1 and best in called
        kept += int(ok)
        per[g] = dict(cluster=best, f1=bf1, cluster_called_specific=best in called, kept=ok)
    sh = np.isin(y, shared)
    split = float(np.isin(z[sh], list(called)).mean()) if sh.any() else np.nan
    # Added 2026-11-10, after the results: the counterpart of specific_kept. A shared group is kept when its
    # best-matching cluster (F1 >= SPECIFIC_MIN_F1) is not batch-specific. shared_split alone rewards a clustering
    # that resolves no group (NMF: 13-16 clusters, cell-type ARI 0.13, no shared cell split).
    shared_kept, per_shared = 0, {}
    for g in shared:
        G = y == g
        best, bf1 = None, -1.0
        for k in sorted(set(z), key=_key):
            K = z == k
            f1 = 2 * float((G & K).sum()) / (G.sum() + K.sum())
            if f1 > bf1:
                best, bf1 = k, f1
        ok = bf1 >= S.SPECIFIC_MIN_F1 and best not in called
        shared_kept += int(ok)
        per_shared[g] = dict(cluster=best, f1=bf1, cluster_called_specific=best in called, kept=ok)
    return dict(n_specific_true=len(spec), specific_kept=kept if spec else np.nan, n_shared_true=len(shared),
                shared_split=split, shared_kept=shared_kept if shared else np.nan, clusters=int(len(size)),
                clusters_called_specific=len(called), specific=per, shared=per_shared)


# --------------------------------------------------------------------------- scDEF
def load_scdef(ds, label, inp):
    a = S.scdef_adata(ds)
    if list(a.obs_names) != list(inp.obs_names) or list(a.var_names) != list(inp.var_names):
        raise ValueError("scdef_adata does not match the shared input")
    m = scd.scDEF.load(os.path.join(S.ds_dir(ds, "scdef", label), "model"), adata=a)
    if label in (S.DECOMPOSED, S.DECOMPOSED_T1) and getattr(m, "decomposition_top_layer", None) is None:
        m.decomposition_top_layer = 2 if label == S.DECOMPOSED else 1   # for a save without the anchor
        m.annotate_adata()
    elif "L0" not in m.adata.obs:
        m.annotate_adata()
    return m


# --------------------------------------------------------------------------- main
def main(ds, force=False):
    import anndata as ad

    p = S.parse_id(ds)
    d = S.ds_dir(ds)
    sdir = S.ds_dir(ds, "scores")
    inp = ad.read_h5ad(os.path.join(d, "input.h5ad"))
    markers = S.read_json(os.path.join(d, "markers.json"))["markers"]
    truth = S.read_json(os.path.join(d, "truth.json"))["up"]
    gpath = os.path.join(d, "data", "groups.csv")
    groups = pd.read_csv(gpath) if (p["multi"] and os.path.exists(gpath)) else None
    if p["multi"] and groups is None:
        print("data/groups.csv missing: shared/specific metric left empty", flush=True)
    brows = []
    base = dict(id=ds, scenario=p["scenario"], de_fscale=p["de_fscale"], similarity=p["similarity"], rep=p["rep"])
    rows, rows_verbatim, lrows, srows, runtimes, per_type, details = [], [], [], [], [], {}, {}

    def cached(label, mtime, fn):
        path = os.path.join(sdir, f"{label}.json")
        prev = S.read_json(path)
        if prev and not force and prev.get("source_mtime") == mtime and prev.get("score_version") == SCORE_VERSION:
            return prev
        res = fn()
        res.update(source_mtime=mtime, score_version=SCORE_VERSION)
        S.write_json(res, path)
        return json.loads(json.dumps(res, default=S.jsonable))

    def add(label, res, runtime_rows):
        for kind, sink in (("fixed", rows), ("verbatim", rows_verbatim)):
            for metric, v in res[kind]["values"].items():
                sink.append(dict(base, method=label, metric=metric, value=v))
            for metric, v in runtime_rows:
                sink.append(dict(base, method=label, metric=metric, value=v))
        per_type[label] = res["fixed"]["per_type"]
        details[label] = res["levels"]
        for tl, r in res["levels"].items():
            for metric, v in (("Level recovery", r["recovery"]), ("Level signatures", r["signatures"]),
                              ("Level signatures AUPRC", r.get("signatures_auprc", np.nan))):
                lrows.append(dict(base, method=label, true_level=tl, n_groups=r["n_groups"], metric=metric,
                                  value=v, method_level=r["method_level"],
                                  method_level_clusters=r["method_level_clusters"]))
        if res.get("batch_groups") is not None:
            bg = res["batch_groups"]
            brows.append(dict(base, method=label, n_specific_true=bg["n_specific_true"],
                              specific_kept=bg["specific_kept"], n_shared_true=bg["n_shared_true"],
                              shared_split=bg["shared_split"], shared_kept=bg.get("shared_kept"), clusters=bg["clusters"],
                              clusters_called_specific=bg["clusters_called_specific"],
                              specific=json.dumps(bg["specific"], default=S.jsonable)))
            for kind, sink in (("batch_groups", rows), ("batch_groups_2024", rows_verbatim)):
                b = res.get(kind) or bg
                sink.append(dict(base, method=label, metric="Specific groups kept", value=b["specific_kept"]))
                sink.append(dict(base, method=label, metric="Shared groups split by batch", value=b["shared_split"]))
                sink.append(dict(base, method=label, metric="Shared groups kept", value=b.get("shared_kept")))
            if res.get("nesting"):
                rows.append(dict(base, method=label, metric="Specific groups nested", value=res["nesting"]["nested"]))
        if res.get("protocol"):
            rows.append(dict(base, method=label, metric="Clustering: batch-metric clusters",
                             value=res["protocol"]["batch_clusters"]))
        ls = res["layer_sizes"]
        srows.append(dict(base, method=label, what=ls["what"], sizes=json.dumps(ls["sizes"]),
                          nonempty_clusters=json.dumps(ls.get("nonempty_clusters")),
                          all_levels=json.dumps(ls.get("all_levels")),
                          true_levels_matched=ls.get("true_levels_matched"), l0_minus_8=ls.get("l0_minus_8"),
                          default_minus_8=ls.get("default_minus_8")))

    # comparators, then the hierarchical-clustering baselines (same format)
    for method in S.scored_methods(ds):
        md = os.path.join(d, "methods", method)
        run = S.read_json(os.path.join(md, "run.json"), {})
        if run.get("status") != "ok":
            runtimes.append(dict(method=method, status=run.get("status", "not run"), error=run.get("error")))
            continue

        def compute(md=md, method=method):
            res = {}
            for fix in (True, False):
                with rs.label_fix(fix):
                    outs, raw = rs.method_outputs(md)
                    adata = rs.scoring_adata(inp, raw["var_names"], raw["obs_names"])
                    vals, pt = evaluate(adata, outs, method, ds, markers)
                res["fixed" if fix else "verbatim"] = dict(values=vals, per_type=pt)
            z = np.load(os.path.join(md, "scores.npz"))
            gsc = {str(nm): z[f"s{i}"] for i, nm in enumerate(raw["score_names"])}
            sig_lm, gsc_lm = dict(raw["signatures"]), dict(gsc)
            if method in rs.FACTOR_SIGNATURE_METHODS:          # finest level: marker scores (pre-declared)
                lat = np.load(os.path.join(md, "latent0.npy"))
                msig, msc = rs.factor_marker_signatures(inp, raw, lat)
                sig_lm.update(msig)
                gsc_lm.update(msc)
            res["levels"] = level_metrics(inp.obs, comparator_levels(method, raw), sig_lm, truth,
                                          gene_scores=gsc_lm, var_names=raw["var_names"])
            if method in rs.FACTOR_SIGNATURE_METHODS:
                fsig = [raw["signatures"][str(k)] for k in range(lat.shape[1])]
                rs.apply_signature_readout(res, lat, adata.obs["GroupC"], markers, fsig)
                enriched_level_signatures(res, inp.obs, lat, [msig[str(k)] for k in range(lat.shape[1])], truth,
                                          factor_scores=[msc[str(k)] for k in range(lat.shape[1])],
                                          var_names=raw["var_names"])
            res["layer_sizes"] = comparator_sizes(method, raw)
            pv, clusters, info = rs.protocol_values(inp.obs, "GroupC", "Batch" if p["multi"] else None, method,
                                                    **({"levels": raw["assignments"]} if method == "nSBM"
                                                       else {"latent": np.load(os.path.join(md, "latent0.npy"))}))
            res["fixed"]["values"].update(pv)
            res["protocol"] = info
            if groups is not None:
                res["batch_groups"] = batch_groups(inp.obs, clusters, groups)
                res["batch_groups_2024"] = batch_groups(inp.obs, raw["assignments"][0], groups)
                res["nesting"] = nesting(inp.obs, comparator_levels(method, raw), groups, res["batch_groups"]["specific"])
            else:
                res["batch_groups"] = res["batch_groups_2024"] = res["nesting"] = None
            return res

        t = time.time()
        try:
            res = cached(method, os.path.getmtime(os.path.join(md, "run.json")), compute)
        except Exception as e:
            print(f"scoring {method} failed: {e!r}", flush=True)
            runtimes.append(dict(method=method, status="scoring failed", error=repr(e)))
            continue
        print(f"scored {method} ({time.time() - t:.0f} s)", flush=True)
        extra = [("Runtime", run["runtime_s"])]
        if "tree_s" in run:
            extra.append(("Runtime (tree step)", run["tree_s"]))
        add(method, res, extra)
        runtimes.append(dict(method=method, status="ok", runtime_s=run["runtime_s"],
                             device="gpu" if run.get("cuda_visible_devices") not in (None, "") else "cpu",
                             torch_device=run.get("torch_device"), levels=run.get("levels")))

    # scDEF (multi-layer fits, and the decomposition with four batches)
    for label in S.scdef_scored(ds):
        fd = os.path.join(d, "scdef", label)
        fit = S.read_json(os.path.join(fd, "fit.json"), {})
        if fit.get("status") != "ok":
            runtimes.append(dict(method=label, status=fit.get("status", "not run")))
            continue

        def compute(label=label):
            m = load_scdef(ds, label, inp)
            res = {}
            for fix in (True, False):            # the hierarchy-accuracy fix applies to scDEF too
                with rs.label_fix(fix):
                    vals, pt = evaluate(m.adata, m, label, ds, markers)
                res["fixed" if fix else "verbatim"] = dict(values=vals, per_type=pt)
            levels = [(ln, m.adata.obs[ln].astype(str).to_numpy()) for ln in m.layer_names]
            sigs = m.get_signatures_dict()
            msig, msc = scd.tl.marker_genes(m)                  # marker scores (pre-declared 2026-11-11)
            res.update(levels=level_metrics(inp.obs, levels, msig, truth, gene_scores=msc,
                                            var_names=list(m.adata.var_names)), layer_sizes=scdef_sizes(m))
            fsig = [sigs[f] for f in m.factor_names[0]]
            rs.apply_signature_readout(res, m.adata.obsm["X_L0"], m.adata.obs["GroupC"], markers, fsig)
            enriched_level_signatures(res, inp.obs, m.adata.obsm["X_L0"], [msig[f] for f in m.factor_names[0]],
                                      truth, factor_scores=[msc[f] for f in m.factor_names[0]],
                                      var_names=list(m.adata.var_names))
            pv, clusters, info = rs.protocol_values(inp.obs, "GroupC", "Batch" if p["multi"] else None, label,
                                                    latent=m.adata.obsm["X_L0"])
            res["fixed"]["values"].update(pv)
            res["protocol"] = info
            if groups is not None:
                res["batch_groups"] = batch_groups(inp.obs, clusters, groups)
                res["batch_groups_2024"] = batch_groups(inp.obs, m.adata.obs["L0"].astype(str).to_numpy(), groups)
                res["nesting"] = nesting(inp.obs, levels, groups, res["batch_groups"]["specific"])
            else:
                res["batch_groups"] = res["batch_groups_2024"] = None
            return res

        t = time.time()
        try:
            res = cached(label, os.path.getmtime(os.path.join(fd, "fit.json")), compute)
        except Exception as e:
            print(f"scoring {label} failed: {e!r}", flush=True)
            runtimes.append(dict(method=label, status="scoring failed", error=repr(e)))
            continue
        print(f"scored {label} ({time.time() - t:.0f} s)", flush=True)
        ls = res["layer_sizes"]
        add(label, res, [("Runtime", 60 * fit["kept_seed_minutes"]),
                         ("Runtime (five seeds)", 60 * fit["five_seed_minutes"]),
                         ("Layer sizes: true levels matched", ls["true_levels_matched"]),
                         ("Layer sizes: layer 0 minus 8", ls["l0_minus_8"])])
        runtimes.append(dict(method=label, status="ok", runtime_s=60 * fit["kept_seed_minutes"],
                             five_seed_s=60 * fit["five_seed_minutes"], chosen_seed=fit["chosen_seed"],
                             device=fit.get("device"), layer_sizes=ls["sizes"]))

    pd.DataFrame(rows).to_csv(os.path.join(d, "scores.csv"), index=False)
    pd.DataFrame(rows_verbatim).to_csv(os.path.join(d, "scores_2024_verbatim.csv"), index=False)
    pd.DataFrame(lrows).to_csv(os.path.join(d, "level_scores.csv"), index=False)
    pd.DataFrame(srows).to_csv(os.path.join(d, "layer_sizes.csv"), index=False)
    if p["multi"]:
        pd.DataFrame(brows).to_csv(os.path.join(d, "batch_groups.csv"), index=False)
    pd.DataFrame(runtimes).to_csv(os.path.join(d, "runtimes.csv"), index=False)
    S.write_json(details, os.path.join(d, "level_details.json"))
    S.write_json(per_type, os.path.join(d, "signature_accuracy_per_type.json"))
    S.write_json(dict(id=ds, design=p, metrics=S.metrics(ds), metric_code="workflow/scripts/_benchmark/evaluate.py "
                      "(unchanged); added metrics in simulation_benchmark/score.py", score_version=SCORE_VERSION,
                      label_naming_fix="on (scores.csv); off in scores_2024_verbatim.csv",
                      hierarchy_accuracy_fix="on (scores.csv, every method); off in scores_2024_verbatim.csv",
                      clustering_protocol=dict(rule="decision-clustering-protocol (2026-11-06): proportions for "
                                                    f"{sorted(rs.PROPORTION_METHODS)} and their Ward baselines; "
                                                    "Cell Type ARI/AMI/homogeneity/completeness by KMeans at k = 8 "
                                                    "(nSBM: level closest to 8); supplementary ARI by Leiden best of "
                                                    f"{rs.PROTOCOL_RESOLUTIONS}; Batch ARI and batch groups by Leiden at "
                                                    "resolution 1.0 (nSBM: finest level); ASW on the same "
                                                    "representation; the 2024 values in scores_2024_verbatim.csv",
                                               kmeans="n_init 10, random_state 0",
                                               neighbors="scanpy defaults, random_state 0",
                                               leiden="flavor igraph, n_iterations 2, random_state 0"),
                      methods={m: S.settings(m, p["rep"]) for m in S.methods(ds)},
                      hc_baselines={v: dict(deps, reference_scdef=S.REFERENCE_SCDEF)
                                    for v, deps in S.hc_variants(ds).items()},
                      scdef={lb: dict(batch_key=bk, rule="best of five seeds by final loss (fit_fixed)")
                             for lb, bk in S.scdef_variants(ds).items()},
                      decomposition=S.DECOMPOSED if p["multi"] else None,
                      batch_groups_rule=dict(min_fraction=S.SPECIFIC_MIN_FRACTION, min_cells=S.SPECIFIC_MIN_CELLS,
                                             min_f1=S.SPECIFIC_MIN_F1) if p["multi"] else None,
                      scored=time.strftime("%Y-%m-%d %H:%M:%S")), os.path.join(d, "settings.json"))
    if rows:
        wide = pd.DataFrame(rows).pivot_table(index="metric", columns="method", values="value", aggfunc="first")
        lw = pd.DataFrame(lrows).pivot_table(index=["metric", "true_level"], columns="method", values="value")
        with pd.option_context("display.width", 220, "display.max_columns", 20):
            print(wide.drop([i for i in wide.index if i.startswith("Learned")], errors="ignore").round(3))
            print(lw.round(3))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("id")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    sys.exit(main(a.id, a.force))
