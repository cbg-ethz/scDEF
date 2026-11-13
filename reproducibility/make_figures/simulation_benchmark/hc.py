"""Hierarchical-clustering baselines on one simulated data set (added 2026-11-05, before any result).

    python hc.py <id> <variant> [--force]          (env: scdef, JAX on CPU)

Variants (design note, "Hierarchical-clustering baselines"); no refits, all from saved outputs:

- ``<M>_Ward`` for M in PCA, NMF, scHPF, f-scLVM (and scVI, Harmony, Scanorama with four batches): Ward linkage on the
  cells' coordinates in M's finest latent space (``methods/<M>/latent0.npy``), the tree cut at the layer sizes of the
  kept scDEF model of the same data set at every level.
- ``Leiden_Ward`` (the 2024 ``run_unintegrated_hclust``, cut changed): the Leiden clusters at resolution 1.0 on the PCA
  neighbour graph and the PCA coordinates, both from the PCA run (its first level and latent0 are exactly that);
  Ward on the cells placed at their cluster's mean PCA coordinates (as 2024: a size-weighted Ward on the cluster
  means); level 0 is the Leiden clusters, the levels above are cut at the upper layer sizes of the kept scDEF model,
  capped at the number of clusters.
- ``scDEF_1L_Ward`` (the 2024 ``run_scdef_hclust``, ported): the kept one-layer scDEF (fit_scdef.py scDEF_1L); cells
  assigned to their largest layer-0 weight (obs['L0']); Ward on the cells placed at their factor's mean layer-0 weights
  (X_L0); level 0 is the factors, the levels above cut at the upper layer sizes of the kept multi-layer scDEF, capped
  at the number of factors; level-0 signatures are the factors' gene rankings, the levels above Wilcoxon (as 2024).
  2024 listed the level-0 signatures in the order of ``obs['factor'].unique()`` while the level-0 clusters were
  numbered by ``cut_tree``; here each level-0 cluster gets the signature of its own factor.

The kept scDEF model whose layer sizes cut the trees is ``scdef/scDEF``: with batch annotations on four batches (for
every variant, including comparisons with scDEF_un), plain on one batch. Wilcoxon (scanpy rank_genes_groups) on
log-normalised counts (normalize_total 1e4, log1p) of every gene, as the comparators. ``cut_tree`` is used as 2024;
each level's cluster count is checked against the requested size.

Writes methods/<variant>/{outputs.json, scores.npz, latent0.npy, run.json} in the comparators' format (via the
real-data run_method.save), so score.py scores them like any other method. Runtime: ``runtime_s`` = the upstream
step (the method's whole run, for the Ward variants and Leiden_Ward the 2024 resolution sweep; for scDEF_1L_Ward the
kept seed's fit) + the tree step; ``tree_s`` = the tree step alone.
"""
import argparse
import os
import platform
import sys
import time
import traceback
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.cluster import hierarchy as sch

import simcommon as S


def ref_sizes(ds):
    fit = S.read_json(os.path.join(S.ds_dir(ds, "scdef", S.REFERENCE_SCDEF), "fit.json"), {})
    if fit.get("status") != "ok":
        raise RuntimeError("the reference scDEF fit is not finished")
    rec = [r for r in fit["seeds"] if r["seed"] == fit["chosen_seed"]][0]
    return [int(x) for x in rec["layer_sizes"]], fit["chosen_seed"]


def lognorm_input(ds):
    a = sc.read_h5ad(os.path.join(S.ds_dir(ds), "input.h5ad"))
    c = a.layers["counts"]
    a.X = (c.toarray() if hasattr(c, "toarray") else np.asarray(c)).astype(np.float64)
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    return a


def wilcoxon(ad, labels):
    """2024 per-level signatures: Wilcoxon scores of every gene for clusters 0..max, ranked."""
    ad.obs["_lvl"] = pd.Categorical(np.asarray(labels).astype(int).astype(str))
    sc.tl.rank_genes_groups(ad, "_lvl", method="wilcoxon")
    scores = []
    for k in range(int(np.max(np.asarray(labels).astype(int))) + 1):
        try:
            v = sc.get.rank_genes_groups_df(ad, str(k)).set_index("names").loc[ad.var_names]["scores"].values
        except KeyError:                      # a number without cells (not expected from cut_tree)
            v = np.zeros(ad.n_vars)
        scores.append(v)
    scores = np.array(scores)
    sigs = [ad.var_names[np.argsort(scores[k])[::-1]] for k in range(len(scores))]
    return scores, sigs


def mean_per_cell(latent, groups):
    """Each cell placed at its group's mean coordinates (2024 run_*_hclust 'avgs')."""
    groups = np.asarray(groups).astype(str)
    out = np.zeros_like(latent, dtype=np.float64)
    for g in np.unique(groups):
        idx = groups == g
        out[idx] = latent[idx].mean(0)
    return out


def assemble(levels, level_scores, level_sigs, latent):
    """The 2024 output dict (run_multiple_resolutions' format) from per-level labels, scores and signatures."""
    assignments, signatures, scores, sizes = [], {}, {}, {}
    for i, (lab, sc_, sg) in enumerate(zip(levels, level_scores, level_sigs)):
        prefix = "h" * i
        a = [f"{prefix}{int(x)}" for x in lab]
        assignments.append(a)
        uq, cts = np.unique(a, return_counts=True)
        cnt = dict(zip(uq, cts))
        for k in range(len(sg)):
            name = f"{prefix}{k}"
            signatures[name] = list(map(str, sg[k]))
            scores[name] = np.asarray(sc_[k], dtype=np.float64).tolist()
            sizes[name] = int(cnt.get(name, 0))
    return dict(latents=[latent], signatures=signatures, assignments=assignments, scores=scores, sizes=sizes)


def cut(Z, sizes):
    tree = sch.cut_tree(Z, n_clusters=sizes)
    got = [int(len(np.unique(tree[:, j]))) for j in range(tree.shape[1])]
    return tree, got


def run_variant(ds, variant, info):
    deps = S.hc_variants(ds)[variant]
    sizes, ref_seed = ref_sizes(ds)
    info.update(reference_scdef=f"scdef/{S.REFERENCE_SCDEF} (seed {ref_seed})", reference_layer_sizes=sizes)
    ad = lognorm_input(ds)
    if variant.endswith("_Ward") and variant not in ("Leiden_Ward", "scDEF_1L_Ward"):
        m = deps["methods"][0]
        md = S.ds_dir(ds, "methods", m)
        run = S.read_json(os.path.join(md, "run.json"), {})
        if run.get("status") != "ok":
            raise RuntimeError(f"{m} is not finished")
        latent = np.load(os.path.join(md, "latent0.npy"))
        upstream = float(run["runtime_s"])
        t = time.time()
        Z = sch.ward(latent)
        tree, got = cut(Z, sizes)
        levels = [tree[:, j] for j in range(tree.shape[1])]
        lv_scores, lv_sigs = zip(*[wilcoxon(ad, lab) for lab in levels])
        info.update(upstream=f"methods/{m} (runtime of its whole run)", requested_sizes=sizes, cut_sizes=got)
    elif variant == "Leiden_Ward":
        md = S.ds_dir(ds, "methods", "PCA")
        run = S.read_json(os.path.join(md, "run.json"), {})
        if run.get("status") != "ok":
            raise RuntimeError("PCA is not finished")
        raw = S.read_json(os.path.join(md, "outputs.json"))
        if raw["obs_names"] != list(ad.obs_names):
            raise RuntimeError("PCA cells differ from the input")
        leiden = np.asarray(raw["assignments"][0]).astype(str)          # resolution 1.0, level prefix ""
        latent = np.load(os.path.join(md, "latent0.npy"))               # X_pca of that run
        upstream = float(run["runtime_s"])
        t = time.time()
        k = int(len(np.unique(leiden)))
        req = [k] + [min(int(s), k) for s in sizes[1:]]
        Z = sch.ward(mean_per_cell(latent, leiden))
        tree, got = cut(Z, req)
        levels = [tree[:, j] for j in range(tree.shape[1])]
        lv_scores, lv_sigs = zip(*[wilcoxon(ad, lab) for lab in levels])
        info.update(upstream="methods/PCA: Leiden resolution 1.0 and its X_pca (runtime of the whole PCA run)",
                    leiden_clusters=k, requested_sizes=req, cut_sizes=got)
    elif variant == "scDEF_1L_Ward":
        import scdef as scd
        fd = S.ds_dir(ds, "scdef", S.ONE_LAYER)
        fit = S.read_json(os.path.join(fd, "fit.json"), {})
        if fit.get("status") != "ok":
            raise RuntimeError("the one-layer scDEF fit is not finished")
        a = S.scdef_adata(ds)
        m = scd.scDEF.load(os.path.join(fd, "model"), adata=a)
        if "L0" not in m.adata.obs:
            m.annotate_adata()
        # S.ONE_LAYER_SIZES is [100, 1]: one layer of factors under a single root (the fixed workflow fails on [100]).
        if m.n_layers != len(S.ONE_LAYER_SIZES) or len(m.factor_names[-1]) != 1:
            raise RuntimeError(f"the one-layer model has layers of sizes {[len(f) for f in m.factor_names]}")
        upstream = 60 * float(fit["kept_seed_minutes"])
        info["five_seed_upstream_s"] = 60 * float(fit["five_seed_minutes"])
        t = time.time()
        latent = np.asarray(m.adata.obsm["X_L0"], dtype=np.float64)
        factor = m.adata.obs["L0"].astype(str).to_numpy()
        k = int(len(np.unique(factor)))
        req = [k] + [min(int(s), k) for s in sizes[1:]]
        Z = sch.ward(mean_per_cell(latent, factor))
        tree, got = cut(Z, req)
        levels = [tree[:, j] for j in range(tree.shape[1])]
        sigs, fscores = m.get_signatures_dict(scores=True)
        var = np.asarray(m.adata.var_names)
        l0_sigs, l0_scores, l0_factor = [], [], []
        for c in range(int(levels[0].max()) + 1):
            fs = np.unique(factor[levels[0] == c])
            if len(fs) != 1:
                raise RuntimeError(f"level-0 cluster {c} holds factors {fs}")
            l0_factor.append(str(fs[0]))
            s_ = np.asarray(fscores[fs[0]], dtype=np.float64)
            if len(s_) != len(var):
                raise RuntimeError("factor scores are not one per gene")
            l0_scores.append(s_)
            l0_sigs.append(np.asarray(sigs[fs[0]]))
        if list(ad.var_names) != list(var):
            raise RuntimeError("genes differ between the model and the input")
        up = [wilcoxon(ad, lab) for lab in levels[1:]]
        lv_scores = [np.array(l0_scores)] + [u[0] for u in up]
        lv_sigs = [l0_sigs] + [u[1] for u in up]
        info.update(upstream=f"scdef/{S.ONE_LAYER} (kept seed {fit['chosen_seed']}: fit minutes)",
                    one_layer_factors=[len(f) for f in m.factor_names], factors_with_cells=k,
                    level0_factor_of_cluster=l0_factor, requested_sizes=req, cut_sizes=got)
    else:
        raise SystemExit(f"unknown variant {variant}")
    outs = assemble(levels, lv_scores, lv_sigs, latent)
    tree_s = time.time() - t
    if info["cut_sizes"] != info["requested_sizes"]:
        info["cut_mismatch"] = True
    return outs, ad, upstream, tree_s


def main(ds, variant, force=False):
    if variant not in S.hc_variants(ds):
        raise SystemExit(f"{variant} is not run on {ds}")
    d = S.ds_dir(ds, "methods", variant)
    status_path = os.path.join(d, "run.json")
    prev = S.read_json(status_path, {})
    if prev.get("status") == "ok" and not force:
        print(f"skip {ds}/{variant}: ok")
        return 0
    info = dict(method=variant, data=ds, status="running", started=time.strftime("%Y-%m-%d %H:%M:%S"),
                host=platform.node(), cpu_affinity=sorted(os.sched_getaffinity(0)),
                settings=dict(linkage="scipy.cluster.hierarchy.ward", cut="cut_tree at the kept scDEF layer sizes",
                              signatures="Wilcoxon (scanpy rank_genes_groups) on log-normalised counts"
                              + ("; level 0: the factors' gene rankings" if variant == "scDEF_1L_Ward" else "")),
                versions=dict(scanpy=sc.__version__, numpy=np.__version__))
    S.write_json(info, status_path)
    try:
        outs, ad, upstream, tree_s = run_variant(ds, variant, info)
        rd = S.rd_run_method()
        rd.save(outs, ad, d, info)
        info.update(status="ok", tree_s=round(tree_s, 2), upstream_s=round(upstream, 2),
                    runtime_s=round(upstream + tree_s, 2), finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        S.write_json(info, status_path)
        print(f"done {ds}/{variant}: tree {tree_s:.1f} s, levels {info['levels']}", flush=True)
        return 0
    except Exception as e:
        info.update(status="failed", error=repr(e), traceback=traceback.format_exc(),
                    finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        S.write_json(info, status_path)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("id")
    ap.add_argument("variant")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    sys.exit(main(a.id, a.variant, a.force))
