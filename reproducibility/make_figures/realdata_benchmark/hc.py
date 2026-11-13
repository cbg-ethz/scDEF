"""Hierarchical-clustering baselines on one real data set (pre-declared 2026-11-05 in
scDEF-markdown/experiments/exp-realdata-benchmark-local.md; computed 2026-11-06).

    python hc.py <data> [<variant> ...] [--force]          (env: scdef, CPU)

As simulation_benchmark/hc.py, from the saved outputs (no refits):

- ``<M>_Ward`` for M in PCA, NMF, scHPF, fscLVM (and scVI, Harmony, Scanorama on the two-batch data sets): Ward linkage
  on the cells' coordinates in M's finest latent space (``methods/<M>/latent0.npy``), the tree cut at the layer sizes
  of the kept scDEF model (``scDEF``: with the batch key on the two-batch data sets) at every level.
- ``Leiden_Ward``: the Leiden clusters at resolution 1.0 of the PCA run (its first level) and its PCA coordinates;
  Ward on the cells placed at their cluster's mean PCA coordinates; level 0 is the Leiden clusters, the levels above
  cut at the upper layer sizes of the kept scDEF model, capped at the number of clusters.
- The one-layer scDEF baseline is run on the simulations only, as in 2024 (design note).

Signatures at every level: Wilcoxon (scanpy rank_genes_groups) on log-normalised counts (normalize_total 1e4, log1p)
of every gene of the shared input. Writes methods/<variant>/{outputs.json, scores.npz, latent0.npy, run.json} in the
comparators' format (run_method.save); score.py scores them with every other method.
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

import common as C


def variants(data):
    spec = C.DATASETS[C.canonical(data)]
    emb = ["PCA", "NMF", "scHPF", "fscLVM"] + (["scVI", "Harmony", "Scanorama"] if spec["batch"] else [])
    return [f"{m}_Ward" for m in emb] + ["Leiden_Ward"]


def ref_sizes(data):
    res = C.read_json(os.path.join(C.out_dir(data, "scores"), "scDEF.json"), {})
    if not res.get("layer_sizes"):
        raise RuntimeError("scDEF is not scored yet (scores/scDEF.json has no layer_sizes)")
    return [int(x) for x in res["layer_sizes"]], res.get("model")


def lognorm_input(data):
    a = sc.read_h5ad(os.path.join(C.out_dir(data), "input.h5ad"))
    c = a.layers["counts"]
    a.X = (c.toarray() if hasattr(c, "toarray") else np.asarray(c)).astype(np.float64)
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    return a


def wilcoxon(ad, labels):
    """Wilcoxon scores of every gene per cluster against the rest; a cluster of one cell cannot be tested and gets
    zero scores (Ward on f-scLVM, 3k, has one)."""
    lab = np.asarray(labels).astype(int).astype(str)
    ad.obs["_lvl"] = pd.Categorical(lab)
    counts = pd.Series(lab).value_counts()
    testable = sorted(counts.index[counts >= 2], key=int)
    sc.tl.rank_genes_groups(ad, "_lvl", groups=testable, reference="rest", method="wilcoxon")
    scores = []
    for k in range(int(np.max(np.asarray(labels).astype(int))) + 1):
        try:
            v = sc.get.rank_genes_groups_df(ad, str(k)).set_index("names").loc[ad.var_names]["scores"].values
        except KeyError:
            v = np.zeros(ad.n_vars)
        scores.append(v)
    scores = np.array(scores)
    return scores, [ad.var_names[np.argsort(scores[k])[::-1]] for k in range(len(scores))]


def mean_per_cell(latent, groups):
    groups = np.asarray(groups).astype(str)
    out = np.zeros_like(latent, dtype=np.float64)
    for g in np.unique(groups):
        idx = groups == g
        out[idx] = latent[idx].mean(0)
    return out


def assemble(levels, level_scores, level_sigs, latent):
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
    return tree, [int(len(np.unique(tree[:, j]))) for j in range(tree.shape[1])]


def run_variant(data, variant, info):
    sizes, model = ref_sizes(data)
    info.update(reference_scdef=model, reference_layer_sizes=sizes)
    ad = lognorm_input(data)
    m = "PCA" if variant == "Leiden_Ward" else variant[:-5]
    md = os.path.join(C.out_dir(data), "methods", m)
    run = C.read_json(os.path.join(md, "run.json"), {})
    if run.get("status") != "ok":
        raise RuntimeError(f"{m} is not finished")
    latent = np.load(os.path.join(md, "latent0.npy")).astype(np.float64)
    upstream = float(run["runtime_s"])
    t = time.time()
    if variant == "Leiden_Ward":
        raw = C.read_json(os.path.join(md, "outputs.json"))
        if raw["obs_names"] != list(ad.obs_names):
            raise RuntimeError("PCA cells differ from the input")
        leiden = np.asarray(raw["assignments"][0]).astype(str)
        k = int(len(np.unique(leiden)))
        req = [k] + [min(int(s), k) for s in sizes[1:]]
        Z = sch.ward(mean_per_cell(latent, leiden))
        info.update(upstream="methods/PCA: Leiden resolution 1.0 and its PCA coordinates", leiden_clusters=k)
    else:
        req = list(sizes)
        Z = sch.ward(latent)
        info.update(upstream=f"methods/{m} (runtime of its whole run)")
    tree, got = cut(Z, req)
    levels = [tree[:, j] for j in range(tree.shape[1])]
    lv_scores, lv_sigs = zip(*[wilcoxon(ad, lab) for lab in levels])
    info.update(requested_sizes=req, cut_sizes=got, levels=got, cut_mismatch=got != req)
    outs = assemble(levels, lv_scores, lv_sigs, latent)
    return outs, ad, upstream, time.time() - t


def main(data, variant, force=False):
    data = C.canonical(data)
    d = os.path.join(C.out_dir(data), "methods", variant)
    os.makedirs(d, exist_ok=True)
    status_path = os.path.join(d, "run.json")
    if C.read_json(status_path, {}).get("status") == "ok" and not force:
        print(f"skip {data}/{variant}: ok")
        return 0
    info = dict(method=variant, data=data, status="running", started=time.strftime("%Y-%m-%d %H:%M:%S"),
                host=platform.node(),
                settings=dict(linkage="scipy.cluster.hierarchy.ward", cut="cut_tree at the kept scDEF layer sizes",
                              signatures="Wilcoxon (scanpy rank_genes_groups) on log-normalised counts"),
                versions=dict(scanpy=sc.__version__, numpy=np.__version__))
    C.write_json(info, status_path)
    try:
        outs, ad, upstream, tree_s = run_variant(data, variant, info)
        import run_method as rd
        rd.save(outs, ad, d, info)
        info.update(status="ok", tree_s=round(tree_s, 2), upstream_s=round(upstream, 2),
                    runtime_s=round(upstream + tree_s, 2), finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        C.write_json(info, status_path)
        print(f"done {data}/{variant}: tree {tree_s:.1f} s, levels {info['levels']}", flush=True)
        return 0
    except Exception as e:
        info.update(status="failed", error=repr(e), traceback=traceback.format_exc(),
                    finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        C.write_json(info, status_path)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("variants", nargs="*")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    rc = 0
    for v in (a.variants or variants(a.data)):
        rc |= main(a.data, v, a.force)
    sys.exit(rc)
