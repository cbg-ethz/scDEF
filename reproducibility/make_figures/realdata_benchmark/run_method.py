"""Run one comparator on one data set's shared input and save its raw outputs (no scoring here).

    python run_method.py <data> <method> [--force]          (env: scdef-bench2)
    python run_method.py <data> <method> --mark-timeout     (record that the time budget ran out)

Methods: PCA, NMF, scHPF, fscLVM (MuVI), nSBM (schist), scVI, Harmony, Scanorama. The 2024
implementations in ``workflow/scripts/_benchmark/other_methods.py`` are reused where they still work with
current packages; the fixes are made here, not in the old files:

- Harmony: harmonypy 2.x returns ``Z_corr`` as cells x dims and ``sc.external.pp.harmony_integrate`` still
  transposes it, so Harmony (and nSBM's batch-corrected graph) are run through harmonypy directly.
- scVI, Harmony, Scanorama: the embedding is computed once and clustered at every resolution (2024 retrained
  or re-integrated at each of the five resolutions, multiplying the runtime by five).
- nSBM: schist 0.10's ``fit_model`` has no ``n_init``; it runs at its defaults with ``random_seed``.

Writes methods/<method>/{outputs.json, scores.npz, latent0.npy, run.json}. Resumable: skips a method whose
run.json says ``ok`` or ``timeout`` unless ``--force``.
"""
import argparse
import json
import os
import platform
import random
import sys
import time
import traceback
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import scanpy as sc

import common as C

C.ensure_scdef_utils()
om = C.bench_module("other_methods")


# --------------------------------------------------------------------------- inputs


def load_input(data):
    ad = sc.read_h5ad(os.path.join(C.out_dir(data), "input.h5ad"))
    counts = ad.layers["counts"]
    counts = counts.toarray() if hasattr(counts, "toarray") else np.asarray(counts)
    ad.layers["counts"] = counts.astype(np.float64)       # the 2024 inputs held dense counts
    ad.X = ad.layers["counts"].copy()
    return ad


def select_genes(ad, n_top, batch_key):
    """seurat_v3 on counts, batch-aware with a batch key, as the 2024 run_*.py (fallback as run_method.py)."""
    if n_top >= ad.n_vars:
        return ad, "all %d shared genes (n_top_genes %d >= genes)" % (ad.n_vars, n_top)
    try:
        sc.pp.highly_variable_genes(ad, n_top_genes=n_top, subset=True, layer="counts", flavor="seurat_v3",
                                    batch_key=batch_key)
        how = "seurat_v3 top %d" % n_top + (" (batch_key)" if batch_key else "")
    except ValueError as e:
        sc.pp.highly_variable_genes(ad, n_top_genes=n_top, subset=True, layer="counts", flavor="seurat_v3")
        how = "seurat_v3 top %d without batch_key (%s)" % (n_top, e)
    return ad, how


def lognorm(ad):
    ad.X = ad.layers["counts"].copy()
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)


def scaled(ad):
    """The X of the 2024 prepared inputs: log-normalised, depth (and mitochondrial share) regressed out,
    scaled with max 10 (prepare_pbmcs*.py: total_counts + pct_counts_mt; prepare_planaria.py: n_counts)."""
    lognorm(ad)
    if "pct_counts_mt" in ad.obs:
        covs = ["total_counts", "pct_counts_mt"]
    elif "n_counts" in ad.obs:
        covs = ["n_counts"]
    else:
        covs = ["total_counts"]
    sc.pp.regress_out(ad, covs)
    sc.pp.scale(ad, max_value=10)
    return covs


# --------------------------------------------------------------------------- local fixes


def harmony(ad, batch_key, seed):
    import harmonypy
    ho = harmonypy.run_harmony(ad.obsm["X_pca"], ad.obs, batch_key, random_state=seed, verbose=False)
    z = np.asarray(ho.Z_corr)
    return z if z.shape[0] == ad.n_obs else z.T


def patch_scanpy_harmony(seed):
    """run_nsbm calls sc.external.pp.harmony_integrate(ad, batch_key); give it a working one."""
    import scanpy.external as sce

    def harmony_integrate(adata, key, basis="X_pca", adjusted_basis="X_pca_harmony", **kw):
        adata.obsm[adjusted_basis] = harmony(adata, key, kw.pop("random_state", seed))

    sce.pp.harmony_integrate = harmony_integrate


def embedded_leiden(latent, rep):
    """A run_multiple_resolutions method for a fixed embedding: the post-embedding steps of run_scvi,
    run_harmony and run_scanorama (neighbours on the embedding, Leiden, Wilcoxon on X), unchanged."""

    def method(ad, resolution=1.0, batch_key=None, **kwargs):
        ad.obsm[rep] = latent
        if "neighbors" not in ad.uns:
            sc.pp.neighbors(ad, use_rep=rep)
        sc.tl.leiden(ad, resolution=resolution)
        sc.tl.rank_genes_groups(ad, "leiden", method="wilcoxon")
        gene_scores = []
        for leiden in range(np.max(ad.obs["leiden"].unique().astype(int)) + 1):
            gene_scores.append(sc.get.rank_genes_groups_df(ad, str(leiden)).set_index("names")
                               .loc[ad.var_names]["scores"].values)
        gene_scores = np.array(gene_scores)
        signatures = [ad.var_names[np.argsort(gene_scores[k])[::-1]] for k in range(len(gene_scores))]
        return [latent, gene_scores, ad, signatures, ad.obs["leiden"].values.tolist()]

    return method


# --------------------------------------------------------------------------- methods


def run(method, ad, batch_key, info):
    """Returns the 2024 output dict (latents, signatures, assignments, scores, sizes) and the runtime (s),
    timed around the method itself as in the 2024 run_*.py (gene selection and normalisation excluded)."""
    seed = C.SEED
    if method == "PCA":
        lognorm(ad)
        t = time.time()
        outs = om.run_multiple_resolutions(om.run_unintegrated, ad, C.LEIDEN_SWEEP, batch_key=batch_key)
    elif method == "NMF":
        t = time.time()
        outs = om.run_multiple_resolutions(om.run_nmf, ad, C.FACTOR_SWEEP, batch_key=batch_key,
                                           random_state=seed)
    elif method == "scHPF":
        t = time.time()
        outs = om.run_multiple_resolutions(om.run_schpf, ad, C.FACTOR_SWEEP, batch_key=batch_key, seed=seed)
    elif method == "fscLVM":
        import torch
        info["regressed_out"] = scaled(ad)
        info["torch_device"] = "cuda" if torch.cuda.is_available() else "cpu"   # muvi's default device
        t = time.time()
        outs = om.run_multiple_resolutions(om.run_muvi, ad, C.FACTOR_SWEEP, batch_key=batch_key, seed=seed)
    elif method == "nSBM":
        lognorm(ad)
        patch_scanpy_harmony(seed)
        t = time.time()
        outs = om.run_nsbm(ad, batch_key=batch_key, random_seed=seed)
    elif method == "Harmony":
        lognorm(ad)
        t = time.time()
        sc.tl.pca(ad)
        latent = harmony(ad, batch_key, seed)
        outs = om.run_multiple_resolutions(embedded_leiden(latent, "X_pca_harmony"), ad, C.LEIDEN_SWEEP,
                                           batch_key=batch_key)
    elif method == "Scanorama":
        lognorm(ad)
        t = time.time()
        sc.tl.pca(ad)
        order = np.argsort(ad.obs[batch_key].astype(str).to_numpy(), kind="stable")
        contiguous = bool(np.all(order == np.arange(ad.n_obs)))
        info["batches_reordered_for_scanorama"] = not contiguous
        b = ad if contiguous else ad[order].copy()          # scanorama needs each batch contiguous
        np.random.seed(seed)
        random.seed(seed)
        sc.external.pp.scanorama_integrate(b, batch_key)
        latent = np.asarray(b.obsm["X_scanorama"])
        if not contiguous:
            latent = latent[np.argsort(order)]
        outs = om.run_multiple_resolutions(embedded_leiden(latent, "X_scanorama"), ad, C.LEIDEN_SWEEP,
                                           batch_key=batch_key)
    elif method == "scVI":
        import scvi
        import torch
        t = time.time()
        a = ad.copy()
        scvi.model.SCVI.setup_anndata(a, layer="counts", batch_key=batch_key)
        scvi.settings.seed = seed
        model = scvi.model.SCVI(a)
        model.train()
        latent = model.get_latent_representation()
        info["scvi_epochs"] = int(len(model.history["elbo_train"]))
        info["torch_device"] = "cuda" if torch.cuda.is_available() else "cpu"
        lognorm(ad)                                        # Wilcoxon on log-normalised counts
        outs = om.run_multiple_resolutions(embedded_leiden(latent, "X_scVI"), ad, C.LEIDEN_SWEEP,
                                           batch_key=batch_key)
    else:
        raise SystemExit(f"unknown method {method}")
    return outs, time.time() - t


def save(outs, ad, d, info):
    sig = {k: list(map(str, v[:200])) for k, v in outs["signatures"].items()}
    names = list(outs["scores"].keys())
    np.savez_compressed(os.path.join(d, "scores.npz"),
                        **{f"s{i}": np.asarray(outs["scores"][n], dtype=np.float64) for i, n in enumerate(names)})
    lat = outs["latents"][0]
    lat = lat.to_numpy() if hasattr(lat, "to_numpy") else np.asarray(lat)
    np.save(os.path.join(d, "latent0.npy"), lat.astype(np.float64))
    C.write_json(dict(assignments=[list(map(str, a)) for a in outs["assignments"]], signatures=sig,
                      score_names=names, sizes={k: int(v) for k, v in outs["sizes"].items()},
                      var_names=list(map(str, ad.var_names)), obs_names=list(map(str, ad.obs_names))),
                 os.path.join(d, "outputs.json"))
    info["levels"] = [int(len(set(a))) for a in outs["assignments"]]
    info["latent0_shape"] = list(lat.shape)


def versions():
    import importlib.metadata as md
    out = {"python": platform.python_version()}
    for p in ("scanpy", "anndata", "numpy", "scipy", "scikit-learn", "leidenalg", "harmonypy", "scanorama",
              "scvi-tools", "torch", "muvi", "pyro-ppl", "scHPF", "schist", "numba"):
        try:
            out[p] = md.version(p)
        except Exception:
            pass
    try:
        import graph_tool
        out["graph-tool"] = graph_tool.__version__.split()[0]
    except Exception:
        pass
    return out


def main(data, method, force=False, mark_timeout=False):
    data = C.canonical(data)
    spec = C.DATASETS[data]
    d = C.out_dir(data, "methods", method)
    status_path = os.path.join(d, "run.json")
    prev = C.read_json(status_path, {})
    if mark_timeout:
        C.write_json(dict(prev, method=method, status="timeout", budget_hours=C.BUDGET_HOURS,
                          finished=time.strftime("%Y-%m-%d %H:%M:%S")), status_path)
        return 0
    if prev.get("status") in ("ok", "timeout") and not force:
        print(f"skip {data}/{method}: {prev['status']}")
        return 0
    info = dict(method=method, data=data, settings=C.SETTINGS[method], seed=C.SEED, status="running",
                started=time.strftime("%Y-%m-%d %H:%M:%S"), host=platform.node(),
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                cpu_affinity=sorted(os.sched_getaffinity(0)), versions=versions())
    C.write_json(info, status_path)
    try:
        t0 = time.time()
        batch_key = "Batch" if spec["batch"] else None
        ad = load_input(data)
        ad, how = select_genes(ad, C.SETTINGS[method]["n_top_genes"], batch_key)
        info.update(genes=how, n_cells=int(ad.n_obs), n_genes=int(ad.n_vars))
        outs, runtime = run(method, ad, batch_key, info)
        info["preprocess_s"] = round(time.time() - t0 - runtime, 2)
        save(outs, outs.get("adata", ad), d, info)
        info.update(status="ok", runtime_s=round(runtime, 2), finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        C.write_json(info, status_path)
        print(f"done {data}/{method}: {runtime:.1f} s, levels {info['levels']}")
        return 0
    except Exception as e:
        info.update(status="failed", error=repr(e), traceback=traceback.format_exc(),
                    finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        C.write_json(info, status_path)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("data")
    p.add_argument("method")
    p.add_argument("--force", action="store_true")
    p.add_argument("--mark-timeout", action="store_true")
    a = p.parse_args()
    sys.exit(main(a.data, a.method, a.force, a.mark_timeout))
