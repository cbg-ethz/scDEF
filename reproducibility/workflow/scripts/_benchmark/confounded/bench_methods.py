"""scVI and Scanorama, run in the separate `scdef-bench` environment.

Reads <workdir>/input.h5ad (raw counts in X, obs['batch']) written by run.py and writes
<workdir>/emb_bench.npz. Kept out of the `scdef` environment so its JAX stack is not disturbed.

    ~/miniconda3/envs/scdef-bench/bin/python -m confounded.bench_methods <workdir> <seed>
"""
import os
import sys
import time

import numpy as np
import scanpy as sc

BATCH = "batch"


def main(workdir, seed):
    import scanorama
    import scvi

    adata = sc.read_h5ad(os.path.join(workdir, "input.h5ad"))
    adata.layers["counts"] = adata.X.copy()
    out, secs = {}, {}

    t0 = time.time()
    scvi.settings.seed = seed
    ad = adata.copy()
    scvi.model.SCVI.setup_anndata(ad, layer="counts", batch_key=BATCH)
    model = scvi.model.SCVI(ad)
    model.train()
    out["scVI"] = model.get_latent_representation()
    secs["scVI"] = round(time.time() - t0, 1)

    t0 = time.time()
    ad = adata.copy()
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    order = np.argsort(ad.obs[BATCH].astype(str).to_numpy(), kind="stable")
    parts = [ad[ad.obs[BATCH] == b].copy() for b in sorted(ad.obs[BATCH].astype(str).unique())]
    scanorama.integrate_scanpy(parts, dimred=100, seed=seed)
    emb = np.vstack([p.obsm["X_scanorama"] for p in parts])
    back = np.empty_like(emb)
    back[order] = emb               # parts are concatenated in sorted-batch order
    out["Scanorama"] = back
    secs["Scanorama"] = round(time.time() - t0, 1)

    np.savez(os.path.join(workdir, "emb_bench.npz"), **out,
             **{f"seconds_{k}": v for k, v in secs.items()})


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
