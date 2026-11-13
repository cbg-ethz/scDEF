"""Run every method and both metrics on one dataset of the confounded-design benchmark.

    ~/miniconda3/envs/scdef/bin/python -m confounded.run D1_loc3_n50_T2_s0 [--out DIR]

Writes DIR/<dataset>/results.json and the embeddings. scVI and Scanorama run first, in the
`scdef-bench` environment, before this process touches the GPU through JAX.
"""
import argparse
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

from . import data, methods, metrics
from .data import BATCH, CTYPE

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_PY = os.path.expanduser("~/miniconda3/envs/scdef-bench/bin/python")
DEFAULT_OUT = os.path.expanduser("~/projects/scDEF/confounded_results_v2")   # Amendment 1


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def main(spec, out_root, workflow="relaxed", reuse=None):
    """``reuse``: a results root whose saved embeddings of the other methods are used instead of
    re-running them (Amendment 2 re-runs scDEF only); their metrics are recomputed from them."""
    t_start = time.time()
    d = data.build(spec)
    seed = d.meta["seed"]
    wd = os.path.join(out_root, d.name)
    os.makedirs(wd, exist_ok=True)

    adata = methods.preprocess(d.adata)
    ctype = adata.obs[CTYPE].astype(str).to_numpy()
    batch = adata.obs[BATCH].astype(str).to_numpy()

    d.adata.write_h5ad(os.path.join(wd, "input.h5ad"))
    if reuse:
        rd = os.path.join(reuse, d.name)
        bench = np.load(os.path.join(rd, "emb_bench.npz"))
        old = np.load(os.path.join(rd, "embeddings.npz"))
        embs = {k: old[k] for k in ("Uncorrected", "Harmony", "scVI", "Scanorama")}
    else:
        # scVI / Scanorama in their own environment, first.
        subprocess.run([BENCH_PY, "-m", "confounded.bench_methods", wd, str(seed)], cwd=HERE,
                       check=True)
        bench = np.load(os.path.join(wd, "emb_bench.npz"))
        embs = {"Uncorrected": adata.obsm["X_pca"]}
        embs["Harmony"] = methods.run_harmony(adata)
        embs["scVI"], embs["Scanorama"] = bench["scVI"], bench["Scanorama"]
    hde, hde_rule = methods.harmony_de(adata, embs["Harmony"])
    scdef_out = methods.run_scdef(adata, seed, workflow=workflow)
    for k, v in scdef_out.items():
        embs[k] = v["emb"]
    np.savez_compressed(os.path.join(wd, "embeddings.npz"), **embs)

    # Metric 1
    types = metrics.eligible_types(ctype, batch)
    weights = {t: int((ctype == t).sum()) for t in np.unique(ctype)}
    if d.meta["design"] == "D3":
        t_c = {t: 0.5 for t in types}
    elif d.meta["design"] == "D2b":
        # within-donor STIM vs CTRL separability, averaged over the donors outside the pair
        per = []
        for cs in d.clean_sets:
            cp = methods.preprocess(cs)
            cty = cp.obs[CTYPE].astype(str).to_numpy()
            cb = cp.obs[BATCH].astype(str).to_numpy()
            ok = [t for t in types if t in metrics.eligible_types(cty, cb)]
            per.append(metrics.per_type_auc(cp.obsm["X_pca"], cty, cb, ok))
        t_c = {t: float(np.mean([p[t] for p in per if t in p]))
               for t in types if any(t in p for p in per)}
    else:
        clean = methods.preprocess(d.clean)
        t_c = metrics.per_type_auc(clean.obsm["X_pca"], ctype, batch, types)
    m1 = {}
    for name, e in embs.items():
        r_c = metrics.per_type_auc(np.asarray(e, dtype=float), ctype, batch, types)
        m1[name] = metrics.calibration(r_c, t_c, weights)

    # Metric 2
    var = adata.var_names
    attr_inputs = {"Harmony_DE": hde, "Harmony_DE_rule": hde_rule, **scdef_out}
    m2 = {name: metrics.attribution(v["bio"], v["tech"], var, d.bio_genes, d.tech_genes,
                                    weights)
          for name, v in attr_inputs.items()}

    res = dict(
        dataset=d.name, meta=d.meta, workflow=workflow, reused=reuse, n_cells=int(adata.n_obs), n_genes=int(adata.n_vars),
        eligible_types=types, calibration=m1, attribution=m2,
        info=dict(harmony_de=hde["info"],
                  **{k: v["info"] for k, v in scdef_out.items()},
                  bench_seconds={k: float(bench[k]) for k in bench.files
                                 if k.startswith("seconds_")},
                  total_seconds=round(time.time() - t_start, 1)),
    )
    with open(os.path.join(wd, "results.json"), "w") as f:
        json.dump(_jsonable(res), f, indent=1)
    summary = {n: round(v["mae"], 3) for n, v in m1.items()}
    print(d.name, "calibration MAE", summary)
    print(d.name, "attribution", {n: {k: round(x, 3) for k, x in v.items()
                                      if isinstance(x, float)} for n, v in m2.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--workflow", choices=["relaxed", "learn"], default="relaxed")
    ap.add_argument("--reuse", default=None)
    args = ap.parse_args()
    main(args.spec, args.out, args.workflow, args.reuse)
