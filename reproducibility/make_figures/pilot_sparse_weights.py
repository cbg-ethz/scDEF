"""Pilot: sparser layer-0 cell weights through a weaker hierarchy prior (pre-declared 2026-11-12 in
scDEF-markdown/experiments/exp-sparse-cell-weights.md, before any fit).

    python pilot_sparse_weights.py fit     (GPU; resumable)  -> scdef-<data>-pilot-hw<value> beside the paper's models
    python pilot_sparse_weights.py score   (CPU)             -> out/pilot_sparse_weights.csv

The paper's workflow (refit_best_of_seeds.fit_fixed: fit, FIXED_FIRST, refit, FIXED_SECOND) from the seed that the
best-of-five kept (3k PBMCs: 2; interferon with batch annotations: 3), with hierarchy_weight 1 (control), 0.3 and 0.1.
hierarchy_weight multiplies alpha, the shape of the Gamma prior on the cell weights (kappa in the Methods).
"""
import json
import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

RUNS = [("pbmcs3k", 2, None), ("ifn", 3, "stim")]
WEIGHTS = [1.0, 0.3, 0.1]
LOG = os.path.join(HERE, "out", "pilot_sparse_weights_fits.json")


def name(data, hw):
    return f"scdef-{data}-pilot-hw{hw:g}"


def fit():
    import scdef as scd
    import refit_best_of_seeds as rb
    import fig4_batch as f4
    log = json.load(open(LOG)) if os.path.exists(LOG) else {}
    for data, seed, batch_key in RUNS:
        a = None
        for hw in WEIGHTS:
            key = name(data, hw)
            if key in log and os.path.isdir(os.path.join(f4.NB, key)):
                print("done already", key, flush=True)
                continue
            if a is None:
                a = rb.tutorial_adata(data)
            t0 = time.time()
            m = scd.scDEF(a.copy(), counts_layer="counts", batch_key=batch_key, seed=seed, hierarchy_weight=hw)
            m.fit(hierarchical_init=True)
            scd.tl.filter(m, batch_key=batch_key, **rb.FIXED_FIRST)
            m.fit(hierarchical_init=True)
            scd.tl.filter(m, batch_key=batch_key, **rb.FIXED_SECOND)
            rb.save(m, key)
            log[key] = dict(data=data, seed=seed, batch_key=batch_key, hierarchy_weight=hw, alpha=float(m.alpha),
                            final_loss=rb.final_loss(m), layer_sizes=[len(f) for f in m.factor_lists],
                            minutes=round((time.time() - t0) / 60, 1))
            os.makedirs(os.path.dirname(LOG), exist_ok=True)
            json.dump(log, open(LOG, "w"), indent=1)
            print(key, json.dumps(log[key]), flush=True)


def score():
    """Benchmark protocol (realdata_benchmark/score.py) on each pilot model; hierarchy accuracy as in score.main."""
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import pandas as pd
    sys.path.insert(0, os.path.join(HERE, "realdata_benchmark"))
    sys.argv = sys.argv[:1]
    import score as S
    import common as C
    rows = []
    for data, seed, batch_key in RUNS:
        bench = C.canonical(data)
        spec = C.DATASETS[bench]
        inp, mk = S.load_input(bench)
        markers = mk["markers"]
        metrics = [x for x in spec["metrics"] if markers or x != "Signature accuracy"]
        for hw in WEIGHTS:
            m = S.load_scdef(bench, name(data, hw), inp)
            with S.label_fix(True):
                ev_vals, _ = S.evaluate(m.adata, m, "scDEF", bench, markers, metrics)
            hier = {k: float(v) for k, v in ev_vals.items() if "ierarch" in k}
            w0 = np.asarray(m.pmeans[f"{m.layer_names[0]}z"])[:, m.factor_lists[0]]
            p = w0 / w0.sum(1, keepdims=True)
            vals, _, _ = S.protocol_values(inp.obs, spec["hierarchy_obs"][0], "Batch" if spec["batch"] else None,
                                           "scDEF", latent=w0)
            from sklearn.metrics import adjusted_rand_score
            y = inp.obs[spec["hierarchy_obs"][0]].astype(str).to_numpy()
            rows.append(dict(data=data, hierarchy_weight=hw, alpha=float(m.alpha),
                             layer_sizes=str([len(f) for f in m.factor_lists]),
                             eff_factors_l0=float(np.median(1 / (p ** 2).sum(1))), top_share_l0=float(np.median(p.max(1))),
                             native_ari_l0=float(adjusted_rand_score(y, m.adata.obs["L0"].astype(str))),
                             **vals, **hier))
            print(rows[-1], flush=True)
    out = os.path.join(HERE, "out", "pilot_sparse_weights.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    {"fit": fit, "score": score}[sys.argv[1]]()
