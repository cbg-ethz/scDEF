"""Where are the layer-0 factors lost? One first fit per variant, inspected before any filtering.

The fixed workflow left 14 layer-0 factors for the 51 planaria cell types (pilot, 2026-11-03) although the model
starts with 100. For each variant: epochs run, layer-0 factors that hold cells, their BRD and ARD, how many pass
the first filter (BRD >= 1, ARD >= 0.001 of the total), and how many cell types get a factor of their own.

    python reproducibility/make_figures/diagnose_factor_count.py planaria [variant ...]
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refit_best_of_seeds as rb  # noqa: E402

import numpy as np
import pandas as pd
import scdef as scd
from sklearn.metrics import adjusted_rand_score

SEED = 42
VARIANTS = {
    "base": dict(hierarchical_init=True),                         # the fixed workflow's first fit
    "noinit": dict(hierarchical_init=False),
    "initall": dict(hierarchical_init=True, max_cells_init=10 ** 6),
    "longer": dict(hierarchical_init=True, tolerance=1e-6, patience=200),
    "base_table": dict(hierarchical_init=True),                   # base again, saving the per-factor table
}
HERE = os.path.dirname(os.path.abspath(__file__))


def inspect(m, a):
    scd.tl.factor_diagnostics(m)
    fo = m.adata.uns["factor_obs"]
    l0 = fo[fo.index.astype(str).str.startswith("L0")].copy()
    ard_frac = l0["ARD"].astype(float) / l0["ARD"].astype(float).sum()
    brd = l0["BRD"].astype(float)
    n_cells = l0["n_cells"].astype(float)
    passes = (brd >= 1.0) & (ard_frac >= 0.001)
    obs = m.adata.obs
    types = a.obs.loc[obs.index, "clusters"].astype(str)
    l0lab = obs["L0"].astype(str)
    ct = pd.crosstab(types, l0lab)
    tp = ct.to_numpy()
    prec = tp / np.maximum(tp.sum(0, keepdims=True), 1)
    rec = tp / np.maximum(tp.sum(1, keepdims=True), 1)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0)
    own = int((f1.max(1) >= 0.5).sum())
    best = pd.DataFrame({"factor": ct.columns, "best_type": ct.index[f1.argmax(0)], "best_f1": f1.max(0).round(3),
                         "n_cells_argmax": tp.sum(0)}).set_index("factor")
    table = pd.DataFrame({"brd": brd.round(3), "ard_frac": ard_frac.round(5), "n_cells": n_cells, "passes": passes})
    names = [m.factor_names[0][i] for i in range(len(m.factor_names[0]))]
    table.index = names if len(names) == len(table) else table.index
    table = table.join(best, how="left").sort_values("ard_frac", ascending=False)
    return table, dict(
        epochs=len(np.asarray(m.elbos[-1]).ravel()) if m.elbos else None,
        final_loss=rb.final_loss(m),
        l0_total=int(len(l0)),
        l0_with_10_cells=int((n_cells >= 10).sum()),
        brd_quantiles=[round(float(q), 3) for q in np.nanquantile(brd, [0.1, 0.25, 0.5, 0.75, 0.9])],
        brd_ge_1=int((brd >= 1).sum()),
        ard_ge_0001=int((ard_frac >= 0.001).sum()),
        pass_first_filter=int(passes.sum()),
        pass_first_filter_with_10_cells=int((passes & (n_cells >= 10)).sum()),
        types_with_own_factor=own,
        n_types=int(ct.shape[0]),
        ari_l0=float(adjusted_rand_score(types, l0lab)),
        upper=[len(f) for f in m.factor_lists],
        ard_frac_quantiles=[float(q) for q in np.nanquantile(ard_frac, [0.1, 0.25, 0.5, 0.75, 0.9, 0.99])],
    )


if __name__ == "__main__":
    data = sys.argv[1]
    variants = sys.argv[2:] or list(VARIANTS)
    out = os.path.join(HERE, "out", f"diagnose_factor_count_{data}.json")
    log = json.load(open(out)) if os.path.exists(out) else {}
    a = rb.tutorial_adata(data)
    for v in variants:
        if v in log:
            continue
        t0 = time.time()
        m = scd.scDEF(a.copy(), counts_layer="counts", seed=SEED)
        m.fit(**VARIANTS[v])
        table, d = inspect(m, a)
        table.to_csv(os.path.join(HERE, "out", f"diagnose_factor_count_{data}_{v}_factors.csv"))
        d["minutes"] = round((time.time() - t0) / 60, 1)
        log[v] = d
        json.dump(log, open(out, "w"), indent=1)
        print(v, json.dumps(d), flush=True)
    print("done", flush=True)
