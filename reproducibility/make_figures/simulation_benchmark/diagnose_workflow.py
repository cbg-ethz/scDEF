"""At which step of the fixed workflow does scDEF lose the 8-group split on simulations with similar groups? (2026-11-06)

    python diagnose_workflow.py <id> <init>       init: hier (hierarchical_init=True, the default) | prior (False)

One seed (42), the fixed workflow of refit_best_of_seeds.fit_fixed, read out after each step: the first fit, the
FIXED_FIRST filter, the refit, the FIXED_SECOND filter. At each step: layer sizes; ARI of every layer against the 8,
4 and 2 true groups; ARI of KMeans(8) on the row-normalised layer-0 weights; and the sibling merges of layer 0: the
true sibling pairs (two 8-level groups with the same 4-level parent) whose cells mostly share one layer-0 factor.
Writes ../out/simulation_benchmark/diagnosis/<id>/<init>.json; the benchmark outputs are not touched.
"""
import json
import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.metrics import adjusted_rand_score as ari  # noqa: E402

import simcommon as S  # noqa: E402

sys.path.insert(0, S.MAKE_FIGURES)
import refit_best_of_seeds as rb  # noqa: E402
import scdef as scd  # noqa: E402


def readout(m, step, t0):
    m.annotate_adata()
    o = m.adata.obs
    true = o["GroupC"].astype(str).to_numpy()
    sizes = [len(f) for f in m.factor_names]
    r = dict(step=step, minutes=round((time.time() - t0) / 60, 1), layer_sizes=sizes)
    for i, name in enumerate(m.layer_names):
        r[f"ARI_{name}"] = [round(ari(o[g].astype(str), o[name].astype(str)), 3) for g in ("GroupC", "GroupB", "GroupA")]
    l0 = o[m.layer_names[0]].astype(str).to_numpy()
    r["L0_factors_with_cells"] = int(len(np.unique(l0)))
    X = np.asarray(m.adata.obsm["X_" + m.layer_names[0]], float)
    X = X / np.maximum(X.sum(1, keepdims=True), 1e-12)
    r["KMeans8_on_L0"] = round(ari(true, KMeans(8, n_init=10, random_state=0).fit_predict(X)), 3)
    ct = pd.crosstab(l0, true)
    main = ct.idxmax(0)                                   # the factor holding most cells of each group
    parent = o.groupby("GroupC", observed=True)["GroupB"].first().astype(str)
    merged = []
    for p in sorted(parent.unique()):
        kids = sorted(parent.index[parent == p])
        if len(kids) == 2 and main[kids[0]] == main[kids[1]]:
            merged.append(f"{kids[0]}+{kids[1]} in {main[kids[0]]}")
    r["sibling_merges"] = merged
    print(json.dumps(r), flush=True)
    return r


def main(ds, init):
    a = S.scdef_adata(ds)
    hier = {"hier": True, "prior": False}[init]
    t0 = time.time()
    m = scd.scDEF(a.copy(), counts_layer="counts", batch_key=None, seed=42)
    steps = []
    m.fit(hierarchical_init=hier)
    steps.append(readout(m, "fit", t0))
    scd.tl.filter(m, batch_key=None, **rb.FIXED_FIRST)
    steps.append(readout(m, "filter 1", t0))
    m.fit(hierarchical_init=hier)
    steps.append(readout(m, "refit", t0))
    scd.tl.filter(m, batch_key=None, **rb.FIXED_SECOND)
    steps.append(readout(m, "filter 2", t0))
    d = S.ds_dir("diagnosis", ds)
    with open(os.path.join(d, f"{init}.json"), "w") as fh:
        json.dump(dict(id=ds, init=init, seed=42, first=rb.FIXED_FIRST, second=rb.FIXED_SECOND, steps=steps), fh,
                  indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
