"""Where does training lose the planaria resolution of its initialisation? First fits only, stopped early.

The hierarchical initialisation (k-means with k = 100 on the PCA, fitted on 5,000 cells) resolves 30 of the 51
cell types (F1 >= 0.5); the full first fit ends at 18 (`planaria_resolution.py`). Each run here is a first fit,
scored over all 100 layer-0 factors, by the largest weight z and by z times the factor's summed loading.

    python reproducibility/make_figures/planaria_trace.py [run ...]
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import planaria_resolution as pr  # noqa: E402
import refit_best_of_seeds as rb  # noqa: E402

import numpy as np
import scdef as scd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "planaria_trace.json")
RUNS = {
    "epochs1": dict(n_epoch=1, min_epochs=1),
    "epochs30": dict(n_epoch=30, min_epochs=1),
    "epochs100": dict(n_epoch=100, min_epochs=1),
    "default": dict(),
    "zoff0.01": dict(z_off=0.01),
    # the initial z is drawn from Gamma(c, c / init) with c = min(0.5, alpha / 10) = 0.058 here: after one epoch
    # the factors resolve no cell type although the k-means labels resolve 30. A tighter draw around the labels:
    "epochs1_zconc10": dict(n_epoch=1, min_epochs=1, z_init_concentration=10.0),
    "zconc10": dict(z_init_concentration=10.0),
    "zconc100": dict(z_init_concentration=100.0),
}


def scaled(m, a):
    """Resolution when each cell goes to the factor with the largest z times the factor's summed loading."""
    z = np.asarray(m.pmeans["L0z"]); W = np.asarray(m.pmeans["L0W"])
    zs = z * W.sum(1)[None, :]
    m.pmeans["L0z"] = zs                     # present the scaled weights to the scorer, then restore
    try:
        out = pr.resolved(m, a, np.arange(z.shape[1]))
    finally:
        m.pmeans["L0z"] = z
    return out, int(len(set(np.argmax(zs, 1))))


if __name__ == "__main__":
    log = json.load(open(OUT)) if os.path.exists(OUT) else {}
    a = rb.tutorial_adata("planaria")
    for name in sys.argv[1:] or list(RUNS):
        if name in log:
            continue
        t0 = time.time()
        m = scd.scDEF(a.copy(), counts_layer="counts", seed=pr.SEED)
        m.fit(hierarchical_init=True, filter=False, **RUNS[name])
        k = np.arange(np.asarray(m.pmeans["L0z"]).shape[1])
        sc_res, n_used = scaled(m, a)
        W = np.asarray(m.pmeans["L0W"]).sum(1)
        log[name] = dict(argmax_z=pr.resolved(m, a, k), argmax_scaled=sc_res, factors_used_scaled=n_used,
                         loading_quantiles=[float(q) for q in np.quantile(W, [0.25, 0.5, 0.75, 1.0])],
                         epochs=len(np.asarray(m.elbos[-1]).ravel()) if m.elbos else None,
                         minutes=round((time.time() - t0) / 60, 1))
        json.dump(log, open(OUT, "w"), indent=1)
        r = log[name]
        print(name, "| types: argmax z", r["argmax_z"]["types"], "scaled", r["argmax_scaled"]["types"],
              "| factors used (scaled)", n_used, "| loading quartiles", [round(x, 3) for x in r["loading_quantiles"]],
              "| epochs", r["epochs"], "|", r["minutes"], "min", flush=True)
    print("done", flush=True)
