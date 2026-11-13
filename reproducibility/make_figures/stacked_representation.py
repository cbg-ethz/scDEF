"""scDEF scored on the proportions of all its layers, stacked, against layer 0 alone (pre-declared 2026-11-13 in
scDEF-markdown/experiments/exp-stacked-representation.md, before computing).

    JAX_PLATFORMS=cpu python stacked_representation.py real   -> out/stacked_representation_real.csv
    JAX_PLATFORMS=cpu python stacked_representation.py sims   -> out/stacked_representation_sims.csv
    python stacked_representation.py report                    -> the declared checks

Stacked: the proportions of the weights of every layer with at least two factors (each row of each layer divided by
its sum), concatenated. Everything else is the accepted protocol (realdata_benchmark/score.py, protocol_values).
"""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
HERE = os.path.dirname(os.path.abspath(__file__))
SIM_DIR = os.path.join(HERE, "simulation_benchmark")
sys.path.insert(0, SIM_DIR)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import simcommon as S  # noqa: E402

rs = S.rd_score()             # realdata_benchmark/score.py
rc = S.rc                     # realdata_benchmark/common.py

REAL = [("pbmcs3k", "scDEF"), ("planaria", "scDEF"), ("pbmcs2b", "scDEF"), ("pbmcsifn", "scDEF")]


def layer_props(m, L):
    w = np.asarray(m.pmeans[f"{m.layer_names[L]}z"])[:, m.factor_lists[L]]
    return w / np.maximum(w.sum(1, keepdims=True), 1e-12)


def stacked(m):
    return np.hstack([layer_props(m, L) for L in range(m.n_layers) if len(m.factor_lists[L]) >= 2])


def real():
    rows = []
    for data, label in REAL:
        spec = rc.DATASETS[data]
        inp, _ = rs.load_input(data)
        m = rs.load_scdef(data, spec["scdef"][label], inp)
        for rep, X in (("layer 0", layer_props(m, 0)), ("stacked", stacked(m))):
            # proportions already taken; the "embedding" route passes them through unchanged
            v, _, how = rs.protocol_values(inp.obs, spec["hierarchy_obs"][0], "Batch" if spec["batch"] else None,
                                           "PCA", latent=X)
            rows.append(dict(data=data, representation=rep, dims=X.shape[1], **v))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "out", "stacked_representation_real.csv"), index=False)


def sims():
    import anndata as ad
    import score as ss        # simulation_benchmark/score.py (load_scdef, batch_groups)
    rows = []
    for ds in S.all_ids():
        p = S.parse_id(ds)
        d = S.ds_dir(ds)
        inp = ad.read_h5ad(os.path.join(d, "input.h5ad"))
        gpath = os.path.join(d, "data", "groups.csv")
        groups = pd.read_csv(gpath) if (p["multi"] and os.path.exists(gpath)) else None
        base = dict(id=ds, scenario=p["scenario"], similarity=p["similarity"], rep=p["rep"])
        labels = ["scDEF"] + ([S.DECOMPOSED_T1] if p["multi"] else [])   # Figure 2D: specific groups from the layer-1 decomposition
        for label in labels:
            fd = S.ds_dir(ds, "scdef", label)
            if S.read_json(os.path.join(fd, "fit.json"), {}).get("status") != "ok":
                continue
            m = ss.load_scdef(ds, label, inp)
            for rep, X in (("layer 0", layer_props(m, 0)), ("stacked", stacked(m))):
                v, clusters, _ = rs.protocol_values(inp.obs, "GroupC", "Batch" if p["multi"] else None, "PCA", latent=X)
                r = dict(base, model=label, representation=rep, ari=v["Cell Type ARI"], asw=v["Cell Type ASW"])
                if groups is not None:
                    bg = ss.batch_groups(inp.obs, clusters, groups)
                    r.update(shared_kept=bg.get("shared_kept"), specific_kept=bg.get("specific_kept"))
                rows.append(r)
        print("done", ds, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "out", "stacked_representation_sims.csv"), index=False)


def report():
    pd.set_option("display.width", 220)
    r = pd.read_csv(os.path.join(HERE, "out", "stacked_representation_real.csv"))
    cols = ["Cell Type ARI", "Cell Type ASW"] + (["Batch ARI"] if "Batch ARI" in r else [])
    t = r.pivot(index="data", columns="representation", values=cols).round(3)
    print(t)
    gain = t[("Cell Type ARI", "stacked")] - t[("Cell Type ARI", "layer 0")]
    asw = t[("Cell Type ASW", "stacked")] - t[("Cell Type ASW", "layer 0")]
    c1 = (gain > 0).sum() >= 3 and gain.mean() >= 0.03
    c2 = (asw >= -0.02).all() and (r[(r.representation == "stacked") & r["Batch ARI"].notna()]["Batch ARI"] <= 0.03).all()
    print("ARI gain per data set:", gain.round(3).to_dict(), "mean", round(gain.mean(), 3), "-> criterion 1", c1)
    print("ASW change:", asw.round(3).to_dict(), "-> criterion 2", c2)
    s = pd.read_csv(os.path.join(HERE, "out", "stacked_representation_sims.csv"))
    a = s[s.model == "scDEF"].groupby(["scenario", "similarity", "representation"])[["ari", "asw", "shared_kept"]].mean().unstack()
    print(a.round(3))
    dari = a[("ari", "stacked")] - a[("ari", "layer 0")]
    dsh = a[("shared_kept", "stacked")] - a[("shared_kept", "layer 0")]
    dec = s[s.model == S.DECOMPOSED_T1].groupby(["scenario", "similarity", "representation"])["specific_kept"].mean().unstack()
    print(dec.round(2))
    dsp = dec["stacked"] - dec["layer 0"]
    c3 = (dari >= -0.01).all() and (dsh.dropna() >= -0.2).all() and (dsp.dropna() >= -0.2).all()
    print("sims: ARI change", dari.round(3).to_dict(), "shared kept change", dsh.round(2).to_dict(),
          "specific kept change", dsp.round(2).to_dict(), "-> criterion 3", c3)
    print("ADOPT" if (c1 and c2 and c3) else "DO NOT ADOPT")


if __name__ == "__main__":
    {"real": real, "sims": sims, "report": report}[sys.argv[1]]()
