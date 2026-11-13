"""Depth pilot of the confounded benchmark: cell_alpha_by_batch=True against v3 on the same data sets (pre-declared
2026-11-13 in scDEF-markdown/experiments/exp-confounded-batch-attribution.md, "Pilot: the prior strength relative to
the batch").

    python pilot_alpha_score.py [--pilot ~/projects/scDEF/confounded_results_pilot_alpha]
                                [--base ~/projects/scDEF/confounded_results_v3]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

DEPTH_D1 = ["D1_loc1_n50_T1_s0", "D1_loc3_n50_T1_s0", "D1_loc3_n50_T1_s1", "D1_loc3_n50_T1_s2", "D1_locall_n50_T1_s0"]
DEPTH_D2 = ["D2_T1_s0", "D2_T1_s1"]
OTHER = ["D1_loc3_n50_T2_s0", "D1_loc3_n50_T3_s0", "D3_s0"]
MODELS = ["scDEF_default", "scDEF_decomposed_raw"]


def values(root, ds):
    f = os.path.join(root, ds, "results.json")
    if not os.path.exists(f):
        return None
    r = json.load(open(f))
    out = {}
    for m in MODELS:
        c = r["calibration"][m]
        pt = c["per_type"].values()
        bio = [max(v["t"] - v["r"], 0) for v in pt if v["t"] >= 0.8]
        tech = [max(v["r"] - v["t"], 0) for v in pt if v["t"] <= 0.6]
        comp = r["info"][m].get("composition", {})
        cs, lib = comp.get("cell_scale_by_batch", {}), comp.get("library_by_batch", {})
        b = sorted(cs)
        out[m] = dict(removed=np.mean(bio) if bio else np.nan, left=np.mean(tech) if tech else np.nan, mae=c["mae"],
                      cs_ratio=cs[b[1]] / cs[b[0]] if len(b) == 2 else np.nan,
                      lib_ratio=lib[b[1]] / lib[b[0]] if len(b) == 2 else np.nan,
                      kwargs=r["info"][m].get("scdef_kwargs"))
    return out


def main(pilot, base):
    rows = []
    for ds in DEPTH_D1 + DEPTH_D2 + OTHER:
        p, b = values(pilot, ds), values(base, ds)
        if p is None:
            print("missing", ds)
            continue
        for m in MODELS:
            rows.append(dict(dataset=ds, model=m, **{f"{k}_v3": b[m][k] for k in ("removed", "left", "mae")},
                             **{f"{k}_pilot": p[m][k] for k in ("removed", "left", "mae", "cs_ratio", "lib_ratio")},
                             kwargs=json.dumps(p[m]["kwargs"])))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(df.drop(columns="kwargs").round(3).to_string(index=False))
    d = df[df.model == "scDEF_default"].set_index("dataset")
    dec = df[df.model == "scDEF_decomposed_raw"].set_index("dataset")
    d1 = d.loc[[x for x in DEPTH_D1 if x in d.index]]
    c1 = (d1.left_pilot.mean() <= d1.left_v3.mean() * 2 / 3) and (d1.removed_pilot.mean() - d1.removed_v3.mean() <= 0.02)
    d2 = d.loc[[x for x in DEPTH_D2 if x in d.index]]
    c2 = bool(((d2.mae_pilot - d2.mae_v3) <= 0.02).all() and ((d2.removed_pilot - d2.removed_v3) <= 0.02).all())
    ot = d.loc[[x for x in OTHER if x in d.index]]
    c3 = bool((((ot.left_pilot - ot.left_v3).fillna(0)) <= 0.02).all() and (((ot.removed_pilot - ot.removed_v3).fillna(0)) <= 0.02).all()
              and ((dec.removed_pilot - dec.removed_v3).fillna(0) <= 0.02).all())   # amended before results: relative to v3
    print(f"1. depth, control cells: technical left {d1.left_v3.mean():.3f} -> {d1.left_pilot.mean():.3f}, "
          f"condition removed {d1.removed_v3.mean():.3f} -> {d1.removed_pilot.mean():.3f}: {c1}")
    print(f"2. depth, interferon: MAE {d2.mae_v3.round(3).tolist()} -> {d2.mae_pilot.round(3).tolist()}, "
          f"removed {d2.removed_v3.round(3).tolist()} -> {d2.removed_pilot.round(3).tolist()}: {c2}")
    print(f"3. no harm: {c3}; decomposition removes at most {dec.removed_pilot.max():.3f} (v3: {dec.removed_v3.max():.3f})")
    print("PASS" if (c1 and c2 and c3) else "FAIL", f"({len(d)} of 10 data sets)")
    df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "pilot_alpha.csv"), index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", default=os.path.expanduser("~/projects/scDEF/confounded_results_pilot_alpha"))
    ap.add_argument("--base", default=os.path.expanduser("~/projects/scDEF/confounded_results_v3"))
    a = ap.parse_args()
    main(a.pilot, a.base)
