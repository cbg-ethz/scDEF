"""One seed of every data set with the fixed workflow, to check the thresholds before the best-of-five refits.

    python reproducibility/make_figures/pilot_fixed_workflow.py [data ...]

Saves ``pilot-<data>-<kind>`` models next to the figure models and ``out/pilot_fixed_workflow.json``.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refit_best_of_seeds as rb  # noqa: E402

import scdef as scd

SEED = 42
KINDS = {"pbmcs3k": ("nobatch",), "visium": ("nobatch",), "pbmcs2b": ("nobatch", "ref"), "hgsoc": ("ref",),
         "ifn": ("nobatch", "ref"), "planaria": ("nobatch",)}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "pilot_fixed_workflow.json")

if __name__ == "__main__":
    log = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for data in sys.argv[1:] or list(KINDS):
        a = rb.tutorial_adata(data)
        for kind in KINDS[data]:
            key = f"{data}-{kind}"
            if key in log:
                continue
            t0 = time.time()
            m = rb.fit_fixed(a, SEED, batch_key=None if kind == "nobatch" else rb.BATCH_KEY[data])
            d = rb.describe(m, SEED, (time.time() - t0) / 60)
            rb.save(m, f"pilot-{key}")
            if kind == "ref":
                t1 = time.time()
                dec = scd.scDEF.decompose_batch_effects(m, top_layer=2)
                scd.tl.filter(dec, batch_key=rb.BATCH_KEY[data], **rb.DECOMPOSED)
                d["decomposed"] = dict(minutes=round((time.time() - t1) / 60, 1), layer_sizes=[len(f) for f in dec.factor_lists])
                rb.save(dec, f"pilot-{data}-dec-top2")
            log[key] = d
            json.dump(log, open(OUT, "w"), indent=1)
            print(key, json.dumps(d), flush=True)
    print("done", flush=True)
