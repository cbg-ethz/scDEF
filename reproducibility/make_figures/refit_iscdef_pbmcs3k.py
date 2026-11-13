"""Refit the two iscDEF models of the 3k PBMC supplement from several seeds; keep the lowest final loss.

The analysis is the one in scDEF-notebooks/iscdef-pbmcs3k.ipynb (2026-05-21), with its marker lists,
model settings, fit arguments and filters unchanged, re-run on the current code. That notebook
predates two changes, which therefore apply here: the hierarchical initialization is used by default
when a PCA is present (bc93012, 2026-08-12), and the filtering API was updated (40959a0, 2026-06-22).

- ``resolved``: the nine cell-type marker sets at layer 0; iscDEF learns the hierarchy above them.
- ``coarse``: T, monocyte and B marker sets at layer 2; iscDEF learns the states below them.

    python reproducibility/make_figures/refit_iscdef_pbmcs3k.py

Saves ``iscdef-pbmcs3k-{resolved,coarse}-best5`` next to the tutorial models and
``out/refit_iscdef_pbmcs3k.json`` (every seed: loss, layer sizes, agreement with the annotations).
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refit_best_of_seeds as rb  # noqa: E402

import scdef as scd
from sklearn.metrics import adjusted_rand_score

# The notebook's marker sets (its cell 2), unchanged.
MARKERS = {"Memory CD4 T": ["IL7R", "CD3D", "CD3E", "IL32", "CD2"],
           "Naive CD4 T": ["IL7R", "CD3D", "CD3E", "TCF7", "CCR7", "CD2"],
           "CD8 T": ["CD8A", "CD8B", "CCL5", "CD2", "CD3D", "CD3E"],
           "NK": ["GNLY", "NKG7", "CD2"],
           "B": ["MS4A1", "CD79A", "CD79B", "CD74"],
           "CD14+ Mono": ["CD14", "LYZ"],
           "FCGR3A+ Mono": ["CD14", "FCGR3A", "MS4A7", "LYZ"],
           "DC": ["CD14", "FCER1A", "CST3", "CD74"],
           "Platelet": ["PPBP", "PF4"]}
TOP_MARKERS = {"T": sorted(set(MARKERS["Memory CD4 T"] + MARKERS["Naive CD4 T"] + MARKERS["CD8 T"] + MARKERS["NK"])),
               "Mono": sorted(set(MARKERS["CD14+ Mono"] + MARKERS["FCGR3A+ Mono"] + MARKERS["DC"])),
               "B": MARKERS["B"]}


def fit_resolved(a, seed):                          # notebook cells 6, 7, 11
    m = scd.iscDEF(a.copy(), counts_layer="counts", markers_dict=MARKERS, markers_layer=0,
                   add_other=1, seed=seed)
    m.fit(tolerance=1e-3, z_init_concentration=2.)
    m.filter_factors(brd_min=0., n_eff_parents_max=10., ard_min=0.00, filter_up=True, min_cells_upper=1)
    return m


def fit_coarse(a, seed):                            # notebook cells 18, 20, 24
    m = scd.iscDEF(a.copy(), counts_layer="counts", markers_dict=TOP_MARKERS, markers_layer=2,
                   add_other=1, decay_factor=2, seed=seed)
    m.fit(n_rounds=1, z_init_concentration=1., tolerance=1e-3, n_epochs=1000)
    m.filter_factors(brd_min=1., clarity_min=0., ard_min=0.0, n_eff_parents_max=10.00, filter_up=True,
                     min_cells_upper=1)
    return m


def describe(m, seed, minutes, label_key):
    obs = m.adata.obs
    agree = {f"ARI_{ln}": float(adjusted_rand_score(obs[label_key].astype(str), obs[ln].astype(str)))
             for ln in m.layer_names if ln in obs and obs[ln].nunique() > 1}
    return dict(seed=seed, final_loss=rb.final_loss(m), minutes=round(minutes, 1),
                layer_sizes=[len(f) for f in m.factor_lists], **agree)


def main():
    a = rb.prep_3k(MARKERS)
    out_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "refit_iscdef_pbmcs3k.json")
    log = dict(seeds=rb.SEEDS, rule="lowest final loss")
    for kind, fit, label_key in (("resolved", fit_resolved, "Cell types"), ("coarse", fit_coarse, "Coarse")):
        log[kind], best = [], None
        for seed in rb.SEEDS:
            t0 = time.time()
            m = fit(a, seed)
            d = describe(m, seed, (time.time() - t0) / 60, label_key)
            log[kind].append(d)
            print(kind, json.dumps(d), flush=True)
            if best is None or d["final_loss"] < best[0]:
                best = (d["final_loss"], seed, m)
            json.dump(log, open(out_json, "w"), indent=1)
        log[f"{kind}_chosen_seed"] = best[1]
        rb.save(best[2], f"iscdef-pbmcs3k-{kind}-best5")
        json.dump(log, open(out_json, "w"), indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
