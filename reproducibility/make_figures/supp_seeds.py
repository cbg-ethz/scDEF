"""Supplementary figure for R1.2b: how much the PBMC results depend on the initialization.

Every PBMC model in the paper is fitted from five seeds and the fit with the lowest final loss is
kept (``refit_best_of_seeds.py``). This figure shows all five fits, from the logs that script writes:

- A: agreement of the layer-0 factors with the annotated cell types, per seed, for each fit; the kept
  seed is marked. Layer 0 is assigned by each cell's largest weight, so this does not depend on the
  upper-layer assignment rule.
- B: number of layer-0 factors left after filtering, per seed.
- C: 3k PBMCs only, where every seed's assignments were saved: agreement between pairs of seeds, and
  of each seed with the cell types, at each layer. (The two-donor and interferon logs were written
  before the upper-layer rule changed, so their upper layers are not compared here.)

    python reproducibility/make_figures/supp_seeds.py [--out DIR]
"""
import argparse
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig4_batch as f4  # noqa: E402  (style)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from sklearn.metrics import adjusted_rand_score

HERE = os.path.dirname(os.path.abspath(__file__))
FITS = [("pbmcs3k", "nobatch", "3k PBMCs"),
        ("pbmcs2b", "nobatch", "two donors,\nno batch annotations"),
        ("pbmcs2b", "ref", "two donors,\nbatch annotations"),
        ("ifn", "nobatch", "interferon,\nno batch annotations"),
        ("ifn", "ref", "interferon,\nbatch annotations")]
KEPT, OTHER = "#2f6f9e", "#9a9a9a"


def logs():
    return {d: json.load(open(os.path.join(HERE, "out", f"refit_best_of_seeds_{d}.json")))
            for d in sorted({d for d, _, _ in FITS})}


def strip(ax, values, kept, y):
    for v, k in zip(values, kept):
        ax.scatter(v, y, s=14 if k else 9, color=KEPT if k else OTHER, lw=0, zorder=3 if k else 2)


def main(out):
    L = logs()
    fig = plt.figure(figsize=(7.2, 4.6))
    G = GridSpec(2, 3, figure=fig, height_ratios=[1, 1], width_ratios=[1.15, 0.8, 1.25],
                 hspace=0.55, wspace=0.55, left=0.17, right=0.98, top=0.95, bottom=0.1)
    ax_a, ax_b = fig.add_subplot(G[:, 0]), fig.add_subplot(G[:, 1])
    numbers = {}
    for i, (d, kind, label) in enumerate(FITS):
        rows, chosen = L[d][kind], L[d].get(f"{kind}_chosen_seed")
        kept = [r["seed"] == chosen for r in rows]
        strip(ax_a, [r["ARI_L0"] for r in rows], kept, i)
        strip(ax_b, [r["layer_sizes"][0] for r in rows], kept, i)
        numbers[f"{d}/{kind}"] = dict(chosen_seed=chosen,
                                      ARI_L0={r["seed"]: round(r["ARI_L0"], 3) for r in rows},
                                      n_L0={r["seed"]: r["layer_sizes"][0] for r in rows},
                                      loss_spread_pct=round(100 * (max(r["final_loss"] for r in rows)
                                                                   / min(r["final_loss"] for r in rows) - 1), 2))
    for ax in (ax_a, ax_b):
        ax.set_yticks(range(len(FITS)), [lab for _, _, lab in FITS] if ax is ax_a else [])
        ax.set_ylim(len(FITS) - 0.5, -0.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    ax_a.set_xlabel("ARI, layer-0 factors vs cell types", labelpad=1)
    ax_b.set_xlabel("layer-0 factors kept", labelpad=1)
    ax_a.scatter([], [], color=KEPT, s=14, label="kept (lowest loss)")
    ax_a.scatter([], [], color=OTHER, s=9, label="other seeds")
    ax_a.legend(frameon=False, fontsize=5.6, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2,
                handletextpad=0.1, columnspacing=0.6)

    # C: 3k PBMCs, per layer
    A = pd.read_csv(os.path.join(HERE, "out", "refit_best_of_seeds_pbmcs3k_assignments.csv"), index_col=0)
    ann = pd.read_csv(os.path.join(f4.DATA, "pbmcs3k", "annotations.csv"), index_col=0)
    ct = ann["seurat_annotations"].reindex(A.index).astype(str).to_numpy()
    seeds = sorted({int(c.split("_seed")[1].split("_")[0]) for c in A.columns})
    layers = sorted({c.rsplit("_", 1)[1] for c in A.columns}, key=lambda s: int(s[1:]))
    # seeds end with different depths: compare the layers every seed has with more than one factor
    layers = [ln for ln in layers if all(f"nobatch_seed{s}_{ln}" in A and A[f"nobatch_seed{s}_{ln}"].nunique() > 1
                                         for s in seeds)]
    ax_c = fig.add_subplot(G[:, 2])
    pair, vs_type = {}, {}
    for j, ln in enumerate(layers):
        p = [adjusted_rand_score(A[f"nobatch_seed{s}_{ln}"], A[f"nobatch_seed{t}_{ln}"])
             for s, t in itertools.combinations(seeds, 2)]
        pair[ln] = p
        ax_c.scatter(np.full(len(p), j) - 0.12, p, s=8, color=OTHER, lw=0)
        v = [adjusted_rand_score(ct, A[f"nobatch_seed{s}_{ln}"]) for s in seeds]
        vs_type[ln] = v
        ax_c.scatter(np.full(len(v), j) + 0.12, v, s=8, color=KEPT, lw=0, marker="D")
    ax_c.set_xticks(range(len(layers)), [f"layer {ln[1:]}" for ln in layers])
    ax_c.set_ylim(0, 1)
    ax_c.set_ylabel("ARI", labelpad=1)
    ax_c.set_title("3k PBMCs, five seeds", fontsize=6.5, pad=2)
    ax_c.scatter([], [], s=8, color=OTHER, label="between pairs of seeds")
    ax_c.scatter([], [], s=8, color=KEPT, marker="D", label="each seed vs cell types")
    ax_c.legend(frameon=False, fontsize=5.6, loc="lower left", handletextpad=0.1)
    for side in ("top", "right"):
        ax_c.spines[side].set_visible(False)
    numbers["pbmcs3k_between_seeds"] = {ln: [round(x, 3) for x in v] for ln, v in pair.items()}
    numbers["pbmcs3k_vs_cell_types"] = {ln: [round(x, 3) for x in v] for ln, v in vs_type.items()}
    for a, letter in ((ax_a, "A"), (ax_b, "B"), (ax_c, "C")):
        bb = a.get_position()
        fig.text(bb.x0 - (0.15 if a is ax_a else 0.06), bb.y1 + 0.03, letter, fontsize=10, fontweight="bold")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"supp_seeds.{ext}"), dpi=400)
    with open(os.path.join(out, "supp_seeds_numbers.json"), "w") as f:
        json.dump(numbers, f, indent=1)
    print(json.dumps(numbers, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    main(ap.parse_args().out)
