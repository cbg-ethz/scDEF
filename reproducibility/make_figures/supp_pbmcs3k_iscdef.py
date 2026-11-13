"""Supplementary figure: informed scDEF (iscDEF) on the 3k PBMCs.

Models: ``iscdef-pbmcs3k-{resolved,coarse}-best5`` (``refit_iscdef_pbmcs3k.py``: the analysis of
scDEF-notebooks/iscdef-pbmcs3k.ipynb from five seeds, lowest final loss kept). Replaces the original
submission's ``pbmcs3k_iscdef``, which came from an older model.

- A, B: marker sets for the nine annotated cell types at layer 0; iscDEF learns the hierarchy above.
- C, D: marker sets for T cells, monocytes and B cells at layer 2; iscDEF learns the states below.

    python reproducibility/make_figures/supp_pbmcs3k_iscdef.py [--out DIR]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import supp_pbmcs3k as s3  # noqa: E402  (loading, panels, style)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scdef.utils import data_utils

KINDS = (("resolved", "Marker sets for the nine annotated cell types, at layer 0"),
         ("coarse", "Marker sets for T cells, monocytes and B cells, at layer 2"))


def main(out):
    os.makedirs(out, exist_ok=True)
    fig = plt.figure(figsize=(7.2, 6.6))
    G = GridSpec(2, 2, figure=fig, width_ratios=[1.3, 1], hspace=0.5, wspace=0.12,
                 left=0.12, right=0.99, top=0.95, bottom=0.08)
    numbers, letters = {}, []
    for r, (kind, title) in enumerate(KINDS):
        m = s3.load(f"iscdef-pbmcs3k-{kind}-best5")
        f1 = data_utils.get_assignment_scores(m, "Cell types", s3.TYPES)
        ax = s3.score_blocks(fig, G[r, 0], m, f1, "F1", "viridis")
        letters.append((ax, "AC"[r]))
        fig.text(0.01, G[r, 0].get_position(fig).y1 + 0.035, title, fontsize=7, fontweight="bold")
        axg = fig.add_subplot(G[r, 1])
        s3.hierarchy_panel(axg, m, size="6,5.5!", fontsize="8")
        letters.append((axg, "BD"[r]))
        numbers[kind] = dict(layer_names=list(m.layer_names), layer_sizes=[len(f) for f in m.factor_lists],
                             best_f1_per_type={m.layer_names[k]: dict(zip(s3.TYPES, np.round(np.nanmax(f1[k], axis=1), 2).tolist()))
                                               for k in s3.shown_layers(m)})
    for a, letter in letters:
        bb = a.get_position()
        fig.text(max(bb.x0 - 0.1, 0.005) if letter in "AC" else bb.x0, bb.y1 + 0.01, letter,
                 fontsize=10, fontweight="bold", va="bottom")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"supp_pbmcs3k_iscdef.{ext}"), dpi=400)
    with open(os.path.join(out, "supp_pbmcs3k_iscdef_numbers.json"), "w") as f:
        json.dump(numbers, f, indent=1)
    print(json.dumps(numbers, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
    main(ap.parse_args().out)
