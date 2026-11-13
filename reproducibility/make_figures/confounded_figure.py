"""Supplementary Figure fig:confounded: the confounded-batch benchmark (scDEF-markdown/experiments/
exp-confounded-batch-attribution.md, Amendment 2: scDEF fitted with scDEF.learn, results in confounded_results_v3).

    python confounded_figure.py [--results ~/projects/scDEF/confounded_results_v3]
        -> out/supp_confounded.{pdf,png}, out/supp_confounded_numbers.json

Per data set and method, from results.json:
- condition removed = mean over the cell types with t >= 0.8 of max(t - r, 0),
- technical difference left = mean over the cell types with t <= 0.6 of max(r - t, 0) (two donors: every type, t = 0.5),
where r is the cross-validated AUC that separates the batches within a cell type in the method's representation and t the
same AUC on the clean data. Biological attribution: the cell-weighted AUPRC of each cell type's true response genes.
"""
import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})

# the palette of the benchmark figures (realdata_benchmark/plot.py)
METHODS = [("Uncorrected", "Uncorrected", "#bdbdbd"), ("Harmony", "Harmony", "#d62728"), ("scVI", "scVI", "#ff7f0e"),
           ("Scanorama", "Scanorama", "#8c564b"), ("scDEF_default", "scDEF", "#2ca02c"),
           ("scDEF_decomposed_raw", "scDEF (decomposed)", "#98df8a")]
ATTR = [("Harmony_DE", "Harmony + DE", "#d62728"), ("scDEF_default", "scDEF", "#2ca02c"),
        ("scDEF_decomposed_raw", "scDEF (decomposed)", "#98df8a")]
DESIGNS = [("D1", "Control cells,\ninjected response"), ("D2", "Interferon\nresponse"), ("D2b", "Pairs of\ndonors")]
NOISE = [("T1", "Depth"), ("T2", "Ambient RNA"), ("T3", "Per-gene\ncapture")]
CHANCE = {"D1": 0.0125, "D2": 0.04, "D2b": 0.04}     # prevalence of response genes (exp-confounded-batch-attribution)


def per_dataset(root):
    rows = []
    for f in sorted(glob.glob(os.path.join(root, "*", "results.json"))):
        r = json.load(open(f))
        meta = r["meta"]
        base = dict(dataset=r["dataset"], design=meta["design"], tech=meta.get("tech"))
        for meth, c in r["calibration"].items():
            pt = c["per_type"].values()
            bio = [max(v["t"] - v["r"], 0) for v in pt if v["t"] >= 0.8]
            tech = [max(v["r"] - v["t"], 0) for v in pt if v["t"] <= 0.6]
            rows.append(dict(base, method=meth, removed=np.mean(bio) if bio else np.nan,
                             left=np.mean(tech) if tech else np.nan))
        for meth, a in r.get("attribution", {}).items():
            rows.append(dict(base, method=meth, attribution=a.get("bio_auprc")))
    return pd.DataFrame(rows)


def dots_bars(ax, groups, methods, values, ylabel):
    """groups: [(key, label, mask)]; one bar (mean) and dots (data sets) per method within each group."""
    w = 0.8 / len(methods)
    for gi, (_, _, sub) in enumerate(groups):
        for mi, (m, _, col) in enumerate(methods):
            v = sub.loc[sub.method == m, values].dropna().to_numpy()
            if not len(v):
                continue
            x = gi + (mi - (len(methods) - 1) / 2) * w
            ax.bar(x, v.mean(), width=w * 0.9, color=col, edgecolor="none", zorder=1)
            jit = (np.random.default_rng(0).random(len(v)) - 0.5) * w * 0.5
            ax.scatter(x + jit, v, s=3, color="0.15", lw=0, zorder=2)
    ax.set_xticks(range(len(groups)), [g[1] for g in groups])
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.55, len(groups) - 0.45)


def main(results, out):
    df = per_dataset(results)
    numbers = {}
    fig = plt.figure(figsize=(7.2, 5.4))
    gs = fig.add_gridspec(2, 2, left=0.08, right=0.98, top=0.93, bottom=0.1, hspace=0.55, wspace=0.28,
                          width_ratios=[1.1, 1])
    # A: condition removed, by design
    ax = fig.add_subplot(gs[0, 0])
    groups = [(d, lab, df[df.design == d]) for d, lab in DESIGNS]
    dots_bars(ax, groups, METHODS, "removed", "Condition signal removed\n(AUC, responding cell types)")
    numbers["condition_removed"] = {d: df[df.design == d].groupby("method")["removed"].mean().round(3).to_dict()
                                    for d, _ in DESIGNS}
    # B: technical difference left, by kind of noise (control cells) and on the two donors
    ax = fig.add_subplot(gs[0, 1])
    groups = [(t, lab, df[(df.design == "D1") & (df.tech == t)]) for t, lab in NOISE] + [("D3", "Two donors", df[df.design == "D3"])]
    dots_bars(ax, groups, METHODS, "left", "Technical difference left\n(AUC, cell types without response)")
    numbers["technical_left"] = {g[0]: g[2].groupby("method")["left"].mean().round(3).to_dict() for g in groups}
    # C: biological attribution
    ax = fig.add_subplot(gs[1, 0])
    groups = [(d, lab, df[df.design == d]) for d, lab in DESIGNS]
    dots_bars(ax, groups, ATTR, "attribution", "Response genes ranked\n(AUPRC)")
    for gi, (d, _) in enumerate(DESIGNS):
        ax.plot([gi - 0.42, gi + 0.42], [CHANCE[d]] * 2, color="0.4", lw=0.6, ls="--", zorder=3)
    numbers["attribution"] = {d: df[df.design == d].groupby("method")["attribution"].mean().round(3).to_dict()
                              for d, _ in DESIGNS}
    # D: the trade-off on the control cells
    ax = fig.add_subplot(gs[1, 1])
    d1 = df[df.design == "D1"]
    for m, lab, col in METHODS:
        s = d1[d1.method == m]
        ax.scatter(s["left"], s["removed"], s=6, color=col, alpha=0.5, lw=0)
        ax.scatter(s["left"].mean(), s["removed"].mean(), s=40, color=col, edgecolor="0.15", lw=0.6, zorder=3, label=lab)
    ax.set_xlabel("Technical difference left (AUC)")
    ax.set_ylabel("Condition signal removed (AUC)")
    ax.set_xlim(left=-0.02)
    ax.set_ylim(bottom=-0.02)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in METHODS] + [plt.Rectangle((0, 0), 1, 1, color="#d62728", alpha=0)]
    fig.legend(handles[:len(METHODS)], [l for _, l, _ in METHODS], loc="upper center", ncol=6, frameon=False,
               bbox_to_anchor=(0.53, 1.0), handlelength=1.0, columnspacing=1.2)
    for axx, letter in zip(fig.axes, "ABCD"):
        pos = axx.get_position()
        fig.text(pos.x0 - 0.07, pos.y1 + 0.02, letter, fontsize=10, fontweight="bold", va="bottom")
    os.makedirs(out, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"supp_confounded.{ext}"), dpi=200)
    verdicts = pd.read_csv(os.path.join(results, "verdicts.csv"))
    v = verdicts[(verdicts.metric == "bio_auprc") & (verdicts.versus == "Harmony_DE")]
    numbers["attribution_verdicts_vs_harmony_de"] = v.groupby("method")["verdict"].value_counts().unstack(fill_value=0).to_dict("index")
    json.dump(numbers, open(os.path.join(out, "supp_confounded_numbers.json"), "w"), indent=1)
    print(json.dumps(numbers, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.expanduser("~/projects/scDEF/confounded_results_v3"))
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    a = ap.parse_args()
    main(a.results, a.out)
