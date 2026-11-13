"""Distribution of the factor weights of each cell (Reviewer 3, point 1a; pre-declared 2026-11-12 in
scDEF-markdown/revisions/revision-r3-1.md before computing).

    python cell_weights.py compute     (env: scdef, CPU)  -> out/simulation_benchmark/cell_weights.csv
    python cell_weights.py plot                           -> figures/supp_cell_weights.{pdf,png}

For each of the 30 benchmark data sets: the batch-corrected scDEF of the benchmark (scdef/scDEF/model; layers 0 and
1) and, for comparison, NMF and scHPF at their finest level (methods/<M>/latent0.npy). For each cell, the weights of
one layer divided by their sum; read off: the share of the top factor, the share of the second factor and the
effective number of factors (inverse Simpson index, 1 / sum of squared shares). No model is re-fitted.

Added after the pre-declared quantities were read (2026-11-12), as a sensitivity check: the weights of a factor are
not comparable across factors without their gene weights (a factor with small gene weights needs large cell weights to
explain the same counts), so the same three quantities are also computed on the share of the expected counts of the
cell that each factor carries (scDEF: z_ck * sum_g W_kg * gene_scale_bg; NMF: W_ck * sum_g H_kg). Columns *_counts;
not available for scHPF, whose saved scores do not give its gene loadings.
"""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import simcommon as S  # noqa: E402

METHODS = [("scDEF", 0), ("scDEF", 1), ("NMF", 0), ("scHPF", 0)]


def per_cell(w):
    w = np.clip(np.asarray(w, dtype=np.float64), 0, None)
    tot = w.sum(axis=1, keepdims=True)
    p = np.divide(w, tot, out=np.zeros_like(w), where=tot > 0)
    s = -np.sort(-p, axis=1)
    second = s[:, 1] if s.shape[1] > 1 else np.zeros(len(s))
    return dict(top=s[:, 0], second=second, eff=1.0 / np.maximum((p ** 2).sum(axis=1), 1e-12), k=w.shape[1])


def compute():
    import scdef as scd
    rows = []
    for ds in S.all_ids():
        p = S.parse_id(ds)
        d = S.ds_dir(ds)
        base = dict(id=ds, scenario=p["scenario"], similarity=p["similarity"], rep=p["rep"])
        m = scd.scDEF.load(os.path.join(d, "scdef", "scDEF", "model"), adata=S.scdef_adata(ds))
        for method, layer in METHODS:
            contrib = None
            if method == "scDEF":
                ln = m.layer_names[layer]
                w = np.asarray(m.pmeans[f"{ln}z"])[:, m.factor_lists[layer]]
                if layer == 0:
                    W = np.asarray(m.pmeans[f"{ln}W"])[m.factor_lists[0]]
                    gs = np.asarray(m.pmeans["gene_scale"])
                    b = (m.adata.obs[m.batch_key].astype("category").cat.codes.to_numpy()
                         if m.batch_key and gs.shape[0] > 1 else np.zeros(len(w), int))
                    contrib = w * np.einsum("kg,bg->bk", W, gs)[b]
            else:
                w = np.load(os.path.join(d, "methods", method, "latent0.npy"))
                if method == "NMF":
                    sc = np.load(os.path.join(d, "methods", method, "scores.npz"))
                    H = np.stack([sc[f"s{k}"] for k in range(w.shape[1])])
                    contrib = w * H.sum(axis=1)[None]
            r = per_cell(w)
            cols = dict(base, method=method, layer=layer, n_factors=r["k"], cell=np.arange(len(w)),
                        top=r["top"], second=r["second"], eff=r["eff"])
            if contrib is not None:
                rc = per_cell(contrib)
                cols.update(top_counts=rc["top"], second_counts=rc["second"], eff_counts=rc["eff"])
            rows.append(pd.DataFrame(cols))
        print("done", ds, flush=True)
    df = pd.concat(rows, ignore_index=True)
    out = os.path.join(S.OUT_ROOT, "cell_weights.csv")
    df.to_csv(out, index=False)
    print("wrote", out)
    summary(df)


def summary(df=None):
    if df is None:
        df = pd.read_csv(os.path.join(S.OUT_ROOT, "cell_weights.csv"))
    out = []
    for by in (["similarity"], ["similarity", "scenario"]):
        for suffix in ("", "_counts"):
            sub = df.dropna(subset=[f"top{suffix}"])
            g = sub.groupby(["method", "layer"] + by)
            s = pd.DataFrame({
                "factors (median per data set)": g["n_factors"].median(),
                "top share, median": g[f"top{suffix}"].median(),
                "top share > 0.5": g[f"top{suffix}"].apply(lambda x: (x > 0.5).mean()),
                "top two > 0.8": (sub[f"top{suffix}"] + sub[f"second{suffix}"] > 0.8).groupby(
                    [sub[c] for c in ["method", "layer"] + by]).mean(),
                "second share, median": g[f"second{suffix}"].median(),
                "effective factors, median": g[f"eff{suffix}"].median(),
            }).round(3)
            s.insert(0, "measure", "weights" if suffix == "" else "expected counts")
            out.append(s.reset_index())
            if by == ["similarity"]:
                with pd.option_context("display.width", 220, "display.max_columns", 20):
                    print(s)
    pd.concat(out, ignore_index=True).to_csv(os.path.join(S.OUT_ROOT, "cell_weights_summary.csv"), index=False)


def plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 6.5, "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6,
                         "axes.titlesize": 7, "axes.linewidth": 0.5, "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    df = pd.read_csv(os.path.join(S.OUT_ROOT, "cell_weights.csv"))
    series = [("scDEF", 0, "scDEF, layer 0", "#1f77b4"), ("scDEF", 1, "scDEF, layer 1", "#9ecae1"),
              ("NMF", 0, "NMF", "#d62728"), ("scHPF", 0, "scHPF", "#ff9896")]
    scen = [("High", "High group similarity"), ("Low", "Low group similarity")]
    quants = [("top", "Share of the top factor", (0, 1)), ("second", "Share of the second factor", (0, 0.6)),
              ("eff", "Effective number of factors", None)]
    fig, axs = plt.subplots(len(quants), len(scen), figsize=(5.6, 5.0), sharey="row")
    fig.subplots_adjust(left=0.11, right=0.78, top=0.94, bottom=0.05, hspace=0.35, wspace=0.12)
    for c, (sk, title) in enumerate(scen):
        sub = df[df["similarity"] == sk]
        for r, (q, ylab, lim) in enumerate(quants):
            ax = axs[r, c]
            data = [sub[(sub["method"] == m) & (sub["layer"] == l)][q].to_numpy() for m, l, _, _ in series]
            vp = ax.violinplot(data, positions=range(len(series)), widths=0.8, showextrema=False)
            for body, (_, _, _, col) in zip(vp["bodies"], series):
                body.set_facecolor(col)
                body.set_edgecolor("none")
                body.set_alpha(0.75)
            meds = [np.median(x) for x in data]
            ax.scatter(range(len(series)), meds, s=6, color="black", zorder=3)
            ax.set_xticks(range(len(series)), ["" for _ in series])
            if lim is not None:
                ax.set_ylim(*lim)
            if q == "top":
                ax.axhline(0.5, color="0.6", lw=0.5, ls="--", zorder=0)
            if q == "eff":
                ax.set_yscale("log", base=2)
                ax.set_yticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
            if c == 0:
                ax.set_ylabel(ylab)
            if r == 0:
                ax.set_title(title, pad=4)
    from matplotlib.patches import Patch
    axs[0, -1].legend(handles=[Patch(color=col, alpha=0.75, label=lab) for _, _, lab, col in series],
                      loc="center left", bbox_to_anchor=(1.03, 0.5), frameon=False, fontsize=6)
    for r in range(len(quants)):
        fig.text(0.01, axs[r, 0].get_position().y1 + 0.01, "ABC"[r], fontsize=10, fontweight="bold", va="bottom")
    out = os.path.join(S.OUT_ROOT, "figures")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"supp_cell_weights.{ext}"), dpi=200)
    print("wrote", os.path.join(out, "supp_cell_weights.pdf"))


if __name__ == "__main__":
    {"compute": compute, "summary": summary, "plot": plot}[sys.argv[1]]()
