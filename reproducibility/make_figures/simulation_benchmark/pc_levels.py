"""Principal components and the levels of the simulated hierarchy (R1.3b; pre-declared 2026-11-11 in
scDEF-markdown/experiments/exp-simulation-benchmark-rerun.md, "Principal components and the levels of the hierarchy").

    python pc_levels.py          (env: scdef, CPU)  -> out/simulation_benchmark/pc_levels.csv, figures/pc_levels.{pdf,png}

For each of the 30 benchmark data sets, the PCA embedding the benchmark scored (methods/PCA/latent0.npy, 50
components). For each PC, eta squared (between-group over total variance) of the 2-group labels (GroupA), and the
further share explained by the 4-group (GroupB - GroupA) and 8-group labels (GroupC - GroupB); with four batches, eta
squared of the batch. Plotted as means over the five replicates.
"""
import os

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import simcommon as S  # noqa: E402

N_PC = 50


def eta2(x, labels):
    """Between-group sum of squares over the total, for every column of x."""
    x = x - x.mean(0)
    tot = (x ** 2).sum(0)
    between = np.zeros(x.shape[1])
    for g in np.unique(labels):
        sel = labels == g
        between += sel.sum() * x[sel].mean(0) ** 2
    return between / tot


def compute():
    rows = []
    for ds in S.all_ids():
        p = S.parse_id(ds)
        d = S.ds_dir(ds)
        obs = ad.read_h5ad(os.path.join(d, "input.h5ad"), backed="r").obs
        z = np.load(os.path.join(d, "methods", "PCA", "latent0.npy"))
        assert z.shape[0] == len(obs), (ds, z.shape, len(obs))
        e = {tl: eta2(z, obs[tl].astype(str).to_numpy()) for tl in S.HIERARCHY_OBS}
        shares = {"2 groups": e["GroupA"], "4 groups": e["GroupB"] - e["GroupA"], "8 groups": e["GroupC"] - e["GroupB"]}
        if p["multi"]:
            shares["batch"] = eta2(z, obs["Batch"].astype(str).to_numpy())
        for name, v in shares.items():
            for k in range(z.shape[1]):
                rows.append(dict(id=ds, scenario=p["scenario"], similarity=p["similarity"], rep=p["rep"], pc=k + 1,
                                 labels=name, share=float(v[k])))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(S.OUT_ROOT, "pc_levels.csv"), index=False)
    return df


COLORS = {"2 groups": "0.1", "4 groups": "0.45", "8 groups": "0.72", "batch": "#d62728"}


def plot(df):
    plt.rcParams.update({"font.size": 6.5, "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6,
                         "axes.titlesize": 7, "axes.linewidth": 0.5, "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    scen = list(S.SCENARIOS)
    sims = ["Low", "High"]
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 3.6), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.92, bottom=0.2, hspace=0.3, wspace=0.08)
    for r, sim in enumerate(sims):
        for c, sc in enumerate(scen):
            ax = axs[r, c]
            sub = df[(df["scenario"] == sc) & (df["similarity"] == sim)]
            for name, col in COLORS.items():
                m = sub[sub["labels"] == name].groupby("pc")["share"].mean()
                if len(m):
                    ax.plot(m.index, m.values, color=col, lw=1.0, ls="--" if name == "batch" else "-", label=name)
            ax.set_xlim(0.5, 20.5)
            ax.set_ylim(-0.02, 1.0)
            ax.set_xticks([1, 5, 10, 15, 20])
            if r == 0:
                ax.set_title(S.SCENARIOS[sc]["label"], pad=4)
            if c == 0:
                ax.set_ylabel(f"{sim} group similarity\nshare of PC variance")
            if r == 1:
                ax.set_xlabel("Principal component")
    handles, labels = axs[0, 2].get_legend_handles_labels()
    fig.legend(handles, ["2-group labels", "4-group labels, beyond the 2 groups", "8-group labels, beyond the 4 groups",
                         "batch"], loc="lower center", ncol=4, frameon=False, fontsize=6.3)
    out = os.path.join(S.OUT_ROOT, "figures")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"pc_levels.{ext}"), dpi=200)
    print("wrote", os.path.join(out, "pc_levels.pdf"))


if __name__ == "__main__":
    plot(compute())
