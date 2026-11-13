"""The kappa grid (pre-declared 2026-11-10 in scDEF-markdown/experiments/exp-simulation-benchmark-rerun.md,
"Pre-declared 2026-11-10", item 4).

    python kappa_grid.py fit [<id> ...]        (env: scdef, GPU)   all ids and weights by default; resumable
    python kappa_grid.py score                 (env: scdef, CPU)   -> out/simulation_benchmark/kappa_grid.csv

The strength of the hierarchy is ``alpha`` = median library size / number of layer-0 factors * ``hierarchy_weight``
(scDEF.update_model_priors; default weight 1). Each data set is fitted with the fixed workflow
(refit_best_of_seeds.fit_fixed: fit, FIXED_FIRST filter, refit, FIXED_SECOND filter) with the batch annotations,
seed 42 only, at every weight in WEIGHTS. Saved to <id>/kappa/w<weight>/{fit.json, model/}.

Read from each fit (the batch-corrected model): per layer, ARI of the layer assignments (obs[layer]) to each true
level (GroupC, GroupB, GroupA) and to the batches; at layer 0, under the agreed protocol (realdata_benchmark/score.py,
"clustering protocol", proportions of the weights): cell-type ARI (KMeans, k = 8), cell-type and batch ASW, and
specific and shared groups kept (Leiden 1.0; score.batch_groups); the layer sizes and the fitted alpha.
"""
import json
import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import simcommon as S  # noqa: E402

IDS = [f"distinct_fs{fs}_r{r}" for fs in ("0.5", "2.0") for r in (0, 1, 2)]
WEIGHTS = [0.1, 0.3, 1.0, 3.0, 10.0]
SEED = 42


def wdir(ds, w):
    return S.ds_dir(ds, "kappa", f"w{w:g}")


def fit(ds, w):
    import jax

    sys.path.insert(0, S.MAKE_FIGURES)
    import refit_best_of_seeds as rb
    import scdef as scd
    from fit_scdef import save

    d = wdir(ds, w)
    os.makedirs(d, exist_ok=True)
    log_path = os.path.join(d, "fit.json")
    if S.read_json(log_path, {}).get("status") == "ok" and os.path.isdir(os.path.join(d, "model")):
        print(f"skip {ds} w={w:g}: ok", flush=True)
        return
    log = dict(id=ds, hierarchy_weight=w, seed=SEED, batch_key="Batch", status="running",
               workflow="refit_best_of_seeds.fit_fixed steps with hierarchy_weight", device=str(jax.devices()[0]))
    S.write_json(log, log_path)
    a = S.scdef_adata(ds)
    t = time.time()
    m = scd.scDEF(a.copy(), counts_layer="counts", batch_key="Batch", seed=SEED, hierarchy_weight=w)
    m.fit(hierarchical_init=True)
    scd.tl.filter(m, batch_key="Batch", **rb.FIXED_FIRST)
    m.fit(hierarchical_init=True)
    scd.tl.filter(m, batch_key="Batch", **rb.FIXED_SECOND)
    log.update(status="ok", minutes=round((time.time() - t) / 60, 2), alpha=float(m.alpha),
               layer_sizes=[len(f) for f in m.factor_lists], final_loss=rb.final_loss(m))
    save(m, os.path.join(d, "model"))
    S.write_json(log, log_path)
    print(f"done {ds} w={w:g}: {log['minutes']} min, layers {log['layer_sizes']}, alpha {log['alpha']:.3g}", flush=True)


def score():
    import anndata as ad
    from sklearn.metrics import adjusted_rand_score

    import score as SC
    rs, scd = SC.rs, SC.scd
    rows = []
    for ds in IDS:
        p = S.parse_id(ds)
        d = S.ds_dir(ds)
        inp = ad.read_h5ad(os.path.join(d, "input.h5ad"))
        groups = pd.read_csv(os.path.join(d, "data", "groups.csv"))
        for w in WEIGHTS:
            fd = wdir(ds, w)
            log = S.read_json(os.path.join(fd, "fit.json"), {})
            if log.get("status") != "ok":
                continue
            a = S.scdef_adata(ds)
            m = scd.scDEF.load(os.path.join(fd, "model"), adata=a)
            if "L0" not in m.adata.obs:
                m.annotate_adata()
            base = dict(id=ds, similarity=p["similarity"], rep=p["rep"], hierarchy_weight=w, alpha=log["alpha"],
                        layer_sizes=json.dumps(log["layer_sizes"]))
            for k, ln in enumerate(m.layer_names[:-1]):
                z = m.adata.obs[ln].astype(str).to_numpy()
                for tl in S.HIERARCHY_OBS:
                    rows.append(dict(base, layer=k, metric=f"ARI to {tl}",
                                     value=float(adjusted_rand_score(inp.obs[tl].astype(str), z))))
                rows.append(dict(base, layer=k, metric="Batch ARI (layer assignments)",
                                 value=float(adjusted_rand_score(inp.obs["Batch"].astype(str), z))))
            pv, clusters, _ = rs.protocol_values(inp.obs, "GroupC", "Batch", "scDEF", latent=m.adata.obsm["X_L0"])
            for metric in ("Cell Type ARI", "Cell Type ASW", "Batch ASW", "Batch ARI"):
                rows.append(dict(base, layer=0, metric=metric, value=pv.get(metric)))
            bg = SC.batch_groups(inp.obs, clusters, groups)
            rows.append(dict(base, layer=0, metric="Specific groups kept", value=bg["specific_kept"]))
            rows.append(dict(base, layer=0, metric="Shared groups kept", value=bg["shared_kept"]))
            print(f"scored {ds} w={w:g}", flush=True)
    out = os.path.join(S.OUT_ROOT, "kappa_grid.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)


def plot():
    """Supplementary figure (fig:kappa_n_layers): against the multiplier of kappa, for four distinct batches with
    similar (left) and separated (right) groups: (A) the number of factors per layer, (B) batch ASW at layer 0,
    (C) cell-type ARI and ASW at layer 0, (D) batch-specific and shared groups kept at layer 0, (E) ARI to each true
    level, at the layer with the best ARI.
    Lines are means over the three replicates, dots the replicates."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 6.5, "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6,
                         "axes.titlesize": 7, "axes.linewidth": 0.5, "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    df = pd.read_csv(os.path.join(S.OUT_ROOT, "kappa_grid.csv"))
    sims = [("High", "High group similarity"), ("Low", "Low group similarity")]
    fig, axs = plt.subplots(5, 2, figsize=(7.2, 8.0), sharex=True)
    fig.subplots_adjust(left=0.1, right=0.8, top=0.96, bottom=0.06, hspace=0.35, wspace=0.25)
    green, light = "#2ca02c", "#98df8a"

    def line(ax, sub, metric, color, label, layer=None, how="mean"):
        d = sub[sub["metric"] == metric] if layer is None else sub[(sub["metric"] == metric) & (sub["layer"] == layer)]
        if how == "best":                                   # best layer per fit
            d = d.groupby(["id", "hierarchy_weight"], as_index=False)["value"].max()
        m = d.groupby("hierarchy_weight")["value"].mean()
        ax.plot(m.index, m.values, "-", color=color, lw=1.0, label=label)
        ax.plot(d["hierarchy_weight"], d["value"], "o", ms=2.2, color=color, mec="none", alpha=0.7)

    for c, (sim, title) in enumerate(sims):
        sub = df[df["similarity"] == sim]
        ax = axs[0, c]
        sizes = sub.drop_duplicates(["id", "hierarchy_weight"])[["id", "hierarchy_weight", "layer_sizes"]]
        for _, r in sizes.iterrows():
            ls = json.loads(r["layer_sizes"])[:-1]
            for k, n in enumerate(ls):
                ax.plot(r["hierarchy_weight"] * (1 + 0.06 * (k - 1)), n, "os^Dv"[k % 5], ms=2.4, mfc="white",
                        mec=["0.1", "0.35", "0.55", "0.7", "0.8"][k % 5], mew=0.6)
        for t in (8, 4, 2):
            ax.axhline(t, color="0.75", lw=0.5, ls="--", zorder=0)
        ax.set_yscale("log", base=2)
        ax.set_yticks([1, 2, 4, 8, 16, 32], ["1", "2", "4", "8", "16", "32"])
        ax.set_title(title, pad=4)
        line(axs[1, c], sub, "Batch ASW", green, "batch ASW", layer=0)
        line(axs[2, c], sub, "Cell Type ARI", "0.15", "cell-type ARI", layer=0)
        line(axs[2, c], sub, "Cell Type ASW", "0.6", "cell-type ASW", layer=0)
        axs[2, c].set_ylim(-0.03, 1.03)
        line(axs[3, c], sub, "Specific groups kept", green, "batch-specific", layer=0)
        line(axs[3, c], sub, "Shared groups kept", light, "shared", layer=0)
        axs[3, c].set_ylim(-0.2, 4.2)
        for tl, col, name in (("GroupC", "0.1", "8 groups"), ("GroupB", "0.45", "4 groups"), ("GroupA", "0.7", "2 groups")):
            line(axs[4, c], sub, f"ARI to {tl}", col, name, how="best")
        axs[4, c].set_ylim(-0.03, 1.03)
        axs[4, c].set_xscale("log")
        axs[4, c].set_xticks(WEIGHTS, [f"{w:g}" for w in WEIGHTS])
        axs[4, c].set_xlabel("multiplier of $\\kappa$ (1 = default)")
    axs[0, 0].set_ylabel("Factors per layer")
    axs[1, 0].set_ylabel("Batch ASW, layer 0")
    axs[2, 0].set_ylabel("Cell types, layer 0")
    axs[3, 0].set_ylabel("Groups kept, layer 0\n(of 4)")
    axs[4, 0].set_ylabel("ARI, best layer")
    from matplotlib.lines import Line2D
    axs[0, 1].legend(handles=[Line2D([], [], ls="none", marker="os^Dv"[k], ms=3, mfc="white",
                                     mec=["0.1", "0.35", "0.55", "0.7", "0.8"][k], label=f"layer {k}") for k in range(4)],
                     loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=6)
    for r in (2, 3, 4):
        axs[r, 1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=6)
    for r in range(5):
        fig.text(0.01, axs[r, 0].get_position().y1 + 0.005, "ABCDE"[r], fontsize=10, fontweight="bold", va="bottom")
    out = os.path.join(S.OUT_ROOT, "figures")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"kappa_grid.{ext}"), dpi=200)
    print("wrote", os.path.join(out, "kappa_grid.pdf"))


if __name__ == "__main__":
    if sys.argv[1] == "fit":
        for ds in (sys.argv[2:] or IDS):
            for w in WEIGHTS:
                try:
                    fit(ds, w)
                except Exception as e:  # keep going; the log records the failure
                    S.write_json(dict(id=ds, hierarchy_weight=w, status="failed", error=repr(e)),
                                 os.path.join(wdir(ds, w), "fit.json"))
                    print(f"FAILED {ds} w={w:g}: {e!r}", flush=True)
    elif sys.argv[1] == "score":
        score()
    elif sys.argv[1] == "plot":
        plot()
