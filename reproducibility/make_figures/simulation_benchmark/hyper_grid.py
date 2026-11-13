"""Hyperparameter and structure sweeps of scDEF on simulated data (pre-declared 2026-11-11 in
scDEF-markdown/experiments/exp-sim-hyperparams.md and exp-sim-structure.md, before any fit was run). They re-run the
2024 sweeps (workflow/simulations/{hyperparams,structure}.smk) with the current workflow and scorer.

    python hyper_grid.py simulate              (env: scdef; simulate.R through the splatter env)  new data sets
    python hyper_grid.py fit [grid ...]        (env: scdef, GPU)  grids: tau mu layers factors; resumable
    python hyper_grid.py score                 (env: scdef, CPU)  -> out/simulation_benchmark/hyper_grid.csv
    python hyper_grid.py plot                                     -> figures/{brd_grid,structure_grid}.{pdf,png}

Grids (one parameter at a time, the others at their defaults; seed 42; the fixed workflow with the batch
annotations, as kappa_grid.py):
- tau (brd_strength) and mu (brd_mean) in {0.1, 1, 10, 100}, on four distinct batches with low group similarity
  and a share of differentially expressed genes (splatter de_prob) of 0.05, 0.2 (the benchmark's) and 0.5,
  replicates 0-2. As the 2024 hyperparams config (de_fscale 2, de_prob .05/.2/.5).
- layers (n_layers, the layers of the geometric schedule from layer 0 to the top; default 6) in {3, 4, 5, 6, 7},
  and factors (n_factors, the layer-0 size the schedule starts from; default 100) in {25, 50, 100, 200}, on four
  distinct batches with low and high similarity, replicates 0-2 (the kappa grid's data sets). One layer is the
  1-layer comparison of the benchmark (Supplementary Figure sims_one_layer).
The default value of each grid is fitted once per data set and shared by the grids of that data set.

Read from each fit (the batch-corrected model): the final layer sizes; for each true level, the best-ARI layer, its
ARI (level recovery) and the precision of the top 20 marker genes and AUPRC against the true up-regulated genes
(score.level_metrics with scd.tl.marker_genes, as the benchmark); at layer 0 under the agreed protocol, cell-type
ARI and ASW, batch ASW, and batch-specific and shared groups kept.
"""
import json
import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import simcommon as S  # noqa: E402

SEED = 42
REPS3 = [0, 1, 2]
DP_VALUES = [0.05, 0.2, 0.5]


def dp_id(dp, rep):
    return f"distinct_fs2.0_r{rep}" + ("" if dp == S.DE_PROB else f"_dp{dp:g}")


BRD_IDS = [dp_id(dp, r) for dp in DP_VALUES for r in REPS3]
STRUCT_IDS = [f"distinct_fs{fs}_r{r}" for fs in ("2.0", "0.5") for r in REPS3]
GRIDS = {   # name: (scDEF argument, values, default, data sets)
    "tau": ("brd_strength", [0.1, 1.0, 10.0, 100.0], 1.0, BRD_IDS),
    "mu": ("brd_mean", [0.1, 1.0, 10.0, 100.0], 1.0, BRD_IDS),
    "layers": ("n_layers", [3, 4, 5, 6, 7], 6, STRUCT_IDS),
    "factors": ("n_factors", [25, 50, 100, 200], 100, STRUCT_IDS),
}
INT_ARGS = {"n_layers", "n_factors"}


def fdir(ds, arg, value, default):
    return S.ds_dir(ds, "grid", "default" if value == default else f"{arg}{value:g}")


def jobs(grids):
    out = []
    for g in grids:
        arg, values, default, ids = GRIDS[g]
        for ds in ids:
            for v in values:
                out.append((g, ds, arg, v, default))
    return out


def simulate():
    import prepare
    import simulate as sim
    for ds in BRD_IDS:
        if S.parse_id(ds)["de_prob"] == S.DE_PROB:
            continue
        rc = sim.main(ds)
        if rc != 0:
            raise SystemExit(f"simulate {ds} failed ({rc})")
        rc = prepare.main(ds)
        if rc not in (0, None):
            raise SystemExit(f"prepare {ds} failed ({rc})")
        print("ready", ds, flush=True)


def fit(ds, arg, value, default):
    import jax

    sys.path.insert(0, S.MAKE_FIGURES)
    import refit_best_of_seeds as rb
    import scdef as scd
    from fit_scdef import save

    d = fdir(ds, arg, value, default)
    log_path = os.path.join(d, "fit.json")
    if S.read_json(log_path, {}).get("status") == "ok" and os.path.isdir(os.path.join(d, "model")):
        print(f"skip {ds} {arg}={value:g}: ok", flush=True)
        return
    kw = {} if value == default else {arg: int(value) if arg in INT_ARGS else float(value)}
    log = dict(id=ds, arg=arg, value=value, default=value == default, kwargs=kw, seed=SEED, batch_key="Batch",
               status="running", workflow="refit_best_of_seeds.fit_fixed steps", device=str(jax.devices()[0]))
    S.write_json(log, log_path)
    a = S.scdef_adata(ds)
    t = time.time()
    m = scd.scDEF(a.copy(), counts_layer="counts", batch_key="Batch", seed=SEED, **kw)
    m.fit(hierarchical_init=True)
    scd.tl.filter(m, batch_key="Batch", **rb.FIXED_FIRST)
    m.fit(hierarchical_init=True)
    scd.tl.filter(m, batch_key="Batch", **rb.FIXED_SECOND)
    log.update(status="ok", minutes=round((time.time() - t) / 60, 2),
               layer_sizes=[len(f) for f in m.factor_lists], final_loss=rb.final_loss(m))
    save(m, os.path.join(d, "model"))
    S.write_json(log, log_path)
    print(f"done {ds} {arg}={value:g}: {log['minutes']} min, layers {log['layer_sizes']}", flush=True)


def score():
    import anndata as ad

    import score as SC
    rs, scd = SC.rs, SC.scd
    rows, cache = [], {}
    for g, ds, arg, v, default in jobs(GRIDS):
        fd = fdir(ds, arg, v, default)
        log = S.read_json(os.path.join(fd, "fit.json"), {})
        if log.get("status") != "ok":
            continue
        p = S.parse_id(ds)
        base = dict(grid=g, id=ds, similarity=p["similarity"], de_prob=p["de_prob"], rep=p["rep"], arg=arg,
                    setting=v, default=v == default, layer_sizes=json.dumps(log["layer_sizes"]))
        if fd not in cache:                      # the default fit is shared by two grids
            d = S.ds_dir(ds)
            inp = ad.read_h5ad(os.path.join(d, "input.h5ad"))
            groups = pd.read_csv(os.path.join(d, "data", "groups.csv"))
            truth = S.read_json(os.path.join(d, "truth.json"))["up"]
            m = scd.scDEF.load(os.path.join(fd, "model"), adata=S.scdef_adata(ds))
            if "L0" not in m.adata.obs:
                m.annotate_adata()
            recs = []
            levels = [(ln, m.adata.obs[ln].astype(str).to_numpy()) for ln in m.layer_names]
            msig, msc = scd.tl.marker_genes(m)
            lm = SC.level_metrics(inp.obs, levels, msig, truth, gene_scores=msc, var_names=list(m.adata.var_names))
            for tl, r in lm.items():
                recs += [dict(level=tl, metric="Level recovery", value=r["recovery"]),
                         dict(level=tl, metric="Best layer", value=r["method_level_index"]),
                         dict(level=tl, metric="Level signatures", value=r["signatures"]),
                         dict(level=tl, metric="Level signatures AUPRC", value=r["signatures_auprc"])]
            pv, clusters, _ = rs.protocol_values(inp.obs, "GroupC", "Batch", "scDEF", latent=m.adata.obsm["X_L0"])
            for metric in ("Cell Type ARI", "Cell Type ASW", "Batch ASW"):
                recs.append(dict(level="layer 0", metric=metric, value=pv.get(metric)))
            bg = SC.batch_groups(inp.obs, clusters, groups)
            recs += [dict(level="layer 0", metric="Specific groups kept", value=bg["specific_kept"]),
                     dict(level="layer 0", metric="Shared groups kept", value=bg["shared_kept"])]
            cache[fd] = recs
            print(f"scored {ds} {arg}={v:g}", flush=True)
        rows += [dict(base, **r) for r in cache[fd]]
    out = os.path.join(S.OUT_ROOT, "hyper_grid.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)


LEVELS = (("GroupC", "0.1", "8 groups"), ("GroupB", "0.45", "4 groups"), ("GroupA", "0.7", "2 groups"))
GREEN, LIGHT = "#2ca02c", "#98df8a"
LAYER_MARK, LAYER_GREY = "os^Dv<", ["0.1", "0.3", "0.45", "0.6", "0.72", "0.82"]


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 6.5, "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6,
                         "axes.titlesize": 6.5, "axes.linewidth": 0.5, "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    return plt


def _column(axs, sub, xticks, xlabel, logx):
    """One column of a sweep figure: rows (A) factors per layer, (B) level recovery, (C) level signature AUPRC,
    (D) cell-type ARI and ASW at layer 0, (E) batch-specific and shared groups kept at layer 0, against the setting.
    Lines are means over the replicates and dots the replicates."""
    def line(ax, d, color, label):
        m = d.groupby("setting")["value"].mean()
        ax.plot(m.index, m.values, "-", color=color, lw=1.0, label=label)
        ax.plot(d["setting"], d["value"], "o", ms=2.0, color=color, mec="none", alpha=0.7)

    sizes = sub.drop_duplicates(["id", "setting"])
    for _, r in sizes.iterrows():
        ls = json.loads(r["layer_sizes"])[:-1]
        for k, n in enumerate(ls[:6]):
            x = r["setting"] * (1 + 0.07 * (k - 1)) if logx else r["setting"] + 0.12 * (k - 1)
            axs[0].plot(x, n, LAYER_MARK[k], ms=2.2, mfc="white", mec=LAYER_GREY[k], mew=0.6)
    for t in (8, 4, 2):
        axs[0].axhline(t, color="0.75", lw=0.5, ls="--", zorder=0)
    axs[0].set_yscale("log", base=2)
    axs[0].set_yticks([1, 2, 4, 8, 16, 32], ["1", "2", "4", "8", "16", "32"])
    for tl, col, name in LEVELS:
        line(axs[1], sub[(sub["level"] == tl) & (sub["metric"] == "Level recovery")], col, name)
        line(axs[2], sub[(sub["level"] == tl) & (sub["metric"] == "Level signatures AUPRC")], col, name)
    l0 = sub[sub["level"] == "layer 0"]
    line(axs[3], l0[l0["metric"] == "Cell Type ARI"], "0.15", "cell-type ARI")
    line(axs[3], l0[l0["metric"] == "Cell Type ASW"], "0.6", "cell-type ASW")
    line(axs[4], l0[l0["metric"] == "Specific groups kept"], GREEN, "batch-specific")
    line(axs[4], l0[l0["metric"] == "Shared groups kept"], LIGHT, "shared")
    axs[1].set_ylim(-0.03, 1.03)
    axs[2].set_ylim(0, 0.6)
    axs[3].set_ylim(-0.03, 1.03)
    axs[4].set_ylim(-0.2, 4.2)
    if logx:
        axs[-1].set_xscale("log")
    axs[-1].set_xticks(xticks, [f"{v:g}" for v in xticks])
    axs[-1].minorticks_off()
    axs[-1].set_xlabel(xlabel)


def _finish(fig, axs, name):
    from matplotlib.lines import Line2D
    labels = ["Factors per layer", "ARI, best layer", "Signature AUPRC,\nbest layer", "Cell types, layer 0",
              "Groups kept, layer 0\n(of 4)"]
    for r, lab in enumerate(labels):
        axs[r, 0].set_ylabel(lab)
        fig.text(0.01, axs[r, 0].get_position().y1 + 0.004, "ABCDE"[r], fontsize=10, fontweight="bold", va="bottom")
    axs[0, -1].legend(handles=[Line2D([], [], ls="none", marker=LAYER_MARK[k], ms=3, mfc="white", mec=LAYER_GREY[k],
                                      label=f"layer {k}") for k in range(6)],
                      loc="center left", bbox_to_anchor=(1.03, 0.5), frameon=False, fontsize=6)
    for r in (1, 3, 4):
        axs[r, -1].legend(loc="center left", bbox_to_anchor=(1.03, 0.5), frameon=False, fontsize=6)
    out = os.path.join(S.OUT_ROOT, "figures")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=200)
    print("wrote", os.path.join(out, f"{name}.pdf"))


def plot():
    """Supplementary figures: brd_grid (fig:brd_simulation): tau (left three columns) and mu (right three) against
    the share of differentially expressed genes; structure_grid (fig:n_factors): the number of layers (left two
    columns) and of layer-0 factors (right two), at high and low group similarity."""
    plt = _style()
    df = pd.read_csv(os.path.join(S.OUT_ROOT, "hyper_grid.csv"))

    fig, axs = plt.subplots(5, 6, figsize=(7.2, 7.4), sharey="row", sharex="col")
    fig.subplots_adjust(left=0.09, right=0.87, top=0.93, bottom=0.07, hspace=0.32, wspace=0.12)
    for c, (g, sym) in enumerate([("tau", "$\\tau$"), ("mu", "$\\mu$")]):
        for j, dp in enumerate(DP_VALUES):
            sub = df[(df["grid"] == g) & (np.isclose(df["de_prob"], dp))]
            ax = axs[:, 3 * c + j]
            _column(ax, sub, GRIDS[g][1], sym, True)
            ax[0].set_title(f"{dp * 100:g}% of genes DE", pad=3)
        x0, x1 = axs[0, 3 * c].get_position().x0, axs[0, 3 * c + 2].get_position().x1
        fig.text((x0 + x1) / 2, 0.965, f"{sym} (BRD prior; default 1)", ha="center", fontsize=7)
    _finish(fig, axs, "brd_grid")

    fig, axs = plt.subplots(5, 4, figsize=(7.2, 7.4), sharey="row", sharex="col")
    fig.subplots_adjust(left=0.09, right=0.85, top=0.93, bottom=0.07, hspace=0.32, wspace=0.14)
    for c, (g, xlabel, logx) in enumerate([("layers", "number of layers", False),
                                           ("factors", "initial layer-0 factors", True)]):
        for j, sim in enumerate(("High", "Low")):
            sub = df[(df["grid"] == g) & (df["similarity"] == sim)]
            ax = axs[:, 2 * c + j]
            _column(ax, sub, GRIDS[g][1], xlabel, logx)
            ax[0].set_title(f"{sim} group similarity", pad=3)
        x0, x1 = axs[0, 2 * c].get_position().x0, axs[0, 2 * c + 1].get_position().x1
        title = "Number of layers (default 6)" if g == "layers" else "Initial number of layer-0 factors (default 100)"
        fig.text((x0 + x1) / 2, 0.965, title, ha="center", fontsize=7)
    _finish(fig, axs, "structure_grid")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "simulate":
        simulate()
    elif mode == "fit":
        for g, ds, arg, v, default in jobs(sys.argv[2:] or list(GRIDS)):
            try:
                fit(ds, arg, v, default)
            except Exception as e:  # keep going; the log records the failure
                S.write_json(dict(id=ds, arg=arg, value=v, status="failed", error=repr(e)),
                             os.path.join(fdir(ds, arg, v, default), "fit.json"))
                print(f"FAILED {ds} {arg}={v:g}: {e!r}", flush=True)
    elif mode == "score":
        score()
    elif mode == "plot":
        plot()
