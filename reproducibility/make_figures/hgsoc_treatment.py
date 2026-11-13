"""Treatment analysis behind the HGSOC text.

- Layer 2, the shared view the decomposition keeps fixed: per patient, how the cells' layer-2 weights
  differ after NACT, without hard labels, against labels shuffled within patients; and the same
  question under three ways of labelling cells at layer 2, and with the mitochondrial read share held fixed.
- Layer 0 of the corrected fit: factors enriched in post-NACT or naive cells within patients.
- Transition paths between a naive- and a post-enriched factor under a shared ancestor (kept for the
  check across seeds; the figure, ``--figure``, is not in the paper).

Treatment is confounded with patient (EOC136 is almost all post-NACT, EOC443 and EOC733 mostly
treatment-naive), so every comparison is made within a patient and then counted across patients.

    python reproducibility/make_figures/hgsoc_treatment.py [--model ref|dec-top2] [--figure]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hgsoc_figures as hf  # noqa: E402

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scdef as scd

MIN_PHASE = 20      # a patient is tested when it has this many cells in each phase
MIN_CELLS = 10      # ... and a factor is tested in a patient when it holds this many of the patient's cells
LOG2_FC = 1.0       # enrichment: the factor's share of the patient's post-NACT cells is 2x its share of naive cells


def within_patient_enrichment(m, k):
    """log2(share of post-NACT cells in factor / share of naive cells in factor), per factor and patient."""
    obs = m.adata.obs
    col = obs[m.layer_names[k]].astype(str)
    rows = {}
    for p, o in obs.groupby("patient_id", observed=True):
        n = o["treatment_phase"].value_counts()
        if n.min() < MIN_PHASE or len(n) < 2:
            continue
        ct = pd.crosstab(col[o.index], o["treatment_phase"]).reindex(columns=hf.TREAT, fill_value=0)
        share = (ct + 0.5) / (n[hf.TREAT].to_numpy() + 0.5)
        lfc = np.log2(share["post-NACT"] / share["treatment-naive"])
        rows[p] = lfc.where(ct.sum(axis=1) >= MIN_CELLS)
    lfc = pd.DataFrame(rows).reindex([f for f in m.factor_names[k] if f in col.values])
    summary = pd.DataFrame({"tested": lfc.notna().sum(axis=1), "up": (lfc >= LOG2_FC).sum(axis=1),
                            "down": (lfc <= -LOG2_FC).sum(axis=1),
                            "post_share": pd.crosstab(col, obs["treatment_phase"], normalize="index")["post-NACT"],
                            "n_cells": col.value_counts()}).reindex(lfc.index)
    return lfc, summary


def recurrent(summary, direction):
    s = summary[(summary["tested"] >= 3)]
    other = "down" if direction == "up" else "up"
    return s[(s[direction] >= 3) & (s[direction] >= 2 * s[other])].index.tolist()


def null_recurrent(m, k, n_perm=100):
    """Recurrent factors found when treatment labels are shuffled within each patient."""
    obs = m.adata.obs
    phase = obs["treatment_phase"].copy()
    rng = np.random.default_rng(0)
    counts = []
    for _ in range(n_perm):
        shuffled = phase.copy()
        for _, idx in obs.groupby("patient_id", observed=True).indices.items():
            shuffled.iloc[idx] = rng.permutation(phase.iloc[idx].to_numpy())
        obs["treatment_phase"] = shuffled
        _, s = within_patient_enrichment(m, k)
        counts.append((len(recurrent(s, "up")), len(recurrent(s, "down"))))
    obs["treatment_phase"] = phase
    c = np.asarray(counts)
    return dict(post_mean=float(c[:, 0].mean()), post_max=int(c[:, 0].max()),
                naive_mean=float(c[:, 1].mean()), naive_max=int(c[:, 1].max()), n_perm=n_perm)


def upper_layer_shift(m, k=2, n_perm=500, min_consistent=8, exclude=(), same_site=()):
    """Treatment on the shared layer ``k`` without hard labels: per patient (at least ``MIN_PHASE`` cells
    per phase), log2 of the mean share of each factor in the cells' layer-``k`` weights, post-NACT over
    naive. Layer 2 is the anchor the decomposition keeps fixed, so this is the same in both fits."""
    obs = m.adata.obs
    names = list(m.factor_names[k])
    z = np.asarray(m.pmeans[f"{m.layer_names[k]}z"])[:, m.factor_lists[k]]
    share = z / z.sum(1, keepdims=True)
    keep = [i for i, f in enumerate(names) if f not in exclude]        # technical factors are left out
    names, share = [names[i] for i in keep], share[:, keep]
    pat = obs["patient_id"].astype(str).to_numpy()
    ph = obs["treatment_phase"].astype(str).to_numpy()

    def lfc(ph):
        rows = {}
        for p in np.unique(pat):
            mk = pat == p
            if (ph[mk] == "post-NACT").sum() >= MIN_PHASE and (ph[mk] == "treatment-naive").sum() >= MIN_PHASE:
                rows[p] = np.log2(share[mk & (ph == "post-NACT")].mean(0) / share[mk & (ph == "treatment-naive")].mean(0))
        return pd.DataFrame(rows, index=names)

    obs_l = lfc(ph)
    best = np.maximum((obs_l > 0).sum(1), (obs_l < 0).sum(1))
    rng = np.random.default_rng(0)
    null = []
    for _ in range(n_perm):
        sh = ph.copy()
        for p in np.unique(pat):
            mk = np.flatnonzero(pat == p)
            sh[mk] = rng.permutation(ph[mk])
        l = lfc(sh)
        null.append(int((np.maximum((l > 0).sum(1), (l < 0).sum(1)) >= min_consistent).sum()))
    n_obs = int((best >= min_consistent).sum())
    ss = [p for p in obs_l.columns if p in set(same_site)]
    return dict(excluded=list(exclude), same_site_patients=ss,
                higher_after_same_site=(obs_l[ss] > 0).sum(1).to_dict(), lower_after_same_site=(obs_l[ss] < 0).sum(1).to_dict(),
                patients=obs_l.columns.tolist(), log2_by_patient=obs_l.round(3).to_dict(),
                higher_after=(obs_l > 0).sum(1).to_dict(), lower_after=(obs_l < 0).sum(1).to_dict(),
                median_log2=obs_l.median(1).round(3).to_dict(), min_consistent=min_consistent,
                n_consistent=n_obs, null_mean=float(np.mean(null)), null_p=float(np.mean(np.array(null) >= n_obs)),
                n_perm=n_perm)


def mito_share(a, cells):
    """Each cell's share of counts in mitochondrial genes (``MT-``), from the counts of the fitted genes."""
    b = a[cells]
    X = b.layers["counts"] if "counts" in b.layers else b.X
    mt = np.asarray(b.var_names.str.startswith("MT-"))
    tot = np.asarray(X.sum(1)).ravel()
    return np.asarray(X[:, mt].sum(1)).ravel() / np.maximum(tot, 1)


def upper_layer_shift_stratified(m, qc, k=2, n_bins=5, min_bin=5, exclude=()):
    """The layer-``k`` comparison with a per-cell QC covariate held fixed: within each patient, cells are split
    into ``n_bins`` quantile bins of ``qc``, the post-NACT minus naive mean weight share is taken in each bin with
    at least ``min_bin`` cells of each phase, and the bins are averaged (weighted by the smaller phase count)."""
    obs = m.adata.obs
    names = list(m.factor_names[k])
    z = np.asarray(m.pmeans[f"{m.layer_names[k]}z"])[:, m.factor_lists[k]]
    share = z / z.sum(1, keepdims=True)
    keep = [i for i, f in enumerate(names) if f not in exclude]
    names, share = [names[i] for i in keep], share[:, keep]
    pat = obs["patient_id"].astype(str).to_numpy()
    ph = obs["treatment_phase"].astype(str).to_numpy()
    rows = {}
    for p in np.unique(pat):
        mk = pat == p
        if min((mk & (ph == t)).sum() for t in hf.TREAT) < MIN_PHASE:
            continue
        edges = np.quantile(qc[mk], np.linspace(0, 1, n_bins + 1)[1:-1])
        b = np.digitize(qc, edges)
        diffs, wts = [], []
        for j in range(n_bins):
            post, naive = mk & (b == j) & (ph == "post-NACT"), mk & (b == j) & (ph == "treatment-naive")
            if min(post.sum(), naive.sum()) >= min_bin:
                diffs.append(share[post].mean(0) - share[naive].mean(0)); wts.append(min(post.sum(), naive.sum()))
        if diffs:
            rows[p] = np.average(np.stack(diffs), axis=0, weights=wts)
    d = pd.DataFrame(rows, index=names)
    return dict(n_bins=n_bins, min_bin=min_bin, patients=d.columns.tolist(),
                higher_after=(d > 0).sum(1).to_dict(), lower_after=(d < 0).sum(1).to_dict())


def upper_layer_label_rules(m, k=2):
    """Within-patient enrichment at layer ``k`` under three ways of labelling cells there."""
    from scdef.tools.factor import hard_assignment_name_indices, tree_assignment_name_indices
    obs = m.adata.obs
    names = list(m.factor_names[k])
    z = np.asarray(m.pmeans[f"{m.layer_names[k]}z"])[:, m.factor_lists[k]]
    rules = {"tree": tree_assignment_name_indices(m, k), "argmax": hard_assignment_name_indices(m, k),
             "scale-adjusted argmax": np.argmax(z / np.median(z, 0), 1)}
    keep = obs[m.layer_names[k]].copy()
    out = {}
    for rule, idx in rules.items():
        obs[m.layer_names[k]] = pd.Categorical([names[i] for i in idx])
        _, s = within_patient_enrichment(m, k)
        out[rule] = dict(post=recurrent(s, "up"), naive=recurrent(s, "down"))
    obs[m.layer_names[k]] = keep
    return out


def transitions(m, sources, targets):
    """Pairs of a naive- and a post-enriched factor under a shared ancestor: among the ancestor's cells,
    do a patient's post-NACT cells sit further towards the post-enriched factor than its naive cells?

    The cell weights are diffuse (a cell's top layer-0 factor holds a few percent of its layer-0 weight),
    so a path's absolute affinity cannot say which cells are on it; the cells assigned to the shared
    ancestor are used instead, and the position (normalised within the path) orders them.
    """
    if not sources or not targets:
        return []
    scd.tl.build_transition_paths(m, sources=sources, targets=targets, max_path_len=5, max_paths_per_pair=5)
    layer_of = {f: k for k, names in enumerate(m.factor_names) for f in names}

    def rises_then_falls(nodes):
        ls = [layer_of[f] for f in nodes]
        top = int(np.argmax(ls))
        return all(np.diff(ls[:top + 1]) == 1) and all(np.diff(ls[top:]) == -1)

    best = {}
    for p in m.adata.uns["transition_paths"]["paths"]:
        pair = (p["nodes"][0], p["nodes"][-1])
        if pair not in best and rises_then_falls(p["nodes"]):
            best[pair] = p
    m.adata.uns["transition_paths"]["paths"] = list(best.values())
    scd.tl.score_paths(m, paths_key="transition_paths")
    obs = m.adata.obs
    pos = m.adata.obsm["transition_paths_positions"]
    out = []
    for i, p in enumerate(m.adata.uns["transition_paths"]["paths"]):
        ancestor = max(p["nodes"], key=lambda f: layer_of[f])
        on = (obs[m.layer_names[layer_of[ancestor]]].astype(str) == ancestor).to_numpy()
        per = {}
        for pt, o in obs[on].groupby("patient_id", observed=True):
            idx = obs.index.get_indexer(o.index)
            ph = o["treatment_phase"].astype(str).to_numpy()
            if (ph == "post-NACT").sum() >= MIN_CELLS and (ph == "treatment-naive").sum() >= MIN_CELLS:
                per[pt] = {t: round(float(np.median(pos[idx][ph == t, i])), 4) for t in hf.TREAT}
        shift = {k: v["post-NACT"] - v["treatment-naive"] for k, v in per.items()}
        out.append(dict(path_id=int(p.get("path_id", i)), nodes=list(p["nodes"]), ancestor=ancestor,
                        ancestor_layer=m.layer_names[layer_of[ancestor]],
                        n_cells_on_path=int(on.sum()), patients_tested=len(per),
                        patients_post_further=int(sum(v > 0 for v in shift.values())),
                        median_shift_by_patient={k: round(v, 3) for k, v in shift.items()},
                        median_position_by_patient=per))
    return out


def transition_figure(m, t, out):
    """Supplementary figure: the path heatmap over the shared ancestor's cells, and each patient's median
    position of treatment-naive and post-NACT cells along the path."""
    patients = list(m.adata.obs["patient_id"].cat.categories)
    pcol = dict(zip(patients, m.adata.uns["patient_id_colors"]))
    fig = scd.pl.path_trajectory_heatmap(m, t["path_id"], paths_key="transition_paths", genes_per_factor=3,
                                         min_sort_affinity=0.0, subset_obs_key=t["ancestor_layer"],
                                         subset_obs=t["ancestor"], figwidth=7.2, gene_height=0.17,
                                         ytick_prefix_layer=False,
                                         annotation_obs_key=["treatment_phase", "patient_id"], show=False)
    for ax in fig.axes:                     # each annotation track draws its own legend over the track above
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        ax.tick_params(labelsize=6)
        names = {"treatment_phase": "treatment", "patient_id": "patient"}
        labels = [tl.get_text() for tl in ax.get_yticklabels()]
        if len(labels) == 1 and labels[0] in names:
            ax.set_yticks(ax.get_yticks(), [names[labels[0]]])
        ax.yaxis.label.set_size(6.5)
        ax.xaxis.label.set_size(6.5)
    if fig._suptitle is not None:
        fig._suptitle.set_text("")
    for txt in list(fig.texts):
        txt.remove()
    w, h = fig.get_size_inches()
    extra = 2.1
    fig.set_size_inches(w, h + extra)
    fig.subplots_adjust(top=0.965, bottom=(extra + 0.25) / (h + extra), left=0.13, right=0.9)
    hm = [ax for ax in fig.axes if ax.get_subplotspec() is not None]
    y0 = (extra - 0.35) / (h + extra)
    axb = fig.add_axes([0.13, 0.05, 0.22, y0 - 0.06])
    per = t["median_position_by_patient"]
    for pt in patients:
        if pt in per:
            v = [per[pt][x] for x in hf.TREAT]
            axb.plot([0, 1], v, "-o", color=pcol[pt], lw=0.9, ms=3)
    axb.set_xticks([0, 1], ["naive", "post-NACT"], fontsize=6)
    axb.set_xlim(-0.3, 1.3)
    axb.set_ylabel("median position\nalong the path", fontsize=6)
    axb.tick_params(axis="y", labelsize=5.5)
    for side in ("top", "right"):
        axb.spines[side].set_visible(False)
    h1 = [plt.Line2D([], [], marker="s", ls="", ms=4, mfc=hf.TCOL[x], mec="none") for x in hf.TREAT]
    h2 = [plt.Line2D([], [], marker="o", ls="", ms=3.5, mfc=pcol[p], mec="none") for p in patients]
    fig.legend(h1 + h2, hf.TREAT + patients, loc="lower left", ncol=4, frameon=False, fontsize=6,
               bbox_to_anchor=(0.45, 0.05), handletextpad=0.2, columnspacing=0.9)
    for ax, letter in ((hm[0], "A"), (axb, "B")):
        bb = ax.get_position()
        fig.text(0.02, bb.y1 + 0.004, letter, fontsize=10, fontweight="bold", va="bottom")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"supp_hgsoc_transition.{ext}"), dpi=300)
    plt.close(fig)


def main(model, out, figure=False, null=True):
    a = hf.data()
    pcol = hf.patient_colors(list(a.obs["patient_id"].cat.categories))
    name = {"ref": "scdef-hgsoc-ref-best5", "dec-top2": "scdef-hgsoc-dec-top2-best5"}.get(model, model)
    m = hf.load(name, a, pcol)
    res = dict(model=name, thresholds=dict(min_phase=MIN_PHASE, min_cells=MIN_CELLS, log2_fc=LOG2_FC))
    patients = a.obs.groupby("patient_id", observed=True)["treatment_phase"].value_counts().unstack()
    res["cells_by_patient"] = patients.astype(int).to_dict("index")
    for k in hf.LAYERS:
        lfc, s = within_patient_enrichment(m, k)
        res[f"L{k}"] = dict(summary=s.round(3).to_dict("index"), post=recurrent(s, "up"), naive=recurrent(s, "down"),
                            null=null_recurrent(m, k) if null else None)
    site = a.obs.groupby("patient_id", observed=True)["anatomical_location"].nunique()
    res["site_by_patient_phase"] = a.obs.groupby(["patient_id", "treatment_phase"], observed=True)[
        "anatomical_location"].agg(lambda x: x.astype(str).mode().iloc[0]).unstack().to_dict("index")
    # The technical labels are names of the kept fit's factors; in another fit the same names are other factors.
    tech = hf.TECHNICAL if name == "scdef-hgsoc-ref-best5" else ()
    res["technical_excluded"] = list(tech)
    res["L2_shift"] = upper_layer_shift(m, 2, exclude=tech, same_site=site[site == 1].index.tolist())
    qc = mito_share(a, m.adata.obs.index)
    ph_ = m.adata.obs["treatment_phase"].astype(str).to_numpy(); pat_ = m.adata.obs["patient_id"].astype(str).to_numpy()
    med = {p: {t: float(np.median(qc[(pat_ == p) & (ph_ == t)])) for t in hf.TREAT if ((pat_ == p) & (ph_ == t)).sum() >= MIN_PHASE}
           for p in np.unique(pat_)}
    res["mito_share_by_patient_phase"] = med
    res["L2_shift_mito_stratified"] = upper_layer_shift_stratified(m, qc, 2, exclude=tech)
    res["L2_label_rules"] = upper_layer_label_rules(m, 2)
    sigs = m.get_signatures_dict(top_genes=10)
    res["top_genes"] = {f: list(sigs[f]) for k in hf.LAYERS for f in m.factor_names[k]}
    res["transitions"] = transitions(m, res["L0"]["naive"], res["L0"]["post"])
    rec = [t for t in res["transitions"] if t["patients_tested"] >= 3
           and t["patients_post_further"] >= max(3, 0.75 * t["patients_tested"])]
    res["recurrent_transitions"] = [t["nodes"] for t in rec]
    if figure and rec:
        best = max(rec, key=lambda t: (t["patients_post_further"] == t["patients_tested"], t["patients_tested"]))
        res["figure_path"] = best["nodes"]
        transition_figure(m, best, out)
    with open(os.path.join(out, f"hgsoc_treatment_{model}.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    for k in hf.LAYERS:
        print(f"L{k} post-enriched:", res[f"L{k}"]["post"], " naive-enriched:", res[f"L{k}"]["naive"],
              " shuffled:", res[f"L{k}"]["null"])
    sh = res["L2_shift"]
    print("layer 2, continuous weights:", {f: f"up {sh['higher_after'][f]}, down {sh['lower_after'][f]}" for f in sh["higher_after"]},
          f"| {sh['n_consistent']} factors in >= {sh['min_consistent']} patients one way; null mean {sh['null_mean']:.2f}, P {sh['null_p']:.3f}")
    print("layer 2, label rules:", res["L2_label_rules"])
    st = res["L2_shift_mito_stratified"]
    print("layer 2, mitochondrial share held fixed:", {f: f"up {st['higher_after'][f]}, down {st['lower_after'][f]}" for f in st["higher_after"]})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ref")
    ap.add_argument("--out", default=os.path.join(hf.HERE, "out"))
    ap.add_argument("--figure", action="store_true", help="also draw the treatment-branch figure (not in the paper)")
    ap.add_argument("--no-null", action="store_true")
    a_ = ap.parse_args()
    main(a_.model, a_.out, figure=a_.figure, null=not a_.no_null)
