"""Build one data set's shared benchmark input: the cells and genes the saved scDEF models were fitted on.

    python prepare.py <data>          # pbmcs3k | planaria | pbmcs2b | pbmcsifn   (env: scdef)

Cells and genes come from ``refit_best_of_seeds.tutorial_adata`` (the fitted models' data), raw counts in
``layers['counts']``; the label columns, the coarse level and the marker lists are those of the 2024
benchmark's prepare_<data>.py. Writes ``input.h5ad`` (X = counts), ``markers.json`` and ``input_info.json``
to out/realdata_benchmark/<data>/. Skips if ``input.h5ad`` exists (``--force`` rebuilds).
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

import common as C


def ci_map(values, mapping):
    """Map labels through a 2024 dictionary ignoring case (the IFN annotation spells 'B Activated', the
    2024 dictionaries 'B activated'); labels without an entry are kept."""
    low = {k.lower(): v for k, v in mapping.items()}
    return values.map(lambda c: low.get(c.lower(), c))


def match_markers(markers, fine_labels):
    """Re-key the marker lists to the labels' own spelling (case-insensitive); keys matching no label are
    dropped and reported, since evaluate.py looks every marker key up among the labels."""
    by_low = {c.lower(): c for c in fine_labels}
    kept = {by_low[k.lower()]: v for k, v in markers.items() if k.lower() in by_low}
    dropped = sorted(k for k in markers if k.lower() not in by_low)
    return kept, dropped


def labels(data, a):
    """The 2024 hierarchy levels: level 0 unprefixed, level 1 prefixed with 'h' (prepare_*.py)."""
    obs = a.obs
    if data == "pbmcs3k":
        fine = obs["Cell types"].astype(str)
        coarse = fine.map(C.coarse_3k)
    elif data == "planaria":
        fine = obs["clusters_neoblasts"].astype(str)
        coarse = obs["clusters"].astype(str).map(C.coarse_planaria)
    elif data == "pbmcs2b":
        fine = obs["CellType"].astype(str)
        coarse = ci_map(fine, C.COARSE_2B)
    else:
        ann = obs["seurat_annotations"].astype(str)
        fine = pd.Series([f"{c}_{s}" if "CD14 Mono" in c else c for c, s in zip(ann, obs["stim"].astype(str))],
                         index=obs.index)
        coarse = ci_map(ann, C.COARSE_IFN)
    lv = C.DATASETS[data]["hierarchy_obs"]
    return {lv[0]: fine.to_numpy(), lv[1]: ("h" + coarse).to_numpy()}


def planaria_markers(a):
    """Markers per fine type from the 2024 marker file, if present (transcript ids mapped to var_names)."""
    path = C.PLANARIA_MARKERS
    if not os.path.exists(path):
        return {}, f"planaria marker file not found ({path}); Signature accuracy left empty"
    mk = pd.read_csv(path, index_col=0)
    tx2name = dict(zip(a.var["transcript_id"].astype(str), a.var_names.astype(str)))
    out = {}
    for ct in pd.unique(a.obs["clusters_neoblasts"].astype(str)):
        genes = mk.index[mk["clusters_neoblasts"].astype(str) == ct].astype(str)
        if len(genes):                  # a type without markers is left out (evaluate.py divides by the list length)
            out[ct] = [tx2name.get(g, g) for g in genes]
    return out, f"from {os.path.relpath(path)}"


def main(data, force=False):
    data = C.canonical(data)
    spec = C.DATASETS[data]
    d = C.out_dir(data)
    path = os.path.join(d, "input.h5ad")
    if os.path.exists(path) and not force:
        print("exists", path)
        return
    t0 = time.time()
    rb = C.refit_module()
    a = rb.tutorial_adata(spec["refit"])
    lab = labels(data, a)
    if data == "planaria":
        markers, marker_source = planaria_markers(a)
    else:
        markers, marker_source = C.MARKERS[data], "2024 prepare_%s.py" % data

    import anndata as ad
    counts = a.layers["counts"]
    counts = sp.csr_matrix(counts.astype(np.float32))
    obs = pd.DataFrame(index=a.obs_names.astype(str))
    for k, v in lab.items():
        obs[k] = pd.Categorical(v)
    if spec["batch"]:
        obs["Batch"] = pd.Categorical(a.obs[spec["batch"]].astype(str).to_numpy())
    for k in ("total_counts", "pct_counts_mt", "n_counts"):       # covariates of the 2024 scaled input
        if k in a.obs:
            obs[k] = a.obs[k].to_numpy().astype(float)
    if "total_counts" not in obs:
        obs["total_counts"] = np.asarray(counts.sum(1)).ravel()
    out = ad.AnnData(X=counts, obs=obs, var=pd.DataFrame(index=a.var_names.astype(str)))
    out.layers["counts"] = counts.copy()
    if "transcript_id" in a.var:
        out.var["transcript_id"] = a.var["transcript_id"].astype(str).to_numpy()
    out.uns["hierarchy_obs"] = spec["hierarchy_obs"]
    out.write_h5ad(path, compression="gzip")

    genes = set(out.var_names)
    markers, dropped = match_markers(markers, set(obs[spec["hierarchy_obs"][0]].astype(str)))
    missing = {ct: [g for g in gs if g not in genes] for ct, gs in markers.items()}
    missing = {ct: gs for ct, gs in missing.items() if gs}
    fine_key = spec["hierarchy_obs"][0]
    C.write_json(dict(markers=markers, source=marker_source, missing_from_genes=missing,
                      marker_keys_matching_no_label=dropped,
                      types_without_markers=sorted(set(obs[fine_key].astype(str)) - set(markers))),
                 os.path.join(d, "markers.json"))
    info = dict(data=data, refit_name=spec["refit"], n_cells=int(out.n_obs), n_genes=int(out.n_vars),
                source="refit_best_of_seeds.tutorial_adata('%s')" % spec["refit"],
                hierarchy_obs=spec["hierarchy_obs"], batch_column=spec["batch"],
                label_counts={k: obs[k].value_counts().to_dict() for k in spec["hierarchy_obs"]},
                batch_counts=obs["Batch"].value_counts().to_dict() if spec["batch"] else None,
                marker_source=marker_source, minutes=round((time.time() - t0) / 60, 2))
    C.write_json(info, os.path.join(d, "input_info.json"))
    print(f"wrote {path}: {out.n_obs} cells x {out.n_vars} genes; missing markers: {missing}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("data")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    main(args.data, args.force)
