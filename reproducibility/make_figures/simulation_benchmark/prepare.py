"""Build one simulated data set's shared benchmark input, check that its three group levels nest, and write the
level-wise ground truth.

    python prepare.py <id> [--force]          (env: scdef)

As the 2024 ``workflow/scripts/prepare_input.py``: counts from simulate.R, ``filter_cells(min_genes=200)``,
``filter_genes(min_cells=3)``, the hierarchy levels GroupC (8 groups), GroupB (4) and GroupA (2) prefixed "", "h"
and "hh", the 2024 Seurat top-10 markers of the GroupC groups, cells ordered by batch (stable sort; 2024 used
pandas' default sort). Cell names are splatter's (2024: the row number).

Nesting (asserted before anything is written): every GroupC group must lie in one GroupB group and every GroupB group
in one GroupA group. Cells outside their group's majority parent are counted; more than MAX_BOUNDARY in total
stops here with exit code 3 (no fix is attempted). The three splatSimulate calls share the seed and draw the group
labels first, so the levels are expected to nest exactly.

Ground truth (truth.json): for each level and group, the genes of that level's block (GroupA genes 1-1000, GroupB
1001-1500, GroupC 1501-1750) with splatter DE factor > 1 for the group, restricted to the genes kept by the 2024
filter, ordered by DE factor; the counts before and after the filter are recorded.

Writes input.h5ad (X = counts), markers.json, truth.json, nesting.json and input_info.json to <id>/.
"""
import argparse
import os
import sys
import time

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

import simcommon as S

MAX_BOUNDARY = 5        # "a handful" of cells outside their group's parent


def nesting(meta):
    """Per child level: for each child group, its cells per parent group; boundary = cells outside the majority."""
    out, total = {}, 0
    for child, parent in (("GroupC", "GroupB"), ("GroupB", "GroupA"), ("GroupC", "GroupA")):
        tab = pd.crosstab(meta[child].astype(str), meta[parent].astype(str))
        rows = {}
        for g, r in tab.iterrows():
            r = r[r > 0]
            rows[g] = dict(parent=str(r.idxmax()), cells=int(r.sum()), outside=int(r.sum() - r.max()),
                           parents={str(k): int(v) for k, v in r.items()})
        n_out = int(sum(v["outside"] for v in rows.values()))
        out[f"{child}->{parent}"] = dict(boundary_cells=n_out, groups=rows)
        if parent != "GroupA" or child != "GroupC":
            total += n_out
    return out, total


def truth(defacs, var_names):
    kept = set(var_names)
    res, counts = {}, {}
    for level, prefix in S.LEVEL_PREFIX.items():
        lo, hi = S.LEVEL_BLOCK[level]
        sub = defacs[defacs["level"] == level]
        nums = sub["gene"].str.replace("Gene", "").astype(int)
        if nums.min() != lo or nums.max() != hi:
            raise SystemExit(f"{level}: DE factors cover genes {nums.min()}-{nums.max()}, expected {lo}-{hi}")
        res[level], counts[level] = {}, {}
        for g, gs in sub.groupby("group"):
            up = gs[gs["de_factor"] > 1].sort_values("de_factor", ascending=False)
            up_kept = [x for x in up["gene"] if x in kept]
            res[level][prefix + g] = up_kept
            counts[level][prefix + g] = dict(up=int(len(up)), up_kept=len(up_kept),
                                             down=int((gs["de_factor"] < 1).sum()), genes=int(len(gs)))
    return res, counts


def main(ds, force=False):
    p = S.parse_id(ds)
    d = S.ds_dir(ds)
    path = os.path.join(d, "input.h5ad")
    if os.path.exists(path) and not force:
        print("exists", path)
        return 0
    t0 = time.time()
    dd = os.path.join(d, "data")
    meta = pd.read_csv(os.path.join(dd, "meta.csv"), index_col=0)
    defacs = pd.read_csv(os.path.join(dd, "defacs.csv"))
    mk = pd.read_csv(os.path.join(dd, "markers.csv"))

    nest, n_boundary = nesting(meta)
    S.write_json(dict(ok=n_boundary <= MAX_BOUNDARY, boundary_cells=n_boundary, max_boundary=MAX_BOUNDARY,
                      cells=int(len(meta)), **nest), os.path.join(d, "nesting.json"))
    summary = {k: v["boundary_cells"] for k, v in nest.items()}
    print(f"{ds} nesting: {summary} over {len(meta)} cells")
    if n_boundary > MAX_BOUNDARY:
        print(f"NESTING FAILED for {ds}: {n_boundary} cells outside their parent group; stopping", file=sys.stderr)
        return 3

    counts = pd.read_csv(os.path.join(dd, "counts.csv"), index_col=0)
    if list(counts.columns) != list(meta.index):
        raise SystemExit("counts columns and meta rows differ")
    if list(counts.index) != [f"Gene{i + 1}" for i in range(counts.shape[0])]:
        raise SystemExit("unexpected gene names in counts.csv")

    # 2024 prepare_input.py
    a = ad.AnnData(X=sp.csr_matrix(counts.values.T.astype(np.float32)), obs=meta.copy(),
                   var=pd.DataFrame(index=counts.index.astype(str)))
    a.obs_names = meta.index.astype(str)
    n0 = a.shape
    sc.pp.filter_cells(a, min_genes=200)
    sc.pp.filter_genes(a, min_cells=3)
    for obs, prefix in S.LEVEL_PREFIX.items():
        a.obs[obs] = pd.Categorical(prefix + a.obs[obs].astype(str))
    a.obs["Batch"] = pd.Categorical(a.obs["Batch"].astype(str))
    a.obs["total_counts"] = np.asarray(a.X.sum(1)).ravel().astype(float)
    a = a[a.obs.sort_values("Batch", kind="stable").index].copy()

    obs = a.obs[S.HIERARCHY_OBS + ["Batch", "total_counts"]].copy()
    obs["Group"] = a.obs["Group"].astype(str).to_numpy()
    out = ad.AnnData(X=a.X.copy(), obs=obs, var=pd.DataFrame(index=a.var_names.astype(str)))
    out.layers["counts"] = a.X.copy()
    out.uns["hierarchy_obs"] = S.HIERARCHY_OBS
    out.write_h5ad(path, compression="gzip")

    genes = set(out.var_names)
    markers = {str(g): mk.loc[mk["cluster"].astype(str) == str(g), "gene"].astype(str).tolist()
               for g in pd.unique(mk["cluster"].astype(str))}
    missing = {k: [g for g in v if g not in genes] for k, v in markers.items()}
    missing = {k: v for k, v in missing.items() if v}
    fine = set(out.obs["GroupC"].astype(str))
    S.write_json(dict(markers=markers, source="2024 Seurat FindAllMarkers on GroupC (simulate.R: only.pos, "
                      "min.pct 0.25, logfc 0.25, top 10 by avg_log2FC, p_val_adj < 0.05)",
                      missing_from_genes=missing, marker_keys_matching_no_label=sorted(set(markers) - fine),
                      types_without_markers=sorted(fine - set(markers))), os.path.join(d, "markers.json"))

    tr, tr_counts = truth(defacs, out.var_names)
    S.write_json(dict(up=tr, counts=tr_counts, rule="genes of the level's block with splatter DE factor > 1 for "
                      "the group, among the genes kept by the 2024 filter, ordered by DE factor"),
                 os.path.join(d, "truth.json"))

    info = dict(id=ds, design={k: v for k, v in p.items() if k != "id"}, cells_genes_simulated=list(n0),
                n_cells=int(out.n_obs), n_genes=int(out.n_vars),
                cells_removed=int(n0[0] - out.n_obs), genes_removed=int(n0[1] - out.n_vars),
                label_counts={k: out.obs[k].value_counts().to_dict() for k in S.HIERARCHY_OBS},
                batch_counts=out.obs["Batch"].value_counts().to_dict(),
                groups_per_batch={b: sorted(set(out.obs.loc[out.obs["Batch"] == b, "GroupC"].astype(str)))
                                  for b in out.obs["Batch"].cat.categories},
                nesting_boundary_cells=summary,
                up_genes_per_level={lv: {g: c["up_kept"] for g, c in v.items()} for lv, v in tr_counts.items()},
                minutes=round((time.time() - t0) / 60, 2))
    S.write_json(info, os.path.join(d, "input_info.json"))
    print(f"wrote {path}: {out.n_obs} cells x {out.n_vars} genes; up genes per level "
          f"{ {lv: [c['up_kept'] for c in v.values()] for lv, v in tr_counts.items()} }; missing markers {missing}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("id")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    sys.exit(main(a.id, a.force))
