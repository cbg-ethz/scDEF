"""Planaria marker genes for signature accuracy, derived from the annotation (2026-11-05).

The 2024 marker file is only on the Euler cluster, and Table S3 of Plass et al. 2018 cannot be downloaded
automatically. The rule here is the one the 2024 simulations used for their true markers (Seurat
``FindAllMarkers`` defaults, top 10 by fold change; see ``workflow/scripts/splatter_hierarchical.R``),
applied to the 39 benchmark types of the shared input. Defined in the vault before computing
(experiments/exp-realdata-benchmark-local.md, "Planaria marker genes").

    python derive_planaria_markers.py          (env: scdef or scdef-bench2; after prepare.py planaria)

Writes out/realdata_benchmark/planaria/derived_markers.csv in the 2024 format (index: gene,
column ``clusters_neoblasts``: the type), plus all tested statistics in derived_markers_full.csv.
"""
import os

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import mannwhitneyu

import common as C

LOGFC_MIN, PCT_MIN, TOP, PADJ_MAX = 0.25, 0.25, 10, 0.05


def main():
    d = C.out_dir("planaria")
    a = ad.read_h5ad(os.path.join(d, "input.h5ad"))                  # X = counts
    X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
    lib = np.asarray(X.sum(1)).ravel()
    norm = sp.diags(1e4 / lib) @ X                                     # normalised counts
    logn = norm.copy()
    logn.data = np.log1p(logn.data)                                    # the data the test runs on
    y = a.obs["Cell types"].astype(str).to_numpy()
    genes = a.var_names.astype(str).to_numpy()
    n_genes = len(genes)
    rows = []
    for t in sorted(set(y)):
        g, r = y == t, y != t
        mean_g = np.asarray(norm[g].mean(0)).ravel()
        mean_r = np.asarray(norm[r].mean(0)).ravel()
        lfc = np.log2(mean_g + 1) - np.log2(mean_r + 1)                 # Seurat: log2 of mean(expm1) + 1
        pct_g = np.asarray((X[g] > 0).mean(0)).ravel()
        pct_r = np.asarray((X[r] > 0).mean(0)).ravel()
        keep = np.where((lfc >= LOGFC_MIN) & (np.maximum(pct_g, pct_r) >= PCT_MIN))[0]   # only.pos, thresholds
        Lg, Lr = logn[g][:, keep].toarray(), logn[r][:, keep].toarray()
        p = mannwhitneyu(Lg, Lr, axis=0, alternative="two-sided").pvalue if len(keep) else np.array([])
        padj = np.minimum(p * n_genes, 1.0)                            # Bonferroni over all genes, as Seurat
        for j, k in enumerate(keep):
            rows.append(dict(gene=genes[k], clusters_neoblasts=t, avg_log2FC=lfc[k], pct_1=pct_g[k], pct_2=pct_r[k],
                             p_val=p[j], p_val_adj=padj[j], n_cells=int(g.sum())))
        print(f"{t}: {len(keep)} genes tested", flush=True)
    full = pd.DataFrame(rows)
    full.to_csv(os.path.join(d, "derived_markers_full.csv"), index=False)
    # slice_max(n = 10, order_by = avg_log2FC) per type, then filter(p_val_adj < 0.05), as the 2024 R script
    top = (full.sort_values(["clusters_neoblasts", "avg_log2FC"], ascending=[True, False])
               .groupby("clusters_neoblasts").head(TOP))
    top = top[top["p_val_adj"] < PADJ_MAX]
    top.set_index("gene")[["clusters_neoblasts"]].to_csv(os.path.join(d, "derived_markers.csv"))
    per = top.groupby("clusters_neoblasts").size()
    missing = sorted(set(y) - set(per.index))
    print(f"wrote derived_markers.csv: {len(top)} rows; markers per type {per.min()}-{per.max()} "
          f"(median {int(per.median())}); types without markers: {missing}")


if __name__ == "__main__":
    main()
