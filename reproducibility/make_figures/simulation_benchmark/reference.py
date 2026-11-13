"""Write the pbmc3k reference counts that simulate.R gives to splatEstimate.

    python reference.py          (env: scdef; needs network once, or scanpy's cached copy)

The 2024 generator read SeuratData's ``pbmc3k`` (Seurat object of the 10x Genomics 3k PBMCs, 13,714 genes x
2,700 cells), which is not installable from conda. scanpy's ``sc.datasets.pbmc3k()`` is the same 10x matrix
(identical, entry for entry, to the filtered_gene_bc_matrices used by the 3k tutorial; checked 2026-11-05); it is
filtered here as Seurat's CreateSeuratObject(min.cells = 3, min.features = 200) made SeuratData's object, which
gives the same 13,714 x 2,700. Written as genes x cells MatrixMarket (integer counts) to out/simulation_benchmark/
reference/, with the gene and cell names and a JSON record.
"""
import os

import numpy as np
import scanpy as sc
import scipy.io
import scipy.sparse as sp

import simcommon as S


def main():
    if os.path.exists(S.REF_MTX):
        print("exists", S.REF_MTX)
        return
    os.makedirs(S.REF_DIR, exist_ok=True)
    sc.settings.datasetdir = S.REF_DIR
    a = sc.datasets.pbmc3k()
    x = sp.csr_matrix(a.X)
    genes_ok = np.asarray((x > 0).sum(0)).ravel() >= 3
    cells_ok = np.asarray((x > 0).sum(1)).ravel() >= 200
    a = a[cells_ok, genes_ok].copy()
    x = sp.csr_matrix(a.X).astype(np.int64)
    if x.shape != (2700, 13714):
        raise SystemExit(f"unexpected reference size {x.shape}; SeuratData's pbmc3k is 2700 x 13714")
    tmp = S.REF_MTX + ".tmp.mtx"
    scipy.io.mmwrite(tmp, x.T.tocoo(), field="integer")
    os.replace(tmp, S.REF_MTX)
    with open(os.path.join(S.REF_DIR, "pbmc3k_genes.txt"), "w") as f:
        f.write("\n".join(map(str, a.var_names)) + "\n")
    with open(os.path.join(S.REF_DIR, "pbmc3k_cells.txt"), "w") as f:
        f.write("\n".join(map(str, a.obs_names)) + "\n")
    S.write_json(dict(source="scanpy.datasets.pbmc3k() (10x Genomics 3k PBMCs, filtered_gene_bc_matrices hg19)",
                      filter="genes detected in >= 3 cells, cells with >= 200 genes (Seurat CreateSeuratObject "
                             "min.cells = 3, min.features = 200, as SeuratData's pbmc3k)",
                      shape_genes_x_cells=[int(x.shape[1]), int(x.shape[0])], total_counts=int(x.sum()),
                      scanpy=sc.__version__),
                 os.path.join(S.REF_DIR, "reference.json"))
    print("wrote", S.REF_MTX, x.shape)


if __name__ == "__main__":
    main()
