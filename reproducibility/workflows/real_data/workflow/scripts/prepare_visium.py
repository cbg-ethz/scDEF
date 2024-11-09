import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import anndata as ad
import scanpy as sc
import squidpy as sq

np.random.seed(seed)
spatial_adata = sc.read_visium(
    params.data_fname,
    count_file="CytAssist_FFPE_Human_Breast_Cancer_filtered_feature_bc_matrix.h5",
)

spatial_adata.obs_names_make_unique()
spatial_adata.var_names_make_unique()

# mitochondrial genes, "MT-" for human, "Mt-" for mouse
spatial_adata.var["mt"] = spatial_adata.var_names.str.startswith("MT-")
# ribosomal genes
spatial_adata.var["ribo"] = spatial_adata.var_names.str.startswith(("RPS", "RPL"))
# hemoglobin genes
spatial_adata.var["hb"] = spatial_adata.var_names.str.contains("^HB[^(P)]")

sc.pp.calculate_qc_metrics(
    spatial_adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True
)

spatial_adata = spatial_adata[spatial_adata.obs["pct_counts_mt"] < 20].copy()
sc.pp.filter_genes(spatial_adata, min_cells=10)

spatial_adata.layers["counts"] = spatial_adata.X.todense()
sc.pp.normalize_total(spatial_adata, inplace=True)
sc.pp.log1p(spatial_adata)
sc.pp.highly_variable_genes(spatial_adata, n_top_genes=params.n_top_genes)

sc.pp.pca(spatial_adata)
sc.pp.neighbors(spatial_adata)
sc.tl.umap(spatial_adata)
sc.tl.leiden(
    spatial_adata, key_added="clusters", flavor="igraph", directed=False, n_iterations=2
)

# Write new h5ad file
adata.write(output.fname)
