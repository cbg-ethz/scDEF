import anndata
import scanpy as sc
import numpy as np
import pandas as pd

np.random.seed(snakemake.params.seed)
adata = anndata.read_h5ad(snakemake.params.data_fname)

# Remove some genes
for gene in snakemake.params.genes_to_remove:
    adata = adata[:, adata.var_names != gene]
sc.pp.filter_cells(adata, min_genes=snakemake.params.min_genes)
sc.pp.filter_genes(adata, min_cells=snakemake.params.min_cells)
adata.var["mt"] = adata.var_names.str.startswith(
    "MT-"
)  # annotate the group of mitochondrial genes as 'mt'
sc.pp.calculate_qc_metrics(
    adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
)
adata = adata[adata.obs.n_genes_by_counts < 2500, :]
adata = adata[adata.obs.pct_counts_mt < 5, :]

adata.layers["counts"] = adata.X.toarray()  # Keep the counts, for scDEF
adata.raw = adata

# Update annotations
adata.obs["Cell types"] = adata.obs["clusters_neoblasts"]
adata.obs["Coarse cell types"] = adata.obs["clusters_coarse"]

adata.uns["hierarchy_obs"] = ["Cell types", "Coarse cell types"]
for i, obs in enumerate(adata.uns["hierarchy_obs"]):
    adata.obs[obs] = "h" * i + adata.obs[obs].astype(str)

# Load marker genes
markers = pd.read_csv(snakemake.params.markers_fname, index_col=0)
markers_dict = dict()
for cell_type in adata.obs["Cell types"].unique():
    markers_dict[cell_type] = markers.loc[
        markers["clusters_neoblasts"] == cell_type
    ].index.tolist()
adata.uns["true_markers"] = markers_dict

# Load gene names
gene_names = pd.read_csv(snakemake.params.gene_names, index_col=0)
adata.var["gene_name"] = ""
for gene in gene_names.index:
    adata.var.loc[gene, "gene_name"] = gene.loc[gene, "name"]

# Keep only HVGs
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(
    adata, flavor="seurat_v3", layer="counts", n_top_genes=snakemake.params.n_top_genes
)  # Not required, but makes scDEF faster
adata = adata[:, adata.var.highly_variable]

# Process and visualize the data
sc.pp.regress_out(adata, ["total_counts", "pct_counts_mt"])
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver="arpack")
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
sc.tl.umap(adata)
sc.tl.leiden(adata)
sc.pl.umap(
    adata, color=adata.uns["hierarchy_obs"] + ["leiden"], show=False
)  # just to make the colors

# Write new h5ad file
adata.write(snakemake.output.fname)
