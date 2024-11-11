import anndata
import pandas as pd
import scanpy as sc
import numpy as np

np.random.seed(snakemake.params.seed)

df = pd.read_csv(snakemake.params.counts_fname, sep="\t", index_col=0)
cellinfo = pd.read_csv(snakemake.params.annotations_fname, sep="\t")
cellinfo = cellinfo.set_index("cell")

adata = anndata.AnnData(df.T, obs=cellinfo)
adata = adata[adata.obs["cell_type"] == "EOC"]  # cancer cells only

sc.pp.filter_cells(adata, min_genes=1552)
sc.pp.filter_genes(adata, min_cells=10)

# As in the paper
for gene in snakemake.params.genes_to_remove:
    adata = adata[:, adata.var_names != gene]
adata.var["mt"] = adata.var_names.str.startswith(
    "MT-"
)  # annotate the group of mitochondrial genes as 'mt'
sc.pp.calculate_qc_metrics(
    adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
)
adata = adata[adata.obs.n_genes_by_counts < 5000, :]
adata = adata[adata.obs.pct_counts_mt < 12, :]
adata.raw = adata

raw_adata = adata.raw
raw_adata = raw_adata.to_adata()
adata.layers["counts"] = raw_adata.X

# Update annotations
adata.obs["Batch"] = adata.obs["patient_id"]
adata.obs["Patient"] = adata.obs["patient_id"]
adata.obs["Treatment"] = adata.obs["treatment_phase"]

adata.uns["true_markers"] = dict()
adata.uns["hierarchy_obs"] = ["patient_id", "treatment_phase"]
for i, obs in enumerate(adata.uns["hierarchy_obs"]):
    adata.obs[obs] = "h" * i + adata.obs[obs].astype(str)

# Keep only HVGs
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(
    adata,
    flavor="seurat_v3",
    layer="counts",
    n_top_genes=snakemake.params.n_top_genes,
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
    adata, color=["Patient", "Treatment", "leiden"], show=False
)  # just to make the colors

# Write new h5ad file
adata.write(snakemake.output.fname)
