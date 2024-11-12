import scanpy as sc
import pandas as pd
import numpy as np
import infercnvpy as cnv

# Load full data
adata = sc.read_h5ad(snakemake.params.data_fname)
donor_1 = snakemake.params.donor_1
donor_2 = snakemake.params.donor_1

# Select donors
sub_adata = adata[
    adata.obs.query(f"donor_id == '{donor_1}' or donor_id == '{donor_2}'").index
]

# Select samples
sub_adata = sub_adata[
    sub_adata.obs.query(
        "author_tumor_subsite == 'Left Adnexa' or author_tumor_subsite == 'Left Ovary'"
    ).index
]

sub_adata = sub_adata.raw.to_adata()

# Annotate genes
annot = pd.read_csv(snakemake.params.annotations_fname, index_col=0)
sub_adata.var[annot.columns] = annot
sub_adata = sub_adata[:, ~np.isnan(sub_adata.var["start_position"])]
sub_adata.var = sub_adata.var.rename(
    columns={
        "start_position": "start",
        "end_position": "end",
        "chromosome_name": "chromosome",
    }
)


# Pre-process
sc.pp.filter_cells(sub_adata, min_genes=20)
sc.pp.filter_genes(sub_adata, min_cells=10)

# Remove some genes
for gene in snakemake.params.genes_to_remove:
    adata = adata[:, adata.var_names != gene]

sub_adata.var["mt"] = sub_adata.var_names.str.startswith(
    "MTRN"
)  # annotate the group of mitochondrial genes as 'mt'
sc.pp.calculate_qc_metrics(
    sub_adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
)
# sub_adata = sub_adata[sub_adata.obs.n_genes_by_counts < 1000, :]
# sub_adata = sub_adata[sub_adata.obs.pct_counts_mt < 8, :]

sub_adata.layers["counts"] = sub_adata.X.toarray()  # Keep the counts, for scDEF
sub_adata.raw = sub_adata

# Keep only HVGs
sc.pp.normalize_total(sub_adata, target_sum=1e4)
sc.pp.log1p(sub_adata)
sc.pp.highly_variable_genes(
    sub_adata,
    flavor="seurat_v3",
    n_top_genes=5000,
    layer="counts",
    batch_key="donor_id",
)  # Not required, but makes scDEF faster
sub_adata = sub_adata[:, sub_adata.var.highly_variable]

# Process and visualize the data
sc.pp.regress_out(sub_adata, ["total_counts", "pct_counts_mt"])
sc.pp.scale(sub_adata, max_value=10)
sc.tl.pca(sub_adata, svd_solver="arpack")
sc.pp.neighbors(sub_adata, n_neighbors=10, n_pcs=40)
sc.tl.umap(sub_adata)
sc.tl.leiden(sub_adata, n_iterations=2)
sc.pl.umap(sub_adata, color=["leiden", "author_cell_type", "donor_id"], frameon=False)

# Call CNVs
# sub_adata.var["chromosome"] = sub_adata.var["chromosome"].astype("category")
# sub_adata.var["chromosome"] = sub_adata.var["chromosome"].cat.rename_categories(
#     dict(
#         zip(
#             [str(i) for i in range(23)] + ["X", "Y"],
#             [f"chr{i}" for i in range(23)] + ["chrX", "chrY"],
#         )
#     )
# )
sub_adata.X = sub_adata.X.todense()
sub_adata.var["start"] = sub_adata.var["start"].astype(int)
sub_adata.var["end"] = sub_adata.var["end"].astype(int)

# cnv.tl.infercnv(
#     sub_adata,
#     reference_key="author_cell_type",
#     reference_cat=[
#         l for l in sub_adata.obs["author_cell_type"].unique() if "cancer" not in l
#     ],
#     window_size=250,
#     calculate_gene_values=True,
# )
# cnv.pl.chromosome_heatmap(
#     sub_adata, groupby="author_cell_type", save=snakemake.output.cnv_full_fname
# )

sub_adata.obs.index.name = "barcode"
# sub_adata.write_h5ad(snakemake.output.adata_full_fname)

# Pick fibroblasts from one patient and cancer cells from another
celltype_1 = snakemake.params.celltype_1
celltype_2 = snakemake.params.celltype_2
sub_sub_adata = sub_adata[
    sub_adata.obs.query(
        f"(donor_id == '{donor_1}' and author_cell_type == '{celltype_1}') \
                            or (donor_id == '{donor_2}' and author_cell_type == '{celltype_2}') "
    ).index
]

# cnv.pl.chromosome_heatmap(
#     sub_sub_adata, groupby="author_cell_type", save=snakemake.output.cnv_subset_fname
# )

sc.pp.filter_cells(sub_sub_adata, min_genes=20)
sc.pp.filter_genes(sub_sub_adata, min_cells=10)

sub_sub_adata.X = sub_sub_adata.layers["counts"]

sub_sub_adata.obs["Cell types"] = sub_sub_adata.obs["author_cell_type"]
sub_sub_adata.obs["Donor"] = sub_sub_adata.obs["donor_id"]
sub_sub_adata.uns["true_markers"] = dict()
adata.uns["hierarchy_obs"] = ["author_cell_type", "donor_id"]
for i, obs in enumerate(adata.uns["hierarchy_obs"]):
    adata.obs[obs] = "h" * i + adata.obs[obs].astype(str)

# Keep only HVGs
sc.pp.normalize_total(sub_sub_adata, target_sum=1e4)
sc.pp.log1p(sub_sub_adata)
sc.pp.highly_variable_genes(
    sub_sub_adata,
    flavor="seurat_v3",
    n_top_genes=snakemake.params.n_top_genes,
    layer="counts",
    batch_key="donor_id",
)  # Not required, but makes scDEF faster
sub_sub_adata = sub_sub_adata[:, sub_sub_adata.var.highly_variable]

# Process and visualize the data
sc.pp.regress_out(sub_sub_adata, ["total_counts", "pct_counts_mt"])
sc.pp.scale(sub_sub_adata, max_value=10)
sc.tl.pca(sub_sub_adata, svd_solver="arpack")
sc.pp.neighbors(sub_sub_adata, n_neighbors=10, n_pcs=50)
sc.tl.umap(sub_sub_adata)
sc.tl.leiden(sub_sub_adata, n_iterations=2)
sc.pl.umap(sub_sub_adata, color=["author_cell_type", "donor_id"], frameon=False)

# Write new h5ad file
adata.write(snakemake.output.adata_subset_fname)
