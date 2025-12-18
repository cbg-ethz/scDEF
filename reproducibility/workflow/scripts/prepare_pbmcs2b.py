import anndata
import scanpy as sc
import numpy as np

np.random.seed(snakemake.params.seed)
adata = anndata.read_h5ad(snakemake.params.data_fname)

adata = adata[adata.obs["Method"] == "Drop-seq"]

# Remove some genes
for gene in snakemake.params.genes_to_remove:
    adata = adata[:, adata.var_names != gene]
sc.pp.filter_cells(adata, min_genes=snakemake.params.min_genes)
sc.pp.filter_genes(adata, min_cells=snakemake.params.min_cells)
adata.var["mt"] = adata.var_names.str.startswith(
    "MTRN"
)  # annotate the group of mitochondrial genes as 'mt'
sc.pp.calculate_qc_metrics(
    adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
)
adata = adata[adata.obs.n_genes_by_counts < 2000, :]
adata = adata[adata.obs.pct_counts_mt < 8, :]

adata.layers["counts"] = adata.X.toarray()  # Keep the counts, for scDEF
adata.raw = adata

stypes = dict(
    {
        "CD4+ T cell": "T cell",
        "Cytotoxic T cell": "T cell",
        "CD14+ monocyte": "Monocyte",
        "CD16+ monocyte": "Monocyte",
        "Dendritic cell": "Monocyte",
        "Plasmacytoid dendritic cell": "Monocyte",
    }
)
adata.obs["SubType"] = adata.obs["CellType"]
adata.obs["CellType"] = adata.obs["CellType"].replace(stypes)

adata.obs["Coarse cell types"] = adata.obs["CellType"]
adata.obs["Cell types"] = adata.obs["SubType"]
adata.obs["Batch"] = adata.obs["Experiment"]

adata.uns["hierarchy_obs"] = ["Cell types", "Coarse cell types"]
for i, obs in enumerate(adata.uns["hierarchy_obs"]):
    adata.obs[obs] = "h" * i + adata.obs[obs].astype(str)

# Store true markers for the cell types: supplementary table S12 from https://www.biorxiv.org/content/10.1101/632216v1.full
markers = {
    "CD4+ T cell": ["CD3D", "CD3E", "CD3G", "TRAC", "CD4", "TCF7", "CD27", "IL7R"],
    "Cytotoxic T cell": [
        "CD3D",
        "CD3E",
        "CD3G",
        "TRAC",
        "CD8A",
        "CD8B",
        "GZMK",
        "CCL5",
        "NKG7",
    ],
    "CD14+ monocyte": [
        "VCAN",
        "FCN1",
        "S100A8",
        "S100A9",
        "CD14",
        "ITGAL",
        "ITGAM",
        "CSF3R",
        "CSF1R",
        "CX3CR1",
        "TYROBP",
        "LYZ",
        "S100A12",
    ],
    "CD16+ monocyte": [
        "FCN1",
        "FCGR3A",
        "FCGR3B",
        "ITGAL",
        "ITGAM",
        "CSF3R",
        "CSF1R",
        "CX3CR1",
        "CDKN1C",
        "MS4A7",
    ],
    "Dendritic cell": [
        "HLA-DPB1",
        "HLA-DPA1",
        "HLA-DQA1",
        "ITGAX",
        "CD1C",
        "CD1E",
        "FCER1A",
        "CLEC10A",
        "FCGR2B",
        "MS4A1",
        "CD79A",
        "CD79B",
    ],
    "Plasmacytoid dendritic cell": [
        "IL3RA",
        "GZMB",
        "JCHAIN",
        "IRF7",
        "TCF4",
        "LILRA4",
        "CLEC4C",
    ],
    "B cell": ["CD19", "MS4A1", "CD79A", "CD79B", "MZB1", "IGHD", "IGHM"],
    "Natural killer cell": [
        "NCAM1",
        "NKG7",
        "KLRB1",
        "KLRD1",
        "KLRF1",
        "KLRC1",
        "KLRC2",
        "KLRC3",
        "KLRC4",
        "FCGR3A",
        "FCGR3B",
        "ITGAL",
        "ITGAM",
        "FCER1G",
    ],
    "Megakaryocyte": [
        "PF4",
        "PPBP",
        "GP5",
        "ITGA2B",
        "NRGN",
        "TUBB1",
        "SPARC",
        "RGS18",
        "MYL9",
        "GNG11",
    ],
}
adata.uns["true_markers"] = markers

# Keep only HVGs
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(
    adata,
    flavor="seurat_v3",
    layer="counts",
    n_top_genes=snakemake.params.n_top_genes,
    batch_key="Batch",
)  # Not required, but makes scDEF faster
markers_list = list(set([x for sub in [markers[n] for n in markers] for x in sub]))
for marker in markers_list:
    adata.var.loc[marker, "highly_variable"] = True
adata = adata[:, adata.var.highly_variable]

# Process and visualize the data
sc.pp.regress_out(adata, ["total_counts", "pct_counts_mt"])
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver="arpack")
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
sc.tl.umap(adata)
sc.tl.leiden(adata)
sc.pl.umap(
    adata, color=adata.uns["hierarchy_obs"] + ["Batch"] + ["leiden"], show=False
)  # just to make the colors

# Write new h5ad file
adata.write(snakemake.output.fname)
