import scanpy as sc
import numpy as np

np.random.seed(snakemake.params.seed)
adata = sc.read_h5ad(snakemake.params.data_fname)
adata.__dict__["_raw"].__dict__["_var"] = (
    adata.__dict__["_raw"].__dict__["_var"].rename(columns={"_index": "features"})
)
adata.var = adata.var.drop(columns="features")

# Remove some genes
for gene in snakemake.params.genes_to_remove:
    adata = adata[:, adata.var_names != gene]

sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=10)

# As in the paper
adata.var["mt"] = adata.var_names.str.startswith(
    "MT-"
)  # annotate the group of mitochondrial genes as 'mt'
sc.pp.calculate_qc_metrics(
    adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
)
adata = adata[adata.obs.n_genes_by_counts < 5000, :]
adata = adata[adata.obs.pct_counts_mt < 5, :]
adata.raw = adata

adata.layers["counts"] = adata.X.toarray()  # Keep the counts, for scDEF
adata.raw = adata

# Update annotations
adata.obs["Cell types"] = adata.obs["seurat_annotations"]
adata.obs["Cell states"] = adata.obs.apply(
    lambda r: f'{r["seurat_annotations"]}_{r["stim"]}'
    if "CD14 Mono" in r["seurat_annotations"]
    else r["seurat_annotations"],
    axis=1,
)
adata.obs["Batch"] = adata.obs["stim"]

stypes = dict(
    {
        "B": "B",
        "B activated": "B",
        "CD4 Memory T": "T",
        "CD4 Naive T": "T",
        "CD8 T": "T",
        "CD14 Mono": "Monocyte",
        "CD16 Mono": "Monocyte",
        "DC": "Monocyte",
        "pDC": "Monocyte",
        "Eryth": "Eryth",
        "Mk": "Mk",
        "NK": "NK",
        "T activated": "T",
    }
)
adata.obs["Coarse cell types"] = adata.obs["Cell types"].replace(stypes)

adata.uns["hierarchy_obs"] = ["Cell states", "Coarse cell types"]
for i, obs in enumerate(adata.uns["hierarchy_obs"]):
    adata.obs[obs] = "h" * i + adata.obs[obs].astype(str)

# Store true markers for the cell types
markers = {
    "B": ["MS4A1", "CD79A"],
    "B activated": ["MS4A1", "CD79A", "MIR155HG", "NME1"],
    "CD4 Memory T": ["CD3D", "CREM"],
    "CD4 Naive T": ["CD3D", "GIMAP5", "CACYBP"],
    "CD8 T": ["CD3D", "CCL5", "CD8A"],
    "CD14 Mono_CTRL": ["FCGR3A", "S100A9"],
    "CD14 Mono_STIM": ["FCGR3A", "CCL2", "S100A9"],
    "CD16 Mono": ["FCGR3A", "VMO1"],
    "DC": ["HLA-DQA1", "GPR183"],
    "Eryth": ["HBA2", "HBB"],
    "Mk": ["PPBP", "GNG11"],
    "NK": ["GNLY", "NKG7", "CCL5"],
    "T activated": ["CD3D", "CREM", "HSPH1", "SELL", "CACYBP", "GPR183"],
    "pDC": ["TSPAN13", "IL3RA", "IGJ"],
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
