import anndata
import scanpy as sc
import numpy as np


def main():
    np.random.seed(snakemake.params.seed)
    adata = anndata.read_h5ad(
        snakemake.params.data_fname
    )

    # Remove some genes
    for gene in ["MALAT1"]:
        adata = adata[:, adata.var_names != gene]
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
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
    adata.obs["Cell types"] = adata.obs["celltypes"]

    map_coarse = {}
    for c in adata.obs["Cell types"].astype("category").cat.categories:
        if c.endswith(" T"):
            map_coarse[c] = "T"
        elif c.endswith("Mono"):
            map_coarse[c] = "Mono"
        else:
            map_coarse[c] = c

    adata.obs["Coarse cell types"] = (
        adata.obs["Cell types"].map(map_coarse).astype("category")
    )
    adata.uns["hierarchy_obs"] = ["Cell types", "Coarse cell types"]
    for i, obs in enumerate(adata.uns["hierarchy_obs"]):
        adata.obs[obs] = "h" * i + adata.obs[obs].astype(str)

    # Store true markers for the cell types
    markers = {'Memory CD4 T': ['IL7R', 'CD3D', 'CD3E', 'IL32', 'CD2'], 'Naive CD4 T': ['IL7R', 'CD3D', 'CD3E', 'TCF7', 'CCR7', 'CD2'],  
            'CD8 T': ['CD8B', 'CCL5', 'CD2', 'CD3D', 'CD3E'], 
            'NK': ['GNLY', 'NKG7', 'CD2'], 
            'B': ['MS4A1', 'CD79A', 'CD79B', 'CD74'], 
            'CD14+ Mono': ['CD14', 'LYZ'],  'FCGR3A+ Mono': ['CD14', 'FCGR3A', 'MS4A7', 'LYZ'],  'DC': ['CD14', 'FCER1A', 'CST3', 'CD74'], 
            'Platelet': ['PPBP', 'PF4']}
    adata.uns["true_markers"] = markers

    # Keep only HVGs
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata, flavor="seurat_v3", layer="counts", n_top_genes=snakemake.params.n_top_genes
    )  # Not required, but makes scDEF faster
    # Make sure marker genes are present
    markers_list = list(set([x for sub in [markers[n] for n in markers] for x in sub]))
    for marker in markers_list:
        adata.var.loc[marker, 'highly_variable'] = True
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

if __name__ == "__main__":
    main()