import anndata
import scanpy as sc
import pandas as pd
import scdef

counts = pd.read_csv(snakemake.input["counts_fname"], index_col=0)
meta = pd.read_csv(snakemake.input["meta_fname"])
markers = pd.read_csv(snakemake.input["markers_fname"])

groups = markers["cluster"].unique()
markers = dict(
    zip(groups, [markers.loc[markers["cluster"] == g]["gene"].tolist() for g in groups])
)

adata = anndata.AnnData(X=counts.values.T, obs=meta)
adata.var_names = [f"Gene{i+1}" for i in range(adata.shape[1])]

adata.layers["counts"] = adata.X.copy()  # preserve counts
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
adata.raw = adata
raw_adata = adata.raw
raw_adata = raw_adata.to_adata()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

adata.uns["true_markers"] = markers

adata.uns["hierarchy_obs"] = ["GroupC", "GroupB", "GroupA"]
for i, obs in enumerate(adata.uns["hierarchy_obs"]):
    adata.obs[obs] = "h" * i + adata.obs[obs].astype(str)

true_hierarchy = scdef.hierarchy_utils.get_hierarchy_from_clusters(
    [
        adata.obs["GroupC"].values,
        adata.obs["GroupB"].values,
        adata.obs["GroupA"].values,
    ],
    use_names=True,
)
adata.uns["true_hierarchy"] = true_hierarchy

if "Batch" in adata.obs:
    # Make batches contiguous
    idx = adata.obs.sort_values("Batch").index
    adata = adata[idx,]

# Write new h5ad file
adata.write(snakemake.output.fname)
