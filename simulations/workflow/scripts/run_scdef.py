import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import scdef
from sklearn.metrics import adjusted_rand_score, silhouette_score, roc_auc_score
from sklearn.model_selection import train_test_split

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
sc.pp.highly_variable_genes(
    adata,
    n_top_genes=2000,
    subset=True,
    layer="counts",
    flavor="seurat_v3",
    batch_key="Batch",
)

adata.obs["GroupA"] = adata.obs["GroupA"].apply(lambda row: f"hh{row}")
adata.obs["GroupB"] = adata.obs["GroupB"].apply(lambda row: f"h{row}")

# Run scDEF
scd = scdef.scDEF(adata, counts_layer="counts", batch_key="Batch")
scd.learn()
scd.filter_factors(iqr_mult=0.0)

metrics_list = [
    "Cell Type ARI",
    "Cell Type ASW",
    "Batch ARI",
    "Batch ASW",
    "Hierarchy accuracy",
    "Hierarchical signature consistency",
    "Signature sparsity",
    "Signature accuracy",
]

true_hierarchy = scdef.hierarchy_utils.get_hierarchy_from_clusters(
    [
        adata.obs["GroupC"].values,
        adata.obs["GroupB"].values,
        adata.obs["GroupA"].values,
    ],
    use_names=True,
)

df = scdef.benchmark.evaluate_methods(
    adata,
    metrics_list,
    {"scDEF": scd},
    true_hierarchy=true_hierarchy,
    hierarchy_obs_keys=["GroupA", "GroupB", "GroupC"],
    markers=markers,
    celltype_obs_key="GroupC",
    batch_obs_key="Batch",
)
df.to_csv(snakemake.output["scores_fname"])
