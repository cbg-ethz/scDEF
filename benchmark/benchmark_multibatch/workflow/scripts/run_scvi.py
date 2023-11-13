import numpy as np
import scdef
import pandas as pd
import scanpy as sc
import anndata
import scvi
from sklearn.metrics import adjusted_rand_score, silhouette_score, roc_auc_score
from sklearn.model_selection import train_test_split

counts = pd.read_csv(snakemake.input["counts_fname"], index_col=0)
meta = pd.read_csv(snakemake.input["meta_fname"])
markers = pd.read_csv(snakemake.input["markers_fname"])
n_top_genes = snakemake.params["n_top_genes"]
chc_reps = snakemake.params["chc_reps"]
true_hierarchy = snakemake.params["true_hrc"]

groups = markers["cluster"].unique()
markers = dict(
    zip(groups, [markers.loc[markers["cluster"] == g]["gene"].tolist() for g in groups])
)

adata = anndata.AnnData(X=counts.values.T, obs=meta)
adata.var_names = [f"Gene{i+1}" for i in range(adata.shape[1])]
sc.pp.filter_genes(adata, min_counts=3)
adata.layers["counts"] = adata.X.copy()  # preserve counts
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata  # freeze the state in `.raw`

sc.pp.highly_variable_genes(
    adata,
    n_top_genes=2000,
    subset=True,
    layer="counts",
    flavor="seurat_v3",
    batch_key="Batch",
)

methods_list = ["scVI"]
methods_results = scdef.benchmark.run_methods(adata, methods_list, batch_key="Batch")

metrics_list = [
    "Cell Type ARI",
    "Cell Type ASW",
    "Batch ARI",
    "Batch ASW",
    "Hierarchical signature consistency",
    "Signature sparsity",
    "Signature accuracy",
]

df = scdef.benchmark.evaluate_methods(
    adata,
    metrics_list,
    methods_results,
    true_hierarchy=true_hierarchy,
    hierarchy_obs_keys=["GroupA", "GroupB", "GroupC"],
    markers=markers,
    celltype_obs_key="GroupC",
    batch_obs_key="Batch",
)
df.to_csv(snakemake.output["scores_fname"])
