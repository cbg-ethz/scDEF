import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import scdef
from sklearn.metrics import adjusted_rand_score, silhouette_score, roc_auc_score
from sklearn.model_selection import train_test_split
import time

tau = snakemake.params["tau"]
mu = snakemake.params["mu"]
kappa = snakemake.params["kappa"]
seed = snakemake.params["seed"]
layer_sizes = snakemake.params["layer_sizes"]
factor_var = snakemake.params["factor_var"]
n_layers = snakemake.params["n_layers"]

model_params = dict(brd_strength=tau, brd_mean=mu, layer_rates=kappa)

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

# Add noise to layer sizes
np.random.seed(seed)
layer_sizes = [l + np.random.randint(low=-factor_var, high=factor_var)*(l>1) for l in layer_sizes][n_layers]
print(layer_sizes)

# Run scDEF
duration = time.time()
scd = scdef.scDEF(adata, counts_layer="counts", batch_key="Batch",
                  seed=seed,
                  layer_sizes=layer_sizes, **model_params)
scd.learn()
scd.filter_factors(iqr_mult=0.0)
duration = time.time() - duration

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
    celltype_obs_key=["GroupC", "GroupB", "GroupA"], # to compute every layer vs every layer
    batch_obs_key="Batch",
)
df["scDEF", "Runtime"] = duration

df.to_csv(snakemake.output["scores_fname"])
