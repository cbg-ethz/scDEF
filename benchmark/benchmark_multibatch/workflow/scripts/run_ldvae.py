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
k_min = snakemake.params["k_min"]
k_max = snakemake.params["k_max"]
n_top_genes = snakemake.params["n_top_genes"]
chc_reps = snakemake.params["chc_reps"]
test_frac = snakemake.params["test_frac"]

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
    n_top_genes=n_top_genes,
    subset=True,
    layer="counts",
    flavor="seurat_v3",
    batch_key="Batch",
)

latent, loadings, nadata = scdef.other_methods.run_ldvae(adata, range(k_min, k_max))
assignments = np.argmax(latent, axis=1)

# Compute clustering scores
asw = silhouette_score(latent, nadata.obs["Group"])
ari = adjusted_rand_score(nadata.obs["Group"], assignments)

# Compute mean cell score per group
mean_cluster_scores = scdef.util.get_mean_cellscore_per_group(
    latent, nadata.obs["Group"].values
)
mod = scdef.util.mod_score(np.abs(mean_cluster_scores.T))
ent = scdef.util.entropy_score(np.abs(mean_cluster_scores.T))

# Compute ROCAUC score for gene signatures
aucs = []
for group in np.unique(nadata.obs["Group"]):
    # True markers
    true_markers = markers[group]
    true_markers = [1 if gene in true_markers else 0 for gene in nadata.var_names]
    # Get cluster that contains the most cells of this group
    cells = np.where(adata.obs["Group"] == group)[0]
    factors, counts = np.unique(assignments[cells], return_counts=True)
    factor = factors[np.argmax(counts)]
    # Compute ROC AUC
    aucs.append(roc_auc_score(true_markers, loadings[factor]))
avg_auc = np.mean(aucs)

# Compute coherence score of gene signatures
coherences = []
for i in range(chc_reps):
    train_set, heldout_set = train_test_split(
        np.arange(adata.shape[0]), test_size=test_frac, random_state=i, shuffle=True
    )
    latent, loadings, nadata = scdef.other_methods.run_ldvae(
        adata[train_set], range(k_min, k_max)
    )
    gene_rankings = []
    for k in range(latent.shape[1]):
        gene_rankings.append(adata.var_names[np.argsort(loadings[k])][:50])
    coherences.append(
        scdef.util.coherence_score(gene_rankings, adata[heldout_set])
    )  # does not need labels, but needs held-out data?!
chc = np.mean(coherences)

# Save scores
with open(snakemake.output["asw_fname"], "w") as file:
    file.write(str(asw))

with open(snakemake.output["ari_fname"], "w") as file:
    file.write(str(ari))

with open(snakemake.output["mod_fname"], "w") as file:
    file.write(str(mod))

with open(snakemake.output["ent_fname"], "w") as file:
    file.write(str(ent))

with open(snakemake.output["auc_fname"], "w") as file:
    file.write(str(avg_auc))

with open(snakemake.output["chc_fname"], "w") as file:
    file.write(str(chc))
