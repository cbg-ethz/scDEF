import numpy as np
import scdef
import pandas as pd
import scanpy as sc
import anndata
import scvi
from sklearn.metrics import adjusted_rand_score

counts = pd.read_csv(snakemake.input["counts_fname"], index_col=0)
meta = pd.read_csv(snakemake.input["meta_fname"])

adata = anndata.AnnData(X=counts.values.T, obs=meta)

scvi.model.LinearSCVI.setup_anndata(
    adata,
)

# Run for range of K and choose best one
models = []
losses = []
for k in range(5, 15):
    model = scvi.model.LinearSCVI(adata)
    model.train()
    models.append(model)
    losses.append(model.history['elbo_train'].values[-1][0])
best = models[np.argmin(losses)]

latent = best.get_latent_representation()
adata.obsm["X_scVI"] = latent
sc.pp.neighbors(adata, use_rep="X_scVI")
sc.tl.leiden(adata)

ari = adjusted_rand_score(adata.obs['Group'], adata.obs['leiden'])

# Compute mean cell score per group
mean_cluster_scores = scdef.util.get_mean_cellscore_per_group(latent, adata.obs['Group'].values)
mod = scdef.util.mod_score(np.abs(mean_cluster_scores.T))

with open(snakemake.output["ari_fname"], "w") as file:
    file.write(str(ari))

with open(snakemake.output["mod_fname"], "w") as file:
    file.write(str(mod))
