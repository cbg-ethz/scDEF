import scdef
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import schpf
from sklearn.metrics import adjusted_rand_score
import scipy

counts = pd.read_csv(snakemake.input["counts_fname"], index_col=0)
meta = pd.read_csv(snakemake.input["meta_fname"])

adata = anndata.AnnData(X=counts.values.T, obs=meta)
X = scipy.sparse.coo_matrix(adata.X)

# Run for range of K and choose best one
models = []
losses = []
for k in range(5, 15):
    sch = schpf.scHPF(k)
    sch.fit(X)
    models.append(sch)
    losses.append(sch.loss[-1])
best = models[np.argmin(losses)]

cscores = best.cell_score()
assignments = np.argmax(cscores,axis=1)

ari = adjusted_rand_score(adata.obs['Group'].values, assignments)

# Compute mean cell score per group
mean_cluster_scores = scdef.util.get_mean_cellscore_per_group(cscores, adata.obs['Group'].values)
mod = scdef.util.mod_score(mean_cluster_scores.T)

with open(snakemake.output["ari_fname"], "w") as file:
    file.write(str(ari))

with open(snakemake.output["mod_fname"], "w") as file:
    file.write(str(mod))
