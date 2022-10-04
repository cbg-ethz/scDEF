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
for k in range(9, 11):
    sch = schpf.scHPF(k)
    sch.fit(X, max_iter=2)
    models.append(sch)
    losses.append(sch.loss[-1])
best = models[np.argmin(losses)]

cscores = best.cell_score()
assignments = np.argmax(cscores,axis=1)

ari = adjusted_rand_score(adata.obs['Group'].values, assignments)

with open(snakemake.output["fname"], "w") as file:
    file.write(str(ari))
