import pandas as pd
import scanpy as sc
import anndata
import scdef
from sklearn.metrics import adjusted_rand_score

counts = pd.read_csv(snakemake.input["counts_fname"], index_col=0)
meta = pd.read_csv(snakemake.input["meta_fname"])

adata = anndata.AnnData(X=counts.values.T, obs=meta)

scd = scdef.scDEF(adata, n_factors=15, n_hfactors=5, shape=.3)
elbos = scd.optimize(n_epochs=1500, batch_size=256, step_size=.05, num_samples=2, seed=42, init=False)

ari = adjusted_rand_score(adata.obs['Group'], scd.adata.obs['X_factor'])

# Compute mean cell score per group
tokeep = scd.filter_factors(annotate=False, q=[0.4, 0.])
mean_cluster_scores = scdef.util.get_mean_cellscore_per_group(scd.pmeans['z'][:,tokeep[0]] * scd.pmeans['cell_scale'], adata.obs['Group'].values)
mod = scdef.util.mod_core(mean_cluster_scores.T)

with open(snakemake.output["ari_fname"], "w") as file:
    file.write(str(ari))

with open(snakemake.output["mod_fname"], "w") as file:
    file.write(str(mod))
