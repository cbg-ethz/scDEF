import pandas as pd
import scanpy as sc
import anndata
import scvi
from sklearn.metrics import adjusted_rand_score

counts = pd.read_csv(snakemake.input["counts_fname"], index_col=0)
meta = pd.read_csv(snakemake.input["meta_fname"])

adata = anndata.AnnData(X=counts.values.T, obs=meta)

scvi.model.SCVI.setup_anndata(
    adata,
)

model = scvi.model.SCVI(adata)
model.train(max_epochs=2)

latent = model.get_latent_representation()
adata.obsm["X_scVI"] = latent
sc.pp.neighbors(adata, use_rep="X_scVI")
sc.tl.leiden(adata)

ari = adjusted_rand_score(adata.obs['Group'], adata.obs['leiden'])

with open(snakemake.output["fname"], "w") as file:
    file.write(str(ari))
