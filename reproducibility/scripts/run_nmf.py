from .benchmark import run_methods, evaluate_methods

import scanpy as sc
import scdef
import time

adata = sc.read_h5ad(snakemake.input["adata"])

batch_key = None
if "Batch" in adata.obs.columns:
    batch_key = "Batch"

sc.pp.highly_variable_genes(
    adata,
    n_top_genes=2000,
    subset=True,
    layer="counts",
    flavor="seurat_v3",
    batch_key=batch_key,
)

methods_list = ["NMF"]
nmf_settings = dict(max_iter=snakemake.params["max_iter"])
duration = time.time()
methods_results = run_methods(adata, methods_list, batch_key="Batch", **nmf_settings)
duration = time.time() - duration

hierarchy_obs = adata.uns["hierarchy_obs"]
true_hierarchy = scdef.hierarchy_utils.get_hierarchy_from_clusters(
    [adata.obs[o].values for o in hierarchy_obs],
    use_names=True,
)

df = evaluate_methods(
    adata,
    snakemake.params["metrics"],
    methods_results,
    true_hierarchy=true_hierarchy,
    hierarchy_obs_keys=hierarchy_obs,
    markers=adata.uns["markers"],
    celltype_obs_key=hierarchy_obs,  # to compute every layer vs every layer
    batch_obs_key=batch_key,
)
df.loc["Runtime", methods_results.keys()[0]] = duration

if snakemake.params["store_anndata"]:
    # Store anndata
    methods_results[methods_list[0]][2].write_h5ad(snakemake.output["out_fname"])

# Store scores
df.to_csv(snakemake.output["scores_fname"])
