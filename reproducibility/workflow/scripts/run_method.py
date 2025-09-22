from benchmark import run_methods, evaluate_methods

import scanpy as sc
import scdef
import time

adata = sc.read_h5ad(snakemake.input["fname"])

batch_key = None
if "Batch" in adata.obs.columns:
    batch_key = "Batch"

try:
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=snakemake.params["n_top_genes"],
        subset=True,
        layer="counts",
        flavor="seurat_v3",
        batch_key=batch_key,
    )
except ValueError as e:
    print(e)
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=snakemake.params["n_top_genes"],
        subset=True,
        layer="counts",
        flavor="seurat_v3",
        batch_key=None,
    )

methods_list = [snakemake.params["method"]]
settings = snakemake.params["settings"]
if settings is None:
    settings = dict()
duration = time.time()
methods_results = run_methods(adata, methods_list, batch_key=batch_key, **settings)
duration = time.time() - duration

methods_results[methods_list[0]].keys()
if snakemake.params["store_full"]:
    # Store anndata
    methods_results[methods_list[0]]["adata"].write_h5ad(snakemake.output["out_fname"])

hierarchy_obs = adata.uns["hierarchy_obs"].tolist()
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
    markers=adata.uns["true_markers"],
    celltype_obs_key=hierarchy_obs,  # to compute every layer vs every layer
    batch_obs_key=batch_key,
)
df.loc["Runtime", methods_list[0]] = duration

# Store scores
df.to_csv(snakemake.output["scores_fname"])
