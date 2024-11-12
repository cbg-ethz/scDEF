import scanpy as sc
import scdef
import time

adata = sc.read_h5ad(snakemake.input["fname"])

batch_key = None
if "Batch" in adata.obs.columns:
    batch_key = "Batch"

# Run scDEF
model_settings = dict(
    brd_strength=snakemake.params["tau"],
    brd_mean=snakemake.params["mu"],
    nmf_init=snakemake.params["nmf_init"],
    layer_sizes=snakemake.params["layer_sizes"],
)
scd = scdef.scDEF(
    adata,
    counts_layer="counts",
    batch_key=batch_key,
    seed=int(snakemake.params["seed"]),
    **model_settings,
)

learning_settings = dict(
    n_epoch=snakemake.params["n_epoch"],
    lr=snakemake.params["lr"],
    batch_size=snakemake.params["batch_size"],
    num_samples=snakemake.params["num_samples"],
)
duration = time.time()
scd.learn(**learning_settings)
duration = time.time() - duration
scd.filter_factors()

hierarchy_obs = adata.uns["hierarchy_obs"].tolist()
true_hierarchy = scdef.hierarchy_utils.get_hierarchy_from_clusters(
    [adata.obs[o].values for o in hierarchy_obs],
    use_names=True,
)


df = scdef.benchmark.evaluate_methods(
    adata,
    snakemake.params["metrics"],
    {"scDEF": scd},
    true_hierarchy=true_hierarchy,
    hierarchy_obs_keys=hierarchy_obs,
    markers=adata.uns["true_markers"],
    celltype_obs_key=hierarchy_obs,  # to compute every layer vs every layer
    batch_obs_key=batch_key,
)
df.loc["Runtime", "scDEF"] = duration

# Store scDEF
if snakemake.params["store_full"]:
    with open(
        snakemake.output["out_fname"], "wb"
    ) as outp:  # Overwrites any existing file.
        pickle.dump(scd, outp, pickle.HIGHEST_PROTOCOL)

# Store scores
df.to_csv(snakemake.output["scores_fname"])
