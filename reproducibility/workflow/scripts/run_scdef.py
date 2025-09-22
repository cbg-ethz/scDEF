from _benchmark import evaluate_methods

import scanpy as sc
import scdef as scd
import time

def main():
    adata = sc.read_h5ad(snakemake.input["adata"])

    batch_key = None
    if "Batch" in adata.obs.columns:
        batch_key = "Batch"

    # Run scDEF
    model_settings = dict(
        brd_strength=snakemake.params["tau"],
        brd_mean=snakemake.params["mu"],
        n_factors=snakemake.params["n_factors"],
        decay_factor=snakemake.params["decay_factor"],
        layer_concentration=snakemake.params["kappa"],
    )
    model = scd.scDEF(
        adata,
        counts_layer="counts",
        batch_key=batch_key,
        seed=int(snakemake.params["seed"]),
        **model_settings,
    )

    learning_settings = dict(
        pretrain=snakemake.params["pretrain"],
        nmf_init=snakemake.params["nmf_init"],
        n_epoch=snakemake.params["n_epoch"],
        lr=snakemake.params["lr"],
        batch_size=snakemake.params["batch_size"],
        num_samples=snakemake.params["num_samples"],
    )
    duration = time.time()
    model.fit(**learning_settings)
    duration = time.time() - duration
    model.filter_factors()

    hierarchy_obs = adata.uns["hierarchy_obs"].tolist()
    true_hierarchy = scd.hierarchy_utils.get_hierarchy_from_clusters(
        [adata.obs[o].values for o in hierarchy_obs],
        use_names=True,
    )


    df = evaluate_methods(
        adata,
        snakemake.params["metrics"],
        {"scDEF": model},
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
            pickle.dump(model, outp, pickle.HIGHEST_PROTOCOL)

    # Store scores
    df.to_csv(snakemake.output["scores_fname"])

if __name__ == "__main__":
    main()