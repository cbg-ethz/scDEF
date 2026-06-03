import scanpy as sc
import scdef as scd
import time


def main():
    adata = sc.read_h5ad(snakemake.input["adata"])

    batch_key = None
    if "Batch" in adata.obs.columns:
        batch_key = "Batch"

    model_settings = dict(
        brd_strength=float(snakemake.params["brd_strength"]),
        brd_mean=float(snakemake.params["brd_mean"]),
        n_factors=int(snakemake.params["n_factors"]),
        n_layers=int(snakemake.params["n_layers"]),
        hierarchy_weight=float(snakemake.params["hierarchy_weight"]),
        top_factors=int(snakemake.params["top_factors"]),
    )
    model = scd.scDEF(
        adata,
        counts_layer="counts",
        batch_key=batch_key,
        seed=int(snakemake.params["seed"]),
        **model_settings,
    )

    learning_settings = dict(
        pretraining=snakemake.params["pretraining"],
        nmf_init=snakemake.params["nmf_init"],
        n_epoch=snakemake.params["n_epoch"],
        lr=snakemake.params["lr"],
        batch_size=int(snakemake.params["batch_size"]),
        num_samples=int(snakemake.params["num_samples"]),
    )
    duration = time.time()
    model.fit(**learning_settings)
    duration = time.time() - duration
    model.filter_factors()

    model.save(snakemake.output["out_dir"], overwrite=True, save_anndata=True)

    with open(snakemake.output["duration_fname"], "w") as f:
        f.write(str(duration))


if __name__ == "__main__":
    main()
