import scdef as scd
import time


def main():
    model_un = scd.scDEF.load(snakemake.input["scdef_un_dir"])

    duration = time.time()
    model_corr = scd.add_batch_correction(
        model_un,
        batch_key="Batch",
        n_epoch=int(snakemake.params["n_epoch"]),
        lr=float(snakemake.params["lr"]),
    )
    model_corr.filter_factors()
    duration = time.time() - duration

    model_corr.save(snakemake.output["out_dir"], overwrite=True, save_anndata=True)

    with open(snakemake.output["duration_fname"], "w") as f:
        f.write(str(duration))


if __name__ == "__main__":
    main()
