from _benchmark import run_methods

import scanpy as sc
import time


def main():
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
    nmf_settings = dict(
        max_iter=snakemake.params["max_iter"],
        random_state=int(snakemake.params["seed"]),
    )
    duration = time.time()
    methods_results = run_methods(
        adata, methods_list, batch_key=batch_key, **nmf_settings
    )
    duration = time.time() - duration

    methods_results[methods_list[0]]["adata"].write_h5ad(snakemake.output["out_fname"])

    with open(snakemake.output["duration_fname"], "w") as f:
        f.write(str(duration))


if __name__ == "__main__":
    main()
