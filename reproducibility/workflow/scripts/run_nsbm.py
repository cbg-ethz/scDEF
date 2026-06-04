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
    adata.X = adata.layers["counts"]
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    methods_list = ["nSBM"]
    nsbm_settings = dict(
        n_init=snakemake.params["n_init"], random_seed=int(snakemake.params["seed"])
    )
    duration = time.time()
    methods_results = run_methods(
        adata, methods_list, batch_key=batch_key, **nsbm_settings
    )
    duration = time.time() - duration

    res = methods_results[methods_list[0]]
    out_adata = res["adata"]
    out_adata.uns["signatures"] = res["signatures"]
    out_adata.uns["scores"] = res["scores"]
    out_adata.uns["sizes"] = res["sizes"]
    out_adata.uns["hierarchy"] = res["simplified_hierarchy"]

    import numpy as np

    for i, assignments in enumerate(res["assignments"]):
        out_adata.obs[f"level_{i}_cluster"] = assignments
    for i, latent in enumerate(res["latents"]):
        out_adata.obsm[f"level_{i}_latent"] = np.array(latent)

    out_adata.write_h5ad(snakemake.output["out_fname"])

    with open(snakemake.output["duration_fname"], "w") as f:
        f.write(str(duration))


if __name__ == "__main__":
    main()
