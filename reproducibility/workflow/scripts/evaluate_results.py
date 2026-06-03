from _benchmark import evaluate_methods
from scdef import scDEF

import scanpy as sc
import numpy as np
import scdef


def load_scdef_model(model_dir):
    return scDEF.load(model_dir)


def load_method_results_from_h5ad(h5ad_path):
    adata = sc.read_h5ad(h5ad_path)
    latent_keys = sorted([k for k in adata.obsm if k.endswith("_latent")])
    cluster_keys = sorted([k for k in adata.obs.columns if k.endswith("_cluster")])
    return {
        "latents": [adata.obsm[k] for k in latent_keys],
        "assignments": [adata.obs[k].values.tolist() for k in cluster_keys],
        "signatures": adata.uns["signatures"],
        "scores": adata.uns["scores"],
        "sizes": adata.uns["sizes"],
        "simplified_hierarchy": adata.uns["hierarchy"],
    }


def main():
    adata = sc.read_h5ad(snakemake.input["adata"])
    method = snakemake.params["method"]
    metrics = snakemake.params["metrics"]
    is_scdef = snakemake.params.get("is_scdef", False)

    hierarchy_obs = adata.uns["hierarchy_obs"].tolist()
    true_hierarchy = scdef.hierarchy_utils.get_hierarchy_from_clusters(
        [adata.obs[o].values for o in hierarchy_obs],
        use_names=True,
    )

    batch_key = None
    if "Batch" in adata.obs.columns:
        batch_key = "Batch"

    if is_scdef:
        model = load_scdef_model(snakemake.input["model_state"])
        methods_results = {method: model}
    else:
        results = load_method_results_from_h5ad(snakemake.input["model_state"])
        methods_results = {method: results}

    df = evaluate_methods(
        adata,
        metrics,
        methods_results,
        true_hierarchy=true_hierarchy,
        hierarchy_obs_keys=hierarchy_obs,
        markers=adata.uns["true_markers"],
        celltype_obs_key=hierarchy_obs,
        batch_obs_key=batch_key,
    )

    duration_fname = snakemake.input.get("duration_fname", None)
    if duration_fname:
        with open(duration_fname) as f:
            duration = float(f.read().strip())
        df.loc["Runtime", method] = duration

    if is_scdef and hasattr(methods_results[method], "alpha"):
        df.loc["alpha", method] = float(methods_results[method].alpha)

    df.to_csv(snakemake.output["scores_fname"])


if __name__ == "__main__":
    main()
