import scanpy as sc
import scdef as scd
from scdef.utils import hierarchy_utils
import numpy as np
from scipy import cluster
import time


def main():
    adata = sc.read_h5ad(snakemake.input["adata"])

    batch_key = None
    if "Batch" in adata.obs.columns:
        batch_key = "Batch"

    resolutions = snakemake.params["resolutions"]

    model_settings = dict(
        brd_strength=float(snakemake.params["brd_strength"]),
        brd_mean=float(snakemake.params["brd_mean"]),
        n_factors=int(snakemake.params["n_factors"]),
        n_layers=1,
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
    model.filter_factors()

    latent = model.adata.obsm["X_L0"]

    avgs = []
    factor_labels = model.adata.obs["L0"].values
    for cl in np.unique(factor_labels):
        cell_idx = np.where(factor_labels == cl)[0]
        avgs.append(np.mean(latent[cell_idx], axis=0))
    avgs = np.vstack(avgs)

    avg_per_cell = np.zeros(latent.shape)
    for cl in np.unique(factor_labels):
        cell_idx = np.where(factor_labels == cl)[0]
        avg_per_cell[cell_idx] = np.mean(latent[cell_idx], axis=0)

    Z = cluster.hierarchy.ward(avg_per_cell)
    n_clusters = [avgs.shape[0]] + [
        max(int(avgs.shape[0] - avgs.shape[0] / res), 1) for res in resolutions
    ]
    cutree = cluster.hierarchy.cut_tree(Z, n_clusters=n_clusters)

    layer_prefix = "h"
    assignments_results = []
    signatures_dict = dict()
    scores_dict = dict()
    sizes_dict = dict()
    latents_results = []

    for i in range(len(resolutions) + 1):
        adata.obs[f"level_{i}"] = cutree[:, i].astype(str)

        if i == 0:
            sigs, gene_scores_d = model.get_signatures_dict(scores=True)
            unique_factors = np.unique(factor_labels)
            signatures = [sigs[f] for f in unique_factors]
            gene_scores = np.array([gene_scores_d[f] for f in unique_factors])
        else:
            level_col = f"level_{i}"
            group_counts = adata.obs[level_col].value_counts()
            valid_groups = group_counts[group_counts >= 2].index.tolist()
            if len(valid_groups) > 0:
                adata_sub = adata[adata.obs[level_col].isin(valid_groups)]
                sc.tl.rank_genes_groups(adata_sub, level_col, method="wilcoxon")

            all_groups = sorted(adata.obs[level_col].unique().astype(int))
            gene_scores = []
            for cl in all_groups:
                if str(cl) in valid_groups and len(valid_groups) > 0:
                    gene_scores.append(
                        sc.get.rank_genes_groups_df(adata_sub, str(cl))
                        .set_index("names")
                        .loc[adata.var_names]["scores"]
                        .values
                    )
                else:
                    gene_scores.append(np.zeros(adata.shape[1]))
            gene_scores = np.array(gene_scores)
            signatures = []
            for k in range(len(gene_scores)):
                signatures.append(adata.var_names[np.argsort(gene_scores[k])[::-1]])

        latents_results.append(latent)
        prefix = layer_prefix * i
        assignments = [f"{prefix}{a}" for a in adata.obs[f"level_{i}"].values]
        assignments_results.append(assignments)
        uq, cts = np.unique(assignments, return_counts=True)
        sizes = dict(zip(uq, cts))
        for k in range(len(signatures)):
            name = f"{prefix}{k}"
            sig = signatures[k]
            signatures_dict[name] = sig.tolist() if hasattr(sig, "tolist") else sig
            scores_dict[name] = gene_scores[k].tolist()
            try:
                sizes_dict[name] = int(sizes[name])
            except KeyError:
                sizes_dict[name] = 0

    duration = time.time() - duration

    hierarchy = hierarchy_utils.get_hierarchy_from_clusters(assignments_results)
    layer_names = [layer_prefix * level for level in range(len(assignments_results))]
    layer_sizes = [len(np.unique(cl)) for cl in assignments_results]
    simplified = hierarchy_utils.simplify_hierarchy(hierarchy, layer_names, layer_sizes)

    for i, prefix in enumerate(layer_names):
        adata.obsm[f"{prefix}_latent"] = latent
        adata.obs[f"{prefix}_cluster"] = assignments_results[i]
    adata.uns["signatures"] = signatures_dict
    adata.uns["scores"] = scores_dict
    adata.uns["sizes"] = sizes_dict
    adata.uns["hierarchy"] = simplified

    adata.write_h5ad(snakemake.output["out_fname"])

    with open(snakemake.output["duration_fname"], "w") as f:
        f.write(str(duration))


if __name__ == "__main__":
    main()
