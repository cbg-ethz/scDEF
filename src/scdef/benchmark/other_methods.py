from scdef.utils import hierarchy_utils
from scdef.benchmark.constants import *
from scdef.models import scDEF

import numpy as np
from anndata import AnnData
import scanpy as sc
import logging

from typing import Optional, Sequence, Mapping, Callable


def run_multiple_resolutions(
    method: Callable,
    ad: AnnData,
    resolution_sweep: Sequence[float],
    layer_prefix: Optional[str] = "h",
    batch_key: Optional[str] = None,
    **kwargs,
) -> Mapping:
    """Run a clustering and gene signature learning method at multiple resolutions.

    Args:
        method: the function that runs the method. Must take a resolution parameter as argument and
            return a list containing at least an AnnData object, a matrix containing the latent space,
            and a list of genes per cluster.
        ad: the data to run the method on.
        resolution_sweep: list of resolution parameters to use.

    Returns:
        outs: dictionary containing all the outputs from the method across all resolutions. Keys: ["latents", "signatures", "assignments", "scores", "sizes", "simplified_hierarchy"]
    """
    # method is a function
    assignments_results = []
    signatures_dict = dict()
    scores_dict = dict()
    sizes_dict = dict()
    latents_results = []
    adata = ad.copy()
    for i, res in enumerate(resolution_sweep):
        outs = method(ad, resolution=res, batch_key=batch_key, **kwargs)
        latents = outs[0]
        latents_results.append(latents)
        scores = outs[1]
        assignments = outs[-1]
        signatures = outs[-2]
        prefix = layer_prefix * i
        assignments = [f"{prefix}{a}" for a in assignments]
        assignments_results.append(assignments)
        uq, cts = np.unique(assignments, return_counts=True)
        sizes = dict(zip(uq, cts))
        for k in range(len(signatures)):
            name = f"{prefix}{k}"
            signatures_dict[name] = signatures[k].tolist()
            scores_dict[name] = scores[k].tolist()
            try:
                sizes_dict[name] = sizes[name]
            except KeyError:
                sizes_dict[name] = 0
        adata.obsm[f"{prefix}_latent"] = latents
        adata.obs[f"{prefix}_cluster"] = assignments
    adata.uns["signatures"] = signatures_dict
    adata.uns["scores"] = scores_dict

    hierarchy = hierarchy_utils.get_hierarchy_from_clusters(assignments_results)
    layer_names = [layer_prefix * level for level in range(len(assignments_results))]
    layer_sizes = [len(np.unique(cluster)) for cluster in assignments_results]
    simplified = hierarchy_utils.simplify_hierarchy(hierarchy, layer_names, layer_sizes)

    adata.uns["hierarchy"] = simplified
    outs = {
        "latents": latents_results,
        "signatures": signatures_dict,
        "assignments": assignments_results,
        "scores": scores_dict,
        "sizes": sizes_dict,
        "simplified_hierarchy": simplified,
        "adata": adata,
    }
    return outs


def run_unintegrated(
    ad,
    resolution=1.0,
    sorted_scores=False,
    return_signatures=True,
    return_cluster_assignments=True,
    batch_key=None,
    **kwargs,
):
    try:
        import leidenalg
    except ImportError:
        raise ImportError(
            "Please install leidenalg: `pip install leidenalg`. Or install scdef with extras: `pip install scdef[extras]`."
        )
    # PCA
    sc.tl.pca(ad)
    latent = ad.obsm["X_pca"]
    # Cluster
    sc.pp.neighbors(ad)
    sc.tl.leiden(ad, resolution=resolution)
    # Get gene signatures
    sc.tl.rank_genes_groups(ad, "leiden", method="wilcoxon")
    gene_scores = []
    for leiden in range(np.max(ad.obs["leiden"].unique().astype(int)) + 1):
        scores = (
            sc.get.rank_genes_groups_df(ad, str(leiden))
            .set_index("names")
            .loc[ad.var_names]["scores"]
            .values
        )
        if sorted_scores:
            scores = np.sort(scores)[::-1]
        gene_scores.append(scores)
    gene_scores = np.array(gene_scores)

    outs = [latent, gene_scores, ad]
    if return_signatures:
        signatures = []
        for k in range(len(gene_scores)):
            signatures.append(ad.var_names[np.argsort(gene_scores[k])[::-1]])
        outs.append(signatures)
    if return_cluster_assignments:
        cluster_assignments = ad.obs["leiden"].values.tolist()
        outs.append(cluster_assignments)

    return outs


def run_nmf(
    ad,
    k_extra=2,
    layer="counts",
    resolution=10.0,
    return_signatures=True,
    return_cluster_assignments=True,
    batch_key=None,
    **kwargs,
):
    try:
        from sklearn.decomposition import NMF
    except ImportError:
        raise ImportError(
            "Please install scikit-learn: `pip install scikit-learn`. Or install scdef with extras: `pip install scdef[extras]`."
        )
    ad = ad.copy()
    X = ad.layers[layer]
    X = np.log(1e4 * X / np.sum(X, axis=1)[:, None] + 1)
    nmfs = []
    n_modules = []
    k_range = (
        np.arange(max(resolution - k_extra, 2), resolution + k_extra + 1)
    ).astype(int)
    for k in k_range:
        # Run NMF
        nmf = NMF(n_components=k, **kwargs)
        W = nmf.fit_transform(X)
        V = nmf.components_  # K x P
        nmfs.append([W, V])
        gene_to_factor_assignments = np.argmax(V, axis=0)
        m = 0
        if k > k_range[0]:
            for factor in range(k):
                # Top genes
                top_genes = np.argsort(V[factor])[::-1]
                genes = []
                for gene in top_genes:
                    if gene_to_factor_assignments[gene] == factor:
                        genes.append(gene)
                        if len(genes) > 5:
                            m += 1
                            break
                    else:
                        break
        else:
            m = k
        n_modules.append(m)

    # Select K: highest K for which n_modules == K
    best = np.where(n_modules == np.array(k_range))[0][-1]

    W = nmfs[best][0]
    V = nmfs[best][1]

    outs = [W, V, ad]
    if return_signatures:
        signatures = []
        for k in range(V.shape[0]):
            signatures.append(ad.var_names[np.argsort(V[k])[::-1]])
        outs.append(signatures)
    if return_cluster_assignments:
        cluster_assignments = np.argmax(W, axis=1).astype(str)
        outs.append(cluster_assignments)

    return outs


def run_schpf(
    ad,
    k_extra=2,
    layer="counts",
    resolution=10.0,
    return_signatures=True,
    return_cluster_assignments=True,
    batch_key=None,
    **kwargs,
):
    try:
        import schpf
    except ImportError:
        raise ImportError(
            "Please install schpf by following the instructions in https://github.com/simslab/scHPF."
        )
    import scipy

    ad = ad.copy()
    X = ad.layers[layer]
    X = scipy.sparse.coo_matrix(X)
    models = []
    losses = []
    k_range = (
        np.arange(max(resolution - k_extra, 2), resolution + k_extra + 1)
    ).astype(int)
    for k in k_range:
        sch = schpf.scHPF(k, **kwargs)
        sch.fit(X)
        models.append(sch)
        losses.append(sch.loss[-1])
    best = models[np.argmin(losses)]

    cscores = best.cell_score()
    gene_scores = np.array(best.gene_score()).T

    outs = [cscores, gene_scores, ad]
    if return_signatures:
        signatures = []
        for k in range(gene_scores.shape[0]):
            signatures.append(ad.var_names[np.argsort(gene_scores[k])[::-1]])
        outs.append(signatures)
    if return_cluster_assignments:
        cluster_assignments = np.argmax(cscores, axis=1).astype(str)
        outs.append(cluster_assignments)

    return outs


def run_muvi(
    ad,
    layer="counts",
    resolution=10.0,
    return_signatures=True,
    return_cluster_assignments=True,
    batch_key=None,
    **kwargs,
):
    try:
        import muvi
    except ImportError:
        raise ImportError("Please install muvi: pip install muvi")

    ad = ad.copy()

    model = muvi.tl.from_adata(
        ad, prior_mask_key=None, n_factors=int(resolution), nmf=True
    )
    model.fit(**kwargs)

    # Filter factors
    r2_view, r2_factor, _ = muvi.tl.variance_explained(model)
    model_cache = muvi.tools.utils.setup_cache(model)
    factor_adata = model_cache.factor_adata
    r2_cols = [f"r2_{vn}" for vn in model.view_names]
    r2_df = model_cache.factor_metadata[r2_cols]
    r2_sorted = r2_df.sum(1).sort_values(ascending=False)
    factor_subset = r2_sorted.index
    r2_thresh = (r2_sorted.cumsum() / r2_sorted.sum() < 0.95).sum() + 1

    factor_subset = r2_sorted.iloc[: int(r2_thresh)].index
    factor_subset = factor_subset.tolist()

    # Get cell and gene scores
    cscores = model.get_factor_scores(factor_idx=factor_subset)
    gene_scores = model.get_factor_loadings(factor_idx=factor_subset)
    gene_scores = gene_scores["view_0"]

    muvi.tl.filter_factors(model, r2_thresh=0.95)
    factor_adata = model_cache.factor_adata

    outs = [cscores, gene_scores, factor_adata]
    if return_signatures:
        signatures = []
        for k in range(gene_scores.shape[0]):
            signatures.append(ad.var_names[np.argsort(gene_scores[k])[::-1]])
        outs.append(signatures)
    if return_cluster_assignments:
        cluster_assignments = np.argmax(cscores, axis=1).astype(str)
        outs.append(cluster_assignments)

    return outs


def run_harmony(
    ad,
    batch_key="Batch",
    resolution=1.0,
    return_signatures=True,
    return_cluster_assignments=True,
    **kwargs,
):
    try:
        import harmonypy
    except ImportError:
        raise ImportError(
            "Please install harmonypy: `pip install harmonypy`. Or install scdef with extras: `pip install scdef[extras]`."
        )
    ad = ad.copy()
    # PCA
    sc.tl.pca(ad)
    # Harmony
    sc.external.pp.harmony_integrate(ad, batch_key, **kwargs)
    latent = ad.obsm["X_pca_harmony"]
    # Compute neighbors and do Leiden clustering
    sc.pp.neighbors(ad, use_rep="X_pca_harmony")
    sc.tl.leiden(ad, resolution=resolution)
    # Get gene signatures
    sc.tl.rank_genes_groups(ad, "leiden", method="wilcoxon")
    gene_scores = []
    for leiden in range(np.max(ad.obs["leiden"].unique().astype(int)) + 1):
        gene_scores.append(
            sc.get.rank_genes_groups_df(ad, str(leiden))
            .set_index("names")
            .loc[ad.var_names]["scores"]
            .values
        )
    gene_scores = np.array(gene_scores)

    outs = [latent, gene_scores, ad]
    if return_signatures:
        signatures = []
        for k in range(len(gene_scores)):
            signatures.append(ad.var_names[np.argsort(gene_scores[k])[::-1]])
        outs.append(signatures)
    if return_cluster_assignments:
        cluster_assignments = ad.obs["leiden"].values.tolist()
        outs.append(cluster_assignments)

    return outs


def run_scanorama(
    ad,
    batch_key="Batch",
    resolution=1.0,
    return_signatures=True,
    return_cluster_assignments=True,
    **kwargs,
):
    try:
        import scanorama
    except ImportError:
        raise ImportError(
            "Please install scanorama: `pip install scanorama`. Or install scdef with extras: `pip install scdef[extras]`."
        )
    ad = ad.copy()
    # PCA
    sc.tl.pca(ad)
    # scanorama
    sc.external.pp.scanorama_integrate(ad, batch_key, **kwargs)
    latent = ad.obsm["X_scanorama"]
    # Compute neighbors and do Leiden clustering
    sc.pp.neighbors(ad, use_rep="X_scanorama")
    sc.tl.leiden(ad, resolution=resolution)
    # Get gene signatures
    sc.tl.rank_genes_groups(ad, "leiden", method="wilcoxon")
    gene_scores = []
    for leiden in range(np.max(ad.obs["leiden"].unique().astype(int)) + 1):
        gene_scores.append(
            sc.get.rank_genes_groups_df(ad, str(leiden))
            .set_index("names")
            .loc[ad.var_names]["scores"]
            .values
        )
    gene_scores = np.array(gene_scores)

    outs = [latent, gene_scores, ad]
    if return_signatures:
        signatures = []
        for k in range(len(gene_scores)):
            signatures.append(ad.var_names[np.argsort(gene_scores[k])[::-1]])
        outs.append(signatures)
    if return_cluster_assignments:
        cluster_assignments = ad.obs["leiden"].values.tolist()
        outs.append(cluster_assignments)

    return outs


def run_scvi(
    ad,
    batch_key="Batch",
    layer="counts",
    resolution=1.0,
    return_signatures=True,
    return_cluster_assignments=True,
    **kwargs,
):
    try:
        import scvi
    except ImportError:
        raise ImportError("Please install scvi-tools: `pip install scvi-tools`.")

    if batch_key not in ad.obs.columns.tolist():
        batch_key = None

    ad = ad.copy()
    scvi.model.SCVI.setup_anndata(
        ad,
        layer=layer,
        batch_key=batch_key,
    )
    model = scvi.model.SCVI(ad)
    model.train(**kwargs)
    latent = model.get_latent_representation()
    ad.obsm["X_scVI"] = latent
    # Cluster
    sc.pp.neighbors(ad, use_rep="X_scVI")
    sc.tl.leiden(ad, resolution=resolution)
    # Get gene signatures
    sc.tl.rank_genes_groups(ad, "leiden", method="wilcoxon")
    gene_scores = []
    for leiden in range(np.max(ad.obs["leiden"].unique().astype(int)) + 1):
        gene_scores.append(
            sc.get.rank_genes_groups_df(ad, str(leiden))
            .set_index("names")
            .loc[ad.var_names]["scores"]
            .values
        )
    gene_scores = np.array(gene_scores)

    outs = [latent, gene_scores, ad]
    if return_signatures:
        signatures = []
        for k in range(len(gene_scores)):
            signatures.append(ad.var_names[np.argsort(gene_scores[k])[::-1]])
        outs.append(signatures)
    if return_cluster_assignments:
        cluster_assignments = ad.obs["leiden"].values.tolist()
        outs.append(cluster_assignments)

    return outs


def run_ldvae(
    ad,
    k_range=[5, 15],
    resolution=1.0,
    batch_key="Batch",
    layer="counts",
    **kwargs,
):
    try:
        import scvi
    except ImportError:
        raise ImportError("Please install scvi-tools: `pip install scvi-tools`.")

    if batch_key not in ad.obs.columns.tolist():
        batch_key = None

    ad = ad.copy()
    scvi.model.LinearSCVI.setup_anndata(
        ad,
        batch_key=batch_key,
        layer=layer,
    )
    # Run for range of K and choose best one
    models = []
    losses = []
    k_range = (np.array(k_range) * resolution).astype(int)
    for k in k_range:
        model = scvi.model.LinearSCVI(ad, n_latent=k)
        model.train(**kwargs)
        models.append(model)
        losses.append(model.history["elbo_train"].values[-1][0])
    best = models[np.argmin(losses)]
    latent = best.get_latent_representation()
    loadings = best.get_loadings().values.T  # factor by gene
    return latent, loadings, ad


def run_scdef_hclust(
    ad,
    n_epochs=10,
    resolutions=[6, 3, 1],
    layer_prefix="h",
    **kwargs,
):
    from scipy import cluster

    scd = scDEF(
        ad,
        layer_sizes=[100, 1],
        layer_shapes=0.3,
        layer_rates=0.3,
        seed=1,
        batch_key="Experiment",
        counts_layer="counts",
    )
    scd.learn(n_epoch=[n_epochs])

    latent = scd.adata.obsm["X_factors"]

    # Hierarchical clustering of factors: average per factor
    avgs = []
    cells = []
    scd.adata.obsm["avgs"] = np.zeros(latent.shape)
    for cl in scd.adata.obs["factor"].unique():
        cell_idx = np.where(scd.adata.obs["factor"] == cl)[0]
        avg = np.mean(latent[cell_idx], axis=0)
        avgs.append(avg)
        cells.append(cell_idx)
        scd.adata.obsm["avgs"][cell_idx] = avg
    avgs = np.vstack(avgs)
    Z = cluster.hierarchy.ward(scd.adata.obsm["avgs"])
    n_clusters = [avgs.shape[0]] + [
        max(avgs.shape[0] - avgs.shape[0] / res, 1) for res in resolutions
    ]
    cutree = cluster.hierarchy.cut_tree(Z, n_clusters=n_clusters)

    for level in range(len(resolutions) + 1):
        ad.obs[f"level_{level}"] = cutree[:, level].astype(str)

    assignments_results = []
    signatures_dict = dict()
    scores_dict = dict()
    sizes_dict = dict()
    latents_results = []
    for i, level in enumerate(range(len(resolutions) + 1)):
        # Get gene signatures for this level
        if i == 0:
            signatures, gene_scores = scd.get_signatures_dict(scores=True)
            signatures = np.array(
                [signatures[f] for f in scd.adata.obs["factor"].unique()]
            )
            gene_scores = np.array(
                [gene_scores[f] for f in scd.adata.obs["factor"].unique()]
            )
        else:
            sc.tl.rank_genes_groups(ad, f"level_{level}", method="wilcoxon")

            gene_scores = []
            for leiden in range(
                np.max(ad.obs[f"level_{level}"].unique().astype(int)) + 1
            ):
                gene_scores.append(
                    sc.get.rank_genes_groups_df(ad, str(leiden))
                    .set_index("names")
                    .loc[ad.var_names]["scores"]
                    .values
                )
            gene_scores = np.array(gene_scores)
            signatures = []
            for k in range(len(gene_scores)):
                signatures.append(ad.var_names[np.argsort(gene_scores[k])[::-1]])

        latent = scd.adata.obsm["X_factors"]

        outs = [latent, gene_scores, ad]
        outs.append(signatures)

        cluster_assignments = ad.obs[f"level_{level}"].values.tolist()
        outs.append(cluster_assignments)

        latents = outs[0]
        latents_results.append(latents)
        scores = outs[1]
        assignments = outs[-1]
        signatures = outs[-2]
        prefix = layer_prefix * i
        assignments = [f"{prefix}{a}" for a in assignments]
        assignments_results.append(assignments)
        uq, cts = np.unique(assignments, return_counts=True)
        sizes = dict(zip(uq, cts))
        for k in range(len(signatures)):
            name = f"{prefix}{k}"
            signatures_dict[name] = signatures[k].tolist()
            scores_dict[name] = scores[k].tolist()
            try:
                sizes_dict[name] = sizes[name]
            except KeyError:
                sizes_dict[name] = 0

    hierarchy = hierarchy_utils.get_hierarchy_from_clusters(assignments_results)
    layer_names = [layer_prefix * level for level in range(len(assignments_results))]
    layer_sizes = [len(np.unique(cluster)) for cluster in assignments_results]
    simplified = hierarchy_utils.simplify_hierarchy(hierarchy, layer_names, layer_sizes)

    outs = {
        "latents": latents_results,
        "signatures": signatures_dict,
        "assignments": assignments_results,
        "scores": scores_dict,
        "sizes": sizes_dict,
        "simplified_hierarchy": simplified,
    }
    return outs


def run_unintegrated_hclust(
    ad,
    resolutions=[6, 3, 1],
    layer_prefix="h",
    **kwargs,
):
    try:
        import leidenalg
    except ImportError:
        raise ImportError(
            "Please install leidenalg: `pip install leidenalg`. Or install scdef with extras: `pip install scdef[extras]`."
        )
    from scipy import cluster

    # PCA
    sc.tl.pca(ad)
    latent = ad.obsm["X_pca"]
    # Cluster
    sc.pp.neighbors(ad)
    sc.tl.leiden(ad)

    # Hierarchical clustering of factors: average per factor
    avgs = []
    cells = []
    ad.obsm["avgs"] = np.zeros(latent.shape)
    for cl in ad.obs["leiden"].unique():
        cell_idx = np.where(ad.obs["leiden"] == cl)[0]
        avg = np.mean(latent[cell_idx], axis=0)
        avgs.append(avg)
        cells.append(cell_idx)
        ad.obsm["avgs"][cell_idx] = avg
    avgs = np.vstack(avgs)
    Z = cluster.hierarchy.ward(ad.obsm["avgs"])
    n_clusters = [avgs.shape[0]] + [
        max(avgs.shape[0] - avgs.shape[0] / res, 1) for res in resolutions
    ]
    cutree = cluster.hierarchy.cut_tree(Z, n_clusters=n_clusters)

    # # Hierarchical clustering of Leiden groups: average per group
    # avgs = []
    # for cl in ad.obs['leiden'].unique():
    #     avgs.append(np.mean(latent[np.where(ad.obs['leiden']==cl)[0]], axis=0))
    # avgs = np.vstack(avgs)
    # Z = cluster.hierarchy.ward(avgs)
    # cutree = cluster.hierarchy.cut_tree(Z, n_clusters=[avgs.shape[0]] + resolutions)
    for level in range(len(resolutions) + 1):
        ad.obs[f"level_{level}"] = cutree[:, level].astype(str)

    assignments_results = []
    signatures_dict = dict()
    scores_dict = dict()
    sizes_dict = dict()
    latents_results = []
    for i, level in enumerate(range(len(resolutions) + 1)):
        # Get gene signatures for this level
        sc.tl.rank_genes_groups(ad, f"level_{level}", method="wilcoxon")
        gene_scores = []
        for cl in range(np.max(ad.obs[f"level_{level}"].unique().astype(int)) + 1):
            gene_scores.append(
                sc.get.rank_genes_groups_df(ad, str(cl))
                .set_index("names")
                .loc[ad.var_names]["scores"]
                .values
            )
        gene_scores = np.array(gene_scores)

        latent = ad.obsm["X_pca"]

        outs = [latent, gene_scores, ad]
        signatures = []
        for k in range(len(gene_scores)):
            signatures.append(ad.var_names[np.argsort(gene_scores[k])[::-1]])
        outs.append(signatures)

        cluster_assignments = ad.obs[f"level_{level}"].values.tolist()
        outs.append(cluster_assignments)

        latents = outs[0]
        latents_results.append(latents)
        scores = outs[1]
        assignments = outs[-1]
        signatures = outs[-2]
        prefix = layer_prefix * i
        assignments = [f"{prefix}{a}" for a in assignments]
        assignments_results.append(assignments)
        uq, cts = np.unique(assignments, return_counts=True)
        sizes = dict(zip(uq, cts))
        for k in range(len(signatures)):
            name = f"{prefix}{k}"
            signatures_dict[name] = signatures[k].tolist()
            scores_dict[name] = scores[k].tolist()
            try:
                sizes_dict[name] = sizes[name]
            except KeyError:
                sizes_dict[name] = 0

    hierarchy = hierarchy_utils.get_hierarchy_from_clusters(assignments_results)
    layer_names = [layer_prefix * level for level in range(len(assignments_results))]
    layer_sizes = [len(np.unique(cluster)) for cluster in assignments_results]
    simplified = hierarchy_utils.simplify_hierarchy(hierarchy, layer_names, layer_sizes)

    outs = {
        "latents": latents_results,
        "signatures": signatures_dict,
        "assignments": assignments_results,
        "scores": scores_dict,
        "sizes": sizes_dict,
        "simplified_hierarchy": simplified,
        "adata": ad,
    }
    return outs


def run_nsbm(
    ad,
    layer_prefix="h",
    batch_key=None,
    **kwargs,
):
    try:
        import schist as scs
    except ImportError:
        raise ImportError("Please install schist: conda install -c conda-forge schist")
    ad = ad.copy()
    # PCA
    sc.tl.pca(ad)
    # Compute neighbors
    if batch_key is not None:
        sc.external.pp.harmony_integrate(ad, batch_key)
        sc.pp.neighbors(ad, use_rep="X_pca_harmony")
    else:
        sc.pp.neighbors(ad)

    scs.inference.fit_model(ad, **kwargs)

    n_levels = len(ad.obs.filter(like="nsbm_level").columns)
    # Get good lowest resolution level for fairness
    min_level = 0
    for level in range(n_levels):
        avg_cluster_size = np.mean(ad.obs[f"nsbm_level_{level}"].value_counts())
        if avg_cluster_size > 50:
            min_level = level

    assignments_results = []
    signatures_dict = dict()
    scores_dict = dict()
    sizes_dict = dict()
    latents_results = []
    for i, level in enumerate(range(min_level, n_levels)):
        # Get gene signatures for this level
        sc.tl.rank_genes_groups(ad, f"nsbm_level_{level}", method="wilcoxon")
        gene_scores = []
        for leiden in range(
            np.max(ad.obs[f"nsbm_level_{level}"].unique().astype(int)) + 1
        ):
            gene_scores.append(
                sc.get.rank_genes_groups_df(ad, str(leiden))
                .set_index("names")
                .loc[ad.var_names]["scores"]
                .values
            )
        gene_scores = np.array(gene_scores)

        latent = ad.obsm["X_pca"]

        outs = [latent, gene_scores, ad]
        signatures = []
        for k in range(len(gene_scores)):
            signatures.append(ad.var_names[np.argsort(gene_scores[k])[::-1]])
        outs.append(signatures)

        cluster_assignments = ad.obs[f"nsbm_level_{level}"].values.tolist()
        outs.append(cluster_assignments)

        latents = outs[0]
        latents_results.append(latents)
        scores = outs[1]
        assignments = outs[-1]
        signatures = outs[-2]
        prefix = layer_prefix * i
        assignments = [f"{prefix}{a}" for a in assignments]
        assignments_results.append(assignments)
        uq, cts = np.unique(assignments, return_counts=True)
        sizes = dict(zip(uq, cts))
        for k in range(len(signatures)):
            name = f"{prefix}{k}"
            signatures_dict[name] = signatures[k].tolist()
            scores_dict[name] = scores[k].tolist()
            try:
                sizes_dict[name] = sizes[name]
            except KeyError:
                sizes_dict[name] = 0

    hierarchy = hierarchy_utils.get_hierarchy_from_clusters(assignments_results)
    layer_names = [layer_prefix * level for level in range(len(assignments_results))]
    layer_sizes = [len(np.unique(cluster)) for cluster in assignments_results]
    simplified = hierarchy_utils.simplify_hierarchy(hierarchy, layer_names, layer_sizes)

    outs = {
        "latents": latents_results,
        "signatures": signatures_dict,
        "assignments": assignments_results,
        "scores": scores_dict,
        "sizes": sizes_dict,
        "simplified_hierarchy": simplified,
        "adata": ad,
    }
    return outs


OTHERS_FUNCS = dict(
    zip(
        OTHERS_LABELS,
        [
            run_unintegrated,
            run_nsbm,
            run_scvi,
            run_harmony,
            run_scanorama,
            run_ldvae,
            run_nmf,
            run_schpf,
            run_muvi,
        ],
    )
)


# TODO: parallelize across methods
def run_methods(adata, methods_list, res_sweeps=None, batch_key=None, **kwargs):
    methods_outs = dict()
    for method in methods_list:
        logging.info(f"Running {method}...")

        # Run method
        func = OTHERS_FUNCS[method]
        res_sweep = OTHERS_RES_SWEEPS[method]
        if res_sweeps is not None:
            res_sweep = res_sweeps[method]
        if method != "nSBM":
            method_outs = run_multiple_resolutions(
                func, adata, res_sweep, batch_key=batch_key, **kwargs
            )
        else:
            method_outs = run_nsbm(adata, batch_key=batch_key, **kwargs)

        methods_outs[method] = method_outs

    return methods_outs
