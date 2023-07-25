import numpy as np
import anndata
import scanpy as sc
import scvi
import schpf
from sklearn.decomposition import NMF


def run_multiple_resolutions(method, ad, resolution_sweep, layer_prefix="h", **kwargs):
    # Only for methods that use Leiden clustering
    # method is a function
    assignments_results = []
    signatures_dict = dict()
    for i, res in enumerate(resolution_sweep):
        outs = method(ad, resolution=res, **kwargs)
        assignments = outs[-1]
        signatures = outs[-2]
        prefix = layer_prefix * i
        assignments = [f"{prefix}{a}" for a in assignments]
        assignments_results.append(assignments)
        for k in range(len(signatures)):
            name = f"{prefix}{k}"
            signatures_dict[name] = signatures[k].tolist()
    return signatures_dict, assignments_results


def run_unintegrated(
    ad, resolution=1.0, return_signatures=True, return_cluster_assignments=True
):
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


def run_harmony(
    ad,
    batch_key="Batch",
    resolution=1.0,
    return_signatures=True,
    return_cluster_assignments=True,
):
    ad = ad.copy()
    # PCA
    sc.tl.pca(ad)
    # Harmony
    sc.external.pp.harmony_integrate(ad, batch_key)
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


def run_ldvae(ad, k_range, batch_key="Batch"):
    ad = ad.copy()
    scvi.model.LinearSCVI.setup_anndata(
        ad,
        batch_key=batch_key,
        layer="counts",
    )
    # Run for range of K and choose best one
    models = []
    losses = []
    for k in k_range:
        model = scvi.model.LinearSCVI(ad, n_latent=k)
        model.train()
        models.append(model)
        losses.append(model.history["elbo_train"].values[-1][0])
    best = models[np.argmin(losses)]
    latent = best.get_latent_representation()
    loadings = best.get_loadings().values.T  # factor by gene
    return latent, loadings, ad


def run_nmf(ad, k_range, return_signatures=True, return_cluster_assignments=True):
    ad = ad.copy()
    X = ad.X
    nmfs = []
    n_modules = []
    for k in k_range:
        # Run NMF
        nmf = NMF(n_components=k, max_iter=5000)
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
        for k in range(len(gene_scores)):
            signatures.append(ad.var_names[np.argsort(V[k])])
        outs.append(signatures)
    if return_cluster_assignments:
        cluster_assignments = np.argmax(W, axis=1).astype(str)
        outs.append(cluster_assignments)

    return outs


def run_scanorama(
    ad,
    batch_key="Batch",
    resolution=1.0,
    return_signatures=True,
    return_cluster_assignments=True,
):
    ad = ad.copy()
    # PCA
    sc.tl.pca(ad)
    # scanorama
    sc.external.pp.scanorama_integrate(ad, batch_key)
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


def run_schpf(ad, k_range, return_signatures=True, return_cluster_assignments=True):
    ad = ad.copy()
    X = scipy.sparse.coo_matrix(ad.X)
    models = []
    losses = []
    for k in k_range:
        sch = schpf.scHPF(k)
        sch.fit(X)
        models.append(sch)
        losses.append(sch.loss[-1])
    best = models[np.argmin(losses)]

    cscores = best.cell_score()
    gene_scores = best.gene_score()

    outs = [cscores, gene_scores, ad]
    if return_signatures:
        signatures = []
        for k in range(len(gene_scores)):
            signatures.append(ad.var_names[np.argsort(gene_scores[k])[::-1]])
        outs.append(signatures)
    if return_cluster_assignments:
        cluster_assignments = np.argmax(cscores, axis=1).astype(str)
        outs.append(cluster_assignments)

    return outs


def run_scvi(
    ad,
    batch_key="Batch",
    resolution=1.0,
    return_signatures=True,
    return_cluster_assignments=True,
):
    ad = ad.copy()
    scvi.model.SCVI.setup_anndata(
        ad,
        layer="counts",
        batch_key=batch_key,
    )
    model = scvi.model.SCVI(ad)
    model.train()
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
