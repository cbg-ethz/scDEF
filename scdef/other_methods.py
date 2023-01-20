import numpy as np
import anndata
import scanpy as sc
import scvi
import schpf
from sklearn.decomposition import NMF

def run_unintegrated(ad):
    # PCA
    sc.tl.pca(ad)
    latent = ad.obsm['X_pca']
    # Cluster
    sc.pp.neighbors(ad)
    sc.tl.leiden(ad)
    # Get gene signatures
    sc.tl.rank_genes_groups(ad, 'leiden', method='wilcoxon')
    gene_scores = []
    for leiden in range(np.max(ad.obs['leiden'].unique().astype(int)) + 1):
        gene_scores.append(sc.get.rank_genes_groups_df(ad, str(leiden)).set_index('names').loc[ad.var_names]['scores'].values)
    gene_scores = np.array(gene_scores)
    return latent, gene_scores, ad



def run_harmony(ad):
    ad = ad.copy()
    # PCA
    sc.tl.pca(ad)
    # Harmony
    sc.external.pp.harmony_integrate(ad, 'Batch')
    latent = ad.obsm["X_pca_harmony"]
    # Compute neighbors and do Leiden clustering
    sc.pp.neighbors(ad, use_rep="X_pca_harmony")
    sc.tl.leiden(ad)
    # Get gene signatures
    sc.tl.rank_genes_groups(ad, 'leiden', method='wilcoxon')
    gene_scores = []
    for leiden in range(np.max(ad.obs['leiden'].unique().astype(int)) + 1):
        gene_scores.append(sc.get.rank_genes_groups_df(ad, str(leiden)).set_index('names').loc[ad.var_names]['scores'].values)
    gene_scores = np.array(gene_scores)
    return latent, gene_scores, ad



def run_ldvae(ad, k_range):
    ad = ad.copy()
    scvi.model.LinearSCVI.setup_anndata(
        ad,
        batch_key="Batch",
        layer="counts",
    )
    # Run for range of K and choose best one
    models = []
    losses = []
    for k in k_range:
        model = scvi.model.LinearSCVI(ad, n_latent=k)
        model.train()
        models.append(model)
        losses.append(model.history['elbo_train'].values[-1][0])
    best = models[np.argmin(losses)]
    latent = best.get_latent_representation()
    loadings = best.get_loadings().values.T # factor by gene
    return latent, loadings, ad



def run_nmf(X, k_range):
    nmfs = []
    n_modules = []
    for k in k_range:
        # Run NMF
        nmf = NMF(n_components=k, max_iter=5000)
        W = nmf.fit_transform(X)
        V = nmf.components_ # K x P
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
    best = np.where(n_modules==np.array(k_range))[0][-1]

    W = nmfs[best][0]
    V = nmfs[best][1]
    X_nmf = np.argmax(W, axis=1).astype(str)
    return X_nmf, W, V



def run_scanorama(ad):
    ad = ad.copy()
    # PCA
    sc.tl.pca(ad)
    # scanorama
    sc.external.pp.scanorama_integrate(ad, 'Batch')
    latent = ad.obsm["X_scanorama"]
    # Compute neighbors and do Leiden clustering
    sc.pp.neighbors(ad, use_rep="X_scanorama")
    sc.tl.leiden(ad)
    # Get gene signatures
    sc.tl.rank_genes_groups(ad, 'leiden', method='wilcoxon')
    gene_scores = []
    for leiden in range(np.max(ad.obs['leiden'].unique().astype(int)) + 1):
        gene_scores.append(sc.get.rank_genes_groups_df(ad, str(leiden)).set_index('names').loc[ad.var_names]['scores'].values)
    gene_scores = np.array(gene_scores)
    return latent, gene_scores, ad


def run_schpf(X, k_range):
    models = []
    losses = []
    for k in k_range:
        sch = schpf.scHPF(k)
        sch.fit(X)
        models.append(sch)
        losses.append(sch.loss[-1])
    best = models[np.argmin(losses)]
    return best, best.cell_score(), best.gene_score()



def run_scvi(ad):
    ad = ad.copy()
    scvi.model.SCVI.setup_anndata(
        ad,
        layer="counts",
        batch_key="Batch",
    )
    model = scvi.model.SCVI(ad)
    model.train()
    latent = model.get_latent_representation()
    ad.obsm["X_scVI"] = latent
    # Cluster
    sc.pp.neighbors(ad, use_rep="X_scVI")
    sc.tl.leiden(ad)
    # Get gene signatures
    sc.tl.rank_genes_groups(ad, 'leiden', method='wilcoxon')
    gene_scores = []
    for leiden in range(np.max(ad.obs['leiden'].unique().astype(int)) + 1):
        gene_scores.append(sc.get.rank_genes_groups_df(ad, str(leiden)).set_index('names').loc[ad.var_names]['scores'].values)
    gene_scores = np.array(gene_scores)
    return latent, gene_scores, ad
