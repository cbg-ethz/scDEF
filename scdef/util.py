import numpy as np
import jax.numpy as jnp
from jax import vmap, random
from jax.scipy.stats import norm, gamma, poisson


def gaussian_sample(rng, mean, log_scale):
    scale = jnp.exp(log_scale)
    return mean + scale * random.normal(rng, mean.shape)


def gaussian_logpdf(x, mean, log_scale):
    scale = jnp.exp(log_scale)
    return jnp.sum(
        vmap(norm.logpdf)(x, mean * jnp.ones(x.shape), scale * jnp.ones(x.shape))
    )


def gamma_sample(rng, shape, rate):
    scale = 1.0 / rate
    return jnp.clip(scale * random.gamma(rng, shape), a_min=1e-15, a_max=1e15)


def gamma_logpdf(x, shape, rate):
    scale = 1.0 / rate
    return jnp.sum(
        vmap(gamma.logpdf)(
            x, shape * jnp.ones(x.shape), scale=scale * jnp.ones(x.shape)
        )
    )


def get_mean_cellscore_per_group(cell_scores, cell_groups):
    unique_cluster_ids = np.unique(cell_groups)
    mean_cluster_scores = []
    for c in unique_cluster_ids:
        cell_idx = np.where(cell_groups == c)[0]
        mean_cluster_scores.append(np.mean(cell_scores[cell_idx], axis=0))
    mean_cluster_scores = np.array(mean_cluster_scores)
    return mean_cluster_scores


def mod_score(factors_by_groups_matrix):
    total_factor_relative_weight = np.sum(factors_by_groups_matrix, axis=1)
    total_factor_relative_weight = total_factor_relative_weight / np.sum(
        total_factor_relative_weight
    )

    # Per factor
    n1 = (
        factors_by_groups_matrix
        / np.sum(factors_by_groups_matrix, axis=1)[:, np.newaxis]
    )
    n1 = np.sum(np.max(n1, axis=1) * total_factor_relative_weight)

    # Per group
    n2 = (
        factors_by_groups_matrix
        / np.sum(factors_by_groups_matrix, axis=0)[np.newaxis, :]
    )
    n2 = np.mean(np.max(n2, axis=0))

    return np.mean([n1, n2])


def entropy_score(factors_by_groups_matrix):
    total_factor_relative_weight = np.sum(factors_by_groups_matrix, axis=1)
    total_factor_relative_weight = total_factor_relative_weight / np.sum(
        total_factor_relative_weight
    )

    # Per factor
    n1 = (
        factors_by_groups_matrix
        / np.sum(factors_by_groups_matrix, axis=1)[:, np.newaxis]
    )
    n1 = np.sum(-np.sum(n1 * np.log(n1), axis=1) * total_factor_relative_weight)

    # Per group
    n2 = (
        factors_by_groups_matrix
        / np.sum(factors_by_groups_matrix, axis=0)[np.newaxis, :]
    )
    n2 = np.mean(-np.sum(n2 * np.log(n2), axis=0))

    return np.mean([n1, n2])


def compute_geneset_coherence(genes, counts_adata):
    # As in Spectra: https://github.com/dpeerlab/spectra/blob/ff0e5c456127a33938b1ea560432f228dc26a08b/spectra/initialization.py
    mat = np.array(counts_adata[:, genes].X)
    n_genes = len(genes)
    score = 0
    for i in range(1, n_genes):
        for j in range(i):
            dw1 = mat[:, i] > 0
            dw2 = mat[:, j] > 0
            dw1w2 = (dw1 & dw2).astype(float).sum()
            dw1 = dw1.astype(float).sum()
            dw2 = dw2.astype(float).sum()
            score += np.log((dw1w2 + 1) / (dw2))

    denom = n_genes * (n_genes - 1) / 2

    return score / denom


def coherence_score(marker_gene_sets, heldout_counts_adata):
    chs = []
    for marker_genes in marker_gene_sets:
        chs.append(compute_geneset_coherence(marker_genes, heldout_counts_adata))
    return np.mean(chs)
