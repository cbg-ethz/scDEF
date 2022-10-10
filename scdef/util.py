import numpy as np
import jax.numpy as jnp
from jax import vmap, random
from jax.scipy.stats import norm, gamma, poisson

def gaussian_sample(rng, mean, log_scale):
    scale = jnp.exp(log_scale)
    return mean + scale * random.normal(rng, mean.shape)

def gaussian_logpdf(x, mean, log_scale):
    scale = jnp.exp(log_scale)
    return jnp.sum(vmap(norm.logpdf)(x, mean * jnp.ones(x.shape), scale * jnp.ones(x.shape)))

def gamma_sample(rng, shape, rate):
    scale = 1./rate
    return jnp.clip(scale * random.gamma(rng, shape), a_min=1e-10, a_max=1e30)

def gamma_logpdf(x, shape, rate):
    scale = 1./rate
    return jnp.sum(vmap(gamma.logpdf)(x, shape * jnp.ones(x.shape), scale=scale * jnp.ones(x.shape)))

def get_mean_cellscore_per_group(cell_scores, cell_groups):
    unique_cluster_ids = np.unique(cell_groups)
    mean_cluster_scores = []
    for c in unique_cluster_ids:
        cell_idx = np.where(cell_groups == c)[0]
        mean_cluster_scores.append(np.mean(cell_scores[cell_idx], axis=0))
    mean_cluster_scores = np.array(mean_cluster_scores)
    return mean_cluster_scores

def mod_score(factors_by_groups_matrix):
    total_factor_relative_weight = np.sum(factors_by_groups_matrix,axis=1)
    total_factor_relative_weight = total_factor_relative_weight / np.sum(total_factor_relative_weight)

    # Per factor
    n1 = factors_by_groups_matrix/np.sum(factors_by_groups_matrix, axis=1)[:,np.newaxis]
    n1 = np.sum(np.max(n1, axis=1) * total_factor_relative_weight)

    # Per group
    n2 = factors_by_groups_matrix/np.sum(factors_by_groups_matrix, axis=0)[np.newaxis,:]
    n2 = np.mean(np.max(n2, axis=0))

    return np.mean([n1,n2])

def entropy_score(factors_by_groups_matrix):
    total_factor_relative_weight = np.sum(factors_by_groups_matrix,axis=1)
    total_factor_relative_weight = total_factor_relative_weight / np.sum(total_factor_relative_weight)

    # Per factor
    n1 = factors_by_groups_matrix/np.sum(factors_by_groups_matrix, axis=1)[:,np.newaxis]
    n1 = np.sum(-np.sum(n1*np.log(n1), axis=1) * total_factor_relative_weight)

    # Per group
    n2 = factors_by_groups_matrix/np.sum(factors_by_groups_matrix, axis=0)[np.newaxis,:]
    n2 = np.mean(-np.sum(n2*np.log(n2), axis=0))

    return np.mean([n1,n2])
