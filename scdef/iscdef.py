from .scdef import scDEF

from functools import partial
import string

from jax import jit, grad, vmap
from jax.example_libraries import optimizers
from jax import random, value_and_grad
from jax.scipy.stats import norm, gamma, poisson
import jax.numpy as jnp
import jax.nn as jnn

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import gseapy as gp
from graphviz import Graph
from tqdm import tqdm
import time

import logging

import scipy
import numpy as np
from anndata import AnnData


class iscDEF(scDEF):
    def __init__(
        self,
        adata,
        markers_matrix,
        n_factors_per_hfactor=2,
        gs_big_scale=2.0,
        cn_big_scale=1.0,
        gene_set_strength=1000,
        **kwargs
    ):
        """
        markers_matrix is a dataframe where the indices are gene symbols, columns are cell groups,
        and each entry is either 0 or 1
        """
        n_hfactors = markers_matrix.shape[1]
        n_factors = n_factors_per_hfactor * markers_matrix.shape[1]
        self.connectivity_matrix = np.ones((n_hfactors, n_factors))
        self.gene_sets = np.ones((n_factors, adata.X.shape[1]))

        super(iscDEF, self).__init__(
            adata, n_hfactors=n_hfactors, n_factors=n_factors, **kwargs
        )

        self.markers_matrix = markers_matrix
        self.n_factors_per_hfactor = n_factors_per_hfactor
        self.gs_big_scale = gs_big_scale
        self.cn_big_scale = cn_big_scale
        self.gene_set_strength = gene_set_strength

        # Build hW and W priors
        gs_small_scale = 1 / self.gs_big_scale
        cn_small_scale = 0.1
        self.connectivity_matrix = cn_small_scale * np.ones(
            (self.n_hfactors, self.n_factors)
        )
        self.gene_sets = np.ones((self.n_factors, self.n_genes))
        self.marker_gene_locs = []
        for i, cellgroup in enumerate(self.markers_matrix.columns):
            self.connectivity_matrix[
                i, i * self.n_factors_per_hfactor : (i + 1) * self.n_factors_per_hfactor
            ] = self.cn_big_scale
            for gene in self.markers_matrix.index:
                loc = np.where(self.adata.var.index == gene)[0]
                self.marker_gene_locs.append(loc)
                if self.markers_matrix[cellgroup].loc[gene] == 1:
                    self.gene_sets[
                        i
                        * self.n_factors_per_hfactor : (i + 1)
                        * self.n_factors_per_hfactor,
                        loc,
                    ] = self.gs_big_scale
                else:
                    self.gene_sets[
                        i
                        * self.n_factors_per_hfactor : (i + 1)
                        * self.n_factors_per_hfactor,
                        loc,
                    ] = gs_small_scale

        self.init_var_params()

    def init_var_params(self):
        self.var_params = [
            jnp.array(
                (
                    np.log(np.random.uniform(0.5, 1.5, size=[self.n_cells, 1])),
                    np.log(np.random.uniform(0.5, 1.5, size=[self.n_cells, 1])),
                )
            ),
            jnp.array(
                (
                    np.log(np.random.uniform(0.5, 1.5, size=[1, self.n_genes])),
                    np.log(np.random.uniform(0.5, 1.5, size=[1, self.n_genes])),
                )
            ),
            jnp.array(
                (
                    np.log(np.random.uniform(0.5, 1.5, size=[self.n_hfactors, 1])),
                    np.log(np.random.uniform(0.5, 1.5, size=[self.n_hfactors, 1])),
                )
            ),  # hfactor_scales
            jnp.array(
                (
                    np.log(np.random.uniform(0.5, 1.5, size=[self.n_factors, 1])),
                    np.log(np.random.uniform(0.5, 1.5, size=[self.n_factors, 1])),
                )
            ),  # factor_scales
            jnp.array(
                (
                    np.log(
                        np.random.uniform(
                            0.5 * self.shape,
                            1.5 * self.shape,
                            size=[self.n_cells, self.n_hfactors],
                        )
                    ),  # hz
                    np.log(
                        np.random.uniform(
                            0.5 * self.shape,
                            1.5 * self.shape,
                            size=[self.n_cells, self.n_hfactors],
                        )
                    ),
                )
            ),
            jnp.array(
                (
                    np.log(
                        np.random.uniform(
                            0.5 * self.connectivity_matrix,
                            1.5 * self.connectivity_matrix,
                        )
                    ),  # hW
                    np.log(
                        np.random.uniform(
                            0.5, 1.5, size=[self.n_hfactors, self.n_factors]
                        )
                    ),
                )
            ),
            jnp.array(
                (
                    np.log(
                        np.random.uniform(
                            0.5 * self.shape,
                            1.5 * self.shape,
                            size=[self.n_cells, self.n_factors],
                        )
                    ),  # z
                    np.log(
                        np.random.uniform(0.5, 1.5, size=[self.n_cells, self.n_factors])
                    ),
                )
            ),
            jnp.array(
                (
                    np.log(
                        np.random.uniform(0.5 * self.gene_sets, 1.5 * self.gene_sets)
                    ),  # W
                    np.log(
                        np.random.uniform(0.5, 1.5, size=[self.n_factors, self.n_genes])
                    ),
                )
            ),
        ]

    def elbo(self, rng, indices, var_params):
        # Single-sample Monte Carlo estimate of the variational lower bound.
        batch_indices_onehot = self.batch_indices_onehot[indices]

        min_loc = jnp.log(1e-10)
        cell_budget_params = jnp.clip(var_params[0], a_min=min_loc)
        gene_budget_params = jnp.clip(var_params[1], a_min=min_loc)
        hz_params = jnp.clip(var_params[4], a_min=min_loc)
        hW_params = jnp.clip(var_params[5], a_min=min_loc)
        z_params = jnp.clip(var_params[6], a_min=min_loc)
        W_params = jnp.clip(var_params[7], a_min=min_loc)

        min_concentration = 1e-10
        min_scale = 1e-10
        min_rate = 1e-10
        max_rate = 1e10
        cell_budget_concentration = jnp.maximum(
            jnp.exp(cell_budget_params[0][indices]), min_concentration
        )
        cell_budget_rate = jnp.minimum(
            jnp.maximum(jnp.exp(cell_budget_params[1][indices]), min_rate), max_rate
        )

        z_concentration = jnp.maximum(jnp.exp(z_params[0][indices]), min_concentration)
        z_rate = jnp.minimum(
            jnp.maximum(jnp.exp(z_params[1][indices]), min_rate), max_rate
        )

        hz_concentration = jnp.maximum(
            jnp.exp(hz_params[0][indices]), min_concentration
        )
        hz_rate = jnp.minimum(
            jnp.maximum(jnp.exp(hz_params[1][indices]), min_rate), max_rate
        )

        gene_budget_concentration = jnp.maximum(
            jnp.exp(gene_budget_params[0]), min_concentration
        )
        gene_budget_rate = jnp.minimum(
            jnp.maximum(jnp.exp(gene_budget_params[1]), min_rate), max_rate
        )

        factor_scale_concentration = jnp.maximum(
            jnp.exp(factor_scale_params[0]), min_concentration
        )
        factor_scale_rate = jnp.minimum(
            jnp.maximum(jnp.exp(factor_scale_params[1]), min_rate), max_rate
        )

        W_concentration = jnp.maximum(jnp.exp(W_params[0]), min_concentration)
        W_rate = jnp.minimum(jnp.maximum(jnp.exp(W_params[1]), min_rate), max_rate)

        hW_concentration = jnp.maximum(jnp.exp(hW_params[0]), min_concentration)
        hW_rate = jnp.minimum(jnp.maximum(jnp.exp(hW_params[1]), min_rate), max_rate)

        # Sample from variational distribution
        cell_budgets = gamma_sample(rng, cell_budget_concentration, cell_budget_rate)
        gene_budgets = gamma_sample(rng, gene_budget_concentration, gene_budget_rate)

        hz = gamma_sample(rng, hz_concentration, hz_rate)
        hW = gamma_sample(rng, hW_concentration, hW_rate)
        mean_top = jnp.matmul(hz, hW)
        z = gamma_sample(rng, z_concentration, z_rate)
        W = gamma_sample(rng, W_concentration, W_rate)
        mean_bottom_bio = jnp.matmul(z, W)
        mean_bottom = mean_bottom_bio

        # Compute log likelihood
        ll = jnp.sum(vmap(poisson.logpmf)(self.X[indices], mean_bottom))

        # Compute KL divergence
        kl = 0.0
        kl += gamma_logpdf(gene_budgets, 1.0, self.gene_ratio) - gamma_logpdf(
            gene_budgets, gene_budget_concentration, gene_budget_rate
        )

        kl += gamma_logpdf(hW, self.connectivity_matrix, 1.0) - gamma_logpdf(
            hW, hW_concentration, hW_rate
        )

        kl += gamma_logpdf(
            W,
            self.gene_set_strength * self.gene_sets,
            self.gene_set_strength * gene_budgets,
        ) - gamma_logpdf(W, W_concentration, W_rate)

        kl *= indices.shape[0] / self.X.shape[0]  # scale by minibatch size

        kl += gamma_logpdf(
            cell_budgets, 1.0, 1.0 * self.batch_lib_ratio[indices]
        ) - gamma_logpdf(cell_budgets, cell_budget_concentration, cell_budget_rate)

        kl += gamma_logpdf(hz, self.shape, self.shape) - gamma_logpdf(
            hz, hz_concentration, hz_rate
        )

        # Tieing the scales avoids the factors with low scales in W learn z's that correlate with cell scales
        kl += gamma_logpdf(
            z, self.shape, cell_budgets * self.shape / mean_top
        ) - gamma_logpdf(z, z_concentration, z_rate)

        return ll + kl
