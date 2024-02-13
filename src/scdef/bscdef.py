from .scdef import *

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


class bscDEF(scDEF):
    def __init__(self, adata, **kwargs):
        """
        markers_matrix is a dataframe where the indices are gene symbols, columns are cell groups,
        and each entry is either 0 or 1
        """
        self.n_hfactors = 1
        super(bscDEF, self).__init__(adata, **kwargs)
        if self.n_batches == 0:
            raise ValueError("bscDEF requires more than one batch")
        self.n_hfactors = self.n_batches
        self.layer_sizes = [self.n_factors, self.n_hfactors]
        self.init_var_params()
        self.set_posterior_means()

    def init_var_params(self):
        rngs = random.split(random.PRNGKey(self.seed), 12)

        self.var_params = [
            jnp.array(
                (
                    jnp.log(
                        random.uniform(
                            rngs[0], minval=0.5, maxval=1.5, shape=[self.n_cells, 1]
                        )
                    ),  # cell_scales
                    jnp.log(
                        random.uniform(
                            rngs[1], minval=0.5, maxval=1.5, shape=[self.n_cells, 1]
                        )
                    ),
                )
            ),
            jnp.array(
                (
                    jnp.log(
                        random.uniform(
                            rngs[2], minval=0.5, maxval=1.5, shape=[1, self.n_genes]
                        )
                    ),  # gene_scales
                    jnp.log(
                        random.uniform(
                            rngs[3], minval=0.5, maxval=1.5, shape=[1, self.n_genes]
                        )
                    ),
                )
            ),
            jnp.array(
                (
                    jnp.log(
                        random.uniform(
                            rngs[4], minval=0.5, maxval=1.5, shape=[self.n_factors, 1]
                        )
                    ),  # factor_scales
                    jnp.log(
                        random.uniform(
                            rngs[5], minval=0.5, maxval=1.5, shape=[self.n_factors, 1]
                        )
                    ),
                )
            ),
            jnp.array(
                (
                    jnp.log(
                        random.uniform(
                            rngs[6],
                            minval=0.5,
                            maxval=1.5,
                            shape=[self.n_hfactors, self.n_factors],
                        )
                    ),  # hW
                    jnp.log(
                        random.uniform(
                            rngs[7],
                            minval=0.5,
                            maxval=1.5,
                            shape=[self.n_hfactors, self.n_factors],
                        )
                    ),
                )
            ),
            jnp.array(
                (
                    jnp.log(
                        random.uniform(
                            rngs[8],
                            minval=0.5,
                            maxval=1.5,
                            shape=[self.n_cells, self.n_factors],
                        )
                    ),  # z
                    jnp.log(
                        random.uniform(
                            rngs[9],
                            minval=0.5,
                            maxval=1.5,
                            shape=[self.n_cells, self.n_factors],
                        )
                    ),
                )
            ),
            jnp.array(
                (
                    jnp.log(
                        random.uniform(
                            rngs[10],
                            minval=0.5,
                            maxval=1.5,
                            shape=[self.n_factors, self.n_genes],
                        )
                    ),  # W
                    jnp.log(
                        random.uniform(
                            rngs[11],
                            minval=0.5,
                            maxval=1.5,
                            shape=[self.n_factors, self.n_genes],
                        )
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
        factor_scale_params = jnp.clip(var_params[2], a_min=min_loc)
        hW_params = jnp.clip(var_params[3], a_min=min_loc)
        z_params = jnp.clip(var_params[4], a_min=min_loc)
        W_params = jnp.clip(var_params[5], a_min=min_loc)

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
        factor_scales = gamma_sample(rng, factor_scale_concentration, factor_scale_rate)

        hW = gamma_sample(rng, hW_concentration, hW_rate)
        mean_top = jnp.matmul(batch_indices_onehot, hW)

        z = gamma_sample(rng, z_concentration, z_rate)
        W = gamma_sample(rng, W_concentration, W_rate)
        mean_bottom = jnp.matmul(z / cell_budgets, W)

        # Compute log likelihood
        ll = jnp.sum(vmap(poisson.logpmf)(self.X[indices], mean_bottom))

        # Compute KL divergence
        kl = 0.0
        kl += gamma_logpdf(gene_budgets, 1.0, self.gene_ratio) - gamma_logpdf(
            gene_budgets, gene_budget_concentration, gene_budget_rate
        )

        kl += gamma_logpdf(factor_scales, 1.0, 1.0) - gamma_logpdf(
            factor_scales, factor_scale_concentration, factor_scale_rate
        )

        kl += gamma_logpdf(hW, 0.3, 1.0) - gamma_logpdf(hW, hW_concentration, hW_rate)

        kl += gamma_logpdf(W, 0.3, 1.0 * gene_budgets / factor_scales) - gamma_logpdf(
            W, W_concentration, W_rate
        )

        kl *= indices.shape[0] / self.X.shape[0]  # scale by minibatch size

        kl += gamma_logpdf(cell_budgets, 1.0, self.batch_lib_ratio[0]) - gamma_logpdf(
            cell_budgets, cell_budget_concentration, cell_budget_rate
        )

        # Tieing the scales avoids the factors with low scales in W learn z's that correlate with cell scales
        kl += gamma_logpdf(z, self.shape, self.shape / (mean_top)) - gamma_logpdf(
            z, z_concentration, z_rate
        )

        return ll + kl

    def get_dimensions(self, threshold=0.1):
        # Look at the scale of the factors and get the ones which are actually used
        return dict(factor=np.where(self.pmeans["factor_scale"] > threshold)[0])

    def set_posterior_means(self):
        cell_scale_params = self.var_params[0]
        gene_scale_params = self.var_params[1]
        factor_scale_params = self.var_params[2]
        hW_params = self.var_params[3]
        z_params = self.var_params[4]
        W_params = self.var_params[5]

        self.pmeans = {
            "cell_scale": np.array(
                jnp.exp(cell_scale_params[0]) / jnp.exp(cell_scale_params[1])
            ),
            "gene_scale": np.array(
                jnp.exp(gene_scale_params[0]) / jnp.exp(gene_scale_params[1])
            ),
            "factor_scale": np.array(
                jnp.exp(factor_scale_params[0]) / jnp.exp(factor_scale_params[1])
            ),
            "hW": np.array(jnp.exp(hW_params[0]) / jnp.exp(hW_params[1])),
            "z": np.array(jnp.exp(z_params[0]) / jnp.exp(z_params[1])),
            "W": np.array(jnp.exp(W_params[0]) / jnp.exp(W_params[1])),
        }

        self.pvars = {
            "cell_scale": np.array(
                jnp.exp(cell_scale_params[0]) * jnp.exp(cell_scale_params[1]) ** 2
            ),
            "gene_scale": np.array(
                jnp.exp(gene_scale_params[0]) * jnp.exp(gene_scale_params[1]) ** 2
            ),
            "factor_scale": np.array(
                jnp.exp(factor_scale_params[0]) * jnp.exp(factor_scale_params[1]) ** 2
            ),
            "hW": np.array(jnp.exp(hW_params[0]) * jnp.exp(hW_params[1]) ** 2),
            "z": np.array(jnp.exp(z_params[0]) * jnp.exp(z_params[1]) ** 2),
            "W": np.array(jnp.exp(W_params[0]) * jnp.exp(W_params[1]) ** 2),
        }

    def filter_factors(self, ard=0.5, annotate=True):
        thres = np.quantile(self.pmeans[f"{self.layer_names[0]}_scale"], q=ard)
        tokeep = np.where(self.pmeans[f"{self.layer_names[0]}_scale"] >= thres)[0]

        if annotate:
            self.annotate_adata(tokeep=tokeep)
        else:
            return tokeep

    def annotate_adata(self, tokeep=None, gene_budgets_correct=True):
        if tokeep is None:
            tokeep = np.arange(self.n_factors)
        elif not isinstance(tokeep, np.array):
            raise TypeError("`tokeep` must be an array!")

        self.hpal = sns.color_palette("husl", len(self.n_hfactors))

        self.adata.obsm["X_hfactors"] = self.batch_indices_onehot
        self.adata.obsm["X_factors"] = (
            self.pmeans["z"][:, tokeep] * self.pmeans["cell_scale"]
        )  # / self.pmeans['factor_budget'].T[:,tokeep[0]]
        #         self.adata.obsm['X_factors_var'] = self.pvars['z'][:,tokeep]
        #         self.adata.obsm['X_factors_mean'] = self.pmeans['hz'].dot(self.pmeans['hW'])
        self.logger.info("Updated adata.obsm: `X_hfactors` and `X_factors`.")

        self.adata.obs["X_hfactor"] = np.argmax(
            self.adata.obsm["X_hfactors"], axis=1
        ).astype(str)
        self.adata.obs["X_factor"] = np.argmax(
            self.adata.obsm["X_factors"], axis=1
        ).astype(str)
        self.adata.obs["cell_scale"] = 1 / self.pmeans["cell_scale"]
        self.adata.uns["X_factor_colors"] = []
        for i in range(len(tokeep)):
            self.adata.uns["X_factor_colors"].append(
                matplotlib.colors.to_hex(self.fpal[i])
            )
            self.adata.obs[f"cell_score_f{i}"] = self.adata.obsm["X_factors"][:, i]
        #             self.adata.obs[f'm{i}'] = self.adata.obsm['X_factors_mean'][:,i]
        #             self.adata.obs[f'v{i}'] = self.adata.obsm['X_factors_var'][:,i]
        self.adata.uns["X_hfactor_colors"] = []
        for i, idx in enumerate(self.n_hfactors):
            self.adata.uns["X_hfactor_colors"].append(
                matplotlib.colors.to_hex(self.hpal[i])
            )
            self.adata.obs[f"cell_score_h{i}"] = self.adata.obsm["X_hfactors"][:, i]
        self.logger.info(
            "Updated adata.obs: `X_hfactor`, `X_factor`, 'cell_scale', and cell weights for each "
            "factor and hierarchical factor."
        )

        self.adata.var["gene_scale"] = 1 / self.pmeans["gene_scale"].T
        for i in range(len(tokeep)):
            if gene_budgets_correct:
                self.adata.var[f"gene_score_{i}"] = (
                    self.pmeans["W"][tokeep, :] * self.pmeans["gene_scale"]
                )[i, :]
            else:
                self.adata.var[f"gene_score_{i}"] = (self.pmeans["W"][tokeep, :])[i, :]
        self.logger.info(
            "Updated adata.var: `gene_scale` and gene weights for each factor."
        )

    def plot_ard(self, **kwargs):
        scDEF.plot_ard(self, layer_idx=0, **kwargs)

    def get_graph(self, factor_list=None, ard=0.0, **kwargs):
        if factor_list is None:
            factor_list = self.filter_factors(ard=ard, annotate=False)
        return scDEF.get_graph(
            self,
            hfactor_list=np.arange(self.n_hfactors),
            factor_list=factor_list,
            **kwargs,
        )
