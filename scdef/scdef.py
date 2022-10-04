from functools import partial
import string

from jax import jit, grad, vmap
from jax.example_libraries import optimizers
from jax import random, value_and_grad
from jax.scipy.stats import norm, gamma, poisson
import jax.numpy as jnp
import jax.nn as jnn

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import gseapy as gp
from graphviz import Graph
from tqdm import tqdm
import time

import logging

import scipy
from anndata import AnnData
from jax.config import config
config.update("jax_debug_nans", False)
config.update("jax_debug_infs", False)

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

class scDEF(object):
    def __init__(self, adata, n_factors=10, n_hfactors=3, shape=.3, batch_key='batch', logginglevel=logging.INFO):
        self.logger = logging.getLogger('scDEF')
        self.logger.setLevel(logginglevel)

        self.n_factors = n_factors
        self.n_hfactors = n_hfactors
        self.n_batches = 0
        self.shape = shape
        self.layer_names = ['factor', 'hfactor']
        self.layer_sizes = [n_factors, n_hfactors]
        self.load_adata(adata, batch_key=batch_key)
        self.batch_key = batch_key
        self.n_cells, self.n_genes = adata.shape
        self.init_var_params()
        self.set_posterior_means()
        self.hpal = sns.color_palette('Set2', n_hfactors)
        self.graph = self.get_graph()

    def load_adata(self, adata, batch_key='batch'):
        if not isinstance(adata, AnnData):
            raise TypeError("adata must be an instance of AnnData.")
        self.adata = adata
        self.adata.raw = self.adata
        if isinstance(self.adata.raw.X, scipy.sparse.csr_matrix):
            self.X = np.array(self.adata.raw.X.toarray())
        else:
            self.X = np.array(self.adata.raw.X)
        self.n_batches = 1
        self.batch_indices_onehot = np.ones((self.adata.shape[0], self.n_batches))
        self.batch_lib_sizes = np.sum(self.X, axis=1)
        self.batch_lib_ratio = np.ones((self.X.shape[0],)) * np.mean(self.batch_lib_sizes)/np.var(self.batch_lib_sizes)
        if batch_key in self.adata.obs.columns:
            batches = np.array(self.adata.obs[batch_key].values.unique())
            self.n_batches = len(batches)
            if self.n_batches == 1:
                self.n_batches = 0
            self.logger.info(f"Found {self.n_batches} values for `{batch_key}` in data: {batches}")
            self.batch_indices_onehot = np.zeros((self.adata.shape[0], self.n_batches))
            if self.n_batches > 1:
                for i, b in enumerate(batches):
                    cells = np.where(self.adata.obs[batch_key] == b)[0]
                    self.batch_indices_onehot[cells, i] = 1
                    self.batch_lib_sizes[cells] = np.sum(self.X, axis=1)[cells]
                    self.batch_lib_ratio[cells] = np.mean(self.batch_lib_sizes[cells])/np.var(self.batch_lib_sizes[cells])
        self.batch_indices_onehot = jnp.array(self.batch_indices_onehot)
        self.batch_lib_sizes = jnp.array(self.batch_lib_sizes)
        self.batch_lib_ratio = jnp.array(self.batch_lib_ratio)
        self.X = jnp.array(self.X)

    def init_var_params(self):
        self.var_params = [
            jnp.array((np.random.normal(0.5, 0.1, size=self.n_cells),
                       np.random.normal(0., 0.1, size=self.n_cells))), # cell scale
            jnp.array((np.random.normal(0.5, 0.1, size=self.n_genes),
                       np.random.normal(0., 0.1, size=self.n_genes))), # gene scale
            jnp.array((np.random.normal(0, 0.1, size=self.n_hfactors),
                       np.random.normal(0., 0.1, size=self.n_hfactors))), # hfactor scale
            jnp.array((np.random.normal(0, 0.1, size=self.n_factors),
                       np.random.normal(0., 0.1, size=self.n_factors))), # factor scales
            jnp.array(((np.random.normal(.5, 0.1, size=(self.n_cells, self.n_hfactors))),  # hz
                        np.random.normal(0, 0.1, size=(self.n_cells, self.n_hfactors)))),
            jnp.array(((np.random.normal(.5, 0.1, size=(self.n_hfactors, self.n_factors))),  # hW
                        np.random.normal(0, 0.1, size=(self.n_hfactors, self.n_factors)))),
            jnp.array((np.random.normal(.5, 0.1, size=(self.n_cells, self.n_factors)),  # z
                        np.random.normal(0, 0.1, size=(self.n_cells, self.n_factors)))),
            jnp.array(((np.random.normal(.5, 0.1, size=(self.n_factors, self.n_genes))),  # W
                        np.random.normal(0, 0.1, size=(self.n_factors, self.n_genes)))),
            jnp.array(((np.random.normal(.5, 0.1, size=(self.n_batches, self.n_genes))),  # W_noise
                        np.random.normal(0, 0.1, size=(self.n_batches, self.n_genes)))),
            jnp.array((np.random.normal(.5, 0.1, size=(self.n_cells, self.n_batches)),  # z_noise
                        np.random.normal(0, 0.1, size=(self.n_cells, self.n_batches)))),
        ]

    def elbo(self, rng, indices, var_params):
        # Single-sample Monte Carlo estimate of the variational lower bound.
        batch_indices_onehot = self.batch_indices_onehot[indices]

        min_loc = jnp.log(1e-10)
        cell_scale_params = jnp.clip(var_params[0], a_min=min_loc)
        gene_scale_params = jnp.clip(var_params[1], a_min=min_loc)
#         unconst_gene_scales = jnp.clip(var_params[1], a_min=min_loc)
        hfactor_scale_params = jnp.clip(var_params[2], a_min=min_loc)
        factor_scale_params = jnp.clip(var_params[3], a_min=min_loc)
        hz_params = jnp.clip(var_params[4], a_min=min_loc)
        hW_params = jnp.clip(var_params[5], a_min=min_loc)
#         unconst_hW = jnp.clip(var_params[5], a_min=min_loc)
        z_params = jnp.clip(var_params[6], a_min=min_loc)
        W_params = jnp.clip(var_params[7], a_min=min_loc)
#         unconst_W = jnp.clip(var_params[7], a_min=min_loc)
        W_noise_params = jnp.clip(var_params[8], a_min=min_loc)
        z_noise_params = jnp.clip(var_params[9], a_min=min_loc)

        min_concentration = 1e-10
        min_scale = 1e-10
        cell_scale_concentration = jnp.maximum(jnn.softplus(cell_scale_params[0][indices]), min_concentration)
        cell_scale_rate = 1./jnp.maximum(jnn.softplus(cell_scale_params[1][indices]), min_scale)

        z_concentration = jnp.maximum(jnn.softplus(z_params[0][indices]), min_concentration)
        z_rate = 1./jnp.maximum(jnn.softplus(z_params[1][indices]), min_scale)

        hz_concentration = jnp.maximum(jnn.softplus(hz_params[0][indices]), min_concentration)
        hz_rate = 1./jnp.maximum(jnn.softplus(hz_params[1][indices]), min_scale)

        gene_scale_concentration = jnp.maximum(jnn.softplus(gene_scale_params[0]), min_concentration)
        gene_scale_rate = 1./jnp.maximum(jnn.softplus(gene_scale_params[1]), min_scale)

        hfactor_scale_concentration = jnp.maximum(jnn.softplus(hfactor_scale_params[0]), min_concentration)
        hfactor_scale_rate = 1./jnp.maximum(jnn.softplus(hfactor_scale_params[1]), min_scale)

        factor_scale_concentration = jnp.maximum(jnn.softplus(factor_scale_params[0]), min_concentration)
        factor_scale_rate = 1./jnp.maximum(jnn.softplus(factor_scale_params[1]), min_scale)

        W_concentration = jnp.maximum(jnn.softplus(W_params[0]), min_concentration)
        W_rate = 1./jnp.maximum(jnn.softplus(W_params[1]), min_scale)

        hW_concentration = jnp.maximum(jnn.softplus(hW_params[0]), min_concentration)
        hW_rate = 1./jnp.maximum(jnn.softplus(hW_params[1]), min_scale)

        z_noise_concentration = jnp.maximum(jnn.softplus(z_noise_params[0][indices]), min_concentration)
        z_noise_rate = 1./jnp.maximum(jnn.softplus(z_noise_params[1][indices]), min_scale)

        W_noise_concentration = jnp.maximum(jnn.softplus(W_noise_params[0]), min_concentration)
        W_noise_rate = 1./jnp.maximum(jnn.softplus(W_noise_params[1]), min_scale)

        # Sample from variational distribution
        cell_scales = gamma_sample(rng, cell_scale_concentration, cell_scale_rate)
#         cell_scales = jnp.exp(log_cell_scales)
        gene_scales = gamma_sample(rng, gene_scale_concentration, gene_scale_rate)
#         gene_scales = jnp.exp(log_gene_scales)
#         gene_scales = jnn.softplus(unconst_gene_scales)

#         log_hfactor_scales = gaussian_sample(rng, hfactor_scale_params[0], hfactor_scale_params[1])
#         hfactor_scales = jnp.exp(log_hfactor_scales)
        hfactor_scales = gamma_sample(rng, hfactor_scale_concentration, hfactor_scale_rate)
        factor_scales = gamma_sample(rng, factor_scale_concentration, factor_scale_rate)
#         factor_scales = jnp.exp(log_factor_scales)

        hz = gamma_sample(rng, hz_concentration, hz_rate)
#         hz = jnp.exp(log_hz)
        hW = gamma_sample(rng, hW_concentration, hW_rate)
#         hW = jnp.exp(log_hW)
#         hW = jnn.softplus(unconst_hW)
        mean_top = jnp.matmul(hz, hW)

        z = gamma_sample(rng, z_concentration, z_rate)
#         z = jnp.exp(log_z)
#         log_z_noise = gaussian_sample(rng, z_noise_params[0][indices], z_noise_params[1][indices])
#         z_noise = jnp.exp(log_z_noise)
#         z_noise = gamma_sample(rng, z_noise_concentration, z_noise_rate)
        W = gamma_sample(rng, W_concentration, W_rate)
#         W = jnp.exp(log_W)
#         W = jnn.softplus(unconst_W)
#         W_noise = jnn.softplus(unconst_W_noise)
#         W_noise = gamma_sample(rng, W_noise_concentration, W_noise_rate)
        mean_bottom_bio = jnp.matmul(z, W)
#         mean_bottom_batch = jnp.matmul(batch_indices_onehot * z_noise, W_noise) # jnn.softplus(jnp.matmul(batch_indices_onehot, unconst_W_noise))
        mean_bottom = mean_bottom_bio #+ mean_bottom_batch

        # Compute log likelihood
        ll = jnp.sum(vmap(poisson.logpmf)(self.X[indices], mean_bottom))

        # Compute KL divergence
        kl = 0.
        gene_size = jnp.sum(self.X, axis=0)
        kl += gamma_logpdf(gene_scales, 1., jnp.mean(gene_size)/jnp.var(gene_size)) -\
                gamma_logpdf(gene_scales, gene_scale_concentration, gene_scale_rate)

        kl += gamma_logpdf(hfactor_scales, 1e-3, 1e-3) -\
                gamma_logpdf(hfactor_scales, hfactor_scale_concentration, hfactor_scale_rate)

        kl += gamma_logpdf(factor_scales, 1e-3, 1e-3) -\
                gamma_logpdf(factor_scales, factor_scale_concentration, factor_scale_rate)
#         normalized_factor_scales = factor_scales / jnp.sum(factor_scales)

        kl += gamma_logpdf(hW, .3, 1. * hfactor_scales.reshape(-1,1)) -\
                gamma_logpdf(hW, hW_concentration, hW_rate)

        kl += gamma_logpdf(W, .3, 1 * gene_scales.reshape(1,-1) * factor_scales.reshape(-1,1)) -\
                gamma_logpdf(W, W_concentration, W_rate)

#         kl += gamma_logpdf(W_noise, 10, 10. * gene_scales.reshape(1,-1)) -\
#                 gamma_logpdf(W_noise, W_noise_concentration, W_noise_rate)

        kl *= indices.shape[0] / self.X.shape[0] # scale by minibatch size

        kl += gamma_logpdf(cell_scales, 1., 1.*self.batch_lib_ratio[indices]) -\
                gamma_logpdf(cell_scales, cell_scale_concentration, cell_scale_rate)

        kl += gamma_logpdf(hz, 1.,  1. / hfactor_scales.reshape(1,-1)) -\
                gamma_logpdf(hz, hz_concentration, hz_rate)

        # Tieing the scales avoids the factors with low scales in W learn z's that correlate with cell scales
        kl += gamma_logpdf(z, self.shape, cell_scales.reshape(-1,1) * self.shape / (factor_scales.reshape(1,-1) * mean_top) ) -\
                gamma_logpdf(z, z_concentration, z_rate)

#         kl += gamma_logpdf(z_noise, 10., 10. * cell_scales.reshape(-1,1)) -\
#                 gamma_logpdf(z_noise, z_noise_concentration, z_noise_rate)

        return ll + kl

    def batch_elbo(self, rng, indices, var_params, num_samples):
        # Average over a batch of random samples.
        rngs = random.split(rng, num_samples)
        vectorized_elbo = vmap(self.elbo, in_axes=(0, None, None))
        return jnp.mean(vectorized_elbo(rngs, indices, var_params))

    def optimize(self, n_epochs=1000, batch_size=128, step_size=0.01, num_samples=1, init=False, seed=42):
        if init:
            self.init_var_params()
        init_params = self.var_params
        opt_init, opt_update, get_params = optimizers.adam(step_size=step_size)

        def objective(indices, var_params, key):
            return -self.batch_elbo(key, indices, var_params, num_samples) # minimize -ELBO

        loss_grad = jit(value_and_grad(objective, argnums=1))

        def update(indices, i, key, opt_state):
            params = get_params(opt_state)
            value, gradient = loss_grad(indices, params, key)
            return value, opt_update(i, gradient, opt_state)

        num_complete_batches, leftover = divmod(self.n_cells, batch_size)
        num_batches = num_complete_batches + bool(leftover)
        logging.debug(f"Each epoch contains {num_batches} batches of size {batch_size}")
        def data_stream():
            rng = np.random.RandomState(0)
            while True:
              perm = rng.permutation(self.n_cells)
              for i in range(num_batches):
                batch_idx = perm[i * batch_size:(i + 1) * batch_size]
                yield jnp.array(batch_idx)
        batches = data_stream()

        opt_state = opt_init(init_params)
        losses = []
        rng = random.PRNGKey(seed)
        t = 0
        pbar = tqdm(range(n_epochs))
        for epoch in pbar:
            epoch_losses = []
            start_time = time.time()
            for it in range(num_batches):
                rng, rng_input = random.split(rng)
                loss, opt_state = update(next(batches), t, rng_input, opt_state)
                epoch_losses.append(loss)
                t += 1
            losses.append(np.mean(epoch_losses))
            epoch_time = time.time() - start_time
            pbar.set_postfix({'Loss': losses[-1]})

        params = get_params(opt_state)
        self.var_params = params
        self.set_posterior_means()
        self.filter_factors(annotate=True)
        return losses


    def get_dimensions(self, threshold=0.1):
        # Look at the scale of the factors and get the ones which are actually used
        return dict(hfactor=np.where(self.pmeans['hfactor_scale'] > threshold)[0],
                    factor=np.where(self.pmeans['factor_scale'] > threshold)[0])

    def set_posterior_means(self):
        cell_scale_params = self.var_params[0]
        gene_scale_params = self.var_params[1]
        hfactor_scale_params = self.var_params[2]
        factor_scale_params = self.var_params[3]
        hz_params = self.var_params[4]
        hW_params = self.var_params[5]
        z_params = self.var_params[6]
        W_params = self.var_params[7]
        W_noise_params = self.var_params[8]
        z_noise_params = self.var_params[9]

        self.pmeans = {
            'cell_scale': np.array(jnn.softplus(cell_scale_params[0]) * jnn.softplus(cell_scale_params[1])),
            'gene_scale': np.array(jnn.softplus(gene_scale_params[0]) * jnn.softplus(gene_scale_params[1])),
            'hfactor_scale': np.array(jnn.softplus(hfactor_scale_params[0]) * jnn.softplus(hfactor_scale_params[1])),
            'factor_scale': np.array(jnn.softplus(factor_scale_params[0]) * jnn.softplus(factor_scale_params[1])),
            'hz': np.array(jnn.softplus(hz_params[0]) * jnn.softplus(hz_params[1])),
            'hW': np.array(jnn.softplus(hW_params[0]) * jnn.softplus(hW_params[1])),
            'z': np.array(jnn.softplus(z_params[0]) * jnn.softplus(z_params[1])),
            'W': np.array(jnn.softplus(W_params[0]) * jnn.softplus(W_params[1])),
            'W_noise': np.array(jnn.softplus(W_noise_params[0]) * jnn.softplus(W_noise_params[1])),
            'z_noise': np.array(jnn.softplus(z_noise_params[0]) * jnn.softplus(z_noise_params[1])),
        }

        self.pvars = {
            'cell_scale': np.array(jnn.softplus(cell_scale_params[0]) * jnn.softplus(cell_scale_params[1])**2),
            'gene_scale': np.array(jnn.softplus(gene_scale_params[0]) * jnn.softplus(gene_scale_params[1])**2),
            'hfactor_scale': np.array(jnn.softplus(hfactor_scale_params[0]) * jnn.softplus(hfactor_scale_params[1])**2),
            'factor_scale': np.array(jnn.softplus(factor_scale_params[0]) * jnn.softplus(factor_scale_params[1])**2),
            'hz': np.array(jnn.softplus(hz_params[0]) * jnn.softplus(hz_params[1])**2),
            'hW': np.array(jnn.softplus(hW_params[0]) * jnn.softplus(hW_params[1])**2),
            'z': np.array(jnn.softplus(z_params[0]) * jnn.softplus(z_params[1])**2),
            'W': np.array(jnn.softplus(W_params[0]) * jnn.softplus(W_params[1])**2),
            'W_noise': np.array(jnn.softplus(W_noise_params[0]) * jnn.softplus(W_noise_params[1])**2),
            'z_noise': np.array(jnn.softplus(z_noise_params[0]) * jnn.softplus(z_noise_params[1])**2),
        }

    def filter_factors(self, q=[0.4, 0.4], annotate=True):
        tokeep_list = []
        for layer in [0, 1]:
            thres = np.quantile(self.pmeans[f'{self.layer_names[layer]}_scale'], q=q[layer])
            tokeep = np.where(self.pmeans[f'{self.layer_names[layer]}_scale'] >= thres)[0]
            tokeep_list.append(tokeep)
        if annotate:
            self.annotate_adata(tokeep=tokeep_list)
        else:
            return tokeep_list

    def annotate_adata(self, tokeep=None):
        if tokeep is None:
            tokeep = [np.arange(self.n_factors), np.arange(self.n_hfactors)]
        elif not isinstance(tokeep, list):
            raise TypeError("`tokeep` must be a list of arrays!")

        self.adata.obsm['X_hfactors'] = self.pmeans['hz'][:,tokeep[1]] * self.pmeans['cell_scale'].reshape(-1,1)
        self.adata.obsm['X_factors'] = self.pmeans['z'][:,tokeep[0]] * self.pmeans['cell_scale'].reshape(-1,1)
#         self.adata.obsm['X_factors_var'] = self.pvars['z'][:,tokeep]
#         self.adata.obsm['X_factors_mean'] = self.pmeans['hz'].dot(self.pmeans['hW'])
        self.logger.info("Updated adata.obsm: `X_hfactors` and `X_factors`.")

        self.adata.obs['X_hfactor'] = np.argmax(self.adata.obsm['X_hfactors'], axis=1).astype(str)
        self.adata.obs['X_factor'] = np.argmax(self.adata.obsm['X_factors'], axis=1).astype(str)
        self.adata.obs['cell_scale'] = 1/self.pmeans['cell_scale']
        for i in range(len(tokeep[0])):
            self.adata.obs[f'{i}'] = self.adata.obsm['X_factors'][:,i]
#             self.adata.obs[f'm{i}'] = self.adata.obsm['X_factors_mean'][:,i]
#             self.adata.obs[f'v{i}'] = self.adata.obsm['X_factors_var'][:,i]
        self.adata.uns['X_hfactor_colors'] = []
        for i, idx in enumerate(tokeep[1]):
            self.adata.uns['X_hfactor_colors'].append(matplotlib.colors.to_hex(self.hpal[idx]))
            self.adata.obs[f'h{i}'] = self.adata.obsm['X_hfactors'][:,i]
        self.logger.info("Updated adata.obs: `X_hfactor`, `X_factor`, 'cell_scale', and cell weights for each "\
                     "factor and hierarchical factor.")

        self.adata.var['gene_scale'] = 1/self.pmeans['gene_scale']
        for i in range(len(tokeep[0])):
             self.adata.var[f'{i}'] = (self.pmeans['W'][tokeep[0],:] * self.pmeans['gene_scale'])[i,:]
        self.logger.info("Updated adata.var: `gene_scale` and gene weights for each factor.")

    def plot_ard(self, layer_idx=0, q=None):
        if isinstance(q, float):
            thres = np.quantile(self.pmeans[f'{self.layer_names[layer_idx]}_scale'], q=q)
            plt.axhline(thres, color='red', ls='--')
            above = np.where(self.pmeans[f'{self.layer_names[layer_idx]}_scale'] >= thres)[0]
            below = np.where(self.pmeans[f'{self.layer_names[layer_idx]}_scale'] < thres)[0]
            plt.bar(np.arange(self.layer_sizes[layer_idx])[above], self.pmeans[f'{self.layer_names[layer_idx]}_scale'][above])
            plt.bar(np.arange(self.layer_sizes[layer_idx])[below], self.pmeans[f'{self.layer_names[layer_idx]}_scale'][below])
        else:
            plt.bar(np.arange(self.layer_sizes[layer_idx]), self.pmeans[f'{self.layer_names[layer_idx]}_scale'])
        plt.ylabel('Contribution')
        plt.xlabel('Factor')
        plt.xticks(np.arange(self.layer_sizes[layer_idx]))
        plt.title(f'Layer {layer_idx}')
        plt.show()

    def get_rankings(self, layer_idx=0, q=.1, return_scores=False):
        term_names = np.array(self.adata.var_names)
        term_scores = self.pmeans['W']*self.pmeans['gene_scale']
        if layer_idx == 1:
            term_names = np.arange(self.n_factors).astype(str)
            term_scores = self.pmeans['hW']

        top_terms = []
        top_scores = []
        for k in range(self.layer_sizes[layer_idx]):
            top_terms_idx = (term_scores[k, :]).argsort()[::-1]
            sorted_term_scores_k = term_scores[k,:][top_terms_idx]
            thres = np.quantile(sorted_term_scores_k, q=q)
            top_terms_idx = top_terms_idx[np.where(sorted_term_scores_k > thres)[0]]
            top_terms_list = term_names[top_terms_idx].tolist()
            top_scores_list = sorted_term_scores_k.tolist()
            top_terms.append(top_terms_list)
            top_scores.append(top_scores_list)

        if return_scores:
            return top_terms, top_scores
        return top_terms

    def get_annotations(self, marker_reference, gene_rankings=None):
        if gene_rankings is None:
            gene_rankings = self.get_rankings(layer_idx=0)

        annotations = []
        keys = list(marker_reference.keys())
        for rank in gene_rankings:
            # Get annotation in marker_reference that contains the highest score
            # score is length of intersection over length of union
            scores = np.array([len(set(rank).intersection(set(marker_reference[a])))/len(set(rank).union(set(marker_reference[a]))) for a in keys])
            sorted_ann_idx = np.argsort(scores)[::-1]
            sorted_scores = scores[sorted_ann_idx]

            # Get only annotations for which score is not zero
            sorted_ann_idx = sorted_ann_idx[np.where(sorted_scores > 0)[0]]

            ann = np.array(keys)[sorted_ann_idx]
            ann = np.array([f'{a} ({sorted_scores[i]:.4f})' for i, a in enumerate(ann)]).tolist()
            annotations.append(ann)
        return annotations

    def get_enrichments(self, libs=['KEGG_2019_Human'], gene_rankings=None):
        if gene_rankings is None:
            gene_rankings = self.get_rankings(layer_idx=0)

        enrichments = []
        for rank in tqdm(gene_rankings):
            enr = gp.enrichr(gene_list=rank,
                         gene_sets=libs,
                         organism='Human',
                         outdir='test/enrichr',
                         cutoff=0.05
                        )
            enrichments.append(enr)
        return enrichments

    def get_graph(self, annotations=None, enrichments=None, top=10, hfactor_list=None, factor_list=None, ard_filter=[0.3, 0.], gene_rankings=None, reindex=True, batch_counts=True):
        if hfactor_list is None:
            hfactor_list = np.arange(self.n_hfactors)
            if ard_filter:
                hfactor_list = self.filter_factors(q=ard_filter, annotate=False)[1]
        if factor_list is None:
            factor_list = np.arange(self.n_factors)
            if ard_filter:
                factor_list = self.filter_factors(q=ard_filter, annotate=False)[0]

        normalized_htopic_weights = self.pmeans['hW'][:,factor_list]/np.sum(self.pmeans['hW'][:, factor_list], axis=1).reshape(-1,1)
        normalized_hfactor_scales = self.pmeans['hfactor_scale']/np.sum(self.pmeans['hfactor_scale'][hfactor_list])
        normalized_factor_scales = self.pmeans['factor_scale']/np.sum(self.pmeans['factor_scale'][factor_list])

        # Assign factors to hierarchical factors to set the plotting order
        assignments = []
        for i, topic in enumerate(factor_list):
            assignments.append(np.argmax(normalized_htopic_weights[:,i]))
        factor_order = np.argsort(np.array(assignments))

        g = Graph()
        for i, htopic in enumerate(hfactor_list):
            if reindex:
                tlab = str(i)
            else:
                tlab = str(htopic)
            g.node('ht' + str(htopic), label=tlab,
                    fillcolor=matplotlib.colors.to_hex(self.hpal[htopic]), ordering='out', style='filled')

        if self.batch_key in self.adata.obs.columns:
            batches = np.unique(self.adata.obs[self.batch_key])
        else:
            batch_counts = False

        for pos in factor_order:
            topic = factor_list[pos]

            if reindex:
                tlab = str(pos)
            else:
                tlab = str(topic)

            if batch_counts and 'X_factor' in self.adata.obs.keys():
                # Get cells assigned to this factor
                cells = np.where(self.adata.obs['X_factor'] == str(topic))[0]
                # Get number of cells in this factor that belong to each batch
                prevs = [str(np.count_nonzero(self.adata.obs[self.batch_key][cells]==b)) for b in batches]
                tlab += '\n' + ', '.join(prevs)

            if enrichments is not None:
                tlab += '\n\n' + "\n".join([enrichments[topic].results['Term'].values[i] + f" ({enrichments[topic].results['Adjusted P-value'][i]:.3f})" for i in range(top)])
                # tlab = enrichments[topic].results['Term'].values[0] + f" ({enrichments[topic].results['Adjusted P-value'][0]:.3f})" + '\n\n' + tlab
            elif annotations is not None:
                tlab += '\n\n' + "\n".join([annotations[topic][i] for i in range(min(len(annotations[topic]), top))])
            else:
                if gene_rankings is None:
                    gene_rankings = self.get_rankings()
                tlab += '\n\n' + "\n".join(gene_rankings[topic][:top])

            # Get top factors from this hierarchical first
            g.node('t' + str(topic), label=tlab, fontsize="11", ordering='in')

        for htopic in hfactor_list:
            for pos in factor_order:
                topic = factor_list[pos]
                g.edge('ht' + str(htopic), 't'+str(topic), penwidth=str(4*normalized_htopic_weights[htopic,pos]), color=matplotlib.colors.to_hex(self.hpal[htopic]))

        return g

    def get_summary(self, q=[0.4, 0.4], top_genes=10, reindex=True):
        tokeep = self.filter_factors(q=q, annotate=False)[0]
        n_factors_eff = len(tokeep)
        genes, scores = self.get_rankings(return_scores=True)

        summary = f"Found {n_factors_eff} factors grouped in the following way:\n"

        # Group the factors
        assignments = []
        for i, factor in enumerate(tokeep):
            assignments.append(np.argmax(self.pmeans['hW'][:,factor]))
        assignments = np.array(assignments)
        factor_order = np.argsort(np.array(assignments))

        for group in np.unique(assignments):
            factors = tokeep[np.where(assignments == group)[0]]
            if len(factors) > 0:
                summary += f"Group {group}:\n"
            for i, factor in enumerate(factors):
                summary += f"    Factor {i}: "
                summary += ", ".join([f"{genes[factor][j]} ({scores[factor][j]:.3f})" for j in range(top_genes)])
                summary += "\n"
            summary += "\n"

        self.logger.info(summary)

        return summary
