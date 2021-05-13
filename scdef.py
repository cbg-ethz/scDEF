from functools import partial
import string

from jax.api import jit, grad, vmap, value_and_grad
from jax.experimental import optimizers
from jax import random
from jax.scipy.stats import norm, gamma, poisson
import jax.numpy as jnp

import matplotlib
import matplotlib.pyplot as plt
import gseapy as gp
from graphviz import Graph
from tqdm import tqdm
import time

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

def gamma_sample(rng, log_shape, log_scale):
    shape, scale = jnp.exp(log_shape), jnp.exp(log_scale)
    return scale * random.gamma(rng, shape)

def gamma_logpdf(x, log_shape, log_scale):
    shape, scale = jnp.exp(log_shape), jnp.exp(log_scale)
    return jnp.sum(vmap(gamma.logpdf)(x, shape * jnp.ones(x.shape), scale=scale * jnp.ones(x.shape)))

class scDPF(object):
    def __init__(self, adata, n_hfactors=10, n_factors=30):
        self.n_factors = n_factors
        self.n_hfactors = n_hfactors
        self.layer_names = ['factor', 'hfactor']
        self.layer_sizes = [n_factors, n_hfactors]
        self.load_adata(adata)
        self.n_cells, self.n_genes = adata.shape
        self.init_var_params()
        self.set_posterior_means()
        self.graph = self.get_graph()

    def load_adata(self, adata):
        if not isinstance(adata, AnnData):
            raise TypeError("adata must be an instance of AnnData.")
        self.adata = adata
        self.adata.raw = self.adata
        self.X = jnp.array(self.adata.raw.X)

    def init_var_params(self):
        self.var_params = [
            jnp.array((jnp.zeros((self.n_cells,)), jnp.zeros((self.n_cells,)))),
            jnp.array((jnp.zeros((self.n_genes,)), jnp.zeros((self.n_genes,)))),
            jnp.array((jnp.zeros((self.n_hfactors,)), jnp.zeros((self.n_hfactors,)))),
            jnp.array((jnp.zeros((self.n_factors,)), jnp.zeros((self.n_factors,)))),
            jnp.array((jnp.zeros((self.n_cells, self.n_hfactors)), jnp.zeros((self.n_cells, self.n_hfactors)))),
            jnp.array((jnp.zeros((self.n_hfactors, self.n_factors)), jnp.zeros((self.n_hfactors, self.n_factors)))),
            jnp.array((jnp.zeros((self.n_cells, self.n_factors)), jnp.zeros((self.n_cells, self.n_factors)))),
            jnp.array((jnp.zeros((self.n_factors, self.n_genes)), jnp.zeros((self.n_factors, self.n_genes)))),
        ]

    def elbo(self, rng, indices, var_params):
        # Single-sample Monte Carlo estimate of the variational lower bound.
        cell_scale_params = var_params[0]
        gene_scale_params = var_params[1]
        hfactor_scale_params = var_params[2]
        factor_scale_params = var_params[3]
        hz_params = var_params[4]
        hW_params = var_params[5]
        z_params = var_params[6]
        W_params = var_params[7]

        # Sample from variational distribution
        cell_scales = gaussian_sample(rng, cell_scale_params[0][indices], cell_scale_params[1][indices])
        gene_scales = gaussian_sample(rng, gene_scale_params[0], gene_scale_params[1])

        hfactor_scales = gaussian_sample(rng, hfactor_scale_params[0], hfactor_scale_params[1])
        factor_scales = gaussian_sample(rng, factor_scale_params[0], factor_scale_params[1])

        hz = gaussian_sample(rng, hz_params[0][indices], hz_params[1][indices])
        hW = gaussian_sample(rng, hW_params[0], hW_params[1])

        z = gaussian_sample(rng, z_params[0][indices], z_params[1][indices])
        W = gaussian_sample(rng, W_params[0], W_params[1])

        # Compute log likelihood
        ll = jnp.sum(vmap(poisson.logpmf)(self.X[indices], jnp.exp(z).dot(jnp.exp(W))))

        # Compute KL divergence
        kl = 0
        kl += gamma_logpdf(jnp.exp(cell_scales), jnp.log(1.), jnp.log(1.)) - gaussian_logpdf(cell_scales, cell_scale_params[0][indices], cell_scale_params[1][indices])
        kl += gamma_logpdf(jnp.exp(gene_scales), jnp.log(1.), jnp.log(1.)) - gaussian_logpdf(gene_scales, gene_scale_params[0], gene_scale_params[1])
        kl += gamma_logpdf(jnp.exp(hfactor_scales), jnp.log(.1), jnp.log(1.)) - gaussian_logpdf(hfactor_scales, hfactor_scale_params[0], hfactor_scale_params[1])
        kl += gamma_logpdf(jnp.exp(factor_scales), jnp.log(.1), jnp.log(1.)) - gaussian_logpdf(factor_scales, factor_scale_params[0], factor_scale_params[1])
        kl += gamma_logpdf(jnp.exp(hz), jnp.log(.1), cell_scales.reshape(-1,1)) - gaussian_logpdf(hz, hz_params[0][indices], hz_params[1][indices])
        kl += gamma_logpdf(jnp.exp(hW), jnp.log(.3), hfactor_scales.reshape(-1,1)) - gaussian_logpdf(hW, hW_params[0], hW_params[1])
        kl += gamma_logpdf(jnp.exp(z), jnp.log(.1), jnp.log(10. * jnp.exp(hz).dot(jnp.exp(hW)))) - gaussian_logpdf(z, z_params[0][indices], z_params[1][indices])
        kl += gamma_logpdf(jnp.exp(W), jnp.log(.3), factor_scales.reshape(-1,1)+gene_scales.reshape(1,-1)) - gaussian_logpdf(W, W_params[0], W_params[1])
        return ll + kl

    def batch_elbo(self, rng, indices, var_params, num_samples):
        # Average over a batch of random samples.
        rngs = random.split(rng, num_samples)
        vectorized_elbo = vmap(self.elbo, in_axes=(0, None, None))
        return jnp.mean(vectorized_elbo(rngs, indices, var_params))

    def optimize(self, n_epochs=1000, batch_size=1, step_size=0.1, num_samples=1, init=False):
        if init:
            self.init_var_params()
        init_params = self.var_params
        opt_init, opt_update, get_params = optimizers.adam(step_size=step_size)

        def objective(indices, var_params, t):
            rng = random.PRNGKey(t)
            return -self.batch_elbo(rng, indices, var_params, num_samples) # minimize -ELBO

        loss_grad = jit(value_and_grad(objective, argnums=1))

        def update(indices, i, opt_state):
            params = get_params(opt_state)
            value, gradient = loss_grad(indices, params, i)
            return value, opt_update(i, gradient, opt_state)

        num_complete_batches, leftover = divmod(self.n_cells, batch_size)
        num_batches = num_complete_batches + bool(leftover)
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
        print("\nStarting training...")
        t = 0
        for epoch in tqdm(range(n_epochs)):
            epoch_losses = []
            start_time = time.time()
            for it in range(num_batches):
                loss, opt_state = update(next(batches), t, opt_state)
                epoch_losses.append(loss)
                t += 1
            losses.append(np.mean(epoch_losses))
            epoch_time = time.time() - start_time
            print("Epoch {} in {:0.2f} sec || Loss {:0.5f}".format(epoch, epoch_time, losses[-1]))

        params = get_params(opt_state)
        self.var_params = params
        self.set_posterior_means()
        self.annotate_adata()
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

        self.pmeans = {
            'cell_scale': np.exp(cell_scale_params[0]),
            'gene_scale': np.exp(gene_scale_params[0]),
            'hfactor_scale': np.exp(hfactor_scale_params[0]),
            'factor_scale': np.exp(factor_scale_params[0]),
            'hz': np.exp(hz_params[0]),
            'hW': np.exp(hW_params[0]),
            'z': np.exp(z_params[0]),
            'W': np.exp(W_params[0]),
        }

    def annotate_adata(self):
        self.adata.obsm['X_hfactors'] = self.pmeans['hz'] / self.pmeans['cell_scale'].reshape(-1,1)
        self.adata.obsm['X_factors'] = self.pmeans['z'] / self.pmeans['cell_scale'].reshape(-1,1)
        self.adata.obs['X_hfactor'] = np.argmax(self.adata.obsm['X_hfactors'], axis=1).astype(str)
        self.adata.obs['X_factor'] = np.argmax(self.adata.obsm['X_factors'], axis=1).astype(str)
        print('Added `X_hfactors` and `X_factors` to adata.obsm and the corresponding cell assignments to adata.obs')

    def plot_ard(self, layer_idx=0):
        plt.bar(np.arange(self.layer_sizes[layer_idx]), self.pmeans[f'{self.layer_names[layer_idx]}_scale'])
        plt.ylabel('Contribution')
        plt.xlabel('Factor')
        plt.xticks(np.arange(self.layer_sizes[layer_idx]))
        plt.title(f'Layer {layer_idx}')
        plt.show()

    def get_rankings(self, layer_idx=0):
        term_names = self.adata.var_names
        term_scores = self.pmeans['W']/self.pmeans['gene_scale']
        if layer_idx == 1:
            term_names = np.arange(self.n_factors).astype(str)
            term_scores = self.pmeans['hW']

        top_terms = []
        for k in range(self.layer_sizes[layer_idx]):
            top_terms_idx = (term_scores[k, :]).argsort()[::-1]
            top_terms_list = [term_names[i] for i in top_terms_idx]
            top_terms.append(top_terms_list)
        return top_terms

    def get_enrichments(self, libs=['KEGG_2019_Human']):
        enrichments = []
        gene_rankings = self.get_rankings(layer_idx=0)
        for rank in tqdm(gene_rankings):
            enr = gp.enrichr(gene_list=rank,
                         gene_sets=libs,
                         organism='Human',
                         outdir='test/enrichr',
                         cutoff=0.05
                        )
            enrichments.append(enr)
        return enrichments

    def get_graph(self, enrichments=None, top=10, hfactor_list=None, factor_list=None, ard_filter=[0., 0.]):
        if hfactor_list is None:
            hfactor_list = np.arange(self.n_hfactors)
            if ard_filter:
                hfactor_list = np.where(self.pmeans[f'{self.layer_names[1]}_scale']  > ard_filter[0])[0]
        if factor_list is None:
            factor_list = np.arange(self.n_factors)
            if ard_filter:
                factor_list = np.where(self.pmeans[f'{self.layer_names[0]}_scale']  > ard_filter[1])[0]

        gene_rankings = self.get_rankings()
        normalized_htopic_weights = self.pmeans['hW']/np.sum(self.pmeans['hW'][:, factor_list], axis=1).reshape(-1,1)
        normalized_hfactor_scales = self.pmeans['hfactor_scale']/np.sum(self.pmeans['hfactor_scale'][hfactor_list])
        normalized_factor_scales = self.pmeans['factor_scale']/np.sum(self.pmeans['factor_scale'][factor_list])
        g = Graph()
        for htopic in hfactor_list:
            g.node('ht' + str(htopic), label=str(htopic),
                    fillcolor=matplotlib.colors.to_hex((64/255, 224/255, 208/255, normalized_hfactor_scales[htopic]), keep_alpha=True), ordering='out')
        for topic in factor_list:
            if enrichments is not None:
                tlab = str(topic) + '\n\n' + "\n".join([enrichments[topic].results['Term'].values[i] + f" ({enrichments[topic].results['Adjusted P-value'][i]:.3f})" for i in range(top)])
                # tlab = enrichments[topic].results['Term'].values[0] + f" ({enrichments[topic].results['Adjusted P-value'][0]:.3f})" + '\n\n' + tlab
            else:
                tlab = str(topic) + '\n\n' + "\n".join(gene_rankings[topic][:top])
            g.node('t' + str(topic), label=tlab, fontsize="11",
                    fillcolor=matplotlib.colors.to_hex((64/255, 224/255, 208/255, normalized_factor_scales[topic]), keep_alpha=True), ordering='in')
        for htopic in hfactor_list:
            for topic in factor_list:
                g.edge('ht' + str(htopic), 't'+str(topic), penwidth=str(normalized_htopic_weights[htopic,topic]*normalized_hfactor_scales[htopic]))
        return g
