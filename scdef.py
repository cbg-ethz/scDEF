from functools import partial
import string

from jax.api import jit, grad, vmap
from jax.experimental import optimizers
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

def gamma_logpdf(x, shape, rate):
    scale = 1./rate
    return jnp.sum(vmap(gamma.logpdf)(x, shape * jnp.ones(x.shape), scale=scale * jnp.ones(x.shape)))

class scDPF(object):
    def __init__(self, adata, n_hfactors=10, n_factors=30, shape=0.1):
        self.n_factors = n_factors
        self.n_hfactors = n_hfactors
        self.shape = shape
        self.layer_names = ['factor', 'hfactor']
        self.layer_sizes = [n_factors, n_hfactors]
        self.load_adata(adata)
        self.n_cells, self.n_genes = adata.shape
        self.init_var_params()
        self.set_posterior_means()
        self.hpal = sns.color_palette('Set2', n_hfactors)
        self.graph = self.get_graph()

    def load_adata(self, adata):
        if not isinstance(adata, AnnData):
            raise TypeError("adata must be an instance of AnnData.")
        self.adata = adata
        self.adata.raw = self.adata
        self.X = jnp.array(self.adata.raw.X)

    def init_var_params(self):
        self.var_params = [
            jnp.array((-np.log(np.sum(self.X, axis=1)), jnp.ones((self.n_cells,)))),
            jnp.array(np.random.normal(5, 0.1, size=self.n_genes)),
            jnp.array((0*np.random.normal(5, 0.1, size=self.n_hfactors), jnp.ones((self.n_hfactors,)))),
            jnp.array((0*np.random.normal(5, 0.1, size=self.n_factors), jnp.ones((self.n_factors,)))),
            jnp.array((np.random.normal(1., 0.1, size=(self.n_cells, self.n_hfactors)),
                        jnp.zeros((self.n_cells, self.n_hfactors)))),
            jnp.array(np.random.normal(1., 0.1, size=(self.n_hfactors, self.n_factors))),
            jnp.array((np.random.normal(0.5, 0.1, size=(self.n_cells, self.n_factors)),
                        jnp.zeros((self.n_cells, self.n_factors)))),
            jnp.array(np.random.normal(0.5, 0.1, size=(self.n_factors, self.n_genes)))
        ]

    def elbo(self, rng, indices, var_params):
        print("Computing ELBO...")
        # Single-sample Monte Carlo estimate of the variational lower bound.
        min_loc = jnp.log(1e-3)
        cell_scale_params = jnp.clip(var_params[0], a_min=min_loc)
#         gene_scale_params = jnp.clip(var_params[1], a_min=min_loc)
        unconst_gene_scales = jnp.clip(var_params[1], a_min=min_loc)
#         hfactor_scale_params = jnp.clip(var_params[2], a_min=min_loc)
#         factor_scale_params = jnp.clip(var_params[3], a_min=min_loc)
        hz_params = jnp.clip(var_params[4], a_min=min_loc)
#         hW_params = jnp.clip(var_params[5], a_min=min_loc)
        unconst_hW = jnp.clip(var_params[5], a_min=min_loc)
        z_params = jnp.clip(var_params[6], a_min=min_loc)
#         W_params = jnp.clip(var_params[7], a_min=min_loc)
        unconst_W = jnp.clip(var_params[7], a_min=min_loc)

        # Sample from variational distribution
        log_cell_scales = gaussian_sample(rng, cell_scale_params[0][indices], cell_scale_params[1][indices])
        cell_scales = jnp.exp(log_cell_scales)
#         log_gene_scales = gaussian_sample(rng, gene_scale_params[0], gene_scale_params[1])
#         gene_scales = jnp.exp(log_gene_scales)
        gene_scales = jnn.softplus(unconst_gene_scales)

#         log_hfactor_scales = gaussian_sample(rng, hfactor_scale_params[0], hfactor_scale_params[1])
#         hfactor_scales = jnp.exp(log_hfactor_scales)
#         log_factor_scales = gaussian_sample(rng, factor_scale_params[0], factor_scale_params[1])
#         factor_scales = jnp.exp(log_factor_scales)

        log_hz = gaussian_sample(rng, hz_params[0][indices], hz_params[1][indices])
        hz = jnp.exp(log_hz)
#         log_hW = gaussian_sample(rng, hW_params[0], hW_params[1])
#         hW = jnp.exp(log_hW)
        hW = jnn.softplus(unconst_hW)
        mean_top = jnp.matmul(hz, hW)

        log_z = gaussian_sample(rng, z_params[0][indices], z_params[1][indices])
        z = jnp.exp(log_z)
#         log_W = gaussian_sample(rng, W_params[0], W_params[1])
#         W = jnp.exp(log_W)
        W = jnn.softplus(unconst_W)
        mean_bottom = jnp.matmul(z, W)

        # Compute log likelihood
        ll = jnp.sum(vmap(poisson.logpmf)(self.X[indices], mean_bottom))

        # Compute KL divergence
        kl = 0.
        kl += gamma_logpdf(gene_scales, 1., 1.) #-\
#                 gaussian_logpdf(log_gene_scales, gene_scale_params[0], gene_scale_params[1])

#         kl += gamma_logpdf(hfactor_scales, .1, 1.) -\
#                 gaussian_logpdf(log_hfactor_scales, hfactor_scale_params[0], hfactor_scale_params[1])

#         kl += gamma_logpdf(factor_scales, 1., 1.) -\
#                 gaussian_logpdf(log_factor_scales, factor_scale_params[0], factor_scale_params[1])

        kl += gamma_logpdf(hW, .3, .1) #-\
#                 gaussian_logpdf(log_hW, hW_params[0], hW_params[1])

        kl += gamma_logpdf(W, .3, .1 * gene_scales.reshape(1,-1)) #-\
#                 gaussian_logpdf(log_W, W_params[0], W_params[1])

        kl *= indices.shape[0] / self.X.shape[0] # scale by minibatch size

        kl += gamma_logpdf(cell_scales, 1., 1.) -\
                gaussian_logpdf(log_cell_scales, cell_scale_params[0][indices], cell_scale_params[1][indices])

        kl += gamma_logpdf(hz, self.shape, self.shape) -\
                gaussian_logpdf(log_hz, hz_params[0][indices], hz_params[1][indices])

        kl += gamma_logpdf(z, self.shape, cell_scales.reshape(-1,1) * self.shape / mean_top) -\
                gaussian_logpdf(log_z, z_params[0][indices], z_params[1][indices])

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
            t = 0
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
            'gene_scale': np.array(jnn.softplus(gene_scale_params)), #np.exp(gene_scale_params[0]),
            'hfactor_scale': np.exp(hfactor_scale_params[0]),
            'factor_scale': np.exp(factor_scale_params[0]),
            'hz': np.exp(hz_params[0]),
            'hW': np.array(jnn.softplus(hW_params)),#np.exp(hW_params[0]),
            'z': np.exp(z_params[0]),
            'W': np.array(jnn.softplus(W_params))#np.exp(W_params[0]),
        }

    def annotate_adata(self):
        self.adata.obsm['X_hfactors'] = self.pmeans['hz'] * self.pmeans['cell_scale'].reshape(-1,1)
        self.adata.obsm['X_factors'] = self.pmeans['z'] * self.pmeans['cell_scale'].reshape(-1,1)
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

    def get_rankings(self, layer_idx=0, thres=.1):
        term_names = np.array(self.adata.var_names)
        term_scores = self.pmeans['W']/self.pmeans['gene_scale']
        if layer_idx == 1:
            term_names = np.arange(self.n_factors).astype(str)
            term_scores = self.pmeans['hW']

        top_terms = []
        for k in range(self.layer_sizes[layer_idx]):
            top_terms_idx = (term_scores[k, :]).argsort()[::-1]
            sorted_term_scores_k = term_scores[k,:][top_terms_idx]
            top_terms_idx = top_terms_idx[np.where(sorted_term_scores_k > thres)[0]]
            top_terms_list = term_names[top_terms_idx].tolist()
            top_terms.append(top_terms_list)
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

    def get_graph(self, annotations=None, enrichments=None, top=10, hfactor_list=None, factor_list=None, ard_filter=[0., 0.], gene_rankings=None):
        if hfactor_list is None:
            hfactor_list = np.arange(self.n_hfactors)
            if ard_filter:
                hfactor_list = np.where(self.pmeans[f'{self.layer_names[1]}_scale']  > ard_filter[0])[0]
        if factor_list is None:
            factor_list = np.arange(self.n_factors)
            if ard_filter:
                factor_list = np.where(self.pmeans[f'{self.layer_names[0]}_scale']  > ard_filter[1])[0]

        normalized_htopic_weights = self.pmeans['hW']/np.sum(self.pmeans['hW'][:, factor_list], axis=1).reshape(-1,1)
        normalized_hfactor_scales = self.pmeans['hfactor_scale']/np.sum(self.pmeans['hfactor_scale'][hfactor_list])
        normalized_factor_scales = self.pmeans['factor_scale']/np.sum(self.pmeans['factor_scale'][factor_list])

        # Assign factors to hierarchical factors to set the plotting order
        assignments = []
        for topic in factor_list:
            assignments.append(np.argmax(normalized_htopic_weights[:,topic]))
        factor_list = np.argsort(np.array(assignments))

        g = Graph()
        for htopic in hfactor_list:
            g.node('ht' + str(htopic), label=str(htopic),
                    fillcolor=matplotlib.colors.to_hex(self.hpal[htopic]), ordering='out', style='filled')
        for topic in factor_list:
            if enrichments is not None:
                tlab = str(topic) + '\n\n' + "\n".join([enrichments[topic].results['Term'].values[i] + f" ({enrichments[topic].results['Adjusted P-value'][i]:.3f})" for i in range(top)])
                # tlab = enrichments[topic].results['Term'].values[0] + f" ({enrichments[topic].results['Adjusted P-value'][0]:.3f})" + '\n\n' + tlab
            elif annotations is not None:
                tlab = str(topic) + '\n\n' + "\n".join([annotations[topic][i] for i in range(min(len(annotations[topic]), top))])
            else:
                if gene_rankings is None:
                    gene_rankings = self.get_rankings()
                tlab = str(topic) + '\n\n' + "\n".join(gene_rankings[topic][:top])

            # Get top factors from this hierarchical first
            g.node('t' + str(topic), label=tlab, fontsize="11",
                    fillcolor=matplotlib.colors.to_hex((64/255, 224/255, 208/255, normalized_factor_scales[topic]), keep_alpha=True), ordering='in')
        for htopic in hfactor_list:
            for topic in factor_list:
                g.edge('ht' + str(htopic), 't'+str(topic), penwidth=str(normalized_htopic_weights[htopic,topic]*2), color=matplotlib.colors.to_hex(self.hpal[htopic]))
        return g
