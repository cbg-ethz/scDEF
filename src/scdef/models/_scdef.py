# Standard library imports
import logging
import time
from typing import Optional, Union, Sequence, Mapping, Literal

# Third-party imports
import jax
import jax.numpy as jnp
from jax import jit, vmap, random, value_and_grad
from jax.scipy.stats import poisson
import optax
import tensorflow_probability.substrates.jax.distributions as tfd

import numpy as np
import pandas as pd
import scipy
from scipy.cluster.hierarchy import ward, leaves_list
from scipy.spatial.distance import pdist

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import seaborn as sns
from tqdm import tqdm

from anndata import AnnData
import scanpy as sc
import decoupler

# Local imports
from scdef.utils import score_utils, hierarchy_utils, color_utils
from scdef.utils.jax_utils import lognormal_sample, lognormal_entropy, gamma_logpdf

# Configure logging
logging.basicConfig()


class scDEF(object):
    """Single-cell Deep Exponential Families model.

    This model learns multi-level gene signatures describing the input scRNA-seq
    data from an AnnData object.

    Args:
        adata: AnnData object containing the gene expression data. scDEF learns a model from
            counts, so they must be present in either adata.X or in adata.layers.
        counts_layer: layer from adata.layers to get the count data from.
        layer_sizes: number of factors per scDEF layer.
        batch_key: key in adata.obs containing batch annotations for batch correction. If None, or not found,
            no batch correction is performed.
        seed: random seed for JAX
        logginglevel: verbosity level for logger
        brd_strength: BRD prior concentration parameter
        brd_mean: BRD prior mean parameter
        use_brd: whether to use the BRD prior for factor relevance estimation
        cell_scale_shape: concentration level in the cell scale prior
        gene_scale_shape: concentration level in the gene scale prior
        factor_shapes: prior parameters for the W concentration to use in each scDEF layer
        batch_cpal: default color palette for batch annotations
        layer_cpal: default color palettes for scDEF layers
        lightness_mult: multiplier to define lightness of color palette at each scDEF layer
    """

    def make_corrected_data(self, layer_name="scdef_corrected"):
        """
        Adds the low-rank approximation to the UMI counts from the lowest scDEF layer
        to the internal AnnData object, accessible via .adata.layers[`layer_name`]
        """
        scdef_layer = self.layer_names[0]
        Z = self.pmeans[f"{scdef_layer}z"][:, self.factor_lists[0]]  # nxk
        W = self.pmeans[f"{scdef_layer}W"][self.factor_lists[0]]  # kxg
        self.adata.layers[layer_name] = np.array(Z.dot(W))

    def __init__(
        self,
        adata: AnnData,
        counts_layer: Optional[str] = None,
        n_factors: Optional[int] = 100,
        decay_factor: Optional[float] = 2.0,
        max_n_layers: Optional[float] = 5,
        layer_sizes: Optional[list] = None,
        layer_names: Optional[list] = None,
        batch_key: Optional[str] = None,
        seed: Optional[int] = 42,
        logginglevel: Optional[int] = logging.INFO,
        layer_concentration: Optional[float] = 10.0,
        factor_shape: Optional[float] = 0.1,
        brd_strength: Optional[float] = 0.1,
        brd_mean: Optional[float] = 10.0,
        use_brd: Optional[bool] = True,
        cell_scale_shape: Optional[float] = 0.1,
        gene_scale_shape: Optional[float] = 0.1,
        batch_cpal: Optional[str] = "Dark2",
        layer_cpal: Optional[str] = "tab10",
        lightness_mult: Optional[float] = 0.15,
    ):
        self.n_cells, self.n_genes = adata.shape

        self.n_batches = 1
        self.batches = [""]
        self.batch_key = batch_key
        
        self.seed = seed
        self.batch_cpal = batch_cpal
        self.layer_cpal = layer_cpal
        self.lightness_mult = lightness_mult

        self.logger = logging.getLogger("scDEF")
        self.logger.setLevel(logginglevel)

        self.load_adata(adata, layer=counts_layer, batch_key=batch_key)

        self.layer_concentration = layer_concentration
        self.factor_shape = factor_shape
        self.brd = brd_strength
        self.brd_mean = brd_mean
        self.cell_scale_shape = cell_scale_shape
        self.gene_scale_shape = gene_scale_shape
        self.use_brd = use_brd

        self.decay_factor = decay_factor
        self.n_factors = n_factors
        self.max_n_layers = max_n_layers
        if layer_sizes is not None:
            self.layer_sizes = [int(x) for x in layer_sizes]
            self.n_factors = int(layer_sizes[0])
            self.n_layers = len(self.layer_sizes)
        else:
            self.update_model_size(n_factors)

        if layer_names is not None:
            self.layer_names = layer_names
        else:
            self.layer_names = [f"L{i}" for i in range(self.n_layers)]
        self.factor_lists = [np.arange(size) for size in self.layer_sizes]

        self.update_model_priors()

        self.make_layercolors(layer_cpal=self.layer_cpal, lightness_mult=lightness_mult)

        self.init_var_params(nmf_init=False)  # just to get stub
        self.set_posterior_means()

    def __repr__(self):
        out = f"scDEF object with {self.n_layers} layers"
        out += (
            "\n\t"
            + "Layer names: "
            + ", ".join([f"{name}" for name in self.layer_names])
        )
        out += (
            "\n\t"
            + "Layer sizes: "
            + ", ".join([str(len(factors)) for factors in self.factor_lists])
        )
        out += (
            "\n\t" + "Layer concentration parameter: " + str(self.layer_concentration)
        )
        out += (
            "\n\t"
            + "Layer factor shape parameters: "
            + ", ".join([str(shape) for shape in self.factor_shapes])
        )
        if self.use_brd == True:
            out += "\n\t" + "Using BRD"
        out += "\n\t" + "Number of batches: " + str(self.n_batches)
        out += "\n" + "Contains " + self.adata.__str__()
        return out

    def make_layercolors(
        self,
        layer_cpal: Optional[str] = "tab10",
        lightness_mult: Optional[float] = 0.15,
    ):
        if isinstance(layer_cpal, str):
            layer_cpal = [layer_cpal] * self.n_layers
        elif isinstance(layer_cpal, list):
            if len(layer_cpal) != self.n_layers:
                raise ValueError("layer_cpal list must be of size scDEF.n_layers")
        else:
            raise TypeError("layer_cpal must be either a cmap str or a list of strs")

        layer_sizes = []
        for i in range(self.n_layers):
            layer_size = len(self.factor_lists[i])
            if layer_size > 10:
                layer_cpal[i] = "tab20"
            elif layer_size > 20:
                layer_cpal[i] = "hsl"
            layer_sizes.append(layer_size)

        self.layer_colorpalettes = [
            sns.color_palette(layer_cpal[idx], n_colors=size)
            for idx, size in enumerate(layer_sizes)
        ]

        if len(np.unique(layer_cpal)) == 1:
            # Make the layers have different lightness
            for layer_idx, size in enumerate(layer_sizes):
                for factor_idx in range(size):
                    col = self.layer_colorpalettes[layer_idx][factor_idx]
                    self.layer_colorpalettes[layer_idx][factor_idx] = (
                        color_utils.adjust_lightness(
                            col, amount=1.0 + lightness_mult * layer_idx
                        )
                    )

    def load_adata(self, adata, layer=None, batch_key=None):
        if not isinstance(adata, AnnData):
            raise TypeError("adata must be an instance of AnnData.")
        self.adata = adata.copy()
        self.adata.raw = self.adata
        X = self.adata.raw.X
        if layer is not None:
            X = self.adata.layers[layer]

        if isinstance(X, scipy.sparse.csr_matrix):
            self.X = np.array(X.toarray())
        else:
            self.X = np.array(X)

        self.X = self.X.astype(float)

        self.batch_indices_onehot = np.ones((self.adata.shape[0], 1))
        self.batch_lib_sizes = np.sum(self.X, axis=1)
        self.batch_lib_ratio = (
            np.ones((self.X.shape[0], 1))
            * np.mean(self.batch_lib_sizes)
            / np.var(self.batch_lib_sizes)
        )
        gene_size = np.sum(self.X, axis=0)
        self.gene_means = np.mean(gene_size)
        self.gene_vars = np.var(gene_size)
        self.gene_ratio = self.gene_means / self.gene_vars
        if batch_key is not None:
            if batch_key in self.adata.obs.columns:
                batches = np.unique(self.adata.obs[batch_key].values)
                self.batches = batches
                self.n_batches = len(batches)
                self.logger.info(
                    f"Found {self.n_batches} values for `{batch_key}` in data: {batches}"
                )
                if f"{batch_key}_colors" in self.adata.uns:
                    self.batch_colors = self.adata.uns[f"{batch_key}_colors"]
                else:
                    batch_colorpalette = sns.color_palette(
                        self.batch_cpal, n_colors=self.n_batches
                    )
                    self.batch_colors = [
                        matplotlib.colors.to_hex(batch_colorpalette[idx])
                        for idx in range(self.n_batches)
                    ]
                    self.adata.uns[f"{batch_key}_colors"] = self.batch_colors
                self.batch_indices_onehot = np.zeros(
                    (self.adata.shape[0], self.n_batches)
                )
                if self.n_batches > 1:
                    self.gene_means = np.ones((self.n_batches, self.adata.shape[1]))
                    self.gene_vars = np.ones((self.n_batches, self.adata.shape[1]))
                    self.gene_ratio = np.ones((self.n_batches, self.adata.shape[1]))
                    for i, b in enumerate(batches):
                        cells = np.where(self.adata.obs[batch_key] == b)[0]
                        self.batch_indices_onehot[cells, i] = 1
                        self.batch_lib_sizes[cells] = np.sum(self.X, axis=1)[cells]
                        self.batch_lib_ratio[cells] = np.mean(
                            self.batch_lib_sizes[cells]
                        ) / np.var(self.batch_lib_sizes[cells])
                        batch_gene_size = np.sum(self.X[cells], axis=0)
                        self.gene_means[i] = np.mean(batch_gene_size)
                        self.gene_vars[i] = np.var(batch_gene_size)
                        self.gene_ratio[i] = self.gene_means[i] / self.gene_vars[i]
        self.batch_indices_onehot = jnp.array(self.batch_indices_onehot)
        self.batch_lib_sizes = jnp.array(self.batch_lib_sizes)
        self.batch_lib_ratio = jnp.array(self.batch_lib_ratio)
        self.gene_ratio = jnp.array(self.gene_ratio)

    def update_model_size(
        self,
        max_n_factors,
        max_n_layers=None,
    ):
        if max_n_layers is None:
            max_n_layers = self.max_n_layers
        n_factors = int(max_n_factors)
        self.layer_sizes = []
        self.layer_sizes.append(n_factors)
        if max_n_layers > 1:
            while len(self.layer_sizes) < max_n_layers:
                n_factors = int(np.floor(n_factors / self.decay_factor))
                self.layer_sizes.append(n_factors)
                if n_factors == 1:
                    break
            if self.layer_sizes[-1] > 1:
                self.layer_sizes[-1] = 1

        self.n_layers = len(self.layer_sizes)
        self.layer_names = [f"L{i}" for i in range(self.n_layers)]
        self.factor_lists = [np.arange(size) for size in self.layer_sizes]

    def update_model_priors(self):
        if self.use_brd:
            self.factor_shapes = [1.0] + [
                self.factor_shape * self.layer_sizes[l] / self.layer_sizes[0]
                for l in range(1, self.n_layers)
            ]
        else:
            self.factor_shapes = [
                self.factor_shape * self.layer_sizes[l] / self.layer_sizes[0]
                for l in range(self.n_layers)
            ]

        self.w_priors = []
        for idx in range(self.n_layers):
            prior_shapes = self.factor_shapes[idx]
            prior_rates = self.factor_shapes[idx]

            if idx > 0:
                prior_shapes = (
                    np.ones((self.layer_sizes[idx], self.layer_sizes[idx - 1]))
                    * self.factor_shapes[idx]
                    * 1.0
                )
                prior_rates = (
                    np.ones((self.layer_sizes[idx], self.layer_sizes[idx - 1]))
                    * self.factor_shapes[idx]
                )
                prior_shapes = jnp.clip(jnp.array(prior_shapes), 1e-12, 1e12)
                prior_rates = jnp.clip(jnp.array(prior_rates), 1e-12, 1e12)

            self.w_priors.append([prior_shapes, prior_rates])

    def get_effective_factors(
        self,
        thres: Optional[float] = None,
        iqr_mult: Optional[float] = 0.0,
        min_cells: Optional[float] = 0.0,
        normalized: Optional[bool] = False,
    ):
        layer_name = self.layer_names[0]
        min_cells = (
            np.maximum(10, min_cells * self.n_cells) if min_cells < 1.0 else min_cells
        )
        ard = []
        if thres is not None:
            ard = thres
        else:
            ard = iqr_mult

        if not self.use_brd:
            ard = 0.0

        normed = (
            self.pmeans[f"{layer_name}z"]
            / np.sum(self.pmeans[f"{layer_name}z"], axis=1)[:, None]
        )
        assignments = np.argmax(normed, axis=1)
        counts = np.array(
            [np.count_nonzero(assignments == a) for a in range(self.layer_sizes[0])]
        )
        masses = np.sum(normed, axis=0)
        keep = np.array(range(self.layer_sizes[0]))[np.where(counts >= min_cells)[0]]
        brd_keep = np.arange(self.layer_sizes[0])
        if self.use_brd:
            rels = self.pmeans[f"brd"].ravel()
            if normalized:
                rels = rels - np.min(rels)
                rels = rels / np.max(rels)
            if thres is None:
                median = np.median(rels)
                q3 = np.percentile(rels, 75)
                cutoff = ard * (q3 - median)
            else:
                cutoff = ard
            brd_keep = np.where(rels >= cutoff)[0]
            if len(brd_keep) == 0:
                brd_keep = np.arange(self.layer_sizes[0]).astype(int)
        keep = np.unique(list(set(brd_keep).intersection(keep)))
        return keep

    def init_var_params(
        self, init_budgets=True, init_z=None, init_w=None, nmf_init=False, **kwargs
    ):
        rngs = random.split(random.PRNGKey(self.seed), self.n_layers)

        if init_budgets:
            m = 1.0
            v = m
            self.local_params = [
                jnp.array(
                    (
                        jnp.log(m**2 / jnp.sqrt(m**2 + v))
                        * jnp.ones((self.n_cells, 1)),  # cell_scales
                        jnp.log(jnp.sqrt(jnp.log(1 + v / (m**2))))
                        * jnp.ones((self.n_cells, 1)),
                    )
                ),
            ]
            m = 1.0
            v = m
            self.global_params = [
                jnp.array(
                    (
                        jnp.log(m**2 / jnp.sqrt(m**2 + v))
                        * jnp.ones((self.n_batches, self.n_genes)),  # gene_scales
                        jnp.log(jnp.sqrt(jnp.log(1 + v / (m**2))))
                        * jnp.ones((self.n_batches, self.n_genes)),
                    )
                ),
            ]
        else:
            self.local_params = [self.local_params[0]]
            self.global_params = [self.global_params[0]]
        # BRD
        m_brd = self.brd_mean
        v_brd = m_brd / 100.0

        self.global_params.append(
            jnp.array(
                (
                    jnp.log(m_brd**2 / jnp.sqrt(m_brd**2 + v_brd))
                    * jnp.ones((self.layer_sizes[0], 1)),
                    jnp.log(jnp.sqrt(jnp.log(1 + v_brd / (m_brd**2))))
                    * jnp.ones((self.layer_sizes[0], 1)),
                )
            ),
        )

        if nmf_init:
            self.logger.info(f"Initializing the factor weights with NMF.")
            init_z, init_w = self.get_nmf_init(**kwargs)

        z_shapes = []
        z_rates = []
        rng_cnt = 0
        for layer_idx in range(
            self.n_layers
        ):  # we go from layer 0 (bottom) to layer L (top)
            # Init z
            a = 1.0
            clip = 1e-6  # alpha=1 and clip=1e-6 allows for many factors to be learned, but leads to BRD becoming kind of big

            if (
                layer_idx > 0
            ):  # If the top layer inits to very small numbers, the loss goes crazy when MC=10...
                clip = 1e-6
                a = 1.0

            if layer_idx == self.n_layers - 1:
                clip = 1e-3

            if init_z is not None and layer_idx == 0 and not nmf_init:
                m = init_z.astype(jnp.float32)
            else:
                m = jnp.clip(
                    tfd.Gamma(a, a / 1.0).sample(
                        seed=rngs[rng_cnt],
                        sample_shape=[self.n_cells, self.layer_sizes[layer_idx]],
                    ),
                    clip,
                    1e1,
                )
                rng_cnt += 1

            v = m / 100.0
            if layer_idx > 0:
                v = m / 100.0
            z_shape = jnp.log(m**2 / jnp.sqrt(m**2 + v)) * jnp.ones(
                (self.n_cells, self.layer_sizes[layer_idx])
            )
            z_shapes.append(z_shape)
            z_rate = jnp.log(jnp.sqrt(jnp.log(1 + v / (m**2)))) * jnp.ones(
                (self.n_cells, self.layer_sizes[layer_idx])
            )
            z_rates.append(z_rate)

        self.local_params.append(jnp.array((jnp.hstack(z_shapes), jnp.hstack(z_rates))))

        for layer_idx in range(
            self.n_layers
        ):  # the w don't have a shared axis across all, so we can't vectorize
            # Init w
            in_layer = self.layer_sizes[layer_idx]
            if layer_idx == 0:
                out_layer = self.n_genes
            else:
                out_layer = self.layer_sizes[layer_idx - 1]

            a = 100.0

            if init_w is not None and layer_idx == 0 and not nmf_init:
                m = init_w.astype(jnp.float32)
                v = m
            else:

                m = 1.0 / self.layer_sizes[layer_idx] * jnp.ones((in_layer, out_layer))
                m = m * self.w_priors[layer_idx][0] / self.w_priors[layer_idx][1]
                v = m / 100.0
            if nmf_init:
                iw = init_w[layer_idx].astype(jnp.float32)
                # Rescale
                iw = 1.0 * iw / np.max(iw, axis=1)[:, None] + 1.0
                # iw = 10. * 100.**iw/100. + 1.
                iw = jnp.clip(iw, 1e-6, 1e1)  # need to set scales to avoid huge BRDs...
                m = tfd.Gamma(
                    100.0, 100.0 / (iw * 1.0 / self.layer_sizes[layer_idx])
                ).sample(seed=rngs[rng_cnt])
                rng_cnt += 1
                v = m / 100.0

            w_shape = jnp.log(m**2 / jnp.sqrt(m**2 + v)) * jnp.ones(
                (in_layer, out_layer)
            )
            w_rate = jnp.log(jnp.sqrt(jnp.log(1 + v / (m**2)))) * jnp.ones(
                (in_layer, out_layer)
            )

            self.global_params.append(jnp.array((w_shape, w_rate)))

        # Gamma(1e3, 1e3/m)
        m = 1.0
        v = m / 100.0

        self.global_params.append(
            jnp.array(
                (
                    jnp.log(m**2 / jnp.sqrt(m**2 + v))
                    * jnp.ones((self.layer_sizes[0], 1)),
                    jnp.log(jnp.sqrt(jnp.log(1 + v / (m**2))))
                    * jnp.ones((self.layer_sizes[0], 1)),
                )
            ),
        )

    def get_nmf_init(self, max_cells=None):
        """Use NMF on the data to init the first layer and then recursively for the other layers"""
        init_z = []
        init_W = []
        from sklearn.decomposition import NMF

        X = self.X
        X = X * 10_000 / np.sum(X, axis=1)[:, None]
        X = np.log(X + 1)
        if max_cells is not None:
            if max_cells < X.shape[0]:
                key = jax.random.PRNGKey(self.seed)
                cells = jax.random.choice(
                    key, X.shape[0], replace=False, shape=(max_cells,)
                )
                X = X[cells]
        for layer_idx in range(self.n_layers):
            model = NMF(
                n_components=self.layer_sizes[layer_idx],
                random_state=self.seed,
            )
            z = model.fit_transform(X)
            W = model.components_
            X = z
            init_z.append(z)
            init_W.append(W)

        return init_z, init_W

    def identify_mixture_factors(self, max_n_genes=20, thres=0.5):
        """Identify factors that might be better if broken apart"""
        # sparse_factors = np.where(self.pmeans['brd'].ravel() > 1.)[0]
        kept_factors = self.factor_lists[0]
        normed_factors = (
            self.pmeans["L0W"][kept_factors]
            / self.pmeans["L0W"][kept_factors].max(axis=1)[:, None]
        )
        n_genes_per_factor = np.sum(normed_factors > thres, axis=1)
        mixture_factors = kept_factors[np.where(n_genes_per_factor > max_n_genes)[0]]
        return mixture_factors

    def reinit_factors(
        self, mixture_factors=None, init_budgets=False, exponent=1.1, **kwargs
    ):
        kept_factors = self.factor_lists[0]
        if mixture_factors is None:
            mixture_factors = self.identify_mixture_factors(**kwargs)
        # Make W0 matrix with sparse factors, and with each mixture factor repeated twice
        W0 = np.ones((self.layer_sizes[0], self.n_genes)) * 1.0 / self.layer_sizes[0]
        for i, factor in enumerate(kept_factors):
            W0[i] = self.pmeans["L0W"][factor] ** exponent

        # If we can fit repetitions
        if len(mixture_factors) <= self.layer_sizes[0] - len(kept_factors):
            for factor in mixture_factors:
                # Add another copy
                W0[len(kept_factors) + i] = self.pmeans["L0W"][factor] ** exponent

        self.init_var_params(init_budgets=init_budgets, init_w=W0, nmf_init=False)
        return W0

    def elbo(
        self,
        rng,
        batch,
        indices,
        local_params,
        global_params,
        annealing_parameter,
        stop_gradients,
        stop_cell_budgets,
        stop_gene_budgets,
        min_shape=jnp.log(1e-10),
        max_shape=jnp.log(1e6),
        min_rate=jnp.log(1e-10),
        max_rate=jnp.log(1e10),
    ):
        # Single-sample Monte Carlo estimate of the variational lower bound.
        batch_indices_onehot = self.batch_indices_onehot[indices]

        cell_budget_params = local_params[0]
        z_params = local_params[1]
        gene_budget_params = global_params[0]
        fscale_params = global_params[1]

        cell_budget_shape = jnp.clip(
            cell_budget_params[0][indices], min_shape, max_shape
        )
        cell_budget_rate = jnp.exp(
            jnp.clip(cell_budget_params[1][indices], min_rate, max_rate)
        )
        cell_budget_shape = jax.lax.cond(
            stop_gradients[0],
            lambda: jax.lax.stop_gradient(cell_budget_shape),
            lambda: cell_budget_shape,
        )
        cell_budget_rate = jax.lax.cond(
            stop_gradients[0],
            lambda: jax.lax.stop_gradient(cell_budget_rate),
            lambda: cell_budget_rate,
        )

        gene_budget_shape = jnp.maximum(gene_budget_params[0], min_shape)
        gene_budget_rate = jnp.exp(jnp.clip(gene_budget_params[1], min_rate, max_rate))
        gene_budget_shape = jax.lax.cond(
            stop_gradients[0],
            lambda: jax.lax.stop_gradient(gene_budget_shape),
            lambda: gene_budget_shape,
        )
        gene_budget_rate = jax.lax.cond(
            stop_gradients[0],
            lambda: jax.lax.stop_gradient(gene_budget_rate),
            lambda: gene_budget_rate,
        )

        fscale_shapes = jnp.clip(fscale_params[0], min_shape, max_shape)
        fscale_rates = jnp.exp(jnp.clip(fscale_params[1], min_rate, max_rate))
        fscale_shapes = jax.lax.cond(
            stop_gradients[0],
            lambda: jax.lax.stop_gradient(fscale_shapes),
            lambda: fscale_shapes,
        )
        fscale_rates = jax.lax.cond(
            stop_gradients[0],
            lambda: jax.lax.stop_gradient(fscale_rates),
            lambda: fscale_rates,
        )

        wm_shape = jnp.clip(global_params[2 + self.n_layers][0], min_shape, max_shape)
        wm_rate = jnp.exp(
            jnp.clip(global_params[2 + self.n_layers][1], min_rate, max_rate)
        )

        z_shapes = jnp.clip(z_params[0][indices], min_shape, max_shape)
        z_rates = jnp.exp(jnp.clip(z_params[1][indices], min_rate, max_rate))

        # Sample from variational distribution
        cell_budget_shape = jax.lax.cond(
            stop_cell_budgets,
            lambda: jax.lax.stop_gradient(cell_budget_shape),
            lambda: cell_budget_shape,
        )
        cell_budget_rate = jax.lax.cond(
            stop_cell_budgets,
            lambda: jax.lax.stop_gradient(cell_budget_rate),
            lambda: cell_budget_rate,
        )
        cell_budget_sample = lognormal_sample(rng, cell_budget_shape, cell_budget_rate)
        gene_budget_shape = jax.lax.cond(
            stop_gene_budgets,
            lambda: jax.lax.stop_gradient(gene_budget_shape),
            lambda: gene_budget_shape,
        )
        gene_budget_rate = jax.lax.cond(
            stop_gene_budgets,
            lambda: jax.lax.stop_gradient(gene_budget_rate),
            lambda: gene_budget_rate,
        )
        gene_budget_sample = lognormal_sample(rng, gene_budget_shape, gene_budget_rate)
        if self.use_brd:
            fscale_samples = lognormal_sample(rng, fscale_shapes, fscale_rates)
            _wm_sample = lognormal_sample(rng, wm_shape, wm_rate)
            brd_samples = fscale_samples / jnp.maximum(_wm_sample, 1.0)
        # w will be sampled in a loop below because it cannot be vectorized

        # Compute ELBO
        global_pl = gamma_logpdf(
            gene_budget_sample,
            self.gene_scale_shape,
            self.gene_scale_shape * self.gene_ratio,
        )
        global_en = lognormal_entropy(gene_budget_shape, gene_budget_rate)
        local_pl = gamma_logpdf(
            cell_budget_sample,
            self.cell_scale_shape,
            self.cell_scale_shape * self.batch_lib_ratio[indices],
        )
        local_en = lognormal_entropy(cell_budget_shape, cell_budget_rate)

        # scale
        if self.use_brd:
            global_pl += gamma_logpdf(
                fscale_samples, self.brd, self.brd / self.brd_mean
            )
            global_en += lognormal_entropy(fscale_shapes, fscale_rates)

            global_pl += gamma_logpdf(_wm_sample, 0.1, 0.1)
            global_en += lognormal_entropy(wm_shape, wm_rate)

        z_mean = 1.0
        for idx in list(np.arange(0, self.n_layers)[::-1]):
            start = np.sum(self.layer_sizes[:idx]).astype(int)
            end = start + self.layer_sizes[idx]

            # w
            _w_shape = jnp.clip(global_params[2 + idx][0], min_shape, max_shape)
            _w_rate = jnp.exp(jnp.clip(global_params[2 + idx][1], min_rate, max_rate))

            _w_shape = jax.lax.cond(
                stop_gradients[idx],
                lambda: jax.lax.stop_gradient(_w_shape),
                lambda: _w_shape,
            )
            _w_rate = jax.lax.cond(
                stop_gradients[idx],
                lambda: jax.lax.stop_gradient(_w_rate),
                lambda: _w_rate,
            )

            _w_sample = lognormal_sample(rng, _w_shape, _w_rate)

            if idx == 0 and self.use_brd:
                global_pl += gamma_logpdf(
                    _w_sample,
                    self.w_priors[idx][0] * (1.0 / fscale_samples),
                    self.w_priors[idx][1]
                    * (1.0 / fscale_samples)
                    * (1.0 / _wm_sample)
                    * self.layer_sizes[idx],
                )
            elif idx == 0:
                global_pl += gamma_logpdf(
                    _w_sample,
                    self.w_priors[idx][0],
                    self.w_priors[idx][1] * self.layer_sizes[idx],
                )
            else:
                if idx == 1 and self.use_brd:
                    global_pl += gamma_logpdf(
                        _w_sample,
                        self.w_priors[idx][0],
                        self.w_priors[idx][1] * self.layer_sizes[idx] / brd_samples.T,
                    )
                else:
                    global_pl += gamma_logpdf(
                        _w_sample,
                        self.w_priors[idx][0],
                        self.w_priors[idx][1] * self.layer_sizes[idx],
                    )
            global_en += lognormal_entropy(_w_shape, _w_rate)

            # z
            _z_shape = z_shapes[:, start:end]
            _z_rate = z_rates[:, start:end]

            _z_shape = jax.lax.cond(
                stop_gradients[idx],
                lambda: jax.lax.stop_gradient(_z_shape),
                lambda: _z_shape,
            )
            _z_rate = jax.lax.cond(
                stop_gradients[idx],
                lambda: jax.lax.stop_gradient(_z_rate),
                lambda: _z_rate,
            )
            _z_sample = lognormal_sample(rng, _z_shape, _z_rate)

            rate_param = 1.0
            if idx > 0:
                rate_param = 1.0
            elif self.n_layers == 1 and self.use_brd:
                rate_param = brd_samples.T

            local_pl += jax.lax.cond(
                idx == self.n_layers - 1,
                lambda: gamma_logpdf(_z_sample, 0.1, 0.1 / rate_param),
                lambda: gamma_logpdf(
                    _z_sample,
                    self.layer_concentration,
                    self.layer_concentration / (rate_param * z_mean),
                ),
            )

            local_en += lognormal_entropy(_z_shape, _z_rate)

            z_mean = jnp.einsum("nk,kp->np", _z_sample, _w_sample)

        mean_bottom = jnp.einsum("nk,kg->ng", _z_sample, _w_sample) * cell_budget_sample
        mean_bottom = mean_bottom * (batch_indices_onehot.dot(gene_budget_sample))

        ll = jnp.sum(vmap(poisson.logpmf)(jnp.array(batch), mean_bottom))
        ll = jax.lax.cond(
            stop_gradients[0],
            lambda: ll * 0.0,
            lambda: ll,
        )

        # Anneal the entropy
        global_en *= annealing_parameter
        local_en *= annealing_parameter

        return (
            (ll + local_pl + local_en) * (self.X.shape[0] / indices.shape[0])
            + global_pl
            + global_en
        )

    def batch_elbo(
        self,
        rng,
        X,
        indices,
        local_params,
        global_params,
        num_samples,
        annealing_parameter,
        stop_gradients,
        stop_cell_budgets,
        stop_gene_budgets,
    ):
        # Average over a batch of random samples.
        rngs = random.split(rng, num_samples)
        vectorized_elbo = vmap(
            self.elbo, in_axes=(0, None, None, None, None, None, None, None, None)
        )
        return jnp.mean(
            vectorized_elbo(
                rngs,
                X,
                indices,
                local_params,
                global_params,
                annealing_parameter,
                stop_gradients,
                stop_cell_budgets,
                stop_gene_budgets,
            )
        )

    def _optimize(
        self,
        local_update_func,
        global_update_func,
        local_params,
        global_params,
        local_opt_state,
        global_opt_state,
        n_epochs=500,
        batch_size=1,
        annealing_parameter=1.0,
        stop_gradients=None,
        stop_cell_budgets=None,
        stop_gene_budgets=None,
        seed=None,
        min_epochs=100,
        tolerance=1e-5,
        patience=10,
        update_locals=True,
        update_globals=True,
    ):
        if seed is None:
            seed = self.seed

        if stop_gradients is None:
            stop_gradients = np.zeros((self.n_layers))
        if stop_cell_budgets is None:
            stop_cell_budgets = 0.0
        if stop_gene_budgets is None:
            stop_gene_budgets = 0.0

        num_complete_batches, leftover = divmod(self.n_cells, batch_size)
        num_batches = num_complete_batches + bool(leftover)
        self.logger.info(
            f"Each epoch contains {num_batches} batches of size {int(min(batch_size, self.n_cells))}"
        )

        def data_stream():
            rng = np.random.RandomState(0)
            while True:
                perm = rng.permutation(self.n_cells)
                for i in range(num_batches):
                    batch_idx = perm[i * batch_size : (i + 1) * batch_size]
                    yield jnp.array(self.X[batch_idx]), jnp.array(batch_idx)

        batches = data_stream()

        losses = []
        global_grads = 0.0
        annealing_parameter = jnp.array(annealing_parameter)
        stop_gradients = jnp.array(stop_gradients)
        stop_cell_budgets = jnp.array(stop_cell_budgets)
        stop_gene_budgets = jnp.array(stop_gene_budgets)
        rng = random.PRNGKey(seed)
        t = 0
        min_loss = np.inf
        early_stop_counter = 0
        stop_early = False
        pbar = tqdm(range(n_epochs))
        try:
            for epoch in pbar:
                epoch_losses = []
                start_time = time.time()
                for it in range(num_batches):
                    rng, rng_input = random.split(rng)
                    X, indices = next(batches)
                    if update_locals:
                        loss, local_params, local_opt_state = local_update_func(
                            X,
                            indices,
                            t,
                            rng_input,
                            local_params,
                            global_params,
                            local_opt_state,
                            global_opt_state,
                            annealing_parameter,
                            stop_gradients,
                            stop_cell_budgets,
                            stop_gene_budgets,
                        )
                    if update_globals:
                        loss, global_params, global_opt_state, global_grads = (
                            global_update_func(
                                X,
                                indices,
                                t,
                                rng_input,
                                local_params,
                                global_params,
                                local_opt_state,
                                global_opt_state,
                                annealing_parameter,
                                stop_gradients,
                                stop_cell_budgets,
                                stop_gene_budgets,
                            )
                        )
                    epoch_losses.append(loss)
                    t += 1
                current_loss = np.mean(epoch_losses)
                losses.append(current_loss)

                if epoch >= min_epochs:
                    if min_loss == np.inf:
                        min_loss = current_loss
                        stop_early = False

                    relative_improvement = (min_loss - current_loss) / np.abs(min_loss)
                    min_loss = min(min_loss, current_loss)

                    if relative_improvement < tolerance:
                        early_stop_counter += 1
                    else:
                        early_stop_counter = 0
                    if early_stop_counter >= patience:
                        stop_early = True

                if stop_early:
                    break

                epoch_time = time.time() - start_time
                pbar.set_postfix({"Loss": losses[-1]})
                if epoch >= min_epochs:
                    pbar.set_postfix(
                        {"Loss": losses[-1], "Rel. improvement": relative_improvement}
                    )

            if stop_early:
                self.logger.info(
                    "Relative improvement of "
                    f"{relative_improvement:0.4g} < {tolerance:0.4g} "
                    f"for {patience} step(s) in a row, stopping early."
                )
                if (
                    relative_improvement < 0
                    and np.abs(relative_improvement) > 100 * tolerance
                ):
                    self.logger.info(
                        "It seems like the loss increased significantly. "
                        "Consider using a lower learning rate."
                    )
            else:
                self.logger.info(
                    "Learning stopped before convergence. "
                    "Consider increasing the number of iterations."
                )

        except KeyboardInterrupt:
            self.logger.info("Interrupted learning. Exiting safely...")

        return (
            losses,
            local_params,
            global_params,
            local_opt_state,
            global_opt_state,
            global_grads,
        )

    def fit(
        self, pretrain=True, nmf_init=False, max_cells_init=1024, unmix=False, **kwargs
    ):
        """Learn a one-layer scDEF with 100 factors with BRD to obtain
        cell and gene scale factors and an estimate of the effective number of factors.
        Update the sizes based on that estimate and re-init everything except the scale factors.
        """
        if pretrain:
            self.logger.info(
                f"Pretraining to find initial estimate of number of factors"
            )
            self.update_model_size(self.n_factors, max_n_layers=1)
            self.update_model_priors()
            self.init_var_params(
                init_budgets=True, nmf_init=nmf_init, max_cells=max_cells_init
            )
            self.elbos = []
            self.step_sizes = []
            self._learn(filter=False, annotate=False, **kwargs)
            eff_factors = self.get_effective_factors(min_cells=0.01)
            mixture_factors = self.identify_mixture_factors()
            if len(mixture_factors) > 0 and unmix:
                self.logger.info(
                    f"Found {len(mixture_factors)} factors that activate more than 10 genes. Re-fitting to enhance sparsity"
                )
                self.reinit_factors(mixture_factors=mixture_factors)
                self._learn(filter=False, annotate=False, **kwargs)
                eff_factors = self.get_effective_factors(min_cells=0.01)
            n_eff_factors = len(eff_factors)
            if n_eff_factors == 0:
                self.logger.info(
                    "No effective factors found. Using all factors."
                )
                n_eff_factors = self.n_factors
                eff_factors = np.arange(n_eff_factors)
            self.logger.info(
                f"scDEF pretraining finished. Found {n_eff_factors} effective factors."
            )
            self.update_model_size(n_eff_factors)
            self.update_model_priors()
            self.logger.info(f"Learning scDEF with layer sizes {self.layer_sizes}.")
            init_budgets = False
            init_w = self.pmeans["L0W"][eff_factors]
        else:
            init_budgets = True
            init_w = None
        self.init_var_params(
            init_budgets=init_budgets,
            init_w=init_w,
            nmf_init=nmf_init,
            max_cells=max_cells_init,
        )
        self.elbos = []
        self.step_sizes = []
        self._learn(**kwargs)

    def _learn(
        self,
        n_epoch: Optional[Union[int, list]] = [1000],
        lr: Optional[Union[float, list]] = 1e-2,
        annealing: Optional[Union[float, list]] = 1.0,
        num_samples: Optional[int] = 10,
        batch_size: Optional[int] = 256,
        layerwise: Optional[bool] = False,
        min_epochs: Optional[int] = 50,
        tolerance: Optional[float] = 1e-5,
        patience: Optional[int] = 50,
        update_locals: Optional[bool] = True,
        update_globals: Optional[bool] = True,
        stop_cell_budgets: Optional[int] = 0,
        stop_gene_budgets: Optional[int] = 0,
        opt_layer: Optional[int] = None,
        filter: Optional[bool] = True,
        annotate: Optional[bool] = True,
        **kwargs,
    ):
        """Fit a variational approximation to the posterior over scDEF parameters.

        Args:
            n_epoch: number of epochs (full passes of the data).
                Can be a list of ints for multi-step learning.
            lr: learning rate.
                Can be a list of floats for multi-step learning.
            annealing: scale factor for the entropy term.
                Can be a list of floats for multi-step learning.
            num_samples: number of Monte Carlo samples to use in the ELBO approximation.
            batch_size: number of data points to use per iteration. If None, uses all.
                Useful for data sets that do not fit in GPU memory.
            layerwise: whether to optimize the model parameters in a step-wise manner:
                first learn only Layer 0 and 1, and then 2, and then 3, and so on. The size of
                the n_epoch or lr schedules will be ignored, only the first value will be used
                and each step will use that n_epoch value.
            min_epochs: minimum number of epochs for early stopping
            tolerance: maximum relative change in loss for early stopping
            patience: number of epochs for which tolerated loss changes must hold for early stopping
            update_locals: whether to optimize the local parameters
            update_globals: whether to optimize the global parameters
        """
        n_steps = 1
        if layerwise:
            n_steps = self.n_layers

        if isinstance(n_epoch, list):
            if layerwise:
                n_epoch_schedule = [n_epoch[0]] * n_steps
            else:
                n_steps = len(n_epoch)
                n_epoch_schedule = n_epoch
        else:
            n_epoch_schedule = [n_epoch] * n_steps

        if isinstance(lr, list):
            if layerwise:
                lr_schedule = [lr[0]] * n_steps
            lr_schedule = lr
            if len(lr_schedule) != n_steps:
                raise ValueError(
                    "lr_schedule list must be of same length as n_epoch_schedule"
                )
        else:
            if layerwise:
                lr_schedule = [lr] * n_steps
            else:
                lr_schedule = [lr * 0.1**step for step in range(n_steps)]

        if isinstance(annealing, list):
            if layerwise:
                annealing_schedule = [annealing[0]] * n_steps
            annealing_schedule = annealing
            if len(annealing_schedule) != n_steps:
                raise ValueError(
                    "annealing_schedule list must be of same length as n_epoch_schedule"
                )
        else:
            annealing_schedule = [annealing] * n_steps

        if batch_size is None:
            batch_size = self.n_cells

        # Set up jitted functions
        stop_gradients = np.ones((self.n_layers))
        layers_to_optimize = np.arange(self.n_layers)

        def objective(
            X,
            indices,
            local_var_params,
            global_var_params,
            key,
            annealing_parameter,
            stop_gradients,
            stop_cell_budgets,
            stop_gene_budgets,
        ):
            return -self.batch_elbo(
                key,
                X,
                indices,
                local_var_params,
                global_var_params,
                num_samples,
                annealing_parameter,
                stop_gradients,
                stop_cell_budgets,
                stop_gene_budgets,
            )  # minimize -ELBO

        def clip_params(
            params, min_mu=-1e10, max_mu=1e2, min_logstd=-1e10, max_logstd=1e1
        ):
            for i in range(len(params))[:-2]:  # skip the last two params (w and s)
                params[i] = (
                    params[i].at[0].set(jnp.clip(params[i][0], min_mu, max_mu))
                )  # mu
                params[i] = (
                    params[i].at[1].set(jnp.clip(params[i][1], min_logstd, max_logstd))
                )  # logstd
            return params

        local_loss_grad = jit(value_and_grad(objective, argnums=2))
        global_loss_grad = jit(value_and_grad(objective, argnums=3))

        for i in range(len(n_epoch_schedule)):
            n_epochs = n_epoch_schedule[i]
            step_size = lr_schedule[i]
            anneal_param = annealing_schedule[i]
            self.logger.info(f"Initializing optimizer with learning rate {step_size}.")
            if anneal_param != 1:
                self.logger.info(f"Set annealing parameter to {anneal_param}.")

            local_optimizer = optax.adam(step_size / 1.0)
            global_optimizer = optax.adam(step_size)

            if layerwise:
                if opt_layer is not None:
                    if i != opt_layer:
                        continue
                    layers_to_optimize = np.array([i])
                else:
                    layers_to_optimize = np.array([i])
                # layers_to_optimize = np.arange(2 + i)
                # if i > 0:
                #     layers_to_optimize = np.array([2+i-1])
                self.logger.info(f"Optimizing layers {layers_to_optimize}")
            stop_gradients[layers_to_optimize] = 0.0

            def local_update(
                X,
                indices,
                i,
                key,
                local_params,
                global_params,
                local_opt_state,
                global_opt_state,
                annealing_parameter,
                stop_gradients,
                stop_cell_budgets,
                stop_gene_budgets,
            ):
                value, gradient = local_loss_grad(
                    X,
                    indices,
                    local_params,
                    global_params,
                    key,
                    annealing_parameter,
                    stop_gradients,
                    stop_cell_budgets,
                    stop_gene_budgets,
                )
                updates, local_opt_state = local_optimizer.update(
                    gradient, local_opt_state, local_params
                )
                local_params = optax.apply_updates(local_params, updates)
                local_params = clip_params(local_params)
                return value, local_params, local_opt_state

            def global_update(
                X,
                indices,
                i,
                key,
                local_params,
                global_params,
                local_opt_state,
                global_opt_state,
                annealing_parameter,
                stop_gradients,
                stop_cell_budgets,
                stop_gene_budgets,
            ):
                value, gradient = global_loss_grad(
                    X,
                    indices,
                    local_params,
                    global_params,
                    key,
                    annealing_parameter,
                    stop_gradients,
                    stop_cell_budgets,
                    stop_gene_budgets,
                )
                updates, global_opt_state = global_optimizer.update(
                    gradient, global_opt_state, global_params
                )
                global_params = optax.apply_updates(global_params, updates)
                global_params = clip_params(global_params)
                return value, global_params, global_opt_state, gradient

            # local_opt_state = local_opt_init(self.local_params)
            local_opt_state = local_optimizer.init(self.local_params)
            # global_opt_state = global_opt_init(self.global_params)
            global_opt_state = global_optimizer.init(self.global_params)

            (
                losses,
                local_params,
                global_params,
                local_opt_state,
                global_opt_state,
                global_gradients,
            ) = self._optimize(
                local_update,
                global_update,
                self.local_params,
                self.global_params,
                local_opt_state,
                global_opt_state,
                n_epochs=n_epochs,
                batch_size=batch_size,
                annealing_parameter=anneal_param,
                stop_gradients=stop_gradients,
                stop_cell_budgets=stop_cell_budgets,
                stop_gene_budgets=stop_gene_budgets,
                min_epochs=min_epochs,
                tolerance=tolerance,
                patience=patience,
                update_locals=update_locals,
                update_globals=update_globals,
                **kwargs,
            )
            if update_locals:
                self.local_params = local_params
            if update_globals:
                self.global_params = global_params
            self.elbos.append(losses)
            self.step_sizes.append(lr_schedule[i])

        self.set_posterior_means()
        if self.use_brd and filter:
            self.filter_factors()
        else:
            if annotate:
                self.make_layercolors(
                    layer_cpal=self.layer_cpal, lightness_mult=self.lightness_mult
                )
                self.annotate_adata()

    def set_posterior_means(self):
        cell_budget_params = self.local_params[0]
        gene_budget_params = self.global_params[0]
        fscale_params = self.global_params[1]
        wm_params = self.global_params[-1]
        z_params = self.local_params[1]

        self.pmeans = {
            "cell_scale": np.array(
                jnp.exp(
                    cell_budget_params[0] + 0.5 * jnp.exp(cell_budget_params[1]) ** 2
                )
            ),
            "gene_scale": np.array(
                jnp.exp(
                    gene_budget_params[0] + 0.5 * jnp.exp(gene_budget_params[1]) ** 2
                )
            ),
            "factor_concentrations": jnp.exp(
                fscale_params[0] + 0.5 * jnp.exp(fscale_params[1]) ** 2
            ),
            "factor_means": jnp.exp(wm_params[0] + 0.5 * jnp.exp(wm_params[1]) ** 2),
        }
        self.pmeans["brd"] = self.pmeans["factor_concentrations"] / jnp.maximum(
            self.pmeans["factor_means"], 1.0
        )

        for idx in range(self.n_layers):
            start = sum(self.layer_sizes[:idx])
            end = start + self.layer_sizes[idx]
            self.pmeans[f"{self.layer_names[idx]}z"] = jnp.exp(
                z_params[0][:, start:end]
                + 0.5 * jnp.exp(z_params[1][:, start:end]) ** 2
            )
            _w_shape = self.global_params[2 + idx][0]
            _w_rate = self.global_params[2 + idx][1]
            self.pmeans[f"{self.layer_names[idx]}W"] = jnp.exp(
                _w_shape + 0.5 * jnp.exp(_w_rate) ** 2
            )

    def set_posterior_variances(self):
        cell_budget_params = self.local_params[0]
        gene_budget_params = self.global_params[0]
        fscale_params = self.global_params[1]
        wm_params = self.global_params[-1]
        z_params = self.local_params[1]

        self.pvars = {
            "cell_scale": np.array(
                np.exp(cell_budget_params[0]) / np.exp(cell_budget_params[1]) ** 2
            ),
            "gene_scale": np.array(
                np.exp(gene_budget_params[0]) / np.exp(gene_budget_params[1]) ** 2
            ),
            "factor_concentrations": np.array(
                np.exp(fscale_params[0]) / np.exp(fscale_params[1]) ** 2
            ),
            "factor_means": jnp.exp(wm_params[0] + 0.5 * jnp.exp(wm_params[1]) ** 2),
        }

        for idx in range(self.n_layers):
            start = sum(self.layer_sizes[:idx])
            end = start + self.layer_sizes[idx]
            self.pvars[f"{self.layer_names[idx]}z"] = np.array(
                np.exp(z_params[0][:, start:end])
                / np.exp(z_params[1][:, start:end]) ** 2
            )
            _w_shape = self.global_params[2 + idx][0]
            _w_rate = self.global_params[2 + idx][1]
            self.pvars[f"{self.layer_names[idx]}W"] = np.array(
                np.exp(_w_shape) / np.exp(_w_rate) ** 2
            )

    def filter_factors(
        self,
        thres: Optional[float] = 1.0,
        iqr_mult: Optional[float] = 0.0,
        min_cells: Optional[float] = 0.001,
        filter_up: Optional[bool] = True,
        normalized: Optional[bool] = False,
    ):
        """Filter our irrelevant factors based on the BRD posterior or the cell attachments.

        Args:
            thres: minimum factor BRD value
            iqr_mult: multiplier of the difference between the third quartile and the median BRD values to set the threshold
            min_cells: minimum number of cells that each factor must have attached to it for it to be kept. If between 0 and 1, fraction. Otherwise, absolute value
            filter_up: whether to remove factors in upper layers via inter-layer attachments
        """
        if min_cells != 0:
            if min_cells < 1.0:
                min_cells = max(min_cells * self.adata.shape[0], 10)

        self.factor_lists = []
        for i, layer_name in enumerate(self.layer_names):
            if i == 0:
                keep = self.get_effective_factors(
                    thres=thres,
                    iqr_mult=iqr_mult,
                    min_cells=min_cells,
                    normalized=normalized,
                )
            else:
                assignments = np.argmax(self.pmeans[f"{layer_name}z"], axis=1)
                counts = np.array(
                    [
                        np.count_nonzero(assignments == a)
                        for a in range(self.layer_sizes[i])
                    ]
                )
                keep = np.array(range(self.layer_sizes[i]))[
                    np.where(counts >= min_cells)[0]
                ]
                if filter_up:
                    mat = self.pmeans[f"{layer_name}W"][keep]
                    assignments = []
                    for factor in self.factor_lists[i - 1]:
                        assignments.append(keep[np.argmax(mat[:, factor])])

                    keep = np.unique(
                        list(set(np.unique(assignments)).intersection(keep))
                    )

            if len(keep) == 0:
                self.logger.info(
                    f"No factors in layer {i} satisfy the filtering criterion. Please adjust the filtering parameters."
                    f"Keeping all factors for layer {i} for now."
                )
                keep = np.arange(self.layer_sizes[i])
            self.factor_lists.append(keep)

        self.make_layercolors(
            layer_cpal=self.layer_cpal, lightness_mult=self.lightness_mult
        )
        self.annotate_adata()

    def set_factor_names(self):
        self.factor_names = [
            [
                f"{self.layer_names[idx]}_{str(i)}"
                for i in range(len(self.factor_lists[idx]))
            ]
            for idx in range(self.n_layers)
        ]

    def annotate_adata(self):
        self.adata.obs["cell_scale"] = self.pmeans["cell_scale"]
        if self.n_batches == 1:
            self.adata.var["gene_scale"] = self.pmeans["gene_scale"][0]
            self.logger.info("Updated adata.var: `gene_scale`")
        else:
            for batch_idx in range(self.n_batches):
                name = f"gene_scale_{batch_idx}"
                self.adata.var[name] = self.pmeans["gene_scale"][batch_idx]
                self.logger.info(f"Updated adata.var: `{name}` for batch {batch_idx}.")

        self.set_factor_names()
        ranked_genes, ranked_scores = self.get_signatures_dict(
            scores=True, sorted_scores=True
        )

        for idx in range(self.n_layers):
            layer_name = self.layer_names[idx]
            self.adata.obsm[f"X_{layer_name}"] = np.array(
                self.pmeans[f"{layer_name}z"][:, self.factor_lists[idx]]
            )
            assignments = np.argmax(self.adata.obsm[f"X_{layer_name}"], axis=1)
            self.adata.obs[f"{layer_name}"] = [
                self.factor_names[idx][a] for a in assignments
            ]
            # Make sure factor colors in UMAP respect the palette
            factor_colors = [
                matplotlib.colors.to_hex(self.layer_colorpalettes[idx][i])
                for i in range(len(self.factor_lists[idx]))
            ]

            if layer_name == "marker":
                self.adata.obs[f"{layer_name}"] = pd.Categorical(
                    self.adata.obs[f"{layer_name}"]
                )
                sorted_factors = self.adata.obs_vector(f"{layer_name}")
                sorted_colors = []
                for fac in sorted_factors.categories:
                    pos = np.where(np.array(self.factor_names[idx]) == fac)[0][0]
                    col = factor_colors[pos]
                    sorted_colors.append(col)
                    self.adata.uns[f"{layer_name}_colors"] = sorted_colors
            else:
                self.adata.uns[f"{layer_name}_colors"] = factor_colors

            scores_names = [f + "_score" for f in self.factor_names[idx]]
            df = pd.DataFrame(
                self.adata.obsm[f"X_{layer_name}"],
                index=self.adata.obs.index,
                columns=scores_names,
            )
            if scores_names[0] not in self.adata.obs.columns:
                self.adata.obs = pd.concat([self.adata.obs, df], axis=1)
            else:
                self.adata.obs = self.adata.obs.drop(
                    columns=[col for col in self.adata.obs.columns if "score" in col]
                )
                self.adata.obs = pd.concat([self.adata.obs, df], axis=1)

            self.logger.info(
                f"Updated adata.obs with layer {idx}: `{layer_name}` and `{layer_name}_score` for all factors in layer {idx}"
            )
            self.logger.info(f"Updated adata.obsm with layer {idx}: `X_{layer_name}`")

            factor_names = self.factor_names[idx]
            names = np.array(
                [
                    tuple(
                        [ranked_genes[factor_name][i] for factor_name in factor_names]
                    )
                    for i in range(self.n_genes)
                ],
                dtype=[(factor_name, "O") for factor_name in factor_names],
            ).view(np.recarray)

            scores = np.array(
                [
                    tuple(
                        [ranked_scores[factor_name][i] for factor_name in factor_names]
                    )
                    for i in range(self.n_genes)
                ],
                dtype=[(factor_name, "<f4") for factor_name in factor_names],
            ).view(np.recarray)

            pvals = np.array(
                [
                    tuple([1.0 for factor_name in factor_names])
                    for i in range(self.n_genes)
                ],
                dtype=[(factor_name, "<f4") for factor_name in factor_names],
            ).view(np.recarray)

            pvals_adj = np.array(
                [
                    tuple([1.0 for factor_name in factor_names])
                    for i in range(self.n_genes)
                ],
                dtype=[(factor_name, "<f4") for factor_name in factor_names],
            ).view(np.recarray)

            logfoldchanges = np.array(
                [
                    tuple([0.0 for factor_name in factor_names])
                    for i in range(self.n_genes)
                ],
                dtype=[(factor_name, "<f4") for factor_name in factor_names],
            ).view(np.recarray)

            self.adata.uns[f"{layer_name}_signatures"] = {
                "params": {
                    "reference": "rest",
                    "method": "scDEF",
                    "groupby": f"{layer_name}",
                },
                "names": names,
                "scores": scores,
                "pvals": pvals,
                "pvals_adj": pvals_adj,
                "logfoldchanges": logfoldchanges,
            }

            self.logger.info(
                f"Updated adata.uns with layer {idx} signatures: `{layer_name}_signatures`."
            )

    def normalize_cellscores(self):
        for idx in range(self.n_layers):
            layer_name = self.layer_names[idx]
            self.adata.obsm[f"X_{layer_name}_probs"] = (
                self.adata.obsm[f"X_{layer_name}"]
                / np.sum(self.adata.obsm[f"X_{layer_name}"], axis=1)[:, None]
            )
            scores_names = [f + "_prob" for f in self.factor_names[idx]]
            df = pd.DataFrame(
                self.adata.obsm[f"X_{layer_name}_probs"],
                index=self.adata.obs.index,
                columns=scores_names,
            )
            if scores_names[0] not in self.adata.obs.columns:
                self.adata.obs = pd.concat([self.adata.obs, df], axis=1)
            else:
                self.adata.obs = self.adata.obs.drop(
                    columns=[col for col in self.adata.obs.columns if "prob" in col]
                )
                self.adata.obs = pd.concat([self.adata.obs, df], axis=1)

    def get_annotations(self, marker_reference, gene_rankings=None):
        if gene_rankings is None:
            gene_rankings = self.get_rankings(layer_idx=0)

        annotations = []
        keys = list(marker_reference.keys())
        for rank in gene_rankings:
            # Get annotation in marker_reference that contains the highest score
            # score is length of intersection over length of union
            scores = np.array(
                [
                    len(set(rank).intersection(set(marker_reference[a])))
                    / len(set(rank).union(set(marker_reference[a])))
                    for a in keys
                ]
            )
            sorted_ann_idx = np.argsort(scores)[::-1]
            sorted_scores = scores[sorted_ann_idx]

            # Get only annotations for which score is not zero
            sorted_ann_idx = sorted_ann_idx[np.where(sorted_scores > 0)[0]]

            ann = np.array(keys)[sorted_ann_idx]
            ann = np.array(
                [f"{a} ({sorted_scores[i]:.4f})" for i, a in enumerate(ann)]
            ).tolist()
            annotations.append(ann)
        return annotations

    def get_rankings(
        self,
        layer_idx=0,
        top_genes=None,
        genes=True,
        return_scores=False,
        sorted_scores=True,
    ):
        if top_genes is None:
            top_genes = len(self.adata.var_names)

        term_names = np.array(self.adata.var_names)
        term_scores = self.pmeans[f"{self.layer_names[0]}W"][self.factor_lists[0]]
        n_factors = len(self.factor_lists[layer_idx])

        if layer_idx > 0:
            if genes:
                term_scores = self.pmeans[f"{self.layer_names[layer_idx]}W"][
                    self.factor_lists[layer_idx]
                ][:, self.factor_lists[layer_idx - 1]]
                for layer in range(layer_idx - 1, 0, -1):
                    lower_mat = self.pmeans[f"{self.layer_names[layer]}W"][
                        self.factor_lists[layer]
                    ][:, self.factor_lists[layer - 1]]
                    term_scores = term_scores.dot(lower_mat)
                term_scores = term_scores.dot(
                    self.pmeans[f"{self.layer_names[0]}W"][self.factor_lists[0]]
                )
            else:
                n_factors_below = len(self.factor_lists[layer_idx - 1])
                term_names = np.arange(n_factors_below).astype(str)
                term_scores = self.pmeans[f"{self.layer_names[layer_idx]}W"][
                    self.factor_lists[layer_idx]
                ][:, self.factor_lists[layer_idx - 1]]

        top_terms = []
        top_scores = []
        for k in range(n_factors):
            top_terms_idx = (term_scores[k, :]).argsort()[::-1]
            top_terms_idx = top_terms_idx[:top_genes]
            top_terms_list = term_names[top_terms_idx].tolist()
            top_terms.append(top_terms_list)
            if sorted_scores:
                sorted_term_scores_k = term_scores[k, :][top_terms_idx]
                top_scores_list = sorted_term_scores_k.tolist()
            else:
                top_scores_list = term_scores[k, :].tolist()
            top_scores.append(top_scores_list)

        if return_scores:
            return top_terms, top_scores
        return top_terms

    def get_signature_sample(
        self, rng, factor_idx, layer_idx, top_genes=10, return_scores=False
    ):
        term_names = np.array(self.adata.var_names)

        term_scores_shape = self.global_params[2 + 0][0][self.factor_lists[0]]
        term_scores_rate = np.exp(self.global_params[2 + 0][1][self.factor_lists[0]])
        term_scores_sample = lognormal_sample(rng, term_scores_shape, term_scores_rate)

        if layer_idx > 0:
            term_scores_shape = self.global_params[2 + layer_idx][0][
                self.factor_lists[layer_idx]
            ][:, self.factor_lists[layer_idx - 1]]

            term_scores_rate = np.exp(
                self.global_params[2 + layer_idx][1][self.factor_lists[layer_idx]][
                    :, self.factor_lists[layer_idx - 1]
                ]
            )
            term_scores_sample = lognormal_sample(
                rng, term_scores_shape, term_scores_rate
            )

            for layer in range(layer_idx - 1, 0, -1):
                lower_mat_shape = self.global_params[2 + layer][0][
                    self.factor_lists[layer]
                ][:, self.factor_lists[layer - 1]]

                lower_mat_rate = np.exp(
                    self.global_params[2 + layer][1][self.factor_lists[layer]][
                        :, self.factor_lists[layer - 1]
                    ]
                )
                lower_mat_sample = lognormal_sample(
                    rng, lower_mat_shape, lower_mat_rate
                )
                term_scores_sample = term_scores_sample.dot(lower_mat_sample)

            lower_term_scores_shape = self.global_params[2 + 0][0][self.factor_lists[0]]

            lower_term_scores_rate = np.exp(
                self.global_params[2 + 0][1][self.factor_lists[0]]
            )
            lower_term_scores_sample = lognormal_sample(
                rng, lower_term_scores_shape, lower_term_scores_rate
            )

            term_scores_sample = term_scores_sample.dot(lower_term_scores_sample)

        top_terms_idx = (term_scores_sample[factor_idx, :]).argsort()[::-1][:top_genes]
        top_terms = term_names[top_terms_idx].tolist()
        top_scores = term_scores_sample[factor_idx, :].tolist()

        if return_scores:
            return top_terms, top_scores
        return top_terms

    def get_signature_confidence(
        self,
        factor_idx,
        layer_idx,
        mc_samples=100,
        top_genes=10,
        pairwise=False,
    ):
        signatures = []
        for i in range(mc_samples):
            rng = random.PRNGKey(i)
            signature_sample = self.get_signature_sample(
                rng,
                factor_idx=factor_idx,
                layer_idx=layer_idx,
                top_genes=top_genes,
            )
            signatures.append(signature_sample)

        if pairwise:
            jaccs = np.zeros((mc_samples, mc_samples))
            for i in range(mc_samples):
                for j in range(mc_samples):
                    jaccs[i, j] = score_utils.jaccard_similarity(
                        [signatures[i], signatures[j]]
                    )
            return np.mean(jaccs)
        else:
            return score_utils.jaccard_similarity(signatures)

    def get_sizes_dict(self):
        sizes_dict = {}
        for layer_idx in range(self.n_layers):
            layer_sizes = self.adata.obs[
                f"{self.layer_names[layer_idx]}"
            ].value_counts()
            for factor_idx, factor_name in enumerate(self.factor_names[layer_idx]):
                if factor_name in layer_sizes.keys():
                    sizes_dict[factor_name] = layer_sizes[factor_name]
                else:
                    sizes_dict[factor_name] = 0
        return sizes_dict

    def get_signatures_dict(
        self, top_genes=None, scores=False, sorted_scores=False, layer_normalize=False
    ):
        signatures_dict = {}
        scores_dict = {}
        for layer_idx in range(self.n_layers):
            layer_signatures, layer_scores = self.get_rankings(
                layer_idx=layer_idx,
                top_genes=top_genes,
                return_scores=True,
                sorted_scores=sorted_scores,
            )
            for factor_idx, factor_name in enumerate(self.factor_names[layer_idx]):
                val = np.array(layer_scores[factor_idx])
                if layer_normalize:
                    val = val - np.min(val)
                    val = val / np.max(val)
                signatures_dict[factor_name] = layer_signatures[factor_idx]
                scores_dict[factor_name] = val

        if scores:
            return signatures_dict, scores_dict
        return signatures_dict

    def get_summary(self, top_genes=10, reindex=True):
        tokeep = self.factor_lists[0]
        n_factors_eff = len(tokeep)
        genes, scores = self.get_rankings(return_scores=True)

        summary = f"Found {n_factors_eff} factors grouped in the following way:\n"

        # Group the factors
        assignments = []
        for i, factor in enumerate(tokeep):
            assignments.append(
                np.argmax(self.pmeans[f"{self.layer_names[1]}W"][:, factor])
            )
        assignments = np.array(assignments)
        factor_order = np.argsort(np.array(assignments))

        for group in np.unique(assignments):
            factors = tokeep[np.where(assignments == group)[0]]
            if len(factors) > 0:
                summary += f"Group {group}:\n"
            for i, factor in enumerate(factors):
                summary += f"    Factor {i}: "
                summary += ", ".join(
                    [
                        f"{genes[factor][j]} ({scores[factor][j]:.3f})"
                        for j in range(top_genes)
                    ]
                )
                summary += "\n"
            summary += "\n"

        self.logger.info(summary)

        return summary

    def get_enrichments(self, libs=["KEGG_2019_Human"], gene_rankings=None):
        import gseapy as gp

        if gene_rankings is None:
            gene_rankings = self.get_rankings(layer_idx=0)

        enrichments = []
        for rank in tqdm(gene_rankings):
            enr = gp.enrichr(
                gene_list=rank,
                gene_sets=libs,
                organism="Human",
                outdir="test/enrichr",
                cutoff=0.05,
            )
            enrichments.append(enr)
        return enrichments

    def get_layer_factor_orders(self):
        layer_factor_orders = []
        for layer_idx in np.arange(0, self.n_layers)[::-1]:  # Go top down
            factors = self.factor_lists[layer_idx]
            n_factors = len(factors)
            if layer_idx < self.n_layers - 1:
                # Assign factors to upper factors to set the plotting order
                mat = self.pmeans[f"{self.layer_names[layer_idx+1]}W"][
                    self.factor_lists[layer_idx + 1]
                ][:, self.factor_lists[layer_idx]]
                normalized_factor_weights = mat / np.sum(mat, axis=1).reshape(-1, 1)
                assignments = []
                for factor_idx in range(n_factors):
                    assignments.append(
                        np.argmax(normalized_factor_weights[:, factor_idx])
                    )
                assignments = np.array(assignments)

                factor_order = []
                for upper_factor_idx in layer_factor_orders[-1]:
                    factor_order.append(np.where(assignments == upper_factor_idx)[0])
                factor_order = np.concatenate(factor_order).astype(int)
                layer_factor_orders.append(factor_order)
            else:
                layer_factor_orders.append(np.arange(n_factors))
        layer_factor_orders = layer_factor_orders[::-1]
        return layer_factor_orders

    def attach_factors_to_obs(self, obs_key):
        attachments = []
        for layer, layer_name in enumerate(self.layer_names):
            layer_attachments = []
            for factor_idx in range(len(self.factor_lists[layer])):
                factor_name = f"{self.factor_names[layer][int(factor_idx)]}"
                # cells attached to this factor
                cells = np.where(self.adata.obs[f"{layer_name}"] == factor_name)[0]
                if len(cells) > 0:
                    # cells in this factor that belong to each obs
                    prevs = [
                        np.count_nonzero(self.adata.obs[obs_key][cells] == b)
                        / len(np.where(self.adata.obs[obs_key] == b)[0])
                        for b in self.adata.obs[obs_key].cat.categories
                    ]
                    obs_idx = np.argmax(prevs)  # obs attachment
                    layer_attachments.append(
                        self.adata.obs[obs_key].cat.categories[obs_idx]
                    )
            attachments.append(layer_attachments)
        return attachments

    def get_hierarchy(
        self, simplified: Optional[bool] = True
    ) -> Mapping[str, Sequence[str]]:
        """Get a dictionary containing the polytree contained in the scDEF graph.

        Args:
            simplified: whether to collapse single-child nodes

        Returns:
            hierarchy: the dictionary containing the hierarchy
        """
        hierarchy = dict()
        for layer_idx in range(0, self.n_layers - 1):
            factors = self.factor_lists[layer_idx]
            n_factors = len(factors)
            if layer_idx < self.n_layers - 1:
                # Assign factors to upper factors to set the plotting order
                mat = self.pmeans[f"{self.layer_names[layer_idx+1]}W"][
                    self.factor_lists[layer_idx + 1]
                ][:, self.factor_lists[layer_idx]]
                normalized_factor_weights = mat / np.sum(mat, axis=1).reshape(-1, 1)
                assignments = []
                for factor_idx in range(n_factors):
                    assignments.append(
                        np.argmax(normalized_factor_weights[:, factor_idx])
                    )
                assignments = np.array(assignments)

                for upper_layer_factor in range(len(self.factor_lists[layer_idx + 1])):
                    upper_layer_factor_name = self.factor_names[layer_idx + 1][
                        upper_layer_factor
                    ]
                    assigned_lower = np.array(self.factor_names[layer_idx])[
                        np.where(assignments == upper_layer_factor)[0]
                    ].tolist()
                    hierarchy[upper_layer_factor_name] = assigned_lower

        if simplified:
            layer_sizes = [len(self.factor_names[idx]) for idx in range(self.n_layers)]
            hierarchy = hierarchy_utils.simplify_hierarchy(
                hierarchy, self.layer_names, layer_sizes, factor_names=self.factor_names
            )

        return hierarchy

    def compute_weight(self, upper_factor_name, lower_factor_name):
        """Compute the weight between two factors across any number of layers."""
        upper_factor_idx = -1
        upper_factor_layer_idx = -1
        for layer_idx in range(self.n_layers):
            layer_factor_names = np.array(self.factor_names[layer_idx])
            if upper_factor_name in layer_factor_names:
                upper_factor_idx = np.where(upper_factor_name == layer_factor_names)[0][
                    0
                ]
                upper_factor_layer_idx = layer_idx
                break

        assert upper_factor_idx != -1

        lower_factor_idx = -1
        lower_factor_layer_idx = -1
        for layer_idx in range(self.n_layers):
            layer_factor_names = np.array(self.factor_names[layer_idx])
            if lower_factor_name in layer_factor_names:
                lower_factor_idx = np.where(lower_factor_name == layer_factor_names)[0][
                    0
                ]
                lower_factor_layer_idx = layer_idx
                break

        assert lower_factor_idx != -1

        upper_layer_name = self.layer_names[upper_factor_layer_idx]
        mat = self.pmeans[f"{upper_layer_name}W"][
            self.factor_lists[upper_factor_layer_idx]
        ][:, self.factor_lists[upper_factor_layer_idx - 1]]
        for layer_idx in range(upper_factor_layer_idx - 1, lower_factor_layer_idx, -1):
            layer_name = self.layer_names[layer_idx]
            lower_mat = self.pmeans[f"{layer_name}W"][self.factor_lists[layer_idx]][
                :, self.factor_lists[layer_idx - 1]
            ]
            mat = mat.dot(lower_mat)

        return mat[upper_factor_idx][lower_factor_idx]

    def compute_factor_obs_association_score(
        self, layer_idx, factor_name, obs_key, obs_val
    ):
        layer_name = self.layer_names[layer_idx]

        # Cells attached to factor
        adata_cells_in_factor = self.adata[
            np.where(self.adata.obs[f"{layer_name}"] == factor_name)[0]
        ]

        # Cells from obs_val
        adata_cells_from_obs = self.adata[
            np.where(self.adata.obs[obs_key] == obs_val)[0]
        ]

        cells_from_obs = float(adata_cells_from_obs.shape[0])

        # Number of cells from obs_val that are not in factor
        cells_not_in_factor_from_obs = float(
            np.count_nonzero(adata_cells_from_obs.obs[f"{layer_name}"] != factor_name)
        )

        # Number of cells in factor that are obs_val
        cells_in_factor_from_obs = float(
            np.count_nonzero(adata_cells_in_factor.obs[obs_key] == obs_val)
        )

        # Number of cells in factor that are not obs_val
        cells_in_factor_not_from_obs = float(
            np.count_nonzero(adata_cells_in_factor.obs[obs_key] != obs_val)
        )

        return score_utils.compute_fscore(
            cells_in_factor_from_obs,
            cells_in_factor_not_from_obs,
            cells_not_in_factor_from_obs,
        )

    def compute_factor_obs_assignment_fracs(
        self, layer_idx, factor_name, obs_key, obs_val
    ):
        layer_name = self.layer_names[layer_idx]

        # Cells attached to factor
        adata_cells_in_factor = self.adata[
            np.where(self.adata.obs[f"{layer_name}"] == factor_name)[0]
        ]

        # Cells in factor
        cells_in_factor = float(adata_cells_in_factor.shape[0])

        # Cells from factor in obs
        cells_in_factor_from_obs = float(
            np.count_nonzero(adata_cells_in_factor.obs[obs_key] == obs_val)
        )

        score = 0.0
        if cells_in_factor != 0:
            score = cells_in_factor_from_obs / cells_in_factor

        return score

    def compute_factor_obs_weight_score(self, layer_idx, factor_name, obs_key, obs_val):
        layer_name = self.layer_names[layer_idx]

        # Cells from obs_val
        adata_cells_from_obs = self.adata[
            np.where(self.adata.obs[obs_key] == obs_val)[0]
        ]
        adata_cells_not_from_obs = self.adata[
            np.where(self.adata.obs[obs_key] != obs_val)[0]
        ]

        # Weight of cells from obs in factor
        avg_in = np.mean(adata_cells_from_obs.obs[f"{factor_name}_score"])

        # Weight of cells not from obs in factor
        avg_out = np.mean(adata_cells_not_from_obs.obs[f"{factor_name}_score"])

        score = avg_in / np.sum(avg_in + avg_out)

        return score
