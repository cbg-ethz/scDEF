from scdef.utils import score_utils, hierarchy_utils, color_utils
from scdef.utils.jax_utils import *

from jax import jit, vmap
import optax
from jax import random, value_and_grad
import jax.numpy as jnp
from jax.scipy.stats import poisson
import jax

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import seaborn as sns
from graphviz import Graph
from tqdm import tqdm
import time

import logging

logging.basicConfig()

import scipy
import numpy as np
import pandas as pd
from anndata import AnnData
import scanpy as sc
import decoupler

from scipy.cluster.hierarchy import ward, leaves_list
from scipy.spatial.distance import pdist

from typing import Optional, Union, Sequence, Mapping, Literal

import tensorflow_probability.substrates.jax.distributions as tfd


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
        layer_shapes: prior parameters for the z shape to use in each scDEF layer
        brd_strength: BRD prior concentration parameter
        brd_mean: BRD prior mean parameter
        use_brd: whether to use the BRD prior for factor relevance estimation
        cell_scale_shape: concentration level in the cell scale prior
        gene_scale_shape: concentration level in the gene scale prior
        factor_shapes: prior parameters for the W shape to use in each scDEF layer
        factor_rates: prior parameters for the W rate to use in each scDEF layer
        layer_diagonals: prior diagonal strengths for the W parameters in each scDEF layer
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
        layer_sizes: Optional[list] = [100, 60, 30, 10, 1],
        layer_shapes=[1.0, 1.0, 1.0, 1.0, 1.0],
        batch_key: Optional[str] = None,
        seed: Optional[int] = 1,
        logginglevel: Optional[int] = logging.INFO,
        layer_concentration: Optional[float] = 1.0,
        brd_strength: Optional[float] = 1.0,
        brd_mean: Optional[float] = 1.0,
        use_brd: Optional[bool] = True,
        cell_scale_shape: Optional[float] = 1.0,
        gene_scale_shape: Optional[float] = 1.0,
        factor_shapes: Optional[list] = None,
        factor_rates: Optional[list] = None,
        layer_diagonals: Optional[list] = None,
        batch_cpal: Optional[str] = "Dark2",
        layer_cpal: Optional[str] = "tab10",
        lightness_mult: Optional[float] = 0.15,
        nmf_init: Optional[bool] = True,
    ):
        self.n_cells, self.n_genes = adata.shape

        self.logger = logging.getLogger("scDEF")
        self.logger.setLevel(logginglevel)

        self.layer_sizes = [int(x) for x in layer_sizes]
        self.n_layers = len(self.layer_sizes)
        self.use_brd = use_brd

        self.factor_lists = [np.arange(size) for size in self.layer_sizes]

        self.n_batches = 1
        self.batches = [""]

        self.seed = seed
        self.layer_names = [f"L{i}" for i in range(self.n_layers)]
        self.batch_cpal = batch_cpal
        self.layer_cpal = layer_cpal
        self.lightness_mult = lightness_mult
        self.make_layercolors(layer_cpal=self.layer_cpal, lightness_mult=lightness_mult)

        self.batch_key = batch_key

        self.load_adata(adata, layer=counts_layer, batch_key=batch_key)

        if factor_shapes is None:
            if self.use_brd:
                factor_shapes = [1.0 / brd_mean] + [1.0] * (self.n_layers - 1)
            else:
                factor_shapes = [0.3] + [1.0] * (self.n_layers - 1)
        elif isinstance(factor_shapes, float) or isinstance(factor_shapes, int):
            factor_shapes = [float(factor_shapes)] * self.n_layers
        if len(factor_shapes) != self.n_layers:
            raise ValueError("factor_shapes list must be of size scDEF.n_layers")
        self.factor_shapes = factor_shapes

        if factor_rates is None:
            if self.use_brd:
                factor_rates = [1.0 / brd_mean] + [1.0] * (self.n_layers - 1)
            else:
                factor_rates = [0.3] + [1.0] * (self.n_layers - 1)
        elif isinstance(factor_rates, float) or isinstance(factor_rates, int):
            factor_rates = [float(factor_rates)] * self.n_layers
        if len(factor_rates) != self.n_layers:
            raise ValueError("factor_rates list must be of size scDEF.n_layers")
        self.factor_rates = factor_rates

        if layer_diagonals is None:
            layer_diagonals = [1.0] * self.n_layers
        elif isinstance(layer_diagonals, float) or isinstance(layer_diagonals, int):
            layer_diagonals = [float(layer_diagonals)] * self.n_layers
        if len(layer_diagonals) != self.n_layers:
            raise ValueError("layer_diagonals list must be of size scDEF.n_layers")
        self.layer_diagonals = layer_diagonals

        self.layer_shapes = layer_shapes
        self.layer_concentration = layer_concentration
        self.brd = brd_strength
        self.brd_mean = brd_mean
        self.cell_scale_shape = cell_scale_shape
        self.gene_scale_shape = gene_scale_shape

        self.w_priors = []
        for idx in range(self.n_layers):
            prior_shapes = self.factor_shapes[idx]
            prior_rates = self.factor_rates[idx]

            if idx > 0:
                prior_shapes = (
                    np.ones((self.layer_sizes[idx], self.layer_sizes[idx - 1]))
                    * self.factor_shapes[idx]
                    * 1.0
                    / self.layer_diagonals[idx]
                )
                prior_rates = (
                    np.ones((self.layer_sizes[idx], self.layer_sizes[idx - 1]))
                    * self.factor_rates[idx]
                    * self.layer_diagonals[idx]
                )
                for l in range(self.layer_sizes[idx]):
                    prior_shapes[l, l] = (
                        self.factor_shapes[idx] * self.layer_diagonals[idx]
                    )
                    prior_rates[l, l] = (
                        self.factor_rates[idx] * self.layer_diagonals[idx]
                    )
                prior_shapes = jnp.clip(jnp.array(prior_shapes), 1e-12, 1e12)
                prior_rates = jnp.clip(jnp.array(prior_rates), 1e-12, 1e12)

            self.w_priors.append([prior_shapes, prior_rates])

        self.init_var_params(nmf_init=nmf_init)
        self.set_posterior_means()

        # Keep these as class attributes to avoid re-compiling the computation graph everytime we run an optimization loop
        self.opt_init = None
        self.get_params = None
        self.opt_update = None
        self.elbos = []
        self.step_sizes = []

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
            "\n\t"
            + "Layer factor shape parameters: "
            + ", ".join([str(shape) for shape in self.factor_shapes])
        )
        out += (
            "\n\t"
            + "Layer factor rate parameters: "
            + ", ".join([str(rate) for rate in self.factor_rates])
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
                    self.layer_colorpalettes[layer_idx][
                        factor_idx
                    ] = color_utils.adjust_lightness(
                        col, amount=1.0 + lightness_mult * layer_idx
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

        self.batch_indices_onehot = np.ones((self.adata.shape[0], 1))
        self.batch_lib_sizes = np.sum(self.X, axis=1)
        self.batch_lib_ratio = (
            np.ones((self.X.shape[0], 1))
            * np.mean(self.batch_lib_sizes)
            / np.var(self.batch_lib_sizes)
        )
        gene_size = np.mean(self.X, axis=0)
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

    def init_var_params(self, minval=0.5, maxval=1.5, nmf_init=False):
        rngs = random.split(random.PRNGKey(self.seed), 6 + 2 * 2 * self.n_layers)

        m = 1.0 / jnp.maximum(self.batch_lib_ratio, 1e-1)
        v = 1  # .*(m)**2 / self.cell_scale_shape
        self.local_params = [
            jnp.array(
                (
                    jnp.log(
                        random.uniform(
                            rngs[0],
                            minval=1.0,
                            maxval=1.0,
                            shape=[self.n_cells, 1],
                        )
                        * 1.0
                    )
                    + jnp.log(m**2 / jnp.sqrt(m**2 + v)),  # cell_scales
                    jnp.log(
                        random.uniform(
                            rngs[1],
                            minval=1.0,
                            maxval=1.0,
                            shape=[self.n_cells, 1],
                        )
                        * jnp.sqrt(jnp.log(1 + v / (m**2)))
                    )
                    * 1.0,  # jnp.clip(10.0 * self.batch_lib_ratio, 1e-8, 1e2),
                )
            ),
        ]
        # m = 1.
        # v = .1
        gene_sizes = 1.0 / np.sum(self.X, axis=0)
        standardized = gene_sizes / np.std(gene_sizes)
        centered = standardized - np.mean(standardized)
        recentered = centered + 1
        m = recentered[None, :]
        m = 1.0 / jnp.maximum(self.gene_ratio, 1e-1)
        v = 1.0  # e-1 #/ (self.gene_ratio)

        m_brd = jnp.clip(
            tfd.Gamma(self.brd, self.brd / (self.brd_mean / 1.0)).sample(
                seed=rngs[2], sample_shape=[self.layer_sizes[0], 1]
            ),
            1e-1,
            1e1,
        )
        v_brd = 0.1  # e-1 #/ (self.gene_ratio)

        self.global_params = [
            jnp.array(
                (
                    jnp.log(
                        random.uniform(
                            rngs[2],
                            minval=1.0,
                            maxval=1.0,
                            shape=[self.n_batches, self.n_genes],
                        )
                        * 1.0
                    )
                    + jnp.log(m**2 / jnp.sqrt(m**2 + v)),  # gene_scales
                    jnp.log(
                        random.uniform(
                            rngs[3],
                            minval=1.0,
                            maxval=1.0,
                            shape=[self.n_batches, self.n_genes],
                        )
                        * jnp.sqrt(
                            jnp.log(1 + v / (m**2))
                        )  # jnp.clip(10.0 * self.gene_ratio, 1e-8, 1e2)
                    ),
                )
            ),
            jnp.array(
                (
                    jnp.log(
                        random.uniform(
                            rngs[4],
                            minval=1.0,
                            maxval=1.0,
                            shape=[self.layer_sizes[0], 1],
                        )
                    )
                    + jnp.log(m_brd**2 / jnp.sqrt(m_brd**2 + v_brd)),
                    jnp.log(
                        random.uniform(
                            rngs[5],
                            minval=1.0,
                            maxval=1.0,
                            shape=[self.layer_sizes[0], 1],
                        )
                        * jnp.sqrt(jnp.log(1 + v_brd / (m_brd**2)))  # * 1./self.brd
                    ),
                )
            ),
        ]

        if nmf_init:
            self.logger.info(f"Initializing the factor weights with KL-NMF.")
            init_z, init_w = self.get_nmf_init()

        z_shapes = []
        z_rates = []
        rng_cnt = 6
        for layer_idx in range(
            self.n_layers
        ):  # we go from layer 0 (bottom) to layer L (top)
            # Init z
            if layer_idx == 0:
                m = jnp.clip(
                    tfd.Gamma(10.0, 10.0).sample(
                        seed=rngs[rng_cnt],
                        sample_shape=[self.n_cells, self.layer_sizes[layer_idx]],
                    ),
                    1e-1,
                    1e1,
                )  # self.layer_shapes[layer_idx]/self.layer_rates[layer_idx]
                v = 1.0  # this is important! #self.layer_shapes[layer_idx]/(self.layer_rates[layer_idx]**2)
            else:
                m = jnp.clip(
                    tfd.Gamma(10.0, 10.0).sample(
                        seed=rngs[rng_cnt],
                        sample_shape=[self.n_cells, self.layer_sizes[layer_idx]],
                    ),
                    1e-1,
                    1e1,
                )
                v = 1.0  # this is important! #self.layer_shapes[layer_idx]/(self.layer_rates[layer_idx]**2)
                # if nmf_init:
                #     m = jnp.clip(init_z[layer_idx], 1e-1, 1e1)#jnp.maximum(init_z[layer_idx] - np.mean(init_z[layer_idx]) + 1., 1.)
            z_shape = jnp.log(
                random.uniform(
                    rngs[rng_cnt],
                    minval=1.0,
                    maxval=1.0,
                    shape=[self.n_cells, self.layer_sizes[layer_idx]],
                )
                * 1.0  # self.layer_shapes[layer_idx]
            ) + jnp.log(m**2 / jnp.sqrt(m**2 + v))
            z_shapes.append(z_shape)
            rng_cnt += 1
            z_rate = jnp.log(
                random.uniform(
                    rngs[rng_cnt],
                    minval=1.0,
                    maxval=1.0,
                    shape=[self.n_cells, self.layer_sizes[layer_idx]],
                )
                * jnp.sqrt(jnp.log(1 + v / (m**2)))
            )
            z_rates.append(z_rate)
            rng_cnt += 1

        self.local_params.append(jnp.array((jnp.hstack(z_shapes), jnp.hstack(z_rates))))

        # for layer_idx in range(
        #     1, self.n_layers
        # ):  # the w don't have a shared axis across all, so we can't vectorize
        #     # Init w
        #     layer_size = self.layer_sizes[layer_idx]

        #     f_shape = jnp.log(
        #         random.uniform(
        #             rngs[rng_cnt],
        #             minval=1.0,
        #             maxval=1.0,
        #             shape=[layer_size, 1],
        #         )
        #     * 1.#jnp.clip(self.w_priors[layer_idx][0], 1e-2, 1e2)
        #     )
        #     rng_cnt += 1

        #     f_rate = jnp.log(
        #         random.uniform(
        #             rngs[rng_cnt],
        #             minval=1.,
        #             maxval=1.,
        #             shape=[layer_size, 1],
        #         )
        #         * 1. # jnp.clip(self.w_priors[layer_idx][1], 1e-2, 1e2)
        #     )
        #     rng_cnt += 1

        #     self.local_params.append(jnp.array((f_shape, f_rate)))

        for layer_idx in range(
            self.n_layers
        ):  # the w don't have a shared axis across all, so we can't vectorize
            # Init w
            in_layer = self.layer_sizes[layer_idx]
            if layer_idx == 0:
                out_layer = self.n_genes
            else:
                out_layer = self.layer_sizes[layer_idx - 1]

            if layer_idx == 0 and self.use_brd:
                fscale_samples = lognormal_sample(
                    rngs[rng_cnt],
                    self.global_params[1][0],
                    jnp.exp(self.global_params[1][1]),
                )
                m = jnp.clip(
                    tfd.Gamma(1.0, 1.0).sample(
                        seed=rngs[rng_cnt], sample_shape=[in_layer, out_layer]
                    ),
                    1e-1,
                    1e1,
                )
                v = 1.0  # fscale_samples
                if nmf_init:
                    m = jnp.clip(
                        tfd.Gamma(1.0, 1.0 / init_w[layer_idx]).sample(
                            seed=rngs[rng_cnt]
                        ),
                        1e-1,
                        1e1,
                    )
                    v = 1.0
                w_shape = jnp.log(
                    random.uniform(
                        rngs[rng_cnt],
                        minval=1.0,
                        maxval=1.0,
                        shape=[in_layer, out_layer],
                    )
                    * 1.0  # jnp.clip(self.w_priors[layer_idx][0], 1e-2, 1e2)
                ) + jnp.log(
                    m**2 / jnp.sqrt(m**2 + v)
                )  # jnp.log(1/jnp.sqrt(1+fscale_samples))
                rng_cnt += 1

                w_rate = jnp.log(
                    random.uniform(
                        rngs[rng_cnt],
                        minval=1.0,
                        maxval=1.0,
                        shape=[in_layer, out_layer],
                    )
                    * jnp.sqrt(
                        jnp.log(1 + v / (m**2))
                    )  # jnp.sqrt(jnp.log(1+fscale_samples))# jnp.clip(self.w_priors[layer_idx][1], 1e-2, 1e2)
                )
            else:
                if layer_idx == 0:
                    m = jnp.clip(
                        tfd.Gamma(1.0, 1.0).sample(
                            seed=rngs[rng_cnt], sample_shape=[in_layer, out_layer]
                        ),
                        1e-1,
                        1e1,
                    )  # self.factor_shapes[layer_idx]/self.factor_rates[layer_idx]
                    v = 1.0
                else:
                    m = jnp.clip(
                        tfd.Gamma(1.0, 1.0).sample(
                            seed=rngs[rng_cnt], sample_shape=[in_layer, out_layer]
                        ),
                        1e-1,
                        1e1,
                    )
                    v = 1.0
                if nmf_init:
                    m = jnp.clip(
                        tfd.Gamma(1.0, 1.0 / init_w[layer_idx]).sample(
                            seed=rngs[rng_cnt]
                        ),
                        1e-1,
                        1e1,
                    )
                    v = 1.0
                if layer_idx == self.n_layers - 1:
                    m = 1.0
                w_shape = jnp.log(
                    random.uniform(
                        rngs[rng_cnt],
                        minval=1.0,
                        maxval=1.0,
                        shape=[in_layer, out_layer],
                    )
                    * 1.0  # jnp.clip(self.w_priors[layer_idx][0], 1e-2, 1e2)
                ) + jnp.log(m**2 / jnp.sqrt(m**2 + v))
                rng_cnt += 1

                w_rate = jnp.log(
                    random.uniform(
                        rngs[rng_cnt],
                        minval=1.0,
                        maxval=1.0,
                        shape=[in_layer, out_layer],
                    )
                    * jnp.sqrt(
                        jnp.log(1 + v / (m**2))
                    )  # jnp.clip(self.w_priors[layer_idx][1], 1e-2, 1e2)
                )
            rng_cnt += 1

            self.global_params.append(jnp.array((w_shape, w_rate)))

        # Gamma(1e3, 1e3/m)
        m = 1.0 * jnp.ones([1, 1])
        v = m**2 / 1e0 * jnp.ones([1, 1])
        self.global_params.append(
            jnp.array(
                (
                    jnp.log(m**2 / jnp.sqrt(m**2 + v)),
                    jnp.log(jnp.sqrt(jnp.log(1 + v / (m**2)))),
                )
            )
        )

        m = self.layer_concentration
        v = 1.0  # m**2 / 1e0
        self.global_params.append(
            jnp.array(
                (
                    [[jnp.log(m**2 / jnp.sqrt(m**2 + v))]],
                    [[jnp.log(jnp.sqrt(jnp.log(1 + v / (m**2))))]],
                )
            )
        )

    def get_nmf_init(self, max_cells=1000):
        """Use NMF on the data to init the first layer and then recursively for the other layers"""
        init_z = []
        init_W = []
        from sklearn.decomposition import NMF

        X = self.X
        X = X * 10_000 / np.sum(X, axis=1)[:, None]
        X = np.log(X + 1)
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
                beta_loss="kullback-leibler",
                solver="mu",
            )
            z = model.fit_transform(X)
            # z = z / np.max(z, axis=1)[:,None] # center each factor at 1
            W = model.components_
            # W = W / np.max(W, axis=1)[:,None]
            # W = W - np.mean(W,axis=1)[:,None] # center each factor at 1
            # W += 1.
            z = jnp.clip(z, 1e-2, 1e2)
            W = W - np.min(W, axis=1)[:, None]
            W = jnp.clip(W, 1e-2, 1e2)
            X = z
            init_z.append(z)
            init_W.append(W)

        return init_z, init_W

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
        # s_params = local_params[2]
        gene_budget_params = global_params[0]
        fscale_params = global_params[1]

        # cell_budget_params = var_params[0]
        # gene_budget_params = var_params[1]
        # fscale_params = var_params[2]
        # z_params = var_params[3]

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
        # w will be sampled in a loop below because it cannot be vectorized

        # Compute ELBO
        global_pl = 0.0
        global_en = 0.0
        global_pl = gamma_logpdf(
            gene_budget_sample,
            self.gene_scale_shape,  # * self.gene_ratio,
            self.gene_scale_shape * self.gene_ratio,
        )
        global_en = lognormal_entropy(gene_budget_shape, gene_budget_rate)
        local_pl = 0.0
        local_en = 0.0
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

            global_pl += gamma_logpdf(_wm_sample, 1.0, 1.0)
            global_en += lognormal_entropy(wm_shape, wm_rate)

        s_shape = jnp.clip(
            global_params[2 + self.n_layers + 1][0], min_shape, max_shape
        )
        s_rate = jnp.exp(
            jnp.clip(global_params[2 + self.n_layers + 1][1], min_rate, max_rate)
        )
        _s_sample = lognormal_sample(rng, s_shape, s_rate)
        global_pl += gamma_logpdf(_s_sample, 1.0, 1.0 / self.layer_concentration)
        global_en += lognormal_entropy(s_shape, s_rate)

        z_mean = 1.0
        for idx in list(np.arange(0, self.n_layers)[::-1]):
            # idx = 0
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

            # if idx == 0:
            #     jax.debug.print("🤯 {a}, {b} 🤯", a=jnp.min(_w_shape), b=jnp.max(_w_shape))
            #     jax.debug.print("🤯🤯 {a}, {b} 🤯🤯", a=jnp.min(_w_rate), b=jnp.max(_w_rate))
            _w_sample = lognormal_sample(rng, _w_shape, _w_rate)

            if idx == 0 and self.use_brd:
                global_pl += gamma_logpdf(
                    _w_sample,
                    1.0 / fscale_samples,
                    (1.0 / fscale_samples) / _wm_sample,
                )
            else:
                # _f_shape = jnp.clip(local_params[2+idx-1][0], min_shape, max_shape)
                # _f_rate = jnp.clip(jnp.exp(local_params[2+idx-1][1]), min_rate, max_rate)
                # _f_sample = lognormal_sample(rng, _f_shape, _f_rate)
                # global_pl += gamma_logpdf(_f_sample, 1., 1.)
                # global_en += lognormal_entropy(_f_shape, _f_rate)
                if idx == 1 and self.use_brd:
                    global_pl += gamma_logpdf(
                        _w_sample,
                        self.w_priors[idx][0],
                        self.w_priors[idx][1] / fscale_samples.T,
                    )
                else:
                    global_pl += gamma_logpdf(
                        _w_sample,
                        self.w_priors[idx][0],
                        self.w_priors[idx][1],  # / _f_sample,
                    )
            global_en += lognormal_entropy(_w_shape, _w_rate)

            # z
            # _z_sample = z_samples[:, start:end]
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

            shape_param = self.layer_shapes[idx]
            rate_param = self.layer_shapes[idx] / 1.0
            if idx < self.n_layers - 1:
                shape_param = jax.lax.cond(
                    stop_gradients[idx],
                    lambda: 3.0,
                    lambda: shape_param,
                )
                rate_param = jax.lax.cond(
                    stop_gradients[idx],
                    lambda: 3.0,
                    lambda: rate_param,
                )
            # if idx > 0:
            #     rate_param = 1./_f_sample.T
            # else:
            #     rate_param = 1./fscale_samples.T
            # local_pl += gamma_logpdf(_z_sample, self.layer_concentration, self.layer_concentration)
            local_pl += jax.lax.cond(
                idx == self.n_layers - 1,
                lambda: gamma_logpdf(_z_sample, _s_sample, _s_sample),
                lambda: gamma_logpdf(_z_sample, _s_sample, _s_sample / z_mean),
            )

            # local_pl += gamma_logpdf(
            #     _z_sample, _s_sample, _s_sample / (z_mean * rate_param)
            # )
            local_en += lognormal_entropy(_z_shape, _z_rate)

            z_mean = jnp.einsum("nk,kp->np", _z_sample, _w_sample)

            z_mean = jax.lax.cond(
                stop_gradients[idx], lambda: z_mean * 0.0 + 1, lambda: z_mean
            )

        # Compute log likelihood
        # mean_bottom = jnp.einsum(
        #     "nk,kg->ng", _z_sample , _w_sample
        # )
        mean_bottom = jnp.einsum(
            "nk,kg->ng", _z_sample / cell_budget_sample, _w_sample
        ) / (batch_indices_onehot.dot(gene_budget_sample))
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

        # return ll + (
        #     local_pl
        #     + local_en
        #     + (indices.shape[0] / self.X.shape[0]) * (global_pl + global_en)
        # )

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
                        loss, global_params, global_opt_state = global_update_func(
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

        return losses, local_params, global_params, local_opt_state, global_opt_state

    def learn(
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
            params, min_mu=-1e6, max_mu=1e2, min_logstd=-1e6, max_logstd=2e0
        ):
            for i in range(len(params)):
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
                return value, global_params, global_opt_state

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
        self.filter_factors()

    def set_posterior_means(self):
        cell_budget_params = self.local_params[0]
        gene_budget_params = self.global_params[0]
        fscale_params = self.global_params[1]
        z_params = self.local_params[1]
        w_params = self.global_params[2]

        self.pmeans = {
            "cell_scale": np.array(
                jnp.exp(
                    cell_budget_params[0] + 0.5 * jnp.exp(cell_budget_params[1]) ** 2
                )
                # np.exp(cell_budget_params[0]) / np.exp(cell_budget_params[1])
            ),
            "gene_scale": np.array(
                jnp.exp(
                    gene_budget_params[0] + 0.5 * jnp.exp(gene_budget_params[1]) ** 2
                )
                # np.exp(gene_budget_params[0]) / np.exp(gene_budget_params[1])
            ),
            "brd": jnp.exp(
                fscale_params[0] + 0.5 * jnp.exp(fscale_params[1]) ** 2
            ),  # np.array(np.exp(fscale_params[0]) / np.exp(fscale_params[1])),
        }

        for idx in range(self.n_layers):
            start = sum(self.layer_sizes[:idx])
            end = start + self.layer_sizes[idx]
            self.pmeans[f"{self.layer_names[idx]}z"] = jnp.exp(
                z_params[0][:, start:end]
                + 0.5 * jnp.exp(z_params[1][:, start:end]) ** 2
            )
            # self.pmeans[f"{self.layer_names[idx]}z"] = np.array(
            #     np.exp(z_params[0][:, start:end]) / np.exp(z_params[1][:, start:end])
            # )
            _w_shape = self.global_params[2 + idx][0]
            _w_rate = self.global_params[2 + idx][1]
            self.pmeans[f"{self.layer_names[idx]}W"] = jnp.exp(
                _w_shape + 0.5 * jnp.exp(_w_rate) ** 2
            )
            # self.pmeans[f"{self.layer_names[idx]}W"] = np.array(
            #     np.exp(_w_shape) / np.exp(_w_rate)
            # )

    def set_posterior_variances(self):
        cell_budget_params = self.var_params[0]
        gene_budget_params = self.var_params[1]
        fscale_params = self.var_params[2]
        z_params = self.var_params[3]
        w_params = self.var_params[4]

        self.pvars = {
            "cell_scale": np.array(
                np.exp(cell_budget_params[0]) / np.exp(cell_budget_params[1]) ** 2
            ),
            "gene_scale": np.array(
                np.exp(gene_budget_params[0]) / np.exp(gene_budget_params[1]) ** 2
            ),
            "brd": np.array(np.exp(fscale_params[0]) / np.exp(fscale_params[1]) ** 2),
        }

        for idx in range(self.n_layers):
            start = sum(self.layer_sizes[:idx])
            end = start + self.layer_sizes[idx]
            self.pvars[f"{self.layer_names[idx]}z"] = np.array(
                np.exp(z_params[0][:, start:end])
                / np.exp(z_params[1][:, start:end]) ** 2
            )
            _w_shape = self.var_params[4 + idx][0]
            _w_rate = self.var_params[4 + idx][1]
            self.pvars[f"{self.layer_names[idx]}W"] = np.array(
                np.exp(_w_shape) / np.exp(_w_rate) ** 2
            )

    def filter_factors(
        self,
        thres: Optional[float] = 1.0,
        iqr_mult: Optional[float] = 0.005,
        min_cells: Optional[float] = 0.0,
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
        ard = []
        if thres is not None:
            ard = thres
        else:
            ard = iqr_mult

        if not self.use_brd:
            ard = 0.0

        if min_cells != 0:
            if min_cells < 1.0:
                min_cells = max(min_cells * self.adata.shape[0], 10)

        self.factor_lists = []
        for i, layer_name in enumerate(self.layer_names):
            if i == 0:
                normed = (
                    self.pmeans[f"{layer_name}z"]
                    / np.sum(self.pmeans[f"{layer_name}z"], axis=1)[:, None]
                )
                assignments = np.argmax(normed, axis=1)
                counts = np.array(
                    [
                        np.count_nonzero(assignments == a)
                        for a in range(self.layer_sizes[i])
                    ]
                )
                masses = np.sum(normed, axis=0)
                keep = np.array(range(self.layer_sizes[i]))[
                    np.where(counts >= min_cells)[0]
                ]
                brd_keep = np.arange(self.layer_sizes[i])
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
                keep = np.unique(list(set(brd_keep).intersection(keep)))
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
        self.make_graph()

    def set_factor_names(self):
        self.factor_names = [
            [
                f"{self.layer_names[idx]}_{str(i)}"
                for i in range(len(self.factor_lists[idx]))
            ]
            for idx in range(self.n_layers)
        ]

    def annotate_adata(self):
        self.adata.obs["cell_scale"] = 1 / self.pmeans["cell_scale"]
        if self.n_batches == 1:
            self.adata.var["gene_scale"] = 1 / self.pmeans["gene_scale"][0]
            self.logger.info("Updated adata.var: `gene_scale`")
        else:
            for batch_idx in range(self.n_batches):
                name = f"gene_scale_{batch_idx}"
                self.adata.var[name] = 1 / self.pmeans["gene_scale"][batch_idx]
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

        term_scores_shape = self.var_params[4 + 0][0][self.factor_lists[0]]
        term_scores_rate = np.exp(self.var_params[4 + 0][1][self.factor_lists[0]])
        term_scores_sample = lognormal_sample(rng, term_scores_shape, term_scores_rate)

        if layer_idx > 0:
            term_scores_shape = self.var_params[4 + layer_idx][0][
                self.factor_lists[layer_idx]
            ][:, self.factor_lists[layer_idx - 1]]

            term_scores_rate = np.exp(
                self.var_params[4 + layer_idx][1][self.factor_lists[layer_idx]][
                    :, self.factor_lists[layer_idx - 1]
                ]
            )
            term_scores_sample = lognormal_sample(
                rng, term_scores_shape, term_scores_rate
            )

            for layer in range(layer_idx - 1, 0, -1):
                lower_mat_shape = self.var_params[4 + layer][0][
                    self.factor_lists[layer]
                ][:, self.factor_lists[layer - 1]]

                lower_mat_rate = np.exp(
                    self.var_params[4 + layer][1][self.factor_lists[layer]][
                        :, self.factor_lists[layer - 1]
                    ]
                )
                lower_mat_sample = lognormal_sample(
                    rng, lower_mat_shape, lower_mat_rate
                )
                term_scores_sample = term_scores_sample.dot(lower_mat_sample)

            lower_term_scores_shape = self.var_params[4 + 0][0][self.factor_lists[0]]

            lower_term_scores_rate = np.exp(
                self.var_params[4 + 0][1][self.factor_lists[0]]
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

    def make_graph(
        self,
        hierarchy: Optional[dict] = None,
        show_all: Optional[bool] = False,
        factor_annotations: Optional[dict] = None,
        top_factor: Optional[str] = None,
        show_signatures: Optional[bool] = True,
        enrichments: Optional[pd.DataFrame] = None,
        top_genes: Optional[int] = None,
        show_batch_counts: Optional[bool] = False,
        filled: Optional[str] = None,
        wedged: Optional[str] = None,
        color_edges: Optional[bool] = True,
        show_confidences: Optional[bool] = False,
        mc_samples: Optional[int] = 100,
        n_cells_label: Optional[bool] = False,
        n_cells: Optional[bool] = False,
        node_size_max: Optional[int] = 2.0,
        node_size_min: Optional[int] = 0.05,
        scale_level: Optional[bool] = False,
        show_label: Optional[bool] = True,
        gene_score: Optional[str] = None,
        gene_cmap: Optional[str] = "viridis",
        shell: Optional[bool] = False,
        r: Optional[float] = 2.0,
        r_decay: Optional[float] = 0.8,
        **fontsize_kwargs,
    ):
        """Make Graphviz-formatted scDEF graph.

        Args:
            hierarchy: a dictionary containing the polytree to draw instead of the whole graph
            show_all: whether to show all factors even post filtering
            factor_annotations: factor annotations to include in the node labels
            top_factor: only include factors below this factor
            show_signatures: whether to show the ranked gene signatures in the node labels
            enrichments: enrichment results from gseapy to include in the node labels
            top_genes: number of genes from each signature to be shown in the node labels
            show_batch_counts: whether to show the number of cells from each batch that attach to each factor
            filled: key from self.adata.obs to use to fill the nodes with, or dictionary of factor scores
            wedged: key from self.adata.obs to use to wedge the nodes with
            color_edges: whether to color the graph edges according to the upper factors
            show_confidences: whether to show the confidence score for each signature
            mc_samples: number of Monte Carlo samples to take from the posterior to compute signature confidences
            n_cells_label: wether to show the number of cells that attach to the factor
            n_cells: wether to scale the node sizes by the number of cells that attach to the factor
            node_size_max: maximum node size when scaled by cell numbers
            node_size_min: minimum node size when scaled by cell numbers
            scale_level: wether to scale node sizes per level instead of across all levels
            show_label: wether to show labels on nodes
            gene_score: color the nodes by the score they attribute to a gene, normalized by layer. Overrides filled and wedged
            gene_cmap: colormap to use for gene_score
            **fontsize_kwargs: keyword arguments to adjust the fontsizes according to the gene scores
        """
        if top_genes is None:
            top_genes = [10] * self.n_layers
        elif isinstance(top_genes, float):
            top_genes = [top_genes] * self.n_layers
        elif len(top_genes) != self.n_layers:
            raise IndexError("top_genes list must be of size scDEF.n_layers")

        gene_cmap = matplotlib.colormaps[gene_cmap]
        gene_scores = dict()
        if gene_score is not None:
            if gene_score not in self.adata.var_names:
                raise ValueError("gene_score must be a gene name in self.adata")
            else:
                style = "filled"
                gene_loc = np.where(self.adata.var_names == gene_score)[0][0]
                scores_dict = self.get_signatures_dict(
                    scores=True, layer_normalize=True
                )[1]
                for n in scores_dict:
                    gene_scores[n] = scores_dict[n][gene_loc]
        else:
            if filled is None:
                style = None
            elif filled == "factor":
                style = "filled"
            else:
                if isinstance(filled, str):
                    if filled not in self.adata.obs:
                        raise ValueError(
                            "filled must be factor or any `obs` in self.adata"
                        )
                else:
                    style = "filled"

            if style is None:
                if wedged is None:
                    style = None
                else:
                    if wedged not in self.adata.obs:
                        raise ValueError("wedged must be any `obs` in self.adata")
                    else:
                        style = "wedged"
            else:
                if wedged is not None:
                    self.logger.info("Filled style takes precedence over wedged")

        hierarchy_nodes = None
        if hierarchy is not None:
            if top_factor is None:
                hierarchy_nodes = hierarchy_utils.get_nodes_from_hierarchy(hierarchy)
            else:
                flattened_hierarchy = hierarchy_utils.flatten_hierarchy(hierarchy)
                hierarchy_nodes = flattened_hierarchy[top_factor] + [top_factor]

        layer_factor_orders = []
        for layer_idx in np.arange(0, self.n_layers)[::-1]:  # Go top down
            if show_all:
                factors = np.arange(self.layer_sizes[layer_idx])
            else:
                factors = self.factor_lists[layer_idx]
            n_factors = len(factors)
            if not show_all and layer_idx < self.n_layers - 1:
                # Assign factors to upper factors to set the plotting order
                if show_all:
                    mat = self.pmeans[f"{self.layer_names[layer_idx+1]}W"]
                else:
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

        def map_scores_to_fontsizes(scores, max_fontsize=11, min_fontsize=5):
            scores = scores - np.min(scores)
            scores = scores / np.max(scores)
            fontsizes = min_fontsize + scores * (max_fontsize - min_fontsize)
            return fontsizes

        g = Graph()
        ordering = "out"
        if shell:
            g.engine = "neato"
        else:
            g.engine = "dot"
        # g.node('root', style = 'invis')
        angle_dict = dict()
        for layer_idx in range(self.n_layers):
            layer_name = self.layer_names[layer_idx]
            if show_all:
                factors = np.arange(self.layer_sizes[layer_idx])
                layer_colors = []
                f_idx = 0
                for i in range(self.layer_sizes[layer_idx]):
                    if i in self.factor_lists[layer_idx]:
                        layer_colors.append(self.layer_colorpalettes[layer_idx][f_idx])
                        f_idx += 1
                    else:
                        layer_colors.append("grey")
            else:
                factors = self.factor_lists[layer_idx]
                layer_colors = self.layer_colorpalettes[layer_idx][: len(factors)]
            n_factors = len(factors)

            if show_signatures:
                gene_rankings, gene_scores = self.get_rankings(
                    layer_idx=layer_idx,
                    genes=True,
                    return_scores=True,
                )

            factor_order = layer_factor_orders[layer_idx]
            for ii, factor_idx in enumerate(factor_order):
                factor_idx = int(factor_idx)
                alpha = "FF"
                color = None
                if show_all:
                    factor_name = f"{self.layer_names[layer_idx]}{int(factor_idx)}"
                else:
                    factor_name = f"{self.factor_names[layer_idx][int(factor_idx)]}"

                if hierarchy is not None and factor_name not in hierarchy_nodes:
                    continue

                label = factor_name
                if factor_annotations is not None:
                    if factor_name in factor_annotations:
                        label = factor_annotations[factor_name]

                cells = np.where(self.adata.obs[f"{layer_name}"] == factor_name)[0]
                node_num_cells = len(cells)

                if n_cells_label:
                    label = f"{label}<br/>({node_num_cells} cells)"

                if color_edges:
                    color = matplotlib.colors.to_hex(layer_colors[factor_idx])
                fillcolor = "#FFFFFF"
                if style == "filled":
                    if filled == "factor":
                        fillcolor = matplotlib.colors.to_hex(layer_colors[factor_idx])
                    elif gene_score is not None:
                        # Color by gene score
                        rgba = gene_cmap(gene_scores[factor_name])
                        fillcolor = matplotlib.colors.rgb2hex(rgba)
                    elif isinstance(filled, str):
                        # cells attached to this factor
                        original_factor_index = self.factor_lists[layer_idx][factor_idx]
                        if len(cells) > 0:
                            # cells in this factor that belong to each obs
                            prevs = [
                                np.count_nonzero(self.adata.obs[filled][cells] == b)
                                / len(np.where(self.adata.obs[filled] == b)[0])
                                for b in self.adata.obs[filled].cat.categories
                            ]
                            obs_idx = np.argmax(prevs)  # obs attachment
                            label = f"{label}<br/>{self.adata.obs[filled].cat.categories[obs_idx]}"
                            alpha = prevs[obs_idx] / np.sum(
                                prevs
                            )  # confidence on obs_idx attachment -- should I account for the number of cells in each batch in total?
                            alpha = matplotlib.colors.rgb2hex(
                                (0, 0, 0, alpha), keep_alpha=True
                            )[-2:].upper()
                            fillcolor = self.adata.uns[f"{filled}_colors"][obs_idx]
                    elif isinstance(filled, dict):
                        # Color by dictionary of signature values with gene_cmap
                        rgba = gene_cmap(filled[factor_name])
                        fillcolor = matplotlib.colors.rgb2hex(rgba)
                    fillcolor = fillcolor + alpha
                    color = fillcolor + alpha
                elif style == "wedged":
                    # cells attached to this factor
                    original_factor_index = self.factor_lists[layer_idx][factor_idx]
                    if len(cells) > 0:
                        # cells in this factor that belong to each obs
                        # normalized by total num of cells in each obs
                        prevs = [
                            np.count_nonzero(self.adata.obs[wedged][cells] == b)
                            / len(np.where(self.adata.obs[wedged] == b)[0])
                            for b in self.adata.obs[wedged].cat.categories
                        ]
                        fracs = prevs / np.sum(prevs)
                        # make color string for pie chart
                        fillcolor = ":".join(
                            [
                                f"{self.adata.uns[f'{wedged}_colors'][obs_idx]};{frac}"
                                for obs_idx, frac in enumerate(fracs)
                            ]
                        )

                if enrichments is not None:
                    label += "<br/><br/>" + "<br/>".join(
                        [
                            enrichments[factor_idx].results["Term"].values[i]
                            + f" ({enrichments[factor_idx].results['Adjusted P-value'][i]:.3f})"
                            for i in range(top_genes[layer_idx])
                        ]
                    )
                elif show_signatures:

                    def print_signature(i):
                        factor_gene_rankings = gene_rankings[i][: top_genes[layer_idx]]
                        factor_gene_scores = gene_scores[i][: top_genes[layer_idx]]
                        fontsizes = map_scores_to_fontsizes(
                            gene_scores[i], **fontsize_kwargs
                        )[: top_genes[layer_idx]]
                        gene_labels = []
                        for j, gene in enumerate(factor_gene_rankings):
                            gene_labels.append(
                                f'<FONT POINT-SIZE="{fontsizes[j]}">{gene}</FONT>'
                            )
                        return "<br/><br/>" + "<br/>".join(gene_labels)

                    idx = factor_idx
                    if show_all:
                        if factor_idx in self.factor_lists[layer_idx]:
                            idx = np.where(
                                factor_idx == np.array(self.factor_lists[layer_idx])
                            )[0][0]
                            label += print_signature(idx)
                            if show_confidences:
                                confidence_score = self.get_signature_confidence(
                                    idx,
                                    layer_idx,
                                    top_genes=top_genes[layer_idx],
                                    mc_samples=mc_samples,
                                )
                                label += f"<br/><br/>({confidence_score:.3f})"
                    else:
                        label += print_signature(idx)
                        if show_confidences:
                            confidence_score = self.get_signature_confidence(
                                idx,
                                layer_idx,
                                top_genes=top_genes[layer_idx],
                                mc_samples=mc_samples,
                            )
                            label += f"<br/><br/>({confidence_score:.3f})"

                elif isinstance(filled, str) and filled != "factor":
                    label += "<br/><br/>" + ""

                label = "<" + label + ">"
                size = node_size_min
                fixedsize = "false"
                if n_cells:
                    max_cells = self.n_cells
                    if scale_level:
                        max_cells = self.adata.obs[f"{layer_name}"].value_counts().max()
                    size = np.maximum(
                        node_size_max * np.sqrt((node_num_cells / max_cells)),
                        node_size_min,
                    )
                    if len(self.factor_lists[layer_idx]) == 1:
                        size = node_size_min
                    fixedsize = "true"
                elif show_all:
                    if (
                        factor_idx not in self.factor_lists[layer_idx]
                        or len(self.factor_lists[layer_idx]) == 1
                    ):
                        size = node_size_min
                        fixedsize = "true"
                        color = "gray"
                        fillcolor = "gray"
                        if len(self.factor_lists[layer_idx]) == 1:
                            label = ""

                if not show_label:
                    label = ""

                if shell:
                    radius = r * (layer_idx + 1) ** r_decay  # distance from root
                    if layer_idx == 0:
                        angle_dict[factor_name] = (
                            ii * 2 * np.pi / len(self.factor_lists[0])
                        )
                    else:
                        children_angles = [
                            angle_dict[f] for f in hierarchy[factor_name]
                        ]
                        angle_dict[factor_name] = np.mean(children_angles)
                    x = radius * np.cos(angle_dict[factor_name])
                    y = radius * np.sin(angle_dict[factor_name])
                    g.node(
                        factor_name,
                        label=label,
                        fillcolor=fillcolor,
                        color=color,
                        ordering=ordering,
                        style=style,
                        width=str(size),
                        height=str(size),
                        fixedsize=fixedsize,
                        pos=f"{x},{y}!",
                        pin="true",
                    )
                else:
                    g.node(
                        factor_name,
                        label=label,
                        fillcolor=fillcolor,
                        color=color,
                        ordering=ordering,
                        style=style,
                        width=str(size),
                        height=str(size),
                        fixedsize=fixedsize,
                    )

                if not color_edges:
                    color = None
                if layer_idx > 0:
                    if hierarchy is not None:
                        if factor_name in hierarchy:
                            lower_factor_names = hierarchy[factor_name]
                            mat = np.array(
                                [
                                    self.compute_weight(factor_name, lower_factor_name)
                                    for lower_factor_name in lower_factor_names
                                ]
                            )
                            normalized_factor_weights = mat / np.sum(mat)
                            for lower_factor_idx, lower_factor_name in enumerate(
                                lower_factor_names
                            ):
                                normalized_weight = normalized_factor_weights[
                                    lower_factor_idx
                                ]
                                g.edge(
                                    factor_name,
                                    lower_factor_name,
                                    penwidth=str(4 * normalized_weight),
                                    color=color,
                                )
                    else:
                        if show_all:
                            mat = self.pmeans[f"{self.layer_names[layer_idx]}W"]
                        else:
                            mat = self.pmeans[f"{self.layer_names[layer_idx]}W"][
                                self.factor_lists[layer_idx]
                            ][:, self.factor_lists[layer_idx - 1]]
                        normalized_factor_weights = mat / np.sum(mat, axis=1).reshape(
                            -1, 1
                        )
                        for lower_factor_idx in layer_factor_orders[layer_idx - 1]:
                            if show_all:
                                lower_factor_name = f"{self.layer_names[layer_idx-1]}{int(lower_factor_idx)}"
                            else:
                                lower_factor_name = self.factor_names[layer_idx - 1][
                                    lower_factor_idx
                                ]

                            normalized_weight = normalized_factor_weights[
                                factor_idx, lower_factor_idx
                            ]

                            if factor_idx not in self.factor_lists[layer_idx] or (
                                len(self.factor_lists[layer_idx]) == 1 and show_all
                            ):
                                normalized_weight = normalized_weight / 5.0

                            g.edge(
                                factor_name,
                                lower_factor_name,
                                penwidth=str(4 * normalized_weight),
                                color=color,
                            )
        self.graph = g

        self.logger.info(f"Updated scDEF graph")

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

    def plot_scales(
        self, figsize=(8, 4), alpha=0.6, fontsize=12, legend_fontsize=10, show=True
    ):
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        self.plot_scale(
            "cell", figsize, alpha, fontsize, legend_fontsize, axes[0], False
        )
        self.plot_scale(
            "gene", figsize, alpha, fontsize, legend_fontsize, axes[1], False
        )
        if show:
            fig.tight_layout()
            plt.show()
        else:
            return fig

    def plot_scale(
        self,
        scale_type,
        figsize=(4, 4),
        alpha=0.6,
        fontsize=12,
        legend_fontsize=10,
        ax=None,
        show=True,
    ):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = ax.get_figure()

        if scale_type == "cell":
            x_data = self.batch_lib_sizes
            x_label = "Observed gene scale"

            def get_x_data_batch(b_cells):
                return self.batch_lib_sizes[np.where(b_cells)[0]]

            def get_y_data_batch(_, b_cells):
                return 1.0 / self.pmeans["cell_scale"].ravel()[np.where(b_cells)[0]]

            x_label = "Observed library size"
        else:
            x_data = np.sum(self.X, axis=0)
            x_label = "Observed gene scale"

            def get_x_data_batch(b_cells):
                return np.sum(self.X[b_cells], axis=0)

            def get_y_data_batch(b_id, _):
                return 1.0 / self.pmeans["gene_scale"][b_id].ravel()

        if len(self.batches) > 1:
            for b_id, b in enumerate(self.batches):
                b_cells = self.adata.obs[self.batch_key] == b
                ax.scatter(
                    get_x_data_batch(b_cells),
                    get_y_data_batch(b_id, b_cells),
                    label=b,
                    alpha=alpha,
                )
            ax.legend(fontsize=legend_fontsize)
        else:
            ax.scatter(
                x_data,
                1.0 / self.pmeans[f"{scale_type}_scale"].ravel(),
                alpha=alpha,
            )
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlabel(x_label, fontsize=fontsize)
        ax.set_ylabel(f"Learned {scale_type} size factor", fontsize=fontsize)

        if show:
            fig.tight_layout()
            plt.show()
        else:
            return ax

    def plot_brd(
        self,
        thres=1.0,
        iqr_mult=None,
        show_yticks=False,
        scale="linear",
        normalize=False,
        fontsize=14,
        legend_fontsize=12,
        xlabel="Factor",
        ylabel="Relevance",
        title="Biological relevance determination",
        color=False,
        show=True,
        ax=None,
        **kwargs,
    ):
        if not self.use_brd:
            raise ValueError("This model instance doesn't use the BRD prior.")

        ard = []
        if thres is not None:
            ard = thres
        else:
            ard = iqr_mult

        layer_size = self.layer_sizes[0]
        scales = self.pmeans[f"brd"].ravel()
        if normalize:
            scales = scales - np.min(scales)
            scales = scales / np.max(scales)
        if thres is None:
            if iqr_mult is not None:
                median = np.median(scales)
                q3 = np.percentile(scales, 75)
                cutoff = ard * (q3 - median)
        else:
            cutoff = ard

        if ax is None:
            fig, ax = plt.subplots(**kwargs)
        else:
            fig = ax.get_figure()

        below = []
        if thres is None and iqr_mult is None:
            l = np.arange(self.layer_sizes[0])
            above = self.factor_lists[0]
            below = np.array([f for f in l if f not in above])
        else:
            plt.axhline(cutoff, color="red", ls="--")
            above = np.where(scales >= cutoff)[0]
            below = np.where(scales < cutoff)[0]

        if color:
            colors = []
            f_idx = 0
            for i in range(self.layer_sizes[0]):
                if i in self.factor_lists[0]:
                    colors.append(self.layer_colorpalettes[0][f_idx])
                    f_idx += 1
                else:
                    colors.append("grey")
            ax.bar(np.arange(layer_size), scales, color=colors)
        else:
            ax.bar(np.arange(layer_size)[above], scales[above], label="Kept")
            if len(below) > 0:
                ax.bar(
                    np.arange(layer_size)[below],
                    scales[below],
                    alpha=0.6,
                    color="gray",
                    label="Removed",
                )
        if len(scales) > 15:
            ax.set_xticks(np.arange(0, layer_size, 2))
        else:
            ax.set_xticks(np.arange(layer_size))
        if not show_yticks:
            ax.set_yticks([])
        ax.set_title(title, fontsize=fontsize)
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_yscale(scale)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        if not color:
            ax.legend(fontsize=legend_fontsize)

        if show:
            plt.show()
        else:
            return ax

    def plot_gini_brd(
        self,
        normalize=False,
        figsize=(4, 4),
        alpha=0.6,
        fontsize=12,
        legend_fontsize=10,
        show=True,
        ax=None,
    ):
        brds = self.pmeans["brd"].ravel()
        if normalize:
            brds = brds - np.min(brds)
            brds = brds / np.max(brds)
        ginis = np.array(
            [
                score_utils.gini(self.pmeans[f"{self.layer_names[0]}W"][k])
                for k in range(self.layer_sizes[0])
            ]
        )
        is_kept = np.zeros((self.layer_sizes[0]))
        is_kept[self.factor_lists[0]] = 1

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        for c in [1, 0]:
            label = "Kept" if c == 1 else "Removed"
            color = "C0" if c == 1 else "gray"
            _alpha = 1 if c == 1 else alpha
            ax.scatter(
                ginis[np.where(is_kept == c)[0]],
                brds[np.where(is_kept == c)[0]],
                label=label,
                color=color,
                alpha=_alpha,
            )
        ax.set_xlabel("Gini index", fontsize=fontsize)
        ax.set_ylabel("BRD posterior mean", fontsize=fontsize)
        ax.legend(fontsize=legend_fontsize)

        if show:
            plt.show()
        else:
            return ax

    def plot_loss(self, figsize=(4, 4), fontsize=12, ax=None, show=True):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        ax.plot(np.concatenate(self.elbos)[:])
        ax.set_xlabel("Epoch", fontsize=fontsize)
        ax.set_yscale("log")
        ax.set_ylabel("Loss [log]", fontsize=fontsize)

        if show:
            plt.show()
        else:
            return ax

    def plot_qc(self, figsize=(8, 12), show=True):
        """Plot QC metrics for scDEF run:
            top left: Loss (-1xELBO [log]) vs. Epoch
            top right: Biological relevance det. (BRD) vs. Gini coefficient
            middle left: Learned cell scale vs. Observed library size
            middle right: Learned gene scale vs. Observed gene scale
            bottom:  Biological relevance determination

        Args:
            figsize (tuple(float, float)): Figure size in inches
            show (bool): whether to show the plot

        Returns:
            fig object if show is False and None otherwise
        """

        if self.use_brd:
            fig = plt.figure(figsize=figsize)
            gs = GridSpec(3, 2)
            # First row
            self.plot_loss(ax=fig.add_subplot(gs[0, 0]), show=False)
            self.plot_gini_brd(ax=fig.add_subplot(gs[0, 1]), show=False)
            # Second row
            self.plot_scale("cell", ax=fig.add_subplot(gs[1, 0]), show=False)
            self.plot_scale("gene", ax=fig.add_subplot(gs[1, 1]), show=False)
            # Third row
            self.plot_brd(ax=fig.add_subplot(gs[2, 0:2]), show=False)
        else:
            fig = plt.figure(figsize=(figsize[0], int(figsize[1] * 2 / 3)))
            gs = GridSpec(2, 2)
            # First row
            self.plot_loss(ax=fig.add_subplot(gs[0, 0:2]), show=False)
            # Second row
            self.plot_scale("cell", ax=fig.add_subplot(gs[1, 0]), show=False)
            self.plot_scale("gene", ax=fig.add_subplot(gs[1, 1]), show=False)

        fig.tight_layout()
        if show:
            plt.show()
        else:
            return fig

    def plot_obs_factor_dotplot(
        self,
        obs_key,
        layer_idx,
        cluster_rows=True,
        cluster_cols=True,
        figsize=(8, 2),
        s_min=100,
        s_max=500,
        titlesize=12,
        labelsize=12,
        legend_fontsize=12,
        legend_titlesize=12,
        cmap="viridis",
        logged=False,
        width_ratios=[5, 1, 1],
        show_ylabel=True,
        show=True,
    ):
        # For each obs, compute the average cell score on each factor among the cells that attach to that obs, use as color
        # And compute the fraction of cells in the obs that attach to each factor, use as circle size
        layer_name = self.layer_names[layer_idx]

        obs = self.adata.obs[obs_key].unique()
        n_obs = len(obs)
        n_factors = len(self.factor_lists[layer_idx])

        df_rows = []
        c = np.zeros((n_obs, n_factors))
        s = np.zeros((n_obs, n_factors))
        for i, obs_val in enumerate(obs):
            cells_from_obs = self.adata.obs.index[
                np.where(self.adata.obs[obs_key] == obs_val)[0]
            ]
            n_cells_obs = len(cells_from_obs)
            for factor in range(n_factors):
                factor_name = self.factor_names[layer_idx][factor]
                cells_attached = self.adata.obs.index[
                    np.where(
                        self.adata.obs.loc[cells_from_obs][f"{layer_name}"]
                        == factor_name
                    )[0]
                ]
                if len(cells_attached) == 0:
                    average_weight = 0  # np.nan
                    fraction_attached = 0  # np.nan
                else:
                    average_weight = np.mean(
                        self.adata.obs.loc[cells_from_obs][f"{factor_name}_score"]
                    )
                    fraction_attached = len(cells_attached) / n_cells_obs
                c[i, factor] = average_weight
                s[i, factor] = fraction_attached

        ylabels = obs
        xlabels = self.factor_names[layer_idx]

        if cluster_rows:
            Z = ward(pdist(s))
            hclust_index = leaves_list(Z)
            s = s[hclust_index]
            c = c[hclust_index]
            ylabels = ylabels[hclust_index]

        if cluster_cols:
            Z = ward(pdist(s.T))
            hclust_index = leaves_list(Z)
            s = s[:, hclust_index]
            c = c[:, hclust_index]
            xlabels = xlabels[hclust_index]

        x, y = np.meshgrid(np.arange(len(xlabels)), np.arange(len(ylabels)))

        fig, axes = plt.subplots(1, 3, figsize=figsize, width_ratios=width_ratios)
        plt.sca(axes[0])
        ax = plt.gca()
        s = s / np.max(s)
        s *= s_max
        s += s_max / s_min
        if logged:
            c = np.log(c)
        plt.scatter(x, y, c=c, s=s, cmap=cmap)

        ax.set(
            xticks=np.arange(n_factors),
            yticks=np.arange(n_obs),
            xticklabels=xlabels,
            yticklabels=ylabels,
        )
        ax.tick_params(axis="both", which="major", labelsize=labelsize)
        ax.set_xticks(np.arange(n_factors + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(n_obs + 1) - 0.5, minor=True)
        ax.grid(which="minor")

        if show_ylabel:
            ax.set_ylabel(obs_key, rotation=270, labelpad=20.0, fontsize=labelsize)
        plt.xlabel("Factor", fontsize=labelsize)
        plt.title(f"Layer {layer_idx}\n", fontsize=titlesize)

        # Make legend
        map_f_to_s = {"0": 5, "25": 7, "50": 9, "75": 11, "100": 13}
        plt.sca(axes[1])
        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        circles = [
            Line2D(
                [],
                [],
                color="white",
                marker="o",
                markersize=map_f_to_s[f],
                markerfacecolor="gray",
            )
            for f in map_f_to_s.keys()
        ]
        lg = plt.legend(
            circles[::-1],
            list(map_f_to_s.keys())[::-1],
            numpoints=1,
            loc=2,
            frameon=False,
            fontsize=legend_fontsize,
        )
        plt.title("Fraction of \ncells in group (%)", fontsize=legend_titlesize)

        plt.sca(axes[2])
        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        cb = plt.colorbar(ax=axes[0], cmap=cmap)
        cb.ax.set_title("Average\ncell score", fontsize=legend_titlesize)
        cb.ax.tick_params(labelsize=legend_fontsize)
        plt.grid("off")
        if show:
            plt.show()

    def plot_multilevel_paga(
        self,
        neighbors_rep: Optional[str] = "X_L0",
        layers: Optional[list] = None,
        figsize: Optional[tuple] = (16, 4),
        reuse_pos: Optional[bool] = True,
        fontsize: Optional[int] = 12,
        show: Optional[bool] = True,
        **paga_kwargs,
    ):
        """Plot a PAGA graph from each scDEF layer.

        Args:
            neighbors_rep: the self.obsm key to use to compute the PAGA graphs
            layers: which layers to plot
            figsize: figure size
            reuse_pos: whether to initialize each PAGA graph with the graph from the layer above
            show: whether to show the plot
            **paga_kwargs: keyword arguments to adjust the PAGA layouts
        """

        if layers is None:
            layers = [
                i
                for i in range(self.n_layers - 1, -1, -1)
                if len(self.factor_lists[i]) > 1
            ]

        if len(layers) == 0:
            self.logger.info("Cannot run PAGA on 0 layers.")
            return

        n_layers = len(layers)

        fig, axes = plt.subplots(1, n_layers, figsize=figsize)
        sc.pp.neighbors(self.adata, use_rep=neighbors_rep)
        pos = None
        for i, layer_idx in enumerate(layers):
            ax = axes[i]
            new_layer_name = f"{self.layer_names[layer_idx]}"

            self.logger.info(f"Computing PAGA graph of layer {layer_idx}")

            # Use previous PAGA as initial positions for new PAGA
            if layer_idx != layers[0] and reuse_pos:
                self.logger.info(
                    f"Re-using PAGA positions from layer {layer_idx+1} to init {layer_idx}"
                )
                matches = sc._utils.identify_groups(
                    self.adata.obs[new_layer_name], self.adata.obs[old_layer_name]
                )
                pos = []
                np.random.seed(0)
                for c in self.adata.obs[new_layer_name].cat.categories:
                    pos_coarse = self.adata.uns["paga"]["pos"]  # previous PAGA
                    coarse_categories = self.adata.obs[old_layer_name].cat.categories
                    idx = coarse_categories.get_loc(matches[c][0])
                    pos_i = pos_coarse[idx] + np.random.random(2)
                    pos.append(pos_i)
                pos = np.array(pos)

            sc.tl.paga(self.adata, groups=new_layer_name)
            sc.pl.paga(
                self.adata,
                init_pos=pos,
                layout="fa",
                ax=ax,
                show=False,
                **paga_kwargs,
            )
            ax.set_title(f"Layer {layer_idx} PAGA", fontsize=fontsize)

            old_layer_name = new_layer_name
        if show:
            plt.show()

    def compute_weight(self, upper_factor_name, lower_factor_name):
        # Computes the weight between two factors across any number of layers
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

    def get_factor_obs_association_scores(self, obs_key, obs_val):
        scores = []
        factors = []
        layers = []
        for layer_idx in range(self.n_layers):
            n_factors = len(self.factor_lists[layer_idx])
            for factor in range(n_factors):
                factor_name = self.factor_names[layer_idx][factor]
                score = self.compute_factor_obs_association_score(
                    layer_idx, factor_name, obs_key, obs_val
                )
                scores.append(score)
                factors.append(factor_name)
                layers.append(layer_idx)
        return scores, factors, layers

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

    def get_factor_obs_assignment_fracs(self, obs_key, obs_val):
        scores = []
        factors = []
        layers = []
        for layer_idx in range(self.n_layers):
            n_factors = len(self.factor_lists[layer_idx])
            for factor in range(n_factors):
                factor_name = self.factor_names[layer_idx][factor]
                score = self.compute_factor_obs_assignment_fracs(
                    layer_idx, factor_name, obs_key, obs_val
                )
                scores.append(score)
                factors.append(factor_name)
                layers.append(layer_idx)
        return scores, factors, layers

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

    def get_factor_obs_weight_scores(self, obs_key, obs_val):
        scores = []
        factors = []
        layers = []
        for layer_idx in range(self.n_layers):
            n_factors = len(self.factor_lists[layer_idx])
            for factor in range(n_factors):
                factor_name = self.factor_names[layer_idx][factor]
                score = self.compute_factor_obs_weight_score(
                    layer_idx, factor_name, obs_key, obs_val
                )
                scores.append(score)
                factors.append(factor_name)
                layers.append(layer_idx)
        return scores, factors, layers

    def compute_factor_obs_entropies(self, obs_key):
        mats = self._get_weight_scores(obs_key, self.adata.obs[obs_key].unique())
        mat = np.concatenate(mats, axis=1)
        factors = [self.factor_names[idx] for idx in range(self.n_layers)]
        flat_list = [item for sublist in factors for item in sublist]
        entropies = scipy.stats.entropy(mat, axis=0)
        return dict(zip(flat_list, entropies))

    def assign_obs_to_factors(self, obs_keys, factor_names=[]):
        if not isinstance(obs_keys, list):
            obs_keys = [obs_keys]

        # Sort obs_keys from broad to specific
        sizes = [len(self.adata.obs[obs_key].unique()) for obs_key in obs_keys]
        obs_keys = np.array(obs_keys)[np.argsort(sizes)].tolist()

        obs_to_factor_assignments = []
        obs_to_factor_matches = []
        for obs_key in obs_keys:
            obskey_to_factor_assignments = dict()
            obskey_to_factor_matches = dict()
            for obs in self.adata.obs[obs_key].unique():
                scores, factors, layers = self.get_factor_obs_association_scores(
                    obs_key, obs
                )
                if len(factor_names) > 0:
                    # Subset to factor_names
                    idx = np.array(
                        [
                            i
                            for i, factor in enumerate(factors)
                            if factor in factor_names
                        ]
                    )
                    scores = np.array(scores)[idx]
                    factors = np.array(factors)[idx]
                    layers = np.array(layers)[idx]
                obskey_to_factor_assignments[obs] = factors[np.argmax(scores)]
                obskey_to_factor_matches[factors[np.argmax(scores)]] = obs
            obs_to_factor_assignments.append(obskey_to_factor_assignments)
            obs_to_factor_matches.append(obskey_to_factor_matches)

        # Join them all up
        from collections import ChainMap

        factor_annotation_assignments = ChainMap(*obs_to_factor_assignments)
        factor_annotation_matches = ChainMap(*obs_to_factor_matches)

        return dict(factor_annotation_assignments), dict(factor_annotation_matches)

    def complete_hierarchy(self, hierarchy, obs_keys):
        obs_vals = [
            self.adata.obs[obs_key].astype("category").cat.categories
            for obs_key in obs_keys
        ]
        obs_vals = list(set([item for sublist in obs_vals for item in sublist]))
        return hierarchy_utils.complete_hierarchy(hierarchy, obs_vals)

    def _get_assignment_fracs(self, obs_key, obs_vals):
        signatures_dict = self.get_signatures_dict()
        n_obs = len(obs_vals)
        mats = [
            np.zeros((n_obs, len(self.factor_names[idx])))
            for idx in range(self.n_layers)
        ]
        for i, obs in enumerate(obs_vals):
            scores, factors, layers = self.get_factor_obs_assignment_fracs(obs_key, obs)
            for j in range(self.n_layers):
                indices = np.where(np.array(layers) == j)[0]
                mats[j][i] = np.array(scores)[indices]
        return mats

    def _get_assignment_scores(self, obs_key, obs_vals):
        signatures_dict = self.get_signatures_dict()
        n_obs = len(obs_vals)
        mats = [
            np.zeros((n_obs, len(self.factor_names[idx])))
            for idx in range(self.n_layers)
        ]
        for i, obs in enumerate(obs_vals):
            scores, factors, layers = self.get_factor_obs_association_scores(
                obs_key, obs
            )
            for j in range(self.n_layers):
                indices = np.where(np.array(layers) == j)[0]
                mats[j][i] = np.array(scores)[indices]
        return mats

    def _get_weight_scores(self, obs_key, obs_vals):
        signatures_dict = self.get_signatures_dict()
        n_obs = len(obs_vals)
        mats = [
            np.zeros((n_obs, len(self.factor_names[idx])))
            for idx in range(self.n_layers)
        ]
        for i, obs in enumerate(obs_vals):
            scores, factors, layers = self.get_factor_obs_weight_scores(obs_key, obs)
            for j in range(self.n_layers):
                indices = np.where(np.array(layers) == j)[0]
                mats[j][i] = np.array(scores)[indices]
        return mats

    def _get_signature_scores(self, obs_key, obs_vals, markers, top_genes=10):
        signatures_dict = self.get_signatures_dict()
        n_obs = len(obs_vals)
        mats = [
            np.zeros((n_obs, len(self.factor_names[idx])))
            for idx in range(self.n_layers)
        ]
        for i, obs in enumerate(obs_vals):
            markers_type = markers[obs]
            nonmarkers_type = [m for m in markers if m not in markers_type]
            for layer_idx in range(self.n_layers):
                for j, factor_name in enumerate(self.factor_names[layer_idx]):
                    signature = signatures_dict[factor_name][:top_genes]
                    mats[layer_idx][i, j] = score_utils.score_signature(
                        signature, markers_type, nonmarkers_type
                    )
        return mats

    def _prepare_obs_factor_scores(
        self, obs_keys, get_scores_func, hierarchy=None, **kwargs
    ):
        if not isinstance(obs_keys, list):
            obs_keys = [obs_keys]

        factors = [self.factor_names[idx] for idx in range(self.n_layers)]
        flat_list = [item for sublist in factors for item in sublist]
        n_factors = len(flat_list)

        obs_mats = dict()
        obs_joined_mats = dict()
        obs_clusters = dict()
        obs_vals_dict = dict()
        for idx, obs_key in enumerate(obs_keys):
            obs_vals = self.adata.obs[obs_key].unique().tolist()

            # Don't keep non-hierarchical levels
            if idx > 0 and hierarchy is not None:
                obs_vals = [val for val in obs_vals if len(hierarchy[val]) > 0]

            obs_vals_dict[obs_key] = obs_vals
            n_obs = len(obs_vals)

            mats = get_scores_func(obs_key, obs_vals, **kwargs)

            if np.max(mats[-1]) > 1.0:
                for i in range(len(mats)):
                    mats[i] = mats[i] / np.max(mats[i])

            # Cluster rows across columns in all mats
            joined_mats = np.hstack(mats)
            Z = ward(pdist(joined_mats))
            hclust_index = leaves_list(Z)

            obs_mats[obs_key] = mats
            obs_joined_mats[obs_key] = joined_mats
            obs_clusters[obs_key] = hclust_index
        return obs_mats, obs_clusters, obs_vals_dict

    def plot_layers_obs(
        self,
        obs_keys,
        obs_mats,
        obs_clusters,
        obs_vals_dict,
        sort_layer_factors=True,
        orders=None,
        layers=None,
        vmax=None,
        vmin=None,
        cb_title="",
        cb_title_fontsize=10,
        fontsize=12,
        title_fontsize=12,
        pad=0.1,
        shrink=0.7,
        figsize=(10, 4),
        xticks_rotation=90.0,
        cmap=None,
        show=True,
        rasterized=False,
    ):
        if not isinstance(obs_keys, list):
            obs_keys = [obs_keys]

        if layers is None:
            layers = [
                i for i in range(0, self.n_layers) if len(self.factor_lists[i]) > 1
            ]

        n_layers = len(layers)

        if sort_layer_factors:
            layer_factor_orders = self.get_layer_factor_orders()
        else:
            if orders is not None:
                layer_factor_orders = orders
            else:
                layer_factor_orders = [
                    np.arange(len(self.factor_lists[i])) for i in range(self.n_layers)
                ]

        n_factors = [len(self.factor_lists[idx]) for idx in layers]
        n_obs = [len(obs_clusters[obs_key]) for obs_key in obs_keys]
        fig, axs = plt.subplots(
            len(obs_keys),
            n_layers,
            figsize=figsize,
            gridspec_kw={"width_ratios": n_factors, "height_ratios": n_obs},
        )
        axs = axs.reshape((len(obs_keys), n_layers))
        for i in layers:
            axs[0][i].set_title(f"Layer {i}", fontsize=title_fontsize)
            for j, obs_key in enumerate(obs_keys):
                ax = axs[j][i]
                mat = obs_mats[obs_key][i]
                mat = mat[obs_clusters[obs_key]][:, layer_factor_orders[i]]
                axplt = ax.pcolormesh(
                    mat, vmax=vmax, vmin=vmin, cmap=cmap, rasterized=rasterized
                )

                if j == len(obs_keys) - 1:
                    xlabels = self.factor_names[i]
                    xlabels = np.array(xlabels)[layer_factor_orders[i]]
                    ax.set_xticks(
                        np.arange(len(xlabels)) + 0.5,
                        xlabels,
                        rotation=xticks_rotation,
                        fontsize=fontsize,
                    )
                else:
                    ax.set(xticks=[])

                if i == 0:
                    ylabels = np.array(obs_vals_dict[obs_keys[j]])[
                        obs_clusters[obs_keys[j]]
                    ]
                    ax.set_yticks(
                        np.arange(len(ylabels)) + 0.5,
                        ylabels,
                    )
                else:
                    ax.set(yticks=[])

                if i == n_layers - 1:
                    ax.yaxis.set_label_position("right")
                    ax.set_ylabel(
                        obs_key, rotation=270, labelpad=20.0, fontsize=fontsize
                    )

        plt.subplots_adjust(wspace=0.05)
        plt.subplots_adjust(hspace=0.05)

        cb = fig.colorbar(axplt, ax=axs.ravel().tolist(), pad=pad, shrink=shrink)
        cb.ax.set_title(cb_title, fontsize=cb_title_fontsize)
        if show:
            plt.show()

    def _prepare_pathway_factor_scores(
        self,
        pathways,
        top_genes=20,
        source="source",
        target="target",
        score="Combined score",
        z_score=True,
    ):
        factors = [self.factor_names[idx] for idx in range(self.n_layers)]
        flat_list = [item for sublist in factors for item in sublist]
        n_factors = len(flat_list)

        obs_mats = dict()
        obs_joined_mats = dict()
        obs_clusters = dict()
        obs_vals_dict = dict()
        obs_vals_dict["Pathway"] = pathways[source].unique().tolist()

        n_pathways = len(obs_vals_dict["Pathway"])

        mats = []
        for layer in range(len(self.factor_names)):
            _n_factors = len(self.factor_names[layer])
            factor_vals = np.zeros((n_pathways, _n_factors))
            for i, factor in enumerate(self.factor_names[layer]):
                df = sc.get.rank_genes_groups_df(
                    self.adata,
                    group=factor,
                    key=f"{self.layer_names[layer]}_signatures",
                )
                df = df.set_index("names")
                df = df.iloc[:top_genes]
                res = decoupler.get_ora_df(
                    df, net=pathways, source=source, target=target, verbose=False
                )
                for term in res["Term"]:
                    term_idx = np.where(np.array(obs_vals_dict["Pathway"]) == term)[0]
                    factor_vals[term_idx, i] = res.loc[res["Term"] == term][
                        score
                    ].values[0]

                if z_score:
                    # Compute z-scores
                    den = np.std(factor_vals, axis=0)
                    den[den == 0] = 1e6
                    factor_vals = (factor_vals - np.mean(factor_vals, axis=0)) / den[
                        None, :
                    ]
            mats.append(factor_vals)

        # Cluster rows across columns in all mats
        joined_mats = np.hstack(mats)
        Z = ward(pdist(joined_mats))
        hclust_index = leaves_list(Z)

        obs_mats["Pathway"] = mats
        obs_joined_mats["Pathway"] = joined_mats
        obs_clusters["Pathway"] = hclust_index
        return obs_mats, obs_clusters, obs_vals_dict, joined_mats

    def plot_pathway_scores(
        self,
        pathways: pd.DataFrame,
        top_genes: Optional[int] = 20,
        **kwargs,
    ):
        """Plot the association between a set of cell annotations and a set of gene signatures.

        Args:
            obs_keys: the keys in self.adata.obs to use
            pathways: a pandas DataFrame containing PROGENy pathways
            **kwargs: plotting keyword arguments
        """
        (
            obs_mats,
            obs_clusters,
            obs_vals_dict,
            joined_mats,
        ) = self._prepare_pathway_factor_scores(
            pathways,
            top_genes=top_genes,
        )

        vmax = joined_mats.max()
        vmin = joined_mats.min()
        self.plot_layers_obs(
            ["Pathway"],
            obs_mats,
            obs_clusters,
            obs_vals_dict,
            vmax=vmax,
            vmin=vmin,
            **kwargs,
        )

    def plot_signatures_scores(
        self,
        obs_keys: Sequence[str],
        markers: Mapping[str, Sequence[str]],
        top_genes: Optional[int] = 10,
        hierarchy: Optional[dict] = None,
        **kwargs,
    ):
        """Plot the association between a set of cell annotations and a set of gene signatures.

        Args:
            obs_keys: the keys in self.adata.obs to use
            markers: a dictionary with keys corresponding to self.adata.obs[obs_keys] and values to gene lists
            top_genes: number of genes to consider in the score computations
            hierarchy: the polytree to restrict the associations to
            **kwargs: plotting keyword arguments
        """
        obs_mats, obs_clusters, obs_vals_dict = self._prepare_obs_factor_scores(
            obs_keys,
            self._get_signature_scores,
            markers=markers,
            top_genes=top_genes,
            hierarchy=hierarchy,
        )
        self.plot_layers_obs(obs_keys, obs_mats, obs_clusters, obs_vals_dict, **kwargs)

    def plot_obs_scores(
        self,
        obs_keys: Sequence[str],
        hierarchy: Optional[dict] = None,
        mode: Literal["f1", "fracs", "weights"] = "fracs",
        **kwargs,
    ):
        """Plot the association between a set of cell annotations and factors.

        Args:
            obs_keys: the keys in self.adata.obs to use
            hierarchy: the polytree to restrict the associations to
            mode: whether to compute scores based on assignments or weights
            **kwargs: plotting keyword arguments
        """
        if mode == "f1":
            f = self._get_assignment_scores
        elif mode == "fracs":
            f = self._get_assignment_fracs
        elif mode == "weights":
            f = self._get_weight_scores
        else:
            raise ValueError("`mode` must be one of ['f1', 'fracs', 'weights']")

        obs_mats, obs_clusters, obs_vals_dict = self._prepare_obs_factor_scores(
            obs_keys,
            f,
            hierarchy=hierarchy,
        )
        vmax = None
        vmin = None
        if mode == "f1" or mode == "fracs":
            vmax = 1.0
            vmin = 0.0

        self.plot_layers_obs(
            obs_keys,
            obs_mats,
            obs_clusters,
            obs_vals_dict,
            vmax=vmax,
            vmin=vmin,
            **kwargs,
        )

    def plot_umaps(
        self,
        color=[],
        layers=None,
        figsize=(16, 4),
        fontsize=12,
        legend_fontsize=10,
        use_log=False,
        metric="euclidean",
        rasterized=True,
        n_legend_cols=1,
        show=True,
    ):
        if layers is None:
            layers = [
                i
                for i in range(self.n_layers - 1, -1, -1)
                if len(self.factor_lists[i]) > 1
            ]

        n_layers = len(layers)

        if "X_umap" in self.adata.obsm:
            self.adata.obsm["X_umap_original"] = self.adata.obsm["X_umap"].copy()

        if not isinstance(color, list):
            color = [color]

        n_rows = len(color)
        if n_rows == 0:
            n_rows = 1

        fig, axes = plt.subplots(n_rows, n_layers, figsize=figsize)
        for layer in layers:
            # Compute UMAP
            self.adata.obsm[f"X_{self.layer_names[layer]}_log"] = np.log(
                self.adata.obsm[f"X_{self.layer_names[layer]}"]
            )
            if use_log:
                sc.pp.neighbors(self.adata, use_rep=f"X_{self.layer_names[layer]}_log")
            else:
                sc.pp.neighbors(
                    self.adata,
                    use_rep=f"X_{self.layer_names[layer]}",
                    metric=metric,
                )
            sc.tl.umap(self.adata)

            for row in range(len(color)):
                if n_rows > 1:
                    ax = axes[row, layer]
                else:
                    ax = axes[layer]
                legend_loc = None
                if layer == n_layers - 1:
                    legend_loc = "right margin"
                ax = sc.pl.umap(
                    self.adata,
                    color=[color[row]],
                    frameon=False,
                    show=False,
                    ax=ax,
                    legend_loc=legend_loc,
                )
                if row == 0:
                    ax.set_title(f"Layer {layer}", fontsize=fontsize)
                else:
                    ax.set_title("")

                if layer == n_layers - 1:
                    leg = ax.legend(
                        loc="center left",
                        bbox_to_anchor=(1, 0.5),
                        frameon=False,
                        title_fontsize=legend_fontsize,
                        fontsize=legend_fontsize,
                        title=color[row],
                        ncols=n_legend_cols,
                    )
                    leg._legend_box.align = "left"

        # Put the original one back
        if "X_umap_original" in self.adata.obsm:
            self.adata.obsm["X_umap"] = self.adata.obsm["X_umap_original"].copy()

        if show:
            plt.show()

    def plot_factors_bars(
        self,
        obs_keys,
        sort_layer_factors=True,
        orders=None,
        sharey=True,
        layers=None,
        vmax=None,
        vmin=None,
        fontsize=12,
        title_fontsize=12,
        legend_fontsize=8,
        pad=0.1,
        figsize=(10, 4),
        xticks_rotation=90.0,
        hspace=0.05,
        wspace=0.05,
        wbox_anchor=2.0,
        show=True,
    ):
        if not isinstance(obs_keys, list):
            obs_keys = [obs_keys]

        if layers is None:
            layers = [
                i for i in range(0, self.n_layers) if len(self.factor_lists[i]) > 1
            ]

        n_layers = len(layers)

        if sort_layer_factors:
            layer_factor_orders = self.get_layer_factor_orders()
        else:
            if orders is not None:
                layer_factor_orders = orders
            else:
                layer_factor_orders = [
                    np.arange(len(self.factor_lists[i])) for i in range(self.n_layers)
                ]

        obs_mats, obs_clusters, obs_vals_dict = self._prepare_obs_factor_scores(
            obs_keys,
            self._get_assignment_fracs,
        )

        n_factors = [len(self.factor_lists[idx]) for idx in layers]
        n_obs = [len(obs_clusters[obs_key]) for obs_key in obs_keys]

        fig, axs = plt.subplots(
            len(obs_keys),
            n_layers,
            figsize=figsize,
            gridspec_kw={"width_ratios": n_factors},
            sharey=sharey,
        )

        axs = axs.reshape((len(obs_keys), n_layers))
        for i in layers:
            axs[0][i].set_title(f"Layer {i}", fontsize=title_fontsize)
            layer_sizes = []
            for factor in self.factor_names[i]:
                layer_sizes.append(
                    len(np.where(self.adata.obs[f"{self.layer_names[i]}"] == factor)[0])
                )
            layer_sizes = np.array(layer_sizes)[layer_factor_orders[i]]
            xlabels = self.factor_names[i]
            xlabels = np.array(xlabels)[layer_factor_orders[i]]
            for j, obs_key in enumerate(obs_keys):
                ax = axs[j][i]
                plt.sca(ax)
                mat = obs_mats[obs_key][i]
                mat = mat[:, layer_factor_orders[i]]
                for idx, obs in enumerate(obs_vals_dict[obs_key]):
                    obs_idx = np.where(self.adata.obs[obs_key].cat.categories == obs)[
                        0
                    ][0]
                    color = self.adata.uns[f"{obs_key}_colors"][obs_idx]
                    y = mat[idx] * layer_sizes
                    if idx == 0:
                        plt.bar(xlabels, y, color=color, label=obs)
                        y_ = y
                    else:
                        plt.bar(xlabels, y, bottom=y_, color=color, label=obs)
                        y_ += y

                if j == len(obs_keys) - 1:
                    ax.set_xticks(
                        np.arange(len(xlabels)),
                        xlabels,
                        rotation=xticks_rotation,
                        fontsize=fontsize,
                    )
                else:
                    ax.set(xticks=[])

                if i == len(layers) - 1:
                    leg = ax.legend(
                        loc="center left",
                        bbox_to_anchor=(wbox_anchor, 0.5),
                        frameon=False,
                        title_fontsize=title_fontsize,
                        fontsize=legend_fontsize,
                        title=obs_key,
                    )
                    leg._legend_box.align = "left"

                if i == 0:
                    ax.set_ylabel("Number of cells", fontsize=fontsize)

                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

        plt.subplots_adjust(wspace=wspace)
        plt.subplots_adjust(hspace=hspace)

        if show:
            plt.show()
