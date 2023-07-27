from jax import jit, grad, vmap
from jax.example_libraries import optimizers
from jax import random, value_and_grad
from jax.scipy.stats import norm, gamma, poisson
import jax.numpy as jnp
import jax.nn as jnn

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import gseapy as gp
from graphviz import Graph
from tqdm import tqdm
import time

import logging

import scipy
import numpy as np
import pandas as pd
from anndata import AnnData
import scanpy as sc

from scipy.cluster.hierarchy import ward, leaves_list
from scipy.spatial.distance import pdist

from ..utils import score_utils, hierarchy_utils, color_utils
from ..utils.jax_utils import *


class scDEF(object):
    def __init__(
        self,
        adata,
        counts_layer=None,
        layer_sizes=[100, 30, 10, 3],
        batch_key="batch",
        seed=42,
        logginglevel=logging.INFO,
        layer_shapes=None,
        brd=1e3,
        use_brd=True,
        cell_scale_shape=1.0,
        gene_scale_shape=1.0,
        factor_shapes=None,
        factor_rates=None,
        layer_diagonals=None,
        batch_cpal="Dark2",
        layer_cpal=None,
        lightness_mult=0.1,
    ):
        self.layer_sizes = [int(x) for x in layer_sizes]
        self.n_layers = len(self.layer_sizes)

        if self.n_layers < 2:
            raise ValueError("scDEF requires at least 2 layers")

        if layer_shapes is None:
            layer_shapes = [1.0] * self.n_layers
        elif isinstance(layer_shapes, float) or isinstance(layer_shapes, int):
            layer_shapes = [float(layer_shapes)] * self.n_layers
        elif len(layer_shapes) != self.n_layers:
            raise ValueError("layer_shapes list must be of size scDEF.n_layers")

        if factor_shapes is None:
            factor_shapes = [1.0] + [0.1] * (self.n_layers - 1)
        elif isinstance(factor_shapes, float) or isinstance(factor_shapes, int):
            factor_shapes = [float(factor_shapes)] * self.n_layers
        elif len(factor_shapes) != self.n_layers:
            raise ValueError("factor_shapes list must be of size scDEF.n_layers")

        if factor_rates is None:
            factor_rates = [10.0] + [0.3] * (self.n_layers - 1)
        elif isinstance(factor_rates, float) or isinstance(factor_rates, int):
            factor_rates = [float(factor_rates)] * self.n_layers
        elif len(factor_rates) != self.n_layers:
            raise ValueError("factor_rates list must be of size scDEF.n_layers")

        if layer_diagonals is None:
            layer_diagonals = [1.0] * self.n_layers
        elif isinstance(layer_diagonals, float) or isinstance(layer_diagonals, int):
            layer_diagonals = [float(layer_diagonals)] * self.n_layers
        elif len(layer_diagonals) != self.n_layers:
            raise ValueError("layer_diagonals list must be of size scDEF.n_layers")

        if layer_cpal is None:
            layer_cpal = ["Set1"] * self.n_layers
            if self.n_layers <= 3:
                layer_cpal = [f"Set{i}" for i in range(1, self.n_layers + 1)]
        elif isinstance(layer_cpal, str):
            layer_cpal = [layer_cpal] * self.n_layers
        elif len(layer_cpal) != self.n_layers:
            raise ValueError("layer_cpal list must be of size scDEF.n_layers")

        self.layer_shapes = layer_shapes
        self.layer_diagonals = layer_diagonals

        self.factor_lists = [np.arange(size) for size in self.layer_sizes]

        self.factor_shapes = factor_shapes
        self.factor_rates = factor_rates

        self.brd = brd
        self.use_brd = use_brd
        self.cell_scale_shape = cell_scale_shape
        self.gene_scale_shape = gene_scale_shape

        self.logger = logging.getLogger("scDEF")
        self.logger.setLevel(logginglevel)

        self.n_batches = 1
        self.batches = [""]

        self.seed = seed
        self.layer_names = ["h" * i for i in range(self.n_layers)]
        self.layer_cpal = layer_cpal
        self.batch_cpal = batch_cpal
        self.layer_colorpalettes = [
            sns.color_palette(layer_cpal[idx], n_colors=size)
            for idx, size in enumerate(layer_sizes)
        ]

        if len(np.unique(layer_cpal)) == 1:
            # Make the layers have different lightness
            for layer_idx, size in enumerate(self.layer_sizes):
                for factor_idx in range(size):
                    col = self.layer_colorpalettes[layer_idx][factor_idx]
                    self.layer_colorpalettes[layer_idx][
                        factor_idx
                    ] = color_utils.adjust_lightness(
                        col, amount=1.0 + lightness_mult * layer_idx
                    )

        self.load_adata(adata, layer=counts_layer, batch_key=batch_key)
        self.batch_key = batch_key
        self.n_cells, self.n_genes = adata.shape

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

        self.init_var_params()
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
            + ", ".join([f"{name}factor" for name in self.layer_names])
        )
        out += (
            "\n\t"
            + "Layer sizes: "
            + ", ".join([str(len(factors)) for factors in self.factor_lists])
        )
        out += (
            "\n\t"
            + "Layer shape parameters: "
            + ", ".join([str(shape) for shape in self.layer_shapes])
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
        out += "\n\t" + "BRD prior parameter: " + str(self.brd)
        out += "\n\t" + "Number of batches: " + str(self.n_batches)
        out += "\n" + "Contains " + self.adata.__str__()
        return out

    def load_adata(self, adata, layer=None, batch_key="batch"):
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
        gene_size = np.sum(self.X, axis=0)
        self.gene_ratio = np.mean(gene_size) / np.var(gene_size)
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
            self.batch_indices_onehot = np.zeros((self.adata.shape[0], self.n_batches))
            if self.n_batches > 1:
                self.gene_ratio = np.ones((self.n_batches, self.adata.shape[1]))
                for i, b in enumerate(batches):
                    cells = np.where(self.adata.obs[batch_key] == b)[0]
                    self.batch_indices_onehot[cells, i] = 1
                    self.batch_lib_sizes[cells] = np.sum(self.X, axis=1)[cells]
                    self.batch_lib_ratio[cells] = np.mean(
                        self.batch_lib_sizes[cells]
                    ) / np.var(self.batch_lib_sizes[cells])
                    batch_gene_size = np.sum(self.X[cells], axis=0)
                    self.gene_ratio[i] = np.mean(batch_gene_size) / np.var(
                        batch_gene_size
                    )
        self.batch_indices_onehot = jnp.array(self.batch_indices_onehot)
        self.batch_lib_sizes = jnp.array(self.batch_lib_sizes)
        self.batch_lib_ratio = jnp.array(self.batch_lib_ratio)
        self.gene_ratio = jnp.array(self.gene_ratio)

    def init_var_params(self, minval=0.5, maxval=1.5):
        rngs = random.split(random.PRNGKey(self.seed), 6 + 2 * 2 * self.n_layers)

        self.var_params = [
            jnp.array(
                (
                    jnp.log(
                        random.uniform(
                            rngs[0], minval=0.5, maxval=1.5, shape=[self.n_cells, 1]
                        )
                        * 10.0
                    ),  # cell_scales
                    jnp.log(
                        random.uniform(
                            rngs[1], minval=0.5, maxval=1.5, shape=[self.n_cells, 1]
                        )
                    )
                    * jnp.clip(10.0 * self.batch_lib_ratio, 1e-6, 1e2),
                )
            ),
            jnp.array(
                (
                    jnp.log(
                        random.uniform(
                            rngs[2],
                            minval=0.5,
                            maxval=1.5,
                            shape=[self.n_batches, self.n_genes],
                        )
                        * 10.0
                    ),  # gene_scales
                    jnp.log(
                        random.uniform(
                            rngs[3],
                            minval=0.5,
                            maxval=1.5,
                            shape=[self.n_batches, self.n_genes],
                        )
                        * jnp.clip(10.0 * self.gene_ratio, 1e-6, 1e2)
                    ),
                )
            ),
        ]

        self.var_params.append(
            jnp.array(
                (
                    jnp.log(
                        random.uniform(
                            rngs[4],
                            minval=1.0,
                            maxval=1.0,
                            shape=[self.layer_sizes[0], 1],
                        )
                    ),  # BRD
                    jnp.log(
                        random.uniform(
                            rngs[5],
                            minval=1.0,
                            maxval=1.0,
                            shape=[self.layer_sizes[0], 1],
                        )
                    ),
                )
            )
        )

        z_shapes = []
        z_rates = []
        rng_cnt = 6
        for layer_idx in range(
            self.n_layers
        ):  # we go from layer 0 (bottom) to layer L (top)
            # Init z
            z_shape = jnp.log(
                random.uniform(
                    rngs[rng_cnt],
                    minval=minval,
                    maxval=maxval,
                    shape=[self.n_cells, self.layer_sizes[layer_idx]],
                )
                * self.layer_shapes[layer_idx]
            )
            z_shapes.append(z_shape)
            rng_cnt += 1
            z_rate = jnp.log(
                random.uniform(
                    rngs[rng_cnt],
                    minval=minval,
                    maxval=maxval,
                    shape=[self.n_cells, self.layer_sizes[layer_idx]],
                )
                * self.layer_shapes[layer_idx]
            )
            z_rates.append(z_rate)
            rng_cnt += 1

        self.var_params.append(jnp.array((jnp.hstack(z_shapes), jnp.hstack(z_rates))))

        for layer_idx in range(
            self.n_layers
        ):  # the w don't have a shared axis across all, so we can't vectorize
            # Init w
            in_layer = self.layer_sizes[layer_idx]
            if layer_idx == 0:
                out_layer = self.n_genes
            else:
                out_layer = self.layer_sizes[layer_idx - 1]
            w_shape = jnp.log(
                random.uniform(
                    rngs[rng_cnt],
                    minval=minval,
                    maxval=maxval,
                    shape=[in_layer, out_layer],
                )
                * jnp.clip(self.w_priors[layer_idx][0], 1e-2, 1e2)
            )
            rng_cnt += 1
            w_rate = jnp.log(
                random.uniform(
                    rngs[rng_cnt],
                    minval=minval,
                    maxval=maxval,
                    shape=[in_layer, out_layer],
                )
                * jnp.clip(self.w_priors[layer_idx][1], 1e-2, 1e2)
            )
            rng_cnt += 1

            self.var_params.append(jnp.array((w_shape, w_rate)))

    def elbo(
        self,
        rng,
        batch,
        indices,
        var_params,
        annealing_parameter,
        min_shape=1e-5,
        min_rate=1e-3,
        max_rate=1e3,
    ):
        # Single-sample Monte Carlo estimate of the variational lower bound.
        batch_indices_onehot = self.batch_indices_onehot[indices]

        cell_budget_params = var_params[0]
        gene_budget_params = var_params[1]
        fscale_params = var_params[2]
        z_params = var_params[3]

        cell_budget_shape = jnp.maximum(
            jnp.exp(cell_budget_params[0][indices]), min_shape
        )
        cell_budget_rate = jnp.minimum(
            jnp.maximum(jnp.exp(cell_budget_params[1][indices]), min_rate), max_rate
        )

        gene_budget_shape = jnp.maximum(jnp.exp(gene_budget_params[0]), min_shape)
        gene_budget_rate = jnp.minimum(
            jnp.maximum(jnp.exp(gene_budget_params[1]), min_rate), max_rate
        )

        fscale_shapes = jnp.maximum(jnp.exp(fscale_params[0]), min_shape)
        fscale_rates = jnp.minimum(
            jnp.maximum(jnp.exp(fscale_params[1]), min_rate), max_rate
        )

        z_shapes = jnp.maximum(jnp.exp(z_params[0][indices]), min_shape)
        z_rates = jnp.minimum(
            jnp.maximum(jnp.exp(z_params[1][indices]), min_rate), max_rate
        )

        # Sample from variational distribution
        cell_budget_sample = gamma_sample(rng, cell_budget_shape, cell_budget_rate)
        gene_budget_sample = gamma_sample(rng, gene_budget_shape, gene_budget_rate)
        if self.use_brd:
            fscale_samples = gamma_sample(rng, fscale_shapes, fscale_rates)
        z_samples = gamma_sample(rng, z_shapes, z_rates)  # vectorized
        # w will be sampled in a loop below because it cannot be vectorized

        # Compute ELBO
        global_pl = gamma_logpdf(
            gene_budget_sample,
            self.gene_scale_shape,
            self.gene_scale_shape * self.gene_ratio,
        )
        global_en = -gamma_logpdf(
            gene_budget_sample, gene_budget_shape, gene_budget_rate
        )
        local_pl = gamma_logpdf(
            cell_budget_sample,
            self.cell_scale_shape,
            self.cell_scale_shape * self.batch_lib_ratio[indices],
        )
        local_en = -gamma_logpdf(
            cell_budget_sample, cell_budget_shape, cell_budget_rate
        )

        # scale
        if self.use_brd:
            global_pl += gamma_logpdf(
                fscale_samples, self.brd, self.brd * self.factor_rates[0]
            )
            global_en += -gamma_logpdf(fscale_samples, fscale_shapes, fscale_rates)

        # Top layer
        idx = self.n_layers - 1
        start = np.sum(self.layer_sizes[:idx])
        end = start + self.layer_sizes[idx]
        # w
        _w_shape = jnp.maximum(jnp.exp(var_params[4 + idx][0]), min_shape)
        _w_rate = jnp.minimum(
            jnp.maximum(jnp.exp(var_params[4 + idx][1]), min_rate), max_rate
        )
        _w_sample = gamma_sample(rng, _w_shape, _w_rate)
        global_pl += gamma_logpdf(
            _w_sample, self.w_priors[idx][0], self.w_priors[idx][1]
        )
        global_en += -gamma_logpdf(_w_sample, _w_shape, _w_rate)
        # z
        _z_sample = z_samples[:, start:end]
        _z_shape = z_shapes[:, start:end]
        _z_rate = z_rates[:, start:end]
        local_pl += gamma_logpdf(
            _z_sample, self.layer_shapes[idx], self.layer_shapes[idx]
        )
        local_en += -gamma_logpdf(_z_sample, _z_shape, _z_rate)

        z_mean = jnp.einsum("nk,kp->np", _z_sample, _w_sample)

        for idx in list(np.arange(0, self.n_layers - 1)[::-1]):
            start = np.sum(self.layer_sizes[:idx]).astype(int)
            end = start + self.layer_sizes[idx]

            # w
            _w_shape = jnp.maximum(jnp.exp(var_params[4 + idx][0]), min_shape)
            _w_rate = jnp.minimum(
                jnp.maximum(jnp.exp(var_params[4 + idx][1]), min_rate), max_rate
            )
            _w_sample = gamma_sample(rng, _w_shape, _w_rate)
            if idx == 0 and self.use_brd:
                global_pl += gamma_logpdf(
                    _w_sample,
                    self.w_priors[idx][0] / fscale_samples,
                    self.w_priors[idx][1] / fscale_samples,
                )
            elif idx == 1 and self.use_brd:
                global_pl += gamma_logpdf(
                    _w_sample,
                    self.w_priors[idx][0],
                    self.w_priors[idx][1] / fscale_samples.T,
                )
            else:
                global_pl += gamma_logpdf(
                    _w_sample,
                    self.w_priors[idx][0],
                    self.w_priors[idx][1],
                )
            global_en += -gamma_logpdf(_w_sample, _w_shape, _w_rate)
            # z
            _z_sample = z_samples[:, start:end]
            _z_shape = z_shapes[:, start:end]
            _z_rate = z_rates[:, start:end]
            local_pl += gamma_logpdf(
                _z_sample, self.layer_shapes[idx], self.layer_shapes[idx] / z_mean
            )
            local_en += -gamma_logpdf(_z_sample, _z_shape, _z_rate)

            z_mean = jnp.einsum("nk,kp->np", _z_sample, _w_sample)

        # Compute log likelihood
        mean_bottom = jnp.einsum(
            "nk,kg->ng", _z_sample / cell_budget_sample, _w_sample
        ) / (batch_indices_onehot.dot(gene_budget_sample))
        ll = jnp.sum(vmap(poisson.logpmf)(jnp.array(batch), mean_bottom))

        # Anneal the entropy
        global_en *= annealing_parameter
        local_en *= annealing_parameter

        return ll + (
            local_pl
            + local_en
            + (indices.shape[0] / self.X.shape[0]) * (global_pl + global_en)
        )

    def batch_elbo(self, rng, X, indices, var_params, num_samples, annealing_parameter):
        # Average over a batch of random samples.
        rngs = random.split(rng, num_samples)
        vectorized_elbo = vmap(self.elbo, in_axes=(0, None, None, None, None))
        return jnp.mean(
            vectorized_elbo(rngs, X, indices, var_params, annealing_parameter)
        )

    def _optimize(
        self,
        update_func,
        opt_state,
        n_epochs=500,
        batch_size=1,
        step_size=0.1,
        annealing_parameter=1.0,
        seed=None,
    ):
        if seed is None:
            seed = self.seed

        num_complete_batches, leftover = divmod(self.n_cells, batch_size)
        num_batches = num_complete_batches + bool(leftover)
        self.logger.info(
            f"Each epoch contains {num_batches} batches of size {batch_size}"
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
        rng = random.PRNGKey(seed)
        t = 0
        pbar = tqdm(range(n_epochs))
        for epoch in pbar:
            epoch_losses = []
            start_time = time.time()
            for it in range(num_batches):
                rng, rng_input = random.split(rng)
                X, indices = next(batches)
                loss, opt_state = update_func(
                    X, indices, t, rng_input, opt_state, annealing_parameter
                )
                epoch_losses.append(loss)
                t += 1
            losses.append(np.mean(epoch_losses))
            epoch_time = time.time() - start_time
            pbar.set_postfix({"Loss": losses[-1]})

        return losses, opt_state

    def learn(
        self,
        n_epoch=[1000, 1000],
        lr=0.1,
        annealing=1.0,
        num_samples=5,
        batch_size=None,
    ):
        n_steps = 1

        if isinstance(n_epoch, list):
            n_steps = len(n_epoch)
            n_epoch_schedule = n_epoch
        else:
            n_epoch_schedule = [n_epoch]

        if isinstance(lr, list):
            lr_schedule = lr
            if len(lr_schedule) != n_steps:
                raise ValueError(
                    "lr_schedule list must be of same length as n_epoch_schedule"
                )
        else:
            lr_schedule = [lr * 0.5**step for step in range(n_steps)]

        if isinstance(annealing, list):
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
        init_params = self.var_params

        def objective(X, indices, var_params, key, annealing_parameter):
            return -self.batch_elbo(
                key, X, indices, var_params, num_samples, annealing_parameter
            )  # minimize -ELBO

        loss_grad = jit(value_and_grad(objective, argnums=2))

        for i in range(len(lr_schedule)):
            step_size = lr_schedule[i]
            anneal_param = annealing_schedule[i]
            self.logger.info(
                f"Initializing optimizer with learning rate {step_size} and annealing parameter {anneal_param}"
            )
            opt_init, opt_update, get_params = optimizers.adam(step_size=step_size)

            def update(X, indices, i, key, opt_state, annealing_parameter):
                params = get_params(opt_state)
                value, gradient = loss_grad(
                    X, indices, params, key, annealing_parameter
                )
                return value, opt_update(i, gradient, opt_state)

            opt_state = opt_init(self.var_params)

            losses, opt_state = self._optimize(
                update,
                opt_state,
                n_epochs=n_epoch_schedule[i],
                step_size=lr_schedule[i],
                batch_size=batch_size,
                annealing_parameter=anneal_param,
            )
            params = get_params(opt_state)
            self.var_params = params
            self.elbos.append(losses)
            self.step_sizes.append(lr_schedule[i])

        self.set_posterior_means()
        self.filter_factors()

    def set_posterior_means(self):
        cell_budget_params = self.var_params[0]
        gene_budget_params = self.var_params[1]
        fscale_params = self.var_params[2]
        z_params = self.var_params[3]
        w_params = self.var_params[4]

        self.pmeans = {
            "cell_scale": np.array(
                np.exp(cell_budget_params[0]) / np.exp(cell_budget_params[1])
            ),
            "gene_scale": np.array(
                np.exp(gene_budget_params[0]) / np.exp(gene_budget_params[1])
            ),
            "brd": np.array(np.exp(fscale_params[0]) / np.exp(fscale_params[1])),
        }

        for idx in range(self.n_layers):
            start = sum(self.layer_sizes[:idx])
            end = start + self.layer_sizes[idx]
            self.pmeans[f"{self.layer_names[idx]}z"] = np.array(
                np.exp(z_params[0][:, start:end]) / np.exp(z_params[1][:, start:end])
            )
            _w_shape = self.var_params[4 + idx][0]
            _w_rate = self.var_params[4 + idx][1]
            self.pmeans[f"{self.layer_names[idx]}W"] = np.array(
                np.exp(_w_shape) / np.exp(_w_rate)
            )

    def filter_factors(self, thres=None, iqr_mult=3.0, min_cells=0):
        """
        The model tends to remove unused factors by itself, but we can remove further
        noisy factors based low BRD posterior means. By default, keeps only factors
        for which the relevance is at least 3 times the difference between the third
        quartile and the median relevances, and to which at least 10 cells attach.
        """
        ard = []
        if thres is not None:
            ard = thres
        else:
            ard = iqr_mult

        if not self.use_brd:
            ard = 0.0

        self.factor_lists = []
        for i, layer_name in enumerate(self.layer_names):
            if i == 0:
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
                rels = self.pmeans[f"brd"].ravel()
                rels = rels - np.min(rels)
                rels = rels / np.max(rels)
                if thres is None:
                    median = np.median(rels)
                    q3 = np.percentile(rels, 75)
                    cutoff = ard * (q3 - median)
                else:
                    cutoff = ard
                keep = np.unique(
                    list(set(np.where(rels >= cutoff)[0]).intersection(keep))
                )
            else:
                mat = self.pmeans[f"{layer_name}W"]
                assignments = []
                for factor in self.factor_lists[i - 1]:
                    assignments.append(np.argmax(mat[:, factor]))
                keep = np.unique(assignments)

            if len(keep) == 0:
                self.logger.info(
                    f"No factors in layer {i} satisfy the filtering criterion. Please adjust the filtering parameters. \
                                Keeping all factors for layer {i} for now."
                )
                keep = np.arange(self.layer_sizes[i])
            self.factor_lists.append(keep)

        self.annotate_adata()
        self.make_graph()

    def set_factor_names(self):
        self.factor_names = [
            [
                f"{self.layer_names[idx]}{str(i)}"
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

        for idx in range(self.n_layers):
            layer_name = self.layer_names[idx]
            self.adata.obsm[f"X_{layer_name}factors"] = self.pmeans[f"{layer_name}z"][
                :, self.factor_lists[idx]
            ]
            assignments = np.argmax(self.adata.obsm[f"X_{layer_name}factors"], axis=1)
            self.adata.obs[f"{layer_name}factor"] = [
                self.factor_names[idx][a] for a in assignments
            ]
            self.adata.uns[f"{layer_name}factor_colors"] = [
                matplotlib.colors.to_hex(self.layer_colorpalettes[idx][i])
                for i in range(len(self.factor_lists[idx]))
            ]

            scores_names = [f + "_score" for f in self.factor_names[idx]]
            df = pd.DataFrame(
                self.adata.obsm[f"X_{layer_name}factors"],
                index=self.adata.obs.index,
                columns=scores_names,
            )
            if scores_names[0] not in self.adata.obs.columns:
                self.adata.obs = pd.concat([self.adata.obs, df], axis=1)
            else:
                self.adata.obs = self.adata.obs.drop(
                    columns=[col for col in self.adata.obs.columns if "score" in col]
                )
                self.adata.obs[df.columns] = df

            self.logger.info(
                f"Updated adata.obs with layer {idx}: `{layer_name}factor` and `{layer_name}_score` for all factors in layer {idx}"
            )
            self.logger.info(
                f"Updated adata.obsm with layer {idx}: `X_{layer_name}factors`"
            )

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

        term_scores_shape = np.exp(self.var_params[4 + 0][0][self.factor_lists[0]])
        term_scores_rate = np.exp(self.var_params[4 + 0][1][self.factor_lists[0]])
        term_scores_sample = gamma_sample(rng, term_scores_shape, term_scores_rate)

        if layer_idx > 0:
            term_scores_shape = np.exp(
                self.var_params[4 + layer_idx][0][self.factor_lists[layer_idx]][
                    :, self.factor_lists[layer_idx - 1]
                ]
            )
            term_scores_rate = np.exp(
                self.var_params[4 + layer_idx][1][self.factor_lists[layer_idx]][
                    :, self.factor_lists[layer_idx - 1]
                ]
            )
            term_scores_sample = gamma_sample(rng, term_scores_shape, term_scores_rate)

            for layer in range(layer_idx - 1, 0, -1):
                lower_mat_shape = np.exp(
                    self.var_params[4 + layer][0][self.factor_lists[layer]][
                        :, self.factor_lists[layer - 1]
                    ]
                )
                lower_mat_rate = np.exp(
                    self.var_params[4 + layer][1][self.factor_lists[layer]][
                        :, self.factor_lists[layer - 1]
                    ]
                )
                lower_mat_sample = gamma_sample(rng, lower_mat_shape, lower_mat_rate)
                term_scores_sample = term_scores_sample.dot(lower_mat_sample)

            lower_term_scores_shape = np.exp(
                self.var_params[4 + 0][0][self.factor_lists[0]]
            )
            lower_term_scores_rate = np.exp(
                self.var_params[4 + 0][1][self.factor_lists[0]]
            )
            lower_term_scores_sample = gamma_sample(
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
        self, factor_idx, layer_idx, mc_samples=100, top_genes=10
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

        jaccs = np.zeros((mc_samples, mc_samples))
        for i in range(mc_samples):
            for j in range(mc_samples):
                jaccs[i, j] = score_utils.jaccard_similarity(
                    signatures[i], signatures[j]
                )

        return np.mean(jaccs)

    def get_sizes_dict(self):
        sizes_dict = {}
        for layer_idx in range(self.n_layers):
            layer_sizes = self.adata.obs[
                f"{self.layer_names[layer_idx]}factor"
            ].value_counts()
            for factor_idx, factor_name in enumerate(self.factor_names[layer_idx]):
                sizes_dict[factor_name] = layer_sizes[factor_name]
        return sizes_dict

    def get_signatures_dict(self, top_genes=None, scores=False, sorted_scores=False):
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
                signatures_dict[factor_name] = layer_signatures[factor_idx]
                scores_dict[factor_name] = layer_scores[factor_idx]

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
        hierarchy=None,
        factor_annotations=None,
        top_factor=None,
        show_signatures=True,
        enrichments=None,
        top_genes=None,
        show_batch_counts=False,
        filled=None,
        wedged=None,
        color_edges=True,
        show_confidences=False,
        mc_samples=100,
        **fontsize_kwargs,
    ):
        if top_genes is None:
            top_genes = [10] * self.n_layers
        elif isinstance(top_genes, float):
            top_genes = [top_genes] * self.n_layers
        elif len(top_genes) != self.n_layers:
            raise IndexError("top_genes list must be of size scDEF.n_layers")

        if filled is None:
            style = None
        elif filled == "factor":
            style = "filled"
        else:
            if filled not in self.adata.obs:
                raise ValueError("filled must be factor or any `obs` in self.adata")
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

        def map_scores_to_fontsizes(scores, max_fontsize=11, min_fontsize=5):
            scores = scores - np.min(scores)
            scores = scores / np.max(scores)
            fontsizes = min_fontsize + scores * (max_fontsize - min_fontsize)
            return fontsizes

        g = Graph()
        ordering = "out"

        for layer_idx in range(self.n_layers):
            layer_name = self.layer_names[layer_idx]
            factors = self.factor_lists[layer_idx]
            n_factors = len(factors)
            layer_colors = self.layer_colorpalettes[layer_idx][:n_factors]

            if show_signatures:
                gene_rankings, gene_scores = self.get_rankings(
                    layer_idx=layer_idx, genes=True, return_scores=True
                )

            factor_order = layer_factor_orders[layer_idx]
            for factor_idx in factor_order:
                factor_idx = int(factor_idx)
                alpha = "FF"
                color = None
                factor_name = f"{self.factor_names[layer_idx][int(factor_idx)]}"

                if hierarchy is not None and factor_name not in hierarchy_nodes:
                    continue

                label = factor_name
                if factor_annotations is not None:
                    if factor_name in factor_annotations:
                        label = factor_annotations[factor_name]

                if color_edges:
                    color = matplotlib.colors.to_hex(layer_colors[factor_idx])
                fillcolor = "#FFFFFF"
                if style == "filled":
                    if filled == "factor":
                        fillcolor = matplotlib.colors.to_hex(layer_colors[factor_idx])
                    elif filled is not None:
                        # cells attached to this factor
                        original_factor_index = self.factor_lists[layer_idx][factor_idx]
                        cells = np.where(
                            self.adata.obs[f"{layer_name}factor"] == factor_name
                        )[0]
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
                    fillcolor = fillcolor + alpha
                elif style == "wedged":
                    # cells attached to this factor
                    original_factor_index = self.factor_lists[layer_idx][factor_idx]
                    cells = np.where(
                        self.adata.obs[f"{layer_name}factor"] == factor_name
                    )[0]
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
                    factor_gene_rankings = gene_rankings[factor_idx][
                        : top_genes[layer_idx]
                    ]
                    factor_gene_scores = gene_scores[factor_idx][: top_genes[layer_idx]]
                    fontsizes = map_scores_to_fontsizes(
                        gene_scores[factor_idx], **fontsize_kwargs
                    )[: top_genes[layer_idx]]
                    gene_labels = []
                    for j, gene in enumerate(factor_gene_rankings):
                        gene_labels.append(
                            f'<FONT POINT-SIZE="{fontsizes[j]}">{gene}</FONT>'
                        )
                    label += "<br/><br/>" + "<br/>".join(gene_labels)

                    if show_confidences:
                        confidence_score = self.get_signature_confidence(
                            factor_idx,
                            layer_idx,
                            top_genes=top_genes[layer_idx],
                            mc_samples=mc_samples,
                        )
                        label += f"<br/><br/>({confidence_score:.3f})"

                elif filled is not None and filled != "factor":
                    label += "<br/><br/>" + ""

                label = "<" + label + ">"
                g.node(
                    factor_name,
                    label=label,
                    fillcolor=fillcolor,
                    color=color,
                    ordering=ordering,
                    style=style,
                )

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
                        mat = self.pmeans[f"{self.layer_names[layer_idx]}W"][
                            self.factor_lists[layer_idx]
                        ][:, self.factor_lists[layer_idx - 1]]
                        normalized_factor_weights = mat / np.sum(mat, axis=1).reshape(
                            -1, 1
                        )
                        for lower_factor_idx in layer_factor_orders[layer_idx - 1]:
                            lower_factor_name = self.factor_names[layer_idx - 1][
                                lower_factor_idx
                            ]
                            normalized_weight = normalized_factor_weights[
                                factor_idx, lower_factor_idx
                            ]
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
                factor_name = f"{self.factor_names[layer_idx][int(factor_idx)]}"
                # cells attached to this factor
                cells = np.where(self.adata.obs[f"{layer_name}factor"] == factor_name)[
                    0
                ]
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

    def plot_brd(
        self,
        thres=None,
        iqr_mult=3.0,
        show_yticks=False,
        scale="linear",
        normalize=True,
        show=True,
        **kwargs,
    ):
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
            median = np.median(scales)
            q3 = np.percentile(scales, 75)
            cutoff = ard * (q3 - median)
        else:
            cutoff = ard

        fig = plt.figure(**kwargs)
        plt.axhline(cutoff, color="red", ls="--")
        above = np.where(scales >= cutoff)[0]
        below = np.where(scales < cutoff)[0]
        plt.bar(np.arange(layer_size)[above], scales[above])
        plt.bar(np.arange(layer_size)[below], scales[below], alpha=0.6, color="gray")
        if len(scales) > 15:
            plt.xticks(np.arange(0, layer_size, 2))
        else:
            plt.xticks(np.arange(layer_size))
        if not show_yticks:
            plt.yticks([])
        plt.title(f"Biological relevance determination")
        plt.xlabel("Factor")
        plt.yscale(scale)
        plt.ylabel("Relevance")
        if show:
            plt.show()

    def plot_obs_factor_dotplot(
        self,
        obs_key,
        layer_idx,
        figsize=(8, 2),
        s_min=100,
        s_max=500,
        titlesize=12,
        labelsize=12,
        legend_fontsize=12,
        legend_titlesize=12,
        cmap="viridis",
        logged=False,
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
                        self.adata.obs.loc[cells_from_obs][f"{layer_name}factor"]
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

        x, y = np.meshgrid(np.arange(len(xlabels)), np.arange(len(ylabels)))

        fig, axes = plt.subplots(1, 3, figsize=figsize, width_ratios=[5, 1, 1])
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

        plt.ylabel(obs_key, fontsize=labelsize)
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
        neighbors_rep="X_factors",
        figsize=(16, 4),
        reuse_pos=True,
        show=True,
        **paga_kwargs,
    ):
        "Plot PAGA graphs at all scDEF levels"

        fig, axes = plt.subplots(1, self.n_layers, figsize=figsize)
        sc.pp.neighbors(self.adata, use_rep=neighbors_rep)
        pos = None
        for i, layer_idx in enumerate(range(self.n_layers - 1, -1, -1)):
            ax = axes[i]
            new_layer_name = f"{self.layer_names[layer_idx]}factor"

            self.logger.info(f"Computing PAGA graph of layer {layer_idx}")

            # Use previous PAGA as initial positions for new PAGA
            if layer_idx != self.n_layers - 1 and reuse_pos:
                self.logger.info(
                    f"Re-using PAGA positions from layer {layer_idx+1} to init {layer_idx}"
                )
                matches = sc._utils.identify_groups(
                    self.adata.obs[new_layer_name], self.adata.obs[old_layer_name]
                )
                pos = []
                np.random.seed(0)
                for i, c in enumerate(self.adata.obs[new_layer_name].cat.categories):
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
                title=f"Layer {layer_idx} PAGA",
                show=False,
                **paga_kwargs,
            )

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

    def get_hierarchy(self):
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
        return hierarchy

    def simplify_hierarchy(self, hierarchy):
        layer_sizes = [len(self.factor_names[idx]) for idx in range(self.n_layers)]
        return hierarchy_utils.simplify_hierarchy(
            hierarchy, self.layer_names, layer_sizes
        )

    def compute_factor_obs_association_score(
        self, layer_idx, factor_name, obs_key, obs_val
    ):
        layer_name = self.layer_names[layer_idx]

        # Cells attached to factor
        adata_cells_in_factor = self.adata[
            np.where(self.adata.obs[f"{layer_name}factor"] == factor_name)[0]
        ]

        # Cells from obs_val
        adata_cells_from_obs = self.adata[
            np.where(self.adata.obs[obs_key] == obs_val)[0]
        ]

        cells_from_obs = float(adata_cells_from_obs.shape[0])

        # Number of cells from obs_val that are not in factor
        cells_not_in_factor_from_obs = float(
            np.count_nonzero(
                adata_cells_from_obs.obs[f"{layer_name}factor"] != factor_name
            )
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

    def assign_obs_to_factors(self, obs_keys, factor_names=None):
        if not isinstance(obs_keys, list):
            obs_keys = [obs_keys]

        obs_to_factor_assignments = []
        obs_to_factor_matches = []
        for obs_key in obs_keys:
            obskey_to_factor_assignments = dict()
            obskey_to_factor_matches = dict()
            for obs in self.adata.obs[obs_key].unique():
                scores, factors, layers = self.get_factor_obs_association_scores(
                    obs_key, obs
                )
                if factor_names is not None:
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
        vmax=1.0,
        vmin=0.0,
        cb_title="",
        cb_title_fontsize=10,
        pad=0.1,
        shrink=0.7,
        show=True,
    ):
        if not isinstance(obs_keys, list):
            obs_keys = [obs_keys]

        if sort_layer_factors:
            layer_factor_orders = self.get_layer_factor_orders()
        else:
            layer_factor_orders = [
                np.arange(len(self.factor_lists[i])) for i in range(self.n_layers)
            ]

        n_factors = [len(self.factor_lists[idx]) for idx in range(self.n_layers)]
        n_obs = [len(obs_clusters[obs_key]) for obs_key in obs_keys]
        fig, axs = plt.subplots(
            len(obs_keys),
            self.n_layers,
            figsize=(10, 4),
            gridspec_kw={"width_ratios": n_factors, "height_ratios": n_obs},
        )
        axs = axs.reshape((len(obs_keys), self.n_layers))
        for i in range(self.n_layers):
            axs[0][i].set_title(f"Layer {i}")
            for j, obs_key in enumerate(obs_keys):
                ax = axs[j][i]
                mat = obs_mats[obs_key][i]
                mat = mat[obs_clusters[obs_key]][:, layer_factor_orders[i]]
                axplt = ax.pcolormesh(mat, vmax=vmax, vmin=vmin)

                if j == len(obs_keys) - 1:
                    xlabels = self.factor_names[i]
                    xlabels = np.array(xlabels)[layer_factor_orders[i]]
                    ax.set(
                        xticks=np.arange(len(xlabels)) + 0.5,
                        xticklabels=xlabels,
                    )
                else:
                    ax.set(xticks=[])

                if i == 0:
                    ylabels = np.array(obs_vals_dict[obs_keys[j]])[
                        obs_clusters[obs_keys[j]]
                    ]
                    ax.set(yticks=np.arange(len(ylabels)) + 0.5, yticklabels=ylabels)
                else:
                    ax.set(yticks=[])

                if i == self.n_layers - 1:
                    ax.yaxis.set_label_position("right")
                    ax.set_ylabel(obs_key, rotation=270, labelpad=20.0)

        plt.subplots_adjust(wspace=0.05)
        plt.subplots_adjust(hspace=0.05)

        cb = fig.colorbar(axplt, ax=axs.ravel().tolist(), pad=pad, shrink=shrink)
        cb.ax.set_title(cb_title, fontsize=cb_title_fontsize)
        if show:
            plt.show()

    def plot_signatures_scores(
        self, obs_keys, markers, top_genes=10, hierarchy=None, **kwargs
    ):
        obs_mats, obs_clusters, obs_vals_dict = self._prepare_obs_factor_scores(
            obs_keys,
            self._get_signature_scores,
            markers=markers,
            top_genes=top_genes,
            hierarchy=hierarchy,
        )
        self.plot_layers_obs(obs_keys, obs_mats, obs_clusters, obs_vals_dict, **kwargs)

    def plot_obs_scores(self, obs_keys, hierarchy=None, **kwargs):
        obs_mats, obs_clusters, obs_vals_dict = self._prepare_obs_factor_scores(
            obs_keys,
            self._get_assignment_scores,
            hierarchy=hierarchy,
        )
        self.plot_layers_obs(obs_keys, obs_mats, obs_clusters, obs_vals_dict, **kwargs)
