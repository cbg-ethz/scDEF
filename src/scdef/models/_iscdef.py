from scdef.utils import score_utils, hierarchy_utils, color_utils
from scdef.utils.jax_utils import *

from jax import jit, grad, vmap
from jax.example_libraries import optimizers
from jax import random, value_and_grad
from jax.scipy.stats import norm, gamma, poisson
import jax.numpy as jnp
import jax.nn as jnn
import jax

from scdef.models._scdef import scDEF

from anndata import AnnData
import jax.numpy as jnp
import numpy as np
import logging

from typing import Optional, Sequence, Mapping


class iscDEF(scDEF):
    """Informed scDEF model.

    This model extends the basic scDEF by using gene sets to guide the factors.
    iscDEF can either set the given sets as top layer factors and learn higher-resolution
    structure, or use them as the lowest resolution and learn a hierarchy that relates them.
    All the methods from scDEF are available in iscDEF.

    Args:
        adata: AnnData object containing the gene expression data. scDEF learns a model from
            counts, so they must be present in either adata.X or in adata.layers.
        markers_dict: dictionary containing named gene lists.
        add_other: whether to add factors for cells which don't express any of the sets in markers_dict.
        markers_layer: scDEF layer at which the gene sets are defined. If > 0, this defines the number of layers.
        n_factors_per_set: number of lower level factors per gene set.
        n_layers: default number of scDEF layers, including a top layer of size 1 if markers_layer is 0.
        cn_small_scale: scale for low connectivity
        cn_big_scale: scale for large connectivity
        cn_small_strength: strength for weak connectivity
        cn_big_strength: strength for large connectivity
        gs_small_scale: scale for genes not in set
        gs_big_scale: scale for genes in set
        marker_strength: strength for marker genes
        nonmarker_strength: strength for non-marker genes
        other_strength: strength for marker genes of other sets
        **kwargs: keyword arguments for base scDEF.
    """

    def __init__(
        self,
        adata: AnnData,
        markers_dict: Mapping[str, Sequence[str]],
        add_other: Optional[
            bool
        ] = False,  # whether to add factors for cells which don't express any of the sets in markers_matrix
        markers_layer: Optional[
            int
        ] = 0,  # by default, use lower layer and learn a hierarchy
        n_factors_per_set: Optional[int] = 3,
        n_sets_per_factor: Optional[int] = 1.5,
        n_layers: Optional[int] = 4,
        cn_small_mean: Optional[float] = 0.01,
        cn_big_mean: Optional[float] = 10.0,
        cn_small_strength: Optional[float] = 100.0,
        cn_big_strength: Optional[float] = 1.0,
        gs_small_scale: Optional[float] = 0.1,
        gs_big_scale: Optional[float] = 10.0,
        marker_strength: Optional[float] = 100.0,
        nonmarker_strength: Optional[float] = 1.0,
        other_strength: Optional[float] = 100.0,
        **kwargs,
    ):
        self.markers_dict = markers_dict
        self.add_other = add_other
        self.markers_layer = markers_layer
        self.n_layers = n_layers
        self.n_factors_per_set = n_factors_per_set
        self.n_sets_per_factor = n_sets_per_factor
        self.gs_big_scale = gs_big_scale
        self.cn_big_strength = cn_big_strength

        self.marker_names = list(self.markers_dict.keys()) + ["other"] * self.add_other
        self.n_markers = len(self.marker_names)

        layer_sizes = []
        layer_names = []
        if markers_layer == 0:
            for layer in range(self.n_layers):
                if layer == 0:
                    size = self.n_markers
                    name = "marker"
                else:
                    size = int(
                        np.ceil(self.n_markers / (self.n_sets_per_factor * layer))
                    )
                    name = "h" * layer
                    if size <= 1:
                        break
                layer_sizes.append(size)
                layer_names.append(name)
            layer_sizes.append(1)
            layer_names.append("h" * (layer + 1))
            self.n_layers = len(layer_sizes)

            if "layer_shapes" not in kwargs:
                kwargs["layer_shapes"] = [0.3] * (self.n_layers - 1) + [1.0]
            if "factor_shapes" not in kwargs:
                kwargs["factor_shapes"] = [1.0] * self.n_layers
            if "factor_rates" not in kwargs:
                kwargs["factor_rates"] = [1.0] * self.n_layers

            if "use_brd" not in kwargs:
                kwargs["use_brd"] = False
            elif kwargs["use_brd"] == True:
                raise ValueError("`use_brd` must be False if markers_layer is 0")
        else:
            self.n_layers = self.markers_layer + 1
            for layer in range(self.n_layers):
                if layer < self.n_layers - 1:
                    rev_layer = (self.n_layers - 1) - layer
                    size = self.n_markers * self.n_factors_per_set * rev_layer
                    name = "h" * layer
                else:
                    size = self.n_markers
                    name = "marker"
                layer_sizes.append(size)
                layer_names.append(name)

            if "layer_shapes" not in kwargs:
                kwargs["layer_shapes"] = 0.3
            if "factor_shapes" not in kwargs:
                kwargs["factor_shapes"] = 1.0
            if "factor_rates" not in kwargs:
                kwargs["factor_rates"] = 1.0

        super(iscDEF, self).__init__(adata, layer_sizes=layer_sizes, **kwargs)

        logginglevel = self.logger.level
        self.logger = logging.getLogger("iscDEF")
        self.logger.setLevel(logginglevel)

        self.layer_names = layer_names

        # Set w_priors
        if self.markers_layer != 0:
            self.set_connectivity_prior(
                cn_small_strength=cn_small_strength,
                cn_big_strength=cn_big_strength,
                cn_small_mean=cn_small_mean,
                cn_big_mean=cn_big_mean,
            )

        # Do gene sets
        self.set_geneset_prior(
            gs_big_scale=gs_big_scale,
            gs_small_scale=gs_small_scale,
            marker_strength=marker_strength,
            nonmarker_strength=nonmarker_strength,
            other_strength=other_strength,
        )

        self.init_var_params()
        self.set_posterior_means()
        self.set_factor_names()

    def __repr__(self):
        out = f"iscDEF object with {self.n_layers} layers"
        out += "\n\t" + "Markers layer: " + str(self.markers_layer)
        out += "\n\t" + "Contains `other` category: " + str(self.add_other)
        out += "\n\t" + "Gene set strength: " + str(self.marker_strength)
        out += "\n\t" + "Gene set mean: " + str(self.gs_big_scale)
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
            + "Layer rate parameters: "
            + ", ".join([str(rate) for rate in self.layer_rates])
        )
        if self.markers_layer == 0:
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
        else:
            out += "\n\t" + "Connectivity mean: " + str(self.cn_big_mean)
        out += "\n\t" + "Number of batches: " + str(self.n_batches)
        out += "\n" + "Contains " + self.adata.__str__()
        return out

    def set_connectivity_prior(
        self,
        cn_small_strength=100.0,
        cn_big_strength=1.0,
        cn_small_mean=0.01,
        cn_big_mean=1.0,
    ):
        self.cn_small_strength = cn_small_strength
        self.cn_big_strength = cn_big_strength
        self.cn_small_mean = cn_small_mean
        self.cn_big_mean = cn_big_mean

        # Do connectivities
        for layer_idx in range(1, self.n_layers):
            connectivity_matrix = cn_small_mean * np.ones(
                (self.layer_sizes[layer_idx], self.layer_sizes[layer_idx - 1])
            )
            strength_matrix = cn_small_strength * np.ones(
                (self.layer_sizes[layer_idx], self.layer_sizes[layer_idx - 1])
            )

            layer_rev_idx = self.n_layers - 1 - layer_idx

            if layer_idx == self.n_layers - 1:
                n_local_factors_per_set = 1
            else:
                n_local_factors_per_set = self.n_factors_per_set * (layer_rev_idx)
            n_lower_factors_per_set = self.n_factors_per_set * (layer_rev_idx + 1)
            for i in range(len(self.marker_names)):
                upper_start = i * n_local_factors_per_set
                upper_end = (i + 1) * n_local_factors_per_set

                local_start = i * n_lower_factors_per_set
                local_end = (i + 1) * n_lower_factors_per_set

                connectivity_matrix[
                    upper_start:upper_end, local_start:local_end
                ] = cn_big_mean

                strength_matrix[
                    upper_start:upper_end, local_start:local_end
                ] = cn_big_strength

            self.w_priors[layer_idx][0] = strength_matrix
            self.w_priors[layer_idx][1] = strength_matrix / connectivity_matrix

    def set_geneset_prior(
        self,
        gs_big_scale=10.0,
        gs_small_scale=0.1,
        marker_strength=100.0,
        nonmarker_strength=1.0,
        other_strength=100.0,
    ):
        self.gs_big_scale = gs_big_scale
        self.gs_small_scale = gs_small_scale
        self.marker_strength = marker_strength
        self.nonmarker_strength = nonmarker_strength
        self.other_strength = other_strength

        # Do gene sets
        self.gene_sets = np.ones((self.layer_sizes[0], self.n_genes)) * gs_small_scale
        self.strengths = (
            np.ones((self.layer_sizes[0], self.n_genes)) * nonmarker_strength
        )
        self.marker_gene_locs = []
        for i, cellgroup in enumerate(self.marker_names):
            factors_start = i
            factors_end = i + 1
            if self.markers_layer != 0:
                factors_start = i * self.n_factors_per_set * (self.n_layers - 1)
                factors_end = (i + 1) * self.n_factors_per_set * (self.n_layers - 1)

            if cellgroup != "other":
                for gene in self.markers_dict[cellgroup]:
                    loc = np.where(self.adata.var.index == gene)[0]
                    if len(loc) == 0:
                        self.logger.warning(
                            f"Did not find gene {gene} for set {cellgroup} in AnnData object."
                        )
                    self.marker_gene_locs.append(loc)
                    self.gene_sets[factors_start:factors_end, loc] = self.gs_big_scale
                    self.strengths[factors_start:factors_end, loc] = marker_strength

            # Make it hard for the factors in this group to give weight to genes in another group
            for group in self.markers_dict:
                if group != cellgroup:
                    for gene in self.markers_dict[group]:
                        if cellgroup != "other":
                            if gene not in self.markers_dict[cellgroup]:
                                loc = np.where(self.adata.var.index == gene)[0]
                                self.gene_sets[factors_start:factors_end, loc] = (
                                    gs_small_scale / 100.0
                                )
                                self.strengths[
                                    factors_start:factors_end, loc
                                ] = other_strength
                        else:
                            loc = np.where(self.adata.var.index == gene)[0]
                            self.gene_sets[factors_start:factors_end, loc] = (
                                gs_small_scale / 100.0
                            )
                            self.strengths[
                                factors_start:factors_end, loc
                            ] = other_strength

        self.w_priors[0][0] = jnp.array(self.strengths)  # shape
        self.w_priors[0][1] = jnp.array(self.strengths / self.gene_sets)  # rate

    def set_factor_names(self):
        self.factor_names = []

        for idx in range(self.n_layers):
            layer_name = self.layer_names[idx]
            if self.markers_layer == 0:
                if idx == 0:
                    self.factor_names.append(
                        [f"{self.marker_names[i]}" for i in self.factor_lists[idx]]
                    )
                else:
                    self.factor_names.append(
                        [
                            f"{self.layer_names[idx]}{str(i)}"
                            for i in range(len(self.factor_lists[idx]))
                        ]
                    )
            else:  # if not zero, it's the top one, so append gene set names to each subsequent layer
                if idx == self.n_layers - 1:
                    factor_names = [
                        marker
                        for i, marker in enumerate(self.marker_names)
                        if i in self.factor_lists[idx]
                    ]
                else:
                    rev_idx = self.n_layers - 1 - idx
                    factor_names = []
                    for marker_idx, marker_name in enumerate(self.marker_names):
                        marker_factor_names = []
                        sub_factors = np.arange(
                            marker_idx * self.n_factors_per_set * rev_idx,
                            (marker_idx + 1) * self.n_factors_per_set * rev_idx,
                        )
                        filtered_sub_factors = [
                            factor
                            for factor in sub_factors
                            if factor in self.factor_lists[idx]
                        ]
                        for sub_factor in range(len(filtered_sub_factors)):
                            marker_factor_names.append(
                                marker_name + f"_{self.layer_names[idx]}{sub_factor}"
                            )
                        factor_names += marker_factor_names
                self.factor_names.append(factor_names)

    def elbo(
        self,
        rng,
        batch,
        indices,
        var_params,
        annealing_parameter,
        stop_gradients,
        min_shape=1e-6,
        min_rate=1e-6,
        max_rate=1e6,
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

        gene_budget_shape = jnp.maximum(jnp.exp(gene_budget_params[0]), min_shape)
        gene_budget_rate = jnp.minimum(
            jnp.maximum(jnp.exp(gene_budget_params[1]), min_rate), max_rate
        )
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

        fscale_shapes = jnp.maximum(jnp.exp(fscale_params[0]), min_shape)
        fscale_rates = jnp.minimum(
            jnp.maximum(jnp.exp(fscale_params[1]), min_rate), max_rate
        )
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

        z_shapes = jnp.maximum(jnp.exp(z_params[0][indices]), min_shape)
        z_rates = jnp.minimum(
            jnp.maximum(jnp.exp(z_params[1][indices]), min_rate), max_rate
        )

        # Sample from variational distribution
        cell_budget_sample = gamma_sample(rng, cell_budget_shape, cell_budget_rate)
        gene_budget_sample = gamma_sample(rng, gene_budget_shape, gene_budget_rate)
        if self.use_brd:
            fscale_samples = gamma_sample(rng, fscale_shapes, fscale_rates)
        # z_samples = gamma_sample(rng, z_shapes, z_rates)  # vectorized
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
            global_pl += gamma_logpdf(fscale_samples, self.brd, self.brd * 100.0)
            global_en += -gamma_logpdf(fscale_samples, fscale_shapes, fscale_rates)

        z_mean = 1.0
        for idx in list(np.arange(0, self.n_layers)[::-1]):
            start = np.sum(self.layer_sizes[:idx]).astype(int)
            end = start + self.layer_sizes[idx]

            # w
            _w_shape = jnp.maximum(jnp.exp(var_params[4 + idx][0]), min_shape)
            _w_rate = jnp.minimum(
                jnp.maximum(jnp.exp(var_params[4 + idx][1]), min_rate), max_rate
            )

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
            _z_sample = gamma_sample(rng, _z_shape, _z_rate)

            if idx == self.n_layers - 1:
                local_pl += gamma_logpdf(
                    _z_sample, self.layer_shapes[idx], self.layer_rates[idx]
                )
            else:
                rate_param = self.layer_rates[idx]
                rate_param = jax.lax.cond(
                    stop_gradients[idx + 1],
                    lambda: rate_param * 0.0 + 1.0,
                    lambda: rate_param,
                )
                local_pl += gamma_logpdf(
                    _z_sample, self.layer_shapes[idx], rate_param / z_mean
                )
            local_en += -gamma_logpdf(_z_sample, _z_shape, _z_rate)

            z_mean = jnp.einsum("nk,kp->np", _z_sample, _w_sample)

            z_mean = jax.lax.cond(
                stop_gradients[idx], lambda: z_mean * 0.0 + 1, lambda: z_mean
            )

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
