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
            int
        ] = 0,  # whether to add factors for cells which don't express any of the sets in markers_matrix
        markers_layer: Optional[
            int
        ] = 0,  # by default, use lower layer and learn a hierarchy
        cn_small_mean: Optional[float] = 1e-6,
        cn_big_mean: Optional[float] = 1.0,
        cn_small_strength: Optional[float] = 1.0,
        cn_big_strength: Optional[float] = 0.1,
        gs_small_scale: Optional[float] = 1.0,
        gs_big_scale: Optional[float] = 100.0,
        marker_strength: Optional[float] = 10.0,
        nonmarker_strength: Optional[float] = 0.1,
        other_strength: Optional[float] = 0.1,
        **kwargs,
    ):
        self.markers_dict = markers_dict
        self.add_other = add_other
        self.markers_layer = markers_layer

        # Set w_priors
        self.cn_small_strength = cn_small_strength
        self.cn_big_strength = cn_big_strength
        self.cn_small_mean = cn_small_mean
        self.cn_big_mean = cn_big_mean

        self.gs_big_scale = gs_big_scale
        self.gs_small_scale = gs_small_scale
        self.marker_strength = marker_strength
        self.nonmarker_strength = nonmarker_strength
        self.other_strength = other_strength

        # Marker names and n_markers logic
        self.other_names = (
            [f"other{i}" for i in range(self.add_other)] if self.add_other > 0 else []
        )
        self.marker_names = list(self.markers_dict.keys()) + self.other_names
        self.n_markers = len(self.marker_names)

        self.decay_factor = (
            2 if "decay_factor" not in kwargs else kwargs["decay_factor"]
        )
        self.max_n_layers = (
            5 if "max_n_layers" not in kwargs else kwargs["max_n_layers"]
        )
        if markers_layer == 0:
            if "use_brd" not in kwargs:
                kwargs["use_brd"] = False
                self.use_brd = False
            elif kwargs["use_brd"] == True:
                raise ValueError("`use_brd` must be False if markers_layer is 0")
        else:
            if "use_brd" not in kwargs:
                self.use_brd = True
            else:
                self.use_brd = kwargs["use_brd"]
            if add_other > 1:
                raise ValueError("`add_other` can be at most 1 if markers_layer is > 0")

        if self.use_brd:
            self.nonmarker_strength = 1.0

        self.set_layer_sizes()

        super(iscDEF, self).__init__(
            adata, layer_sizes=self.layer_sizes, layer_names=self.layer_names, **kwargs
        )

        logginglevel = self.logger.level
        self.logger = logging.getLogger("iscDEF")
        self.logger.setLevel(logginglevel)

        self.set_geneset_prior(
            gs_big_scale=self.gs_big_scale,
            gs_small_scale=self.gs_small_scale,
            marker_strength=self.marker_strength,
            nonmarker_strength=self.nonmarker_strength,
            other_strength=self.other_strength,
        )

        self.init_var_params()
        self.set_posterior_means()
        self.set_factor_names()

    def set_layer_sizes(self):
        layer_sizes = []
        layer_names = []
        if self.markers_layer == 0:
            self.update_model_size(self.n_markers, self.max_n_layers)
            layer_names = self.layer_names
            layer_sizes = self.layer_sizes
            layer_names[0] = "marker"

            if self.use_brd:
                raise ValueError("`use_brd` must be False if markers_layer is 0")
        else:
            # For markers_layer > 0, only the bottom layer (layer 0) gets "other" factors
            self.n_layers = self.markers_layer + 1
            self.n_factors_per_marker = int(self.decay_factor) * (self.n_layers - 1)
            if self.max_n_layers == 1:
                self.n_layers = 1
            if self.n_layers == 1:
                size = self.n_markers * self.n_factors_per_marker
                name = "marker"
                layer_sizes.append(size)
                layer_names.append(name)
            else:
                for layer in range(self.n_layers):
                    if layer < self.n_layers - 1:
                        rev_layer = (self.n_layers - 1) - layer
                        size = self.n_markers * int(self.decay_factor) * rev_layer
                        name = f"L{layer}"
                    else:
                        size = self.n_markers
                        name = "marker"
                    layer_sizes.append(size)
                    layer_names.append(name)

        self.layer_sizes = layer_sizes
        self.layer_names = layer_names

    def update_model_priors(self):
        super(iscDEF, self).update_model_priors()
        if self.markers_layer != 0 and self.n_layers > 1:
            self.set_connectivity_prior(
                cn_small_strength=self.cn_small_strength,
                cn_big_strength=self.cn_big_strength,
                cn_small_mean=self.cn_small_mean,
                cn_big_mean=self.cn_big_mean,
            )
        self.set_geneset_prior(
            gs_big_scale=self.gs_big_scale,
            gs_small_scale=self.gs_small_scale,
            marker_strength=self.marker_strength,
            nonmarker_strength=self.nonmarker_strength,
            other_strength=self.other_strength,
        )

    def __repr__(self):
        out = f"iscDEF object with {self.n_layers} layers"
        out += "\n\t" + "Markers layer: " + str(self.markers_layer)
        out += "\n\t" + "Contains `other` category: " + str(self.add_other)
        out += "\n\t" + "Gene set strength: " + str(self.marker_strength)
        out += "\n\t" + "Gene set mean: " + str(self.gs_big_scale)
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
        if self.markers_layer == 0:
            out += (
                "\n\t"
                + "Layer factor shape parameters: "
                + ", ".join([str(shape) for shape in self.factor_shapes])
            )
        else:
            out += "\n\t" + "Connectivity mean: " + str(self.cn_big_mean)
        if self.use_brd == True:
            out += "\n\t" + "Using BRD"
        out += "\n\t" + "Number of batches: " + str(self.n_batches)
        out += "\n" + "Contains " + self.adata.__str__()
        return out

    def set_connectivity_prior(
        self,
        cn_small_mean: Optional[float] = 1e-6,
        cn_big_mean: Optional[float] = 1.0,
        cn_small_strength: Optional[float] = 1.0,
        cn_big_strength: Optional[float] = 0.1,
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
                n_local_factors_per_set = int(self.decay_factor) * (layer_rev_idx)
            n_lower_factors_per_set = int(self.decay_factor) * (layer_rev_idx + 1)

            n_marker_factors = self.n_markers
            for i in range(n_marker_factors):
                upper_start = i * n_local_factors_per_set
                upper_end = (i + 1) * n_local_factors_per_set

                local_start = i * n_lower_factors_per_set
                local_end = (i + 1) * n_lower_factors_per_set

                connectivity_matrix[upper_start:upper_end, local_start:local_end] = (
                    cn_big_mean
                )

                strength_matrix[upper_start:upper_end, local_start:local_end] = (
                    cn_big_strength * self.layer_sizes[layer_idx] / self.layer_sizes[0]
                )

            # If "other" factors are present in layer 0, connect them weakly to upper layers (or not at all)
            if self.add_other > 0 and layer_idx > 0:
                other_start = (
                    self.layer_sizes[layer_idx]
                    - self.add_other * n_local_factors_per_set
                )
                other_end = self.layer_sizes[layer_idx]
                connectivity_matrix[other_start:other_end, :] = cn_big_mean
                strength_matrix[other_start:other_end, :] = (
                    cn_big_strength * self.layer_sizes[layer_idx] / self.layer_sizes[0]
                )

            self.w_priors[layer_idx][0] = strength_matrix
            self.w_priors[layer_idx][1] = strength_matrix / connectivity_matrix

    def set_geneset_prior(
        self,
        gs_small_scale: Optional[float] = 1.0,
        gs_big_scale: Optional[float] = 100.0,
        marker_strength: Optional[float] = 10.0,
        nonmarker_strength: Optional[float] = 0.1,
        other_strength: Optional[float] = 0.1,
    ):
        self.gs_big_scale = gs_big_scale
        self.gs_small_scale = gs_small_scale
        self.marker_strength = marker_strength
        self.nonmarker_strength = nonmarker_strength
        self.other_strength = other_strength

        # Do gene sets
        n_factors_layer0 = self.layer_sizes[0]
        self.gene_sets = np.ones((n_factors_layer0, self.n_genes)) * gs_small_scale
        self.strengths = np.ones((n_factors_layer0, self.n_genes)) * nonmarker_strength
        self.marker_gene_locs = []

        if self.markers_layer == 0:
            marker_names = self.marker_names
            marker_dict = self.markers_dict
            n_marker_factors = len(marker_names)
        else:
            marker_names = self.marker_names
            marker_dict = self.markers_dict
            n_marker_factors = self.n_markers

        # Assign marker gene priors for marker factors
        for i, cellgroup in enumerate(marker_names):
            factors_start = i
            factors_end = i + 1
            if self.markers_layer != 0:
                factors_start = i * self.n_factors_per_marker
                factors_end = (i + 1) * self.n_factors_per_marker

            if "other" not in cellgroup:
                for gene in marker_dict[cellgroup]:
                    loc = np.where(self.adata.var.index == gene)[0]
                    if len(loc) == 0:
                        self.logger.warning(
                            f"Did not find gene {gene} for set {cellgroup} in AnnData object."
                        )
                    self.marker_gene_locs.append(loc)
                    self.gene_sets[factors_start:factors_end, loc] = self.gs_big_scale
                    self.strengths[factors_start:factors_end, loc] = marker_strength

            # Make it hard for the factors in this group to give weight to genes in another group
            for group in marker_dict:
                if group != cellgroup:
                    for gene in marker_dict[group]:
                        if "other" not in cellgroup:
                            if gene not in marker_dict[cellgroup]:
                                loc = np.where(self.adata.var.index == gene)[0]
                                self.gene_sets[factors_start:factors_end, loc] = 1e-6
                                self.strengths[factors_start:factors_end, loc] = (
                                    other_strength
                                )
                        else:
                            loc = np.where(self.adata.var.index == gene)[0]
                            self.gene_sets[factors_start:factors_end, loc] = 1e-6
                            self.strengths[factors_start:factors_end, loc] = (
                                other_strength
                            )

        if self.n_layers == 1:
            self.w_priors = [
                [jnp.array(self.strengths), jnp.array(self.strengths / self.gene_sets)]
            ]
        else:
            self.w_priors[0] = [
                jnp.array(self.strengths),
                jnp.array(self.strengths / self.gene_sets),
            ]

    def set_factor_names(self):
        self.factor_names = []

        for idx in range(self.n_layers):
            if self.markers_layer == 0:
                if idx == 0:
                    self.factor_names.append(
                        [f"{self.marker_names[i]}" for i in self.factor_lists[idx]]
                    )
                else:
                    self.factor_names.append(
                        [
                            f"{self.layer_names[idx]}_{str(i)}"
                            for i in range(len(self.factor_lists[idx]))
                        ]
                    )
            else:
                # For markers_layer > 0, only marker_names are in upper layers, "other" only in layer 0
                if self.n_layers == 1:
                    factor_names = []
                    for marker_idx, marker_name in enumerate(self.marker_names):
                        marker_factor_names = []
                        sub_factors = np.arange(
                            marker_idx * self.n_factors_per_marker,
                            (marker_idx + 1) * self.n_factors_per_marker,
                        )
                        filtered_sub_factors = [
                            factor
                            for factor in sub_factors
                            if factor in self.factor_lists[idx]
                        ]
                        for sub_factor in range(len(filtered_sub_factors)):
                            marker_factor_names.append(
                                marker_name + f"_{self.layer_names[idx]}_{sub_factor}"
                            )
                        factor_names += marker_factor_names
                elif idx == self.n_layers - 1:
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
                            marker_idx * int(self.decay_factor) * rev_idx,
                            (marker_idx + 1) * int(self.decay_factor) * rev_idx,
                        )
                        filtered_sub_factors = [
                            factor
                            for factor in sub_factors
                            if factor in self.factor_lists[idx]
                        ]
                        for sub_factor in range(len(filtered_sub_factors)):
                            marker_factor_names.append(
                                marker_name + f"_{self.layer_names[idx]}_{sub_factor}"
                            )
                        factor_names += marker_factor_names
                    # For layer 0, add "other" factors at the end
                self.factor_names.append(factor_names)

    def fit(self, pretrain=False, nmf_init=False, max_cells_init=1024, **kwargs):
        if pretrain:
            self.logger.info(f"Pretraining to find initial set of factors")
            max_n_layers = self.max_n_layers
            self.max_n_layers = 1
            self.set_layer_sizes()
            self.update_model_priors()
            self.init_var_params(
                init_budgets=True, nmf_init=nmf_init, max_cells=max_cells_init
            )
            self.elbos = []
            self.step_sizes = []
            self._learn(filter=False, annotate=False, **kwargs)
            self.logger.info(f"scDEF pretraining finished.")
            self.max_n_layers = max_n_layers
            self.set_layer_sizes()
            # self.update_model_size(self.n_factors)
            self.update_model_priors()
            init_budgets = False
            init_w = self.pmeans["markerW"]
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
