from ._scdef import scDEF

import jax.numpy as jnp
import numpy as np
import logging


class iscDEF(scDEF):
    def __init__(
        self,
        adata,
        markers_dict,
        add_other=True,  # whether to add factors for cells which don't express any of the sets in markers_matrix
        markers_layer=0,  # by default, use lower layer and learn a hierarchy
        n_factors_per_set=2,
        n_layers=2,
        gs_big_scale=1000.0,
        cn_big_scale=1000.0,
        gene_set_strength=1000.0,
        **kwargs,
    ):
        self.markers_dict = markers_dict
        self.add_other = add_other
        self.markers_layer = markers_layer
        self.n_layers = n_layers
        self.n_factors_per_set = n_factors_per_set
        self.gs_big_scale = gs_big_scale
        self.cn_big_scale = cn_big_scale
        self.gene_set_strength = gene_set_strength

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
                        np.ceil(self.n_markers / (self.n_factors_per_set * layer))
                    )
                    name = "h" * layer
                    if size <= 1:
                        break
                layer_sizes.append(size)
                layer_names.append(name)
            self.n_layers = len(layer_sizes)
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

        super(iscDEF, self).__init__(
            adata, layer_sizes=layer_sizes, layer_shapes=0.3, use_brd=False, **kwargs
        )

        logginglevel = self.logger.level
        self.logger = logging.getLogger("iscDEF")
        self.logger.setLevel(logginglevel)

        self.layer_names = layer_names

        # Set w_priors
        if self.markers_layer != 0:
            # Do connectivities
            cn_small_scale = 0.1
            for layer_idx in range(1, self.n_layers):
                connectivity_matrix = cn_small_scale * np.ones(
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
                    ] = self.cn_big_scale
                self.w_priors[layer_idx][0] = jnp.array(connectivity_matrix)
                self.w_priors[layer_idx][1] = jnp.array(
                    np.ones(connectivity_matrix.shape)
                )

        # Do gene sets
        gs_small_scale = 1.0 / self.gs_big_scale
        self.gene_sets = np.ones((self.layer_sizes[0], self.n_genes)) * gs_small_scale
        self.marker_gene_locs = []
        for i, cellgroup in enumerate(self.marker_names):
            if cellgroup == "other":
                continue

            factors_start = i
            factors_end = i + 1
            if self.markers_layer != 0:
                factors_start = i * self.n_factors_per_set * self.n_layers
                factors_end = (i + 1) * self.n_factors_per_set * self.n_layers
            for gene in self.markers_dict[cellgroup]:
                loc = np.where(self.adata.var.index == gene)[0]
                if len(loc) == 0:
                    self.logger.warning(
                        f"Did not find gene {gene} for set {cellgroup} in AnnData object."
                    )
                self.marker_gene_locs.append(loc)
                self.gene_sets[factors_start:factors_end, loc] = self.gs_big_scale

        self.w_priors[0][0] = jnp.array(
            self.gene_set_strength * self.gene_sets
        )  # shape
        self.w_priors[0][1] = jnp.array(
            self.gene_set_strength * np.ones(self.gene_sets.shape)
        )  # rate

        self.init_var_params()
        self.set_posterior_means()
        self.set_factor_names()

    def __repr__(self):
        out = f"iscDEF object with {self.n_layers} layers"
        out += "\n\t" + "Markers layer: " + str(self.markers_layer)
        out += "\n\t" + "Contains `other` category: " + str(self.add_other)
        out += "\n\t" + "Gene set strength: " + str(self.gene_set_strength)
        out += "\n\t" + "Gene set scale: " + str(self.gs_big_scale)
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
            out += "\n\t" + "Connectivity scale: " + str(self.cn_big_scale)
        out += "\n\t" + "Number of batches: " + str(self.n_batches)
        out += "\n" + "Contains " + self.adata.__str__()
        return out

    def set_factor_names(self):
        self.factor_names = []

        for idx in range(self.n_layers):
            layer_name = self.layer_names[idx]
            if self.markers_layer == 0:
                if idx == 0:
                    self.factor_names.append(self.marker_names)
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
                        sub_factors = np.arange(self.n_factors_per_set * rev_idx)
                        filtered_sub_factors = [
                            factor
                            for factor in sub_factors
                            if factor * marker_idx in self.factor_lists[idx]
                        ]
                        for sub_factor in range(len(filtered_sub_factors)):
                            marker_factor_names.append(
                                marker_name + f"_{self.layer_names[idx]}{sub_factor}"
                            )
                        factor_names += marker_factor_names
                self.factor_names.append(factor_names)
