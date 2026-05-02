from scdef.models._scdef import scDEF
from anndata import AnnData
import jax.numpy as jnp
import numpy as np
import logging

from typing import Optional, Sequence, Mapping


class iscDEF(scDEF):
    """Informed Single-cell Deep Exponential Families (iscDEF) model.

    iscDEF extends the scDEF framework by incorporating prior biological knowledge in the form of gene sets ("markers").
    This model can guide the discovery of factors along known biology, either by using gene sets as the highest-resolution
    (top) factors and learning finer substructure beneath them or as the coarsest layer to learn how they relate hierarchically.

    All methods and functionality available in scDEF are inherited by iscDEF. Additional logic allows for flexible
    integration of marker sets at a chosen model layer, custom prior settings for marker versus non-marker genes,
    and automatic handling of cells/gene sets that do not fall into any marker category (via the `add_other` option).

    Args:
        adata: AnnData object containing the gene expression count matrix. Counts must be present
            in either `adata.X` or a specified layer.
        markers_dict: dictionary mapping marker/factor names to gene lists (gene sets). These guide
            the formation of factors in the chosen layer.
        add_other: if > 0, adds one or more "other" factors for cells/observations not matching any
            marker set. Only one "other" factor is supported for `markers_layer > 0`.
        markers_layer: index of the layer at which gene sets are enforced as factors (0 = lowest/finest,
            higher = top layer). If > 0, total layers determined by this value.
        cn_small_mean: mean prior connectivity for "small" (weakly-connected) genes between factors and gene sets.
        cn_big_mean: mean prior connectivity for "big" (strongly-connected) genes between factors and gene sets.
        cn_small_strength: concentration parameter for low connectivity (see scDEF prior specification).
        cn_big_strength: concentration parameter for high connectivity.
        gs_small_scale: scale parameter for genes *not* in the marker gene set.
        gs_big_scale: scale parameter for genes *in* the marker gene set (encourages large factor loadings).
        marker_strength: multiplier for the prior strength for marker genes.
        nonmarker_strength: multiplier for non-marker gene prior strength.
        other_strength: prior strength for marker genes belonging to "other" sets.
        **kwargs: additional arguments passed to the scDEF base model.
    """

    def __init__(
        self,
        adata: AnnData,
        markers_dict: Mapping[str, Sequence[str]],
        add_other: Optional[int] = 0,
        markers_layer: Optional[int] = 0,
        cn_small_mean: Optional[float] = 1e-2,
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
            elif kwargs["use_brd"]:
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
        # Keep marker-aware names stable during annotation calls.
        self._preserve_factor_names_on_annotate = True

        self.set_geneset_prior(
            gs_big_scale=self.gs_big_scale,
            gs_small_scale=self.gs_small_scale,
            marker_strength=self.marker_strength,
            nonmarker_strength=self.nonmarker_strength,
            other_strength=self.other_strength,
        )

        self.init_var_params()
        self.set_posterior_means()
        self.set_posterior_variances()
        self.set_factor_names()
        self._top_factor_names = list(self.factor_names[-1])

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
            # Updated so that the layer sizes are exponential in decay_factor:
            # Layer 0 (lowest): self.n_markers * decay_factor**(n_layers-1), then decay_factor**(n_layers-2), ..., final (top) layer is n_markers.
            if self.max_n_layers == 1:
                self.n_layers = 1
            if self.n_layers == 1:
                # special case, all factors flat for each marker
                self.n_factors_per_marker = int(self.decay_factor)
                size = self.n_markers * self.n_factors_per_marker
                name = "marker"
                layer_sizes.append(size)
                layer_names.append(name)
            else:
                for layer in range(self.n_layers):
                    rev_layer = (self.n_layers - 1) - layer
                    size = self.n_markers * int(self.decay_factor) ** rev_layer
                    layer_sizes.append(size)
                    if layer == self.n_layers - 1:
                        name = "marker"
                    else:
                        name = f"L{layer}"
                    layer_names.append(name)
                # number of factors per marker in layer 0
                self.n_factors_per_marker = int(self.decay_factor) ** (
                    self.n_layers - 1
                )

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
        if self.use_brd:
            out += "\n\t" + "Using BRD"
        out += "\n\t" + "Number of batches: " + str(self.n_batches)
        out += "\n" + "Contains " + self.adata.__str__()
        return out

    def set_connectivity_prior(
        self,
        cn_small_mean: Optional[float] = 1e-3,
        cn_big_mean: Optional[float] = 1.0,
        cn_small_strength: Optional[float] = 1.0,
        cn_big_strength: Optional[float] = 0.1,
    ):
        self.cn_small_strength = cn_small_strength
        self.cn_big_strength = cn_big_strength
        self.cn_small_mean = cn_small_mean
        self.cn_big_mean = cn_big_mean

        # Ensure connectivity follows an exponential hierarchy: Each factor at layer L is connected to decay_factor child factors for the same marker at layer (L-1)
        for layer_idx in range(1, self.n_layers):
            upper_kept = np.asarray(self.factor_lists[layer_idx], dtype=int)
            lower_kept = np.asarray(self.factor_lists[layer_idx - 1], dtype=int)
            n_upper = len(upper_kept)
            n_lower = len(lower_kept)
            connectivity_matrix = cn_small_mean * np.ones((n_upper, n_lower))
            strength_matrix = cn_small_strength * np.ones((n_upper, n_lower))
            upper_pos = {factor_idx: pos for pos, factor_idx in enumerate(upper_kept)}
            lower_pos = {factor_idx: pos for pos, factor_idx in enumerate(lower_kept)}

            layer_rev_idx = self.n_layers - 1 - layer_idx  # How far from the bottom

            n_marker_factors = self.n_markers

            upper_factors_per_marker = int(self.decay_factor) ** layer_rev_idx
            lower_factors_per_marker = int(self.decay_factor) ** (layer_rev_idx + 1)

            for i in range(n_marker_factors):
                upper_start = i * upper_factors_per_marker
                upper_end = (i + 1) * upper_factors_per_marker
                lower_start = i * lower_factors_per_marker
                lower_end = (i + 1) * lower_factors_per_marker  # noqa: F841

                # For each upper factor for marker i, connect to its children in the lower layer
                for upper_factor in range(upper_start, upper_end):
                    if upper_factor not in upper_pos:
                        continue
                    # Each upper_factor is responsible for a group of decay_factor lower factors
                    child_block_size = (
                        lower_factors_per_marker // upper_factors_per_marker
                    )
                    child_start = (
                        lower_start + (upper_factor - upper_start) * child_block_size
                    )
                    child_end = child_start + child_block_size
                    child_positions = [
                        lower_pos[child_factor]
                        for child_factor in range(child_start, child_end)
                        if child_factor in lower_pos
                    ]
                    if len(child_positions) == 0:
                        continue
                    upper_row = upper_pos[upper_factor]
                    connectivity_matrix[upper_row, child_positions] = cn_big_mean
                    strength_matrix[upper_row, child_positions] = (
                        cn_big_strength * n_upper / self.layer_sizes[0]
                    )

            # If "other" factors are present in layer 0, connect them weakly to upper layers (or not at all)
            # (This logic is for "other" in the lowest layer only)
            if self.add_other > 0 and layer_idx > 0:
                # "Other" factors are always appended last in each layer
                # Compute for lower/upper layer sizes
                n_other_upper = 0  # noqa: F841
                n_other_lower = 0  # noqa: F841
                # Only the lowest layer gets extra "other" factors, but we allow for safety at all
                if layer_idx == 1:
                    # "other" at layer 0 (lowest)
                    other_start_lower = (
                        n_lower - self.add_other * lower_factors_per_marker
                    )
                    other_end_lower = n_lower
                    # All upper factors connect (weakly) to all "other" factors
                    connectivity_matrix[
                        :, other_start_lower:other_end_lower
                    ] = cn_big_mean
                    strength_matrix[:, other_start_lower:other_end_lower] = (
                        cn_big_strength * n_upper / self.layer_sizes[0]
                    )
                elif layer_idx == self.n_layers - 1:  # top layer can't have "other"
                    pass
                else:
                    # handle recursively if "other" present at each layer, but in current design, only at layer 0.
                    pass

            self.w_priors[layer_idx][0] *= strength_matrix
            self.w_priors[layer_idx][1] *= strength_matrix / np.maximum(
                connectivity_matrix, 1e-12
            )

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
            n_marker_factors = len(marker_names)  # noqa: F841
        else:
            marker_names = self.marker_names
            marker_dict = self.markers_dict
            n_marker_factors = self.n_markers  # noqa: F841

        # Assign marker gene priors for marker factors
        kept_l0 = np.asarray(self.factor_lists[0], dtype=int)
        for i, cellgroup in enumerate(marker_names):
            if self.markers_layer == 0:
                factors_rows = np.where(kept_l0 == i)[0]
            else:
                factors_start = i * self.n_factors_per_marker
                factors_end = (i + 1) * self.n_factors_per_marker
                factors_rows = np.where(
                    (kept_l0 >= factors_start) & (kept_l0 < factors_end)
                )[0]
            if len(factors_rows) == 0:
                continue

            if "other" not in cellgroup:
                for gene in marker_dict[cellgroup]:
                    loc = np.where(self.adata.var.index == gene)[0]
                    if len(loc) == 0:
                        self.logger.warning(
                            f"Did not find gene {gene} for set {cellgroup} in AnnData object."
                        )
                        continue
                    self.marker_gene_locs.append(loc)
                    self.gene_sets[factors_rows, loc] = self.gs_big_scale
                    self.strengths[factors_rows, loc] = marker_strength

            # Make it hard for the factors in this group to give weight to genes in another group
            for group in marker_dict:
                if group != cellgroup:
                    for gene in marker_dict[group]:
                        if "other" not in cellgroup:
                            if gene not in marker_dict[cellgroup]:
                                loc = np.where(self.adata.var.index == gene)[0]
                                if len(loc) == 0:
                                    continue
                                self.gene_sets[factors_rows, loc] = 1e-3
                                self.strengths[factors_rows, loc] = other_strength
                        else:
                            loc = np.where(self.adata.var.index == gene)[0]
                            if len(loc) == 0:
                                continue
                            self.gene_sets[factors_rows, loc] = 1e-3
                            self.strengths[factors_rows, loc] = other_strength

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
                    if hasattr(self, "_top_factor_names") and len(
                        self._top_factor_names
                    ) == len(self.factor_lists[idx]):
                        factor_names = list(self._top_factor_names)
                    else:
                        factor_names = [
                            marker
                            for i, marker in enumerate(self.marker_names)
                            if i in self.factor_lists[idx]
                        ]
                else:
                    rev_idx = self.n_layers - 1 - idx
                    factor_names = []
                    factors_per_marker = int(self.decay_factor) ** rev_idx
                    for marker_idx, marker_name in enumerate(self.marker_names):
                        marker_factor_names = []
                        sub_factors = np.arange(
                            marker_idx * factors_per_marker,
                            (marker_idx + 1) * factors_per_marker,
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
        if len(self.factor_names) > 0:
            self._top_factor_names = list(self.factor_names[-1])

    def filter_factors(
        self,
        brd_min: Optional[float] = 1.0,
        ard_min: Optional[float] = 0.001,
        clarity_min: Optional[float] = 0.5,
        n_eff_parents_max: float = 1.5,
        local_l0_scores: bool = False,
        min_cells_upper: Optional[float] = 0.001,
        min_cells_lower: Optional[float] = 0.0,
        filter_up: Optional[bool] = True,
        annotate: Optional[bool] = True,
        upper_only: Optional[bool] = False,
    ):
        """Filter factors while preserving existing marker-based factor names.

        This override keeps the base filtering behavior but restores names by
        subsetting the previous ``factor_names``. This avoids marker-prefix
        relabeling across filter/refit workflows.
        """
        prev_factor_names = (
            [list(names) for names in self.factor_names]
            if hasattr(self, "factor_names")
            else None
        )
        super().filter_factors(
            brd_min=brd_min,
            ard_min=ard_min,
            clarity_min=clarity_min,
            n_eff_parents_max=n_eff_parents_max,
            local_l0_scores=local_l0_scores,
            min_cells_upper=min_cells_upper,
            min_cells_lower=min_cells_lower,
            filter_up=filter_up,
            annotate=False,
            upper_only=upper_only,
        )

        if prev_factor_names is not None and len(prev_factor_names) == len(
            self.factor_lists
        ):
            corrected_names = []
            for idx, keep in enumerate(self.factor_lists):
                keep = np.asarray(keep, dtype=int)
                prev_names = prev_factor_names[idx]
                if len(prev_names) > 0 and np.all(
                    (keep >= 0) & (keep < len(prev_names))
                ):
                    corrected_names.append([prev_names[int(k)] for k in keep])
                else:
                    corrected_names.append(self.factor_names[idx])
            self.factor_names = corrected_names
            if len(self.factor_names) > 0:
                self._top_factor_names = list(self.factor_names[-1])

        if annotate:
            self.annotate_adata()

    def fit(
        self,
        nmf_init=False,
        max_cells_init=1024,
        z_init_concentration=100.0,
        **kwargs,
    ):
        """Fit iscDEF, warm-starting from previous fit when available.

        On refit, all layers are initialized from the previous posterior means
        (``z`` and ``W``), while BRD/ARD are initialized from layer 0. Existing
        marker-aware names are preserved through the refit path.
        """
        if getattr(self, "_has_fit", False):
            old_factor_lists = [
                np.array(factors, dtype=int).copy() for factors in self.factor_lists
            ]
            old_factor_names = [list(names) for names in self.factor_names]

            self.layer_sizes = [len(factors) for factors in old_factor_lists]
            self.n_layers = len(self.layer_sizes)
            # Keep current factor names stable across refits. Recomputing names
            # from index patterns here can relabel factors even when n_epoch=0.
            self.update_model_priors()
            self.logger.info(
                f"Continuing iscDEF from previous fit with layer sizes {self.layer_sizes}."
            )
            nmf_init = False
            init_budgets = False
            init_alpha = False
            init_z = []
            init_w = []
            for layer_idx, layer_name in enumerate(self.layer_names):
                keep_idx = old_factor_lists[layer_idx]
                init_z.append(np.array(self.pmeans[f"{layer_name}z"])[:, keep_idx])
                if layer_idx == 0:
                    init_w.append(np.array(self.pmeans[f"{layer_name}W"])[keep_idx])
                else:
                    parent_keep_idx = old_factor_lists[layer_idx - 1]
                    init_w.append(
                        np.array(self.pmeans[f"{layer_name}W"])[
                            np.ix_(keep_idx, parent_keep_idx)
                        ]
                    )
            l0_keep = old_factor_lists[0]
            init_brd = np.array(self.pmeans["brd"])[l0_keep] if self.use_brd else None
            init_ard = np.array(self.pmeans["factor_means"])[l0_keep]
            z_init_concentration = 100.0

            # After extracting priors/inits in original index space, switch to
            # compact local indexing for learning/annotation consistency.
            self.factor_lists = [
                np.arange(size, dtype=int) for size in self.layer_sizes
            ]
            self.factor_names = old_factor_names
        else:
            init_budgets = True
            init_alpha = True
            init_z = None
            init_w = None
            init_brd = None
            init_ard = None
        self.init_var_params(
            init_budgets=init_budgets,
            init_alpha=init_alpha,
            init_z=init_z,
            init_w=init_w,
            init_brd=init_brd,
            init_ard=init_ard,
            nmf_init=nmf_init,
            max_cells=max_cells_init,
            z_init_concentration=z_init_concentration,
        )
        self.elbos = []
        self.step_sizes = []
        self._learn(**kwargs)
        self._has_fit = True
        self._fit_revision = getattr(self, "_fit_revision", 0) + 1
