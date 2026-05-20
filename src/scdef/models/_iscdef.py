from scdef.models._scdef import scDEF
from anndata import AnnData
import jax.numpy as jnp
import numpy as np
import logging

from typing import Any, Dict, List, Optional, Sequence, Mapping


class iscDEF(scDEF):
    """Informed Single-cell Deep Exponential Families (iscDEF) model.

    iscDEF extends scDEF with marker gene sets that shape the layer-0 loading prior ``W``.
    Each typed factor is encouraged to use its own marker genes, may load other genes
    more weakly, and can share biology with ``add_other`` residual factors.

    **Tuning how much the model relies on markers vs. augments signatures**

    Marker reliance is controlled mainly by the Gamma prior on ``W`` at the markers layer
    (for ``markers_layer=0``, that is L0). For a typed factor, each listed marker gene has
    prior mean loading ``≈ gs_big_scale`` (tighter when ``marker_strength`` is high);
    all other genes default to ``≈ gs_small_scale``. Fitted signatures can still add
    genes beyond your list if the data and these priors allow it.

    - **Stay close to the input marker lists** (typing, strict gene programs):
      increase ``gs_big_scale`` and ``marker_strength``;
      keep the default ``penalize_other=True`` so off-type markers are discouraged on the wrong factor;
      keep ``add_other`` small when you only have a few types to avoid ignoring the marker factors.

    - **Augment markers with data-driven genes**:
      decrease ``gs_big_scale`` and ``marker_strength``;
      increase ``gs_small_scale`` or ``nonmarker_strength`` so non-marker genes are not
      overly suppressed on typed factors;
      use ``add_other`` ≥ 1 for programs not in ``markers_dict``;
      set ``penalize_other=False`` if overlapping lists should not hard-reject shared genes.

    List design matters as much as numeric knobs: non-overlapping marker sets per type
    separate factors more reliably than tuning alone. Use ``markers_layer=0`` for one
    factor per type; use ``markers_layer>0`` for coarse types at the top and finer
    substructure at L0.

    Args:
        adata: AnnData object containing the gene expression count matrix. Counts must be present
            in either `adata.X` or a specified layer.
        markers_dict: dictionary mapping marker/factor names to gene lists (gene sets). These guide
            the formation of factors in the chosen layer.
        add_other: if > 0, adds one or more ``other{i}`` residual categories. At
            ``markers_layer=0``, each is a separate L0 factor. At ``markers_layer>0``,
            each gets a block of L0 sub-factors (``add_other * n_factors_per_marker``
            columns at L0) and one coarse factor at the marker layer.
        markers_layer: index of the layer at which gene sets are enforced as factors (0 = lowest/finest,
            higher = top layer). If > 0, total layers determined by this value.
        add_root: when ``markers_layer > 0`` (default ``True``), append a width-1 root above the
            marker layer. Fitting runs in two phases (main fit with frozen root, then
            ``root_epochs`` on the root only); see :meth:`fit`. Ignored when ``markers_layer=0``.
        cn_small_mean: mean prior connectivity for "small" (weakly-connected) genes between factors and gene sets.
        cn_big_mean: mean prior connectivity for "big" (strongly-connected) genes between factors and gene sets.
        cn_small_strength: concentration parameter for low connectivity (see scDEF prior specification).
        cn_big_strength: concentration parameter for high connectivity.
        gs_small_scale: prior mean scale for genes not in a factor's marker list (higher → more non-marker loading).
        gs_big_scale: prior mean scale for genes in that factor's marker list (higher → stronger marker reliance).
        marker_strength: Gamma prior concentration on marker-gene loadings (higher → less augmentation away from markers).
        nonmarker_strength: prior concentration on non-marker loadings (higher → tighter; overridden to 1.0 if ``use_brd``).
        other_strength: prior concentration when penalizing marker genes on the wrong factor or on ``other`` rows.
        penalize_other: if True (default), typed factors penalize other groups' marker genes;
            ``other`` factors penalize all typed markers.
        **kwargs: additional arguments passed to scDEF. ``hierarchy_fraction`` defaults to ``0.25``
            (scales coverage-derived ``alpha`` when ``set_alpha_from_cov=True``).
    """

    def __init__(
        self,
        adata: AnnData,
        markers_dict: Mapping[str, Sequence[str]],
        add_other: Optional[int] = 0,
        markers_layer: Optional[int] = 0,
        add_root: Optional[bool] = None,
        cn_small_mean: Optional[float] = 1e-2,
        cn_big_mean: Optional[float] = 1.0,
        cn_small_strength: Optional[float] = 1.0,
        cn_big_strength: Optional[float] = 0.1,
        gs_small_scale: Optional[float] = 1.0,
        gs_big_scale: Optional[float] = 10.0,
        marker_strength: Optional[float] = 1.0,
        nonmarker_strength: Optional[float] = 0.1,
        other_strength: Optional[float] = 0.1,
        penalize_other: Optional[bool] = True,
        **kwargs,
    ):
        self.markers_dict = markers_dict
        self.penalize_other = penalize_other
        self.add_other = int(add_other) if add_other is not None else 0
        if self.add_other < 0:
            raise ValueError("add_other must be >= 0.")
        self.markers_layer = int(markers_layer)
        if self.markers_layer == 0:
            if add_root is True:
                raise ValueError("add_root=True requires markers_layer > 0.")
            self.add_root = False
        elif add_root is None:
            self.add_root = True
        else:
            self.add_root = bool(add_root)

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
        self.n_layers_schedule = kwargs.pop("n_layers", 6)
        if "use_brd" not in kwargs and markers_layer != 0:
            kwargs["use_brd"] = True
        self.use_brd = kwargs.get("use_brd", True)

        if self.use_brd:
            self.nonmarker_strength = 1.0

        self.set_layer_sizes()

        kwargs.pop("decay_factor", None)
        kwargs.setdefault("hierarchy_fraction", 0.25)
        super(iscDEF, self).__init__(
            adata,
            layer_sizes=self.layer_sizes,
            layer_names=self.layer_names,
            n_layers=len(self.layer_sizes),
            **kwargs,
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
        self._refresh_top_factor_names()

    def set_layer_sizes(self):
        layer_sizes = []
        layer_names = []
        if self.markers_layer == 0:
            self.update_model_size(
                self.n_markers, self.n_layers_schedule, use_decay_factor_schedule=True
            )
            layer_names = self.layer_names
            layer_sizes = self.layer_sizes
            layer_names[0] = "marker"
        else:
            # For markers_layer > 0, only the bottom layer (layer 0) gets "other" factors
            self.n_layers = self.markers_layer + 1
            # Updated so that the layer sizes are exponential in decay_factor:
            # Layer 0 (lowest): self.n_markers * decay_factor**(n_layers-1), then decay_factor**(n_layers-2), ..., final (top) layer is n_markers.
            if self.n_layers_schedule == 1:
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

            if self.add_root:
                layer_sizes.append(1)
                layer_names.append("root")

        self.layer_sizes = layer_sizes
        self.layer_names = layer_names
        self.n_layers = len(layer_sizes)

    def _refresh_top_factor_names(self) -> None:
        """Marker-layer names used for annotation (not the optional width-1 root)."""
        if self.markers_layer != 0 and len(self.factor_names) > self.markers_layer:
            self._top_factor_names = list(self.factor_names[self.markers_layer])
        elif len(self.factor_names) > 0:
            self._top_factor_names = list(self.factor_names[-1])

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
        out += "\n\t" + "Layer concentration parameter: " + str(self.alpha)
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
        connectivity_layers = self.n_layers
        if getattr(self, "add_root", False) and self.markers_layer > 0:
            connectivity_layers = self.markers_layer + 1
        for layer_idx in range(1, connectivity_layers):
            upper_kept = np.asarray(self.factor_lists[layer_idx], dtype=int)
            lower_kept = np.asarray(self.factor_lists[layer_idx - 1], dtype=int)
            n_upper = len(upper_kept)
            n_lower = len(lower_kept)
            connectivity_matrix = cn_small_mean * np.ones((n_upper, n_lower))
            strength_matrix = cn_small_strength * np.ones((n_upper, n_lower))
            upper_pos = {factor_idx: pos for pos, factor_idx in enumerate(upper_kept)}
            lower_pos = {factor_idx: pos for pos, factor_idx in enumerate(lower_kept)}

            layer_rev_idx = self._hierarchy_content_layers() - 1 - layer_idx

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
                    strength_matrix[
                        upper_row, child_positions
                    ] = cn_big_strength  # * n_upper / self.layer_sizes[0]

            # # If "other" factors are present in layer 0, connect them weakly to upper layers (or not at all)
            # # (This logic is for "other" in the lowest layer only)
            # if self.add_other > 0 and layer_idx > 0:
            #     # "Other" factors are always appended last in each layer
            #     # Compute for lower/upper layer sizes
            #     n_other_upper = 0  # noqa: F841
            #     n_other_lower = 0  # noqa: F841
            #     # Only the lowest layer gets extra "other" factors, but we allow for safety at all
            #     if layer_idx == 1:
            #         # "other" at layer 0 (lowest)
            #         other_start_lower = (
            #             n_lower - self.add_other * lower_factors_per_marker
            #         )
            #         other_end_lower = n_lower
            #         # All upper factors connect (weakly) to all "other" factors
            #         connectivity_matrix[
            #             :, other_start_lower:other_end_lower
            #         ] = cn_big_mean
            #         strength_matrix[:, other_start_lower:other_end_lower] = (
            #             cn_big_strength #* n_upper / self.layer_sizes[0]
            #         )
            #     elif layer_idx == self.n_layers - 1:  # top layer can't have "other"
            #         pass
            #     else:
            #         # handle recursively if "other" present at each layer, but in current design, only at layer 0.
            #         pass

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
            if self.penalize_other:
                for group in marker_dict:
                    if group != cellgroup:
                        for gene in marker_dict[group]:
                            if "other" not in cellgroup:
                                if gene not in marker_dict[cellgroup]:
                                    loc = np.where(self.adata.var.index == gene)[0]
                                    if len(loc) == 0:
                                        continue
                                    self.gene_sets[factors_rows, loc] = 1e-6
                                    self.strengths[factors_rows, loc] = other_strength
                            else:
                                loc = np.where(self.adata.var.index == gene)[0]
                                if len(loc) == 0:
                                    continue
                                self.gene_sets[factors_rows, loc] = 1e-6
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
                elif idx == self.markers_layer:
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
                elif getattr(self, "add_root", False) and idx == self.n_layers - 1:
                    factor_names = [
                        f"{self.layer_names[idx]}_{i}"
                        for i in range(len(self.factor_lists[idx]))
                    ]
                else:
                    rev_idx = self._hierarchy_content_layers() - 1 - idx
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
        self._refresh_top_factor_names()

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
            self._refresh_top_factor_names()

        if annotate:
            self.annotate_adata()

    def _compute_marker_score_matrix(
        self,
        layer: Optional[str] = None,
        score_genes_kwargs: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Per-cell Scanpy ``score_genes`` scores for each typed entry in ``markers_dict``."""
        import scanpy as sc

        score_genes_kwargs = dict(score_genes_kwargs or {})
        typed_names = list(self.markers_dict.keys())
        n_cells = self.adata.n_obs
        scores = np.zeros((n_cells, len(typed_names)), dtype=np.float32)
        if len(typed_names) == 0:
            return scores

        score_layer = layer
        use_adata = self.adata
        if score_layer is not None:
            if score_layer not in self.adata.layers:
                raise ValueError(
                    f"score_genes layer `{score_layer}` not found in adata.layers."
                )

        for idx, name in enumerate(typed_names):
            genes = [g for g in self.markers_dict[name] if g in use_adata.var_names]
            if len(genes) == 0:
                self.logger.warning(
                    f"No marker genes for `{name}` found in var_names; "
                    "using zero scores for z initialization."
                )
                continue
            score_key = f"_scdef_zinit_{idx}_{name}"
            score_call_kwargs = dict(score_genes_kwargs)
            score_call_kwargs.setdefault("score_name", score_key)
            if score_layer is not None:
                score_call_kwargs["layer"] = score_layer
            sc.tl.score_genes(use_adata, gene_list=genes, **score_call_kwargs)
            scores[:, idx] = np.asarray(
                use_adata.obs[score_key].values, dtype=np.float32
            )
        return scores

    def _union_marker_genes(self) -> List[str]:
        """Unique genes in typed ``markers_dict`` entries (union for ``other`` z init)."""
        seen = set()
        genes: List[str] = []
        for gene_list in self.markers_dict.values():
            for gene in gene_list:
                if gene in seen:
                    continue
                seen.add(gene)
                genes.append(gene)
        return [g for g in genes if g in self.adata.var_names]

    def _compute_union_marker_scores(
        self,
        layer: Optional[str] = None,
        score_genes_kwargs: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Per-cell ``score_genes`` on the union of typed marker genes."""
        import scanpy as sc

        n_cells = self.adata.n_obs
        genes = self._union_marker_genes()
        if len(genes) == 0:
            self.logger.warning(
                "No typed marker genes found in var_names for union other z init; "
                "using zero union scores."
            )
            return np.zeros(n_cells, dtype=np.float32)

        score_genes_kwargs = dict(score_genes_kwargs or {})
        score_key = str(
            score_genes_kwargs.setdefault("score_name", "_scdef_zinit_union")
        )
        score_call_kwargs = dict(score_genes_kwargs)
        use_adata = self.adata
        if layer is not None:
            if layer not in self.adata.layers:
                raise ValueError(
                    f"score_genes layer `{layer}` not found in adata.layers."
                )
            score_call_kwargs["layer"] = layer
        sc.tl.score_genes(use_adata, gene_list=genes, **score_call_kwargs)
        return np.asarray(use_adata.obs[score_key].values, dtype=np.float32)

    def _other_affinity_from_union_scores(
        self,
        union_scores: np.ndarray,
        temperature: float = 1.0,
    ) -> np.ndarray:
        """Per-cell weight for ``other`` init: high when union marker score is low."""
        scores = np.asarray(union_scores, dtype=np.float32).reshape(-1)
        temp = max(float(temperature), 1e-8)
        affinity = 1.0 / (1.0 + np.maximum(scores, 0.0) / temp)
        return affinity.astype(np.float32)

    def _resolve_other_affinity(
        self,
        other_mode: str,
        temperature: float,
        layer: Optional[str] = None,
        score_genes_kwargs: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Per-cell multipliers for ``other`` blocks (``uniform`` or ``inverse_union``)."""
        mode = str(other_mode).strip().lower()
        n_cells = self.adata.n_obs
        if self.add_other <= 0:
            return np.ones(n_cells, dtype=np.float32)
        if mode == "uniform":
            return np.ones(n_cells, dtype=np.float32)
        if mode == "inverse_union":
            union_scores = self._compute_union_marker_scores(
                layer=layer, score_genes_kwargs=score_genes_kwargs
            )
            return self._other_affinity_from_union_scores(
                union_scores, temperature=temperature
            )
        raise ValueError(
            f"z_init_other_mode must be 'uniform' or 'inverse_union', got {other_mode!r}."
        )

    def _hierarchy_content_layers(self) -> int:
        """Layer count used for marker block geometry (excludes optional width-1 root)."""
        n = int(self.n_layers)
        if getattr(self, "add_root", False):
            n -= 1
        return n

    def _hierarchical_block_size(self, layer_idx: int) -> int:
        """Factors per marker block at ``layer_idx`` when ``markers_layer > 0``."""
        n_content = self._hierarchy_content_layers()
        if n_content == 1:
            return int(self.n_factors_per_marker)
        rev_layer = (n_content - 1) - layer_idx
        return int(self.decay_factor**rev_layer)

    def _typed_props_from_marker_scores(
        self,
        temperature: float,
        layer: Optional[str] = None,
        score_genes_kwargs: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        scores = self._compute_marker_score_matrix(
            layer=layer, score_genes_kwargs=score_genes_kwargs
        )
        temp = max(float(temperature), 1e-8)
        shifted = scores / temp
        shifted = shifted - np.max(shifted, axis=1, keepdims=True)
        exp_s = np.exp(shifted)
        denom = np.maximum(exp_s.sum(axis=1, keepdims=True), 1e-12)
        return exp_s / denom

    def _build_z_init_hierarchical_layer(
        self,
        typed_props: np.ndarray,
        layer_idx: int,
        other_mass: Optional[float] = None,
        other_affinity: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """One layer of score-based ``z`` init for ``markers_layer > 0`` layout."""
        n_cells = self.adata.n_obs
        k_layer = int(self.layer_sizes[layer_idx])
        block_size = max(self._hierarchical_block_size(layer_idx), 1)
        n_typed = len(self.markers_dict)
        m = np.zeros((n_cells, k_layer), dtype=np.float32)
        n_other_factors = int(self.add_other * block_size)

        if n_other_factors > 0:
            if other_mass is None:
                other_fraction = float(n_other_factors) / float(k_layer)
            else:
                other_fraction = float(np.clip(other_mass, 0.0, 0.5))
            other_per_factor = (other_fraction * k_layer) / float(n_other_factors)
        else:
            other_fraction = 0.0
            other_per_factor = 0.0

        typed_scale = float(k_layer) * (1.0 - other_fraction)
        typed_per_factor = typed_scale / float(block_size)

        if other_affinity is None:
            other_affinity = np.ones(n_cells, dtype=np.float32)
        other_affinity = np.asarray(other_affinity, dtype=np.float32).reshape(-1)
        if other_affinity.shape[0] != n_cells:
            raise ValueError(
                f"other_affinity must have length n_obs={n_cells}, "
                f"got shape {other_affinity.shape}."
            )

        for i in range(n_typed):
            start = i * block_size
            end = start + block_size
            m[:, start:end] = (typed_props[:, i] * typed_per_factor)[:, None]
        for j in range(self.add_other):
            start = (n_typed + j) * block_size
            end = start + block_size
            m[:, start:end] = (other_per_factor * other_affinity)[:, None]

        m = np.clip(m, 1e-3, 10.0)
        if float(np.ptp(other_affinity)) > 1e-6:
            row_sum = np.maximum(m.sum(axis=1, keepdims=True), 1e-12)
            m = m * (float(k_layer) / row_sum)
        return m.astype(np.float32)

    def _build_z_init_all_hierarchical_layers_from_marker_scores(
        self,
        temperature: float = 1.0,
        other_mass: Optional[float] = None,
        other_mode: str = "inverse_union",
        layer: Optional[str] = None,
        score_genes_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[np.ndarray]:
        typed_props = self._typed_props_from_marker_scores(
            temperature=temperature,
            layer=layer,
            score_genes_kwargs=score_genes_kwargs,
        )
        other_affinity = self._resolve_other_affinity(
            other_mode=other_mode,
            temperature=temperature,
            layer=layer,
            score_genes_kwargs=score_genes_kwargs,
        )
        n_init_layers = self.n_layers
        if getattr(self, "add_root", False):
            n_init_layers -= 1
        inits = [
            self._build_z_init_hierarchical_layer(
                typed_props, ell, other_mass, other_affinity=other_affinity
            )
            for ell in range(n_init_layers)
        ]
        if getattr(self, "add_root", False):
            inits.append(None)
        return inits

    def _l0_other_factor_count(self) -> int:
        """Number of L0 factor columns reserved for ``other`` blocks."""
        if self.add_other <= 0:
            return 0
        if self.markers_layer == 0:
            return int(self.add_other)
        return int(self.add_other * self.n_factors_per_marker)

    def _build_z_init_l0_from_marker_scores(
        self,
        temperature: float = 1.0,
        other_mass: Optional[float] = None,
        other_mode: str = "inverse_union",
        layer: Optional[str] = None,
        score_genes_kwargs: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Build layer-0 ``z`` means from marker scores and optional ``other`` mass.

        Typed groups use a softmax of ``score_genes``. If ``other_mass`` is ``None``,
        the fraction of mass on ``other`` L0 columns is ``n_other / K0``. With
        ``other_mode='inverse_union'`` (default), ``other`` mass is higher on cells
        with low ``score_genes`` on the union of typed marker genes; use
        ``other_mode='uniform'`` for constant per-cell ``other`` init.
        When ``markers_layer > 0``, the same block structure as
        :meth:`_build_z_init_hierarchical_layer` at layer 0 is used.
        """
        n_cells = self.adata.n_obs
        n_typed = len(self.markers_dict)
        if self.markers_layer != 0:
            typed_props = self._typed_props_from_marker_scores(
                temperature=temperature,
                layer=layer,
                score_genes_kwargs=score_genes_kwargs,
            )
            other_affinity = self._resolve_other_affinity(
                other_mode=other_mode,
                temperature=temperature,
                layer=layer,
                score_genes_kwargs=score_genes_kwargs,
            )
            return self._build_z_init_hierarchical_layer(
                typed_props,
                layer_idx=0,
                other_mass=other_mass,
                other_affinity=other_affinity,
            )

        scores = self._compute_marker_score_matrix(
            layer=layer, score_genes_kwargs=score_genes_kwargs
        )
        k0 = int(self.layer_sizes[0])
        m = np.zeros((n_cells, k0), dtype=np.float32)
        n_other = self._l0_other_factor_count()

        temp = max(float(temperature), 1e-8)
        shifted = scores / temp
        shifted = shifted - np.max(shifted, axis=1, keepdims=True)
        exp_s = np.exp(shifted)
        denom = np.maximum(exp_s.sum(axis=1, keepdims=True), 1e-12)
        typed_props = exp_s / denom

        other_affinity = self._resolve_other_affinity(
            other_mode=other_mode,
            temperature=temperature,
            layer=layer,
            score_genes_kwargs=score_genes_kwargs,
        )

        if n_other > 0:
            if other_mass is None:
                other_fraction = float(n_other) / float(k0)
            else:
                other_fraction = float(np.clip(other_mass, 0.0, 0.5))
            other_per_factor = (other_fraction * k0) / float(n_other)
        else:
            other_fraction = 0.0
            other_per_factor = 0.0

        typed_scale = float(k0) * (1.0 - other_fraction)

        for i in range(n_typed):
            m[:, i] = typed_props[:, i] * typed_scale
        for j in range(self.add_other):
            m[:, n_typed + j] = other_per_factor * other_affinity

        m = np.clip(m, 1e-3, 10.0)
        if float(np.ptp(other_affinity)) > 1e-6:
            row_sum = np.maximum(m.sum(axis=1, keepdims=True), 1e-12)
            m = m * (float(k0) / row_sum)
        return m.astype(np.float32)

    def fit(
        self,
        nmf_init: bool = False,
        max_cells_init: int = 1024,
        z_init_concentration: float = 100.0,
        z_init_from_score_genes: bool = True,
        z_init_score_temperature: float = 1.0,
        z_init_other_mass: Optional[float] = None,
        z_init_other_mode: str = "inverse_union",
        score_genes_layer: Optional[str] = None,
        score_genes_kwargs: Optional[Dict[str, Any]] = None,
        root_epochs: int = 0,
        **kwargs,
    ):
        """Fit iscDEF, warm-starting from previous fit when available.

        On refit, all layers are initialized from the previous posterior means
        (``z`` and ``W``), while BRD/ARD are initialized from layer 0. Existing
        marker-aware names are preserved through the refit path.

        On the first fit, ``z`` can be initialized from Scanpy ``score_genes`` on
        typed marker sets. For ``markers_layer == 0`` only layer 0 is set this way.
        For ``markers_layer > 0``, every layer uses the same score softmax among
        typed markers, replicated uniformly within each marker block at that layer
        (and uniform ``other`` blocks when ``add_other`` is used). With
        ``z_init_other_mass=None`` (default), the fraction of mass on ``other``
        columns at each layer equals the number of other factors at that layer
        divided by that layer width. With ``z_init_other_mode='inverse_union'``
        (default), per-cell ``other`` init is scaled by ``1 / (1 + union_marker_score)``
        from ``score_genes`` on the union of typed marker genes; use
        ``z_init_other_mode='uniform'`` for constant ``other`` init. ``nmf_init`` is
        not used by iscDEF.

        When ``add_root=True`` (``markers_layer > 0`` only), a width-1 root is appended
        and fitting runs in two phases: all layers except the root, then ``root_epochs``
        on the root only (default ``10`` when ``add_root`` and ``root_epochs`` is 0).
        """
        if nmf_init:
            self.logger.warning(
                "iscDEF does not use nmf_init; z is initialized from "
                "marker score_genes when z_init_from_score_genes=True."
            )
        nmf_init = False
        if getattr(self, "add_root", False) and int(root_epochs) == 0:
            root_epochs = 10
        self.root_epochs = int(root_epochs)
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
            if z_init_from_score_genes:
                if self.markers_layer > 0:
                    init_z = (
                        self._build_z_init_all_hierarchical_layers_from_marker_scores(
                            temperature=z_init_score_temperature,
                            other_mass=z_init_other_mass,
                            other_mode=z_init_other_mode,
                            layer=score_genes_layer,
                            score_genes_kwargs=score_genes_kwargs,
                        )
                    )
                    self.logger.info(
                        "Initialized z at all hierarchy layers from "
                        "scanpy.tl.score_genes on typed marker sets."
                    )
                else:
                    init_l0 = self._build_z_init_l0_from_marker_scores(
                        temperature=z_init_score_temperature,
                        other_mass=z_init_other_mass,
                        other_mode=z_init_other_mode,
                        layer=score_genes_layer,
                        score_genes_kwargs=score_genes_kwargs,
                    )
                    init_z = [init_l0] + [None] * (self.n_layers - 1)
                    self.logger.info(
                        "Initialized layer-0 z from scanpy.tl.score_genes on typed marker sets."
                    )
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
        self._invalidate_cached_diagnostics()

        optimize_layers = list(range(self.n_layers))
        if self.root_epochs > 0:
            optimize_layers = list(range(self.n_layers - 1))

        main_learn_kwargs = dict(kwargs)
        if self.root_epochs > 0:
            main_learn_kwargs["filter"] = False
            main_learn_kwargs["annotate"] = False

        self._learn(
            optimize_layers=optimize_layers,
            **main_learn_kwargs,
        )
        self.qc_elbos = [np.asarray(x).copy() for x in self.elbos]
        preserved_traces = {
            "entropy_annealing_trace": np.asarray(
                getattr(self, "entropy_annealing_trace", np.array([]))
            ).copy(),
            "entropy_annealing_trace_epochs": np.asarray(
                getattr(self, "entropy_annealing_trace_epochs", np.array([]))
            ).copy(),
        }

        if self.root_epochs > 0:
            root_kwargs = dict(kwargs)
            root_kwargs.pop("n_rounds", None)
            root_kwargs["n_epoch"] = self.root_epochs
            root_kwargs["annealing"] = 1.0
            root_kwargs["entropy_anneal"] = False
            self._learn(
                n_rounds=1,
                optimize_layers=[self.n_layers - 1],
                **root_kwargs,
            )
            self.entropy_annealing_trace = preserved_traces[
                "entropy_annealing_trace"
            ].copy()
            self.entropy_annealing_trace_epochs = preserved_traces[
                "entropy_annealing_trace_epochs"
            ].copy()
            self.adata.uns.pop("alpha_trace", None)
            self.adata.uns.pop("alpha_trace_epochs", None)
            self.adata.uns.pop("n_eff_parents_trace", None)
            self.adata.uns.pop("n_eff_parents_trace_epochs", None)
            self.adata.uns.pop("active_l0_factor_counts_trace", None)
            self.adata.uns.pop("alpha_schedule_alphas", None)
            self.adata.uns.pop("alpha_schedule_losses", None)
            self.adata.uns.pop("alpha_schedule_epochs", None)

        self.clear_runtime_cache(clear_jax_cache=False)
        self._has_fit = True
        self._fit_revision = getattr(self, "_fit_revision", 0) + 1
