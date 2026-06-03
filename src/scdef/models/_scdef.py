# Standard library imports
import logging
import time
import json
import pickle
from pathlib import Path
from typing import Optional, Union, Sequence, Mapping, Dict, List, Tuple, Any

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

import matplotlib
import seaborn as sns
from tqdm import tqdm

from anndata import AnnData
import anndata as ad

# Local imports
from scdef.utils import score_utils, color_utils
from scdef.utils.jax_utils import lognormal_sample, lognormal_entropy, gamma_logpdf

# Configure logging
logging.basicConfig()


class scDEF(object):
    """Single-cell Deep Exponential Families (scDEF) model.

    scDEF learns hierarchical, multi-level gene expression signatures from single-cell
    RNA-seq data provided in an AnnData object. This model can be used for a variety
    of analyses including dimensionality reduction, batch correction, clustering,
    and visualization of cell states and gene programs.

    The model fits multiple layers of latent factors ("gene signatures") to describe
    cellular heterogeneity at different resolutions. It supports batch correction,
    prior specification, and generation of corrected gene expression matrices.

    Model fitting, inference routines, and additional plotting utilities are
    implemented as methods of this class. The stored AnnData object is updated with
    model results during training.

    Args:
        adata: AnnData object containing the single-cell gene expression count matrix. Counts
            should be present in either `adata.X` or in the specified `adata.layers`.
        counts_layer: key for `adata.layers` specifying which layer to use as expression counts (if not `adata.X`).
        batch_key: key in `adata.obs` containing batch annotations; if provided, batch correction is performed.
            If None or not found, no batch correction is used.
        seed: random seed for model initialization and stochastic routines (uses JAX's pseudo-random number generator).
        n_factors: number of latent factors at the lowest layer (L0), denoted ``K_0`` in
            the geometric layer-size schedule when ``layer_sizes`` is None.
        n_layers: number of layers in the geometric schedule from L0 through the layer of
            width ``top_factors`` when ``layer_sizes`` is None. A size-``1`` root appended
            when ``top_factors > 1`` does not count toward ``n_layers``.
        top_factors: target width at the coarsest non-root layer (default ``1``), used with
            ``n_factors`` and ``n_layers`` in the geometric ladder
            ``K_l = K_0 * (K_top / K_0) ** (l / (n_layers - 1))`` for ``l = 0, ..., n_layers - 1``
            (last rung fixed at ``K_top``). When ``top_factors > 1``, a final root layer of
            width ``1`` is appended and is not included in ``n_layers``.
        layer_sizes: explicit list of the number of factors in each scDEF layer. If None, layer sizes are set automatically.
        layer_names: list of custom names for the layers. If None, layer names are enumerated as ["L0", "L1", ...].
        logginglevel: verbosity level for the logger.
        alpha: concentration parameter for the Gamma prior on z.
        shrinkage_shape: shape parameter for shrinkage prior controlling factor usage.
        shrinkage_rate: rate parameter for shrinkage prior controlling factor usage.
        shrinkage_mean: target prior mean for shrinkage/factor relevance.
        top_alpha: concentration parameter for the top layer Dirichlet prior over factor proportions.
        factor_shape: shape of the prior distribution for factor-gene loadings matrix W.
        brd_strength: BRD (Batch Relevance Determination) prior concentration parameter for factor relevance estimation.
        brd_mean: mean of the BRD prior for factor relevance estimation.
        use_brd: if True, use BRD prior for automatic selection of active factors.
        cell_scale_shape: precision/concentration parameter for cell-specific scaling priors.
        gene_scale_shape: precision/concentration parameter for gene-specific scaling priors.
        batch_cpal: default matplotlib color palette name used for batches.
        layer_cpal: matplotlib color palette for factors/colors at each scDEF layer.
        lightness_mult: lightness multiplier to define the base color for each new scDEF layer.
        set_alpha_from_cov: if True, set the alpha parameter from the data coverage.
        hierarchy_weight: multiplier applied to coverage-derived ``alpha`` when
            ``set_alpha_from_cov`` is True (ignored otherwise).
        marginalize_alpha: if True, infer alpha from a variational posterior during training.
    """

    def make_corrected_data(self, layer_name: str = "scdef_corrected") -> None:
        """Compute and store the low-rank reconstruction of the UMI count matrix.

        The reconstructed matrix is saved to adata.layers[layer_name], providing a
        denoised, batch-corrected version of the expression data.

        Args:
            layer_name: name for the AnnData layer where the reconstructed matrix is stored
        """
        scdef_layer = self.layer_names[0]
        Z = self.pmeans[f"{scdef_layer}z"][:, self.factor_lists[0]]  # nxk
        W = self.pmeans[f"{scdef_layer}W"][self.factor_lists[0]]  # kxg
        self.adata.layers[layer_name] = np.array(Z.dot(W))

    def __init__(
        self,
        adata: AnnData,
        counts_layer: Optional[str] = None,
        batch_key: Optional[str] = None,
        seed: Optional[int] = 42,
        n_factors: Optional[int] = 100,
        top_factors: int = 1,
        n_layers: Optional[float] = 6,
        layer_sizes: Optional[list] = None,
        layer_names: Optional[list] = None,
        logginglevel: Optional[int] = logging.INFO,
        alpha: Optional[float] = 1.0,
        shrinkage_shape: Optional[float] = 1.0,
        shrinkage_rate: Optional[float] = 1.0,
        shrinkage_mean: Optional[float] = 1.0,
        top_alpha: Optional[float] = 1.0,
        factor_shape: Optional[float] = 0.1,
        brd_strength: Optional[float] = 1.0,
        brd_mean: Optional[float] = 1.0,
        use_brd: Optional[bool] = True,
        cell_scale_shape: Optional[float] = 1.0,
        gene_scale_shape: Optional[float] = 1.0,
        batch_cpal: Optional[str] = "Dark2",
        layer_cpal: Optional[str] = "tab10",
        lightness_mult: Optional[float] = 0.15,
        set_alpha_from_cov: Optional[bool] = True,
        hierarchy_weight: Optional[float] = 1.0,
        marginalize_alpha: Optional[bool] = False,
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

        self.top_alpha = top_alpha
        self.alpha = alpha
        self.factor_shape = factor_shape
        self.brd = brd_strength
        self.brd_mean = brd_mean
        self.shrinkage_shape = shrinkage_shape
        self.shrinkage_rate = shrinkage_rate
        self.shrinkage_mean = shrinkage_mean
        self.cell_scale_shape = cell_scale_shape
        self.gene_scale_shape = gene_scale_shape
        self.use_brd = use_brd
        self.set_alpha_from_cov = set_alpha_from_cov
        self.hierarchy_weight = float(hierarchy_weight)
        self.marginalize_alpha = bool(marginalize_alpha)
        self._marginalize_alpha_init = bool(marginalize_alpha)

        if n_layers is None:
            n_layers = 6.0
        self.n_layers_schedule = n_layers
        k0 = int(n_factors if n_factors is not None else 100)
        if k0 < 1:
            raise ValueError("n_factors must be >= 1.")
        self.top_factors = int(top_factors)
        if self.top_factors < 1:
            raise ValueError("top_factors must be >= 1.")
        if self.top_factors > k0:
            raise ValueError(
                "top_factors must be <= n_factors (layer sizes are non-increasing "
                "from L0 toward the top layer)."
            )
        self.n_factors = k0

        if layer_sizes is not None:
            self.layer_sizes = [int(x) for x in layer_sizes]
            self.n_factors = int(layer_sizes[0])
            self.n_layers = len(self.layer_sizes)
        else:
            self.update_model_size(
                n_layers=int(self.n_layers_schedule),
                use_decay_factor_schedule=False,
            )

        if layer_names is not None:
            self.layer_names = layer_names
        else:
            self.layer_names = [f"L{i}" for i in range(self.n_layers)]
        self.factor_lists = [np.arange(size) for size in self.layer_sizes]
        self.set_factor_names()

        self.update_model_priors()

        self.make_layercolors(layer_cpal=self.layer_cpal, lightness_mult=lightness_mult)

        self.init_var_params(nmf_init=False)  # just to get stub
        self.set_posterior_means()
        self.set_posterior_variances()
        self._has_fit = False
        self._fit_revision = 0
        self._learn_jit_cache = None
        self._pending_reference_init = None

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
        out += "\n\t" + "alpha parameter: " + str(self.alpha)
        if self.use_brd:
            out += "\n\t" + "Using BRD"
        out += "\n\t" + "Number of batches: " + str(self.n_batches)
        out += "\n" + "Contains " + self.adata.__str__()
        return out

    @staticmethod
    def _resolve_init_gene_scale_array(
        reference_model: "scDEF",
        init_gene_scale: Union[str, np.ndarray],
        n_batches: int,
        n_genes: int,
    ) -> np.ndarray:
        """Build ``(n_batches, n_genes)`` gene-scale means for warm-starting pass 2.

        Returns:
            Per-batch gene-scale means to pass to :meth:`init_var_params`.

        Raises:
            ValueError: If ``init_gene_scale`` is invalid or the reference lacks
                fitted ``gene_scale`` when ``init_gene_scale='reference'``.
        """
        if isinstance(init_gene_scale, str):
            if init_gene_scale == "batch":
                raise ValueError(
                    "_resolve_init_gene_scale_array called with init_gene_scale='batch'."
                )
            if init_gene_scale != "reference":
                raise ValueError(
                    "init_gene_scale must be 'batch', 'reference', or a float array; "
                    f"got {init_gene_scale!r}."
                )
            if "gene_scale" not in reference_model.pmeans:
                raise ValueError(
                    "reference_model has no fitted gene_scale in pmeans; run fit() on "
                    "the reference model before from_reference with init_gene_scale='reference'."
                )
            gs = np.asarray(reference_model.pmeans["gene_scale"], dtype=np.float32)
            if gs.ndim == 1:
                gs = gs[None, :]
            if gs.shape[1] != n_genes:
                raise ValueError(
                    f"reference gene_scale has {gs.shape[1]} genes but adata has {n_genes}."
                )
            if gs.shape[0] == 1:
                profile = gs[0]
            else:
                profile = np.exp(np.mean(np.log(np.clip(gs, 1e-6, None)), axis=0))
            profile = np.clip(profile, 1e-6, 1e6)
            return np.tile(profile[None, :], (n_batches, 1))

        arr = np.asarray(init_gene_scale, dtype=np.float32)
        if arr.ndim == 1:
            arr = np.tile(arr[None, :], (n_batches, 1))
        elif arr.ndim == 2:
            if arr.shape[0] == 1 and n_batches > 1:
                arr = np.tile(arr, (n_batches, 1))
            elif arr.shape[0] != n_batches:
                raise ValueError(
                    f"init_gene_scale array has {arr.shape[0]} batch rows but "
                    f"model expects {n_batches}."
                )
        else:
            raise ValueError(
                "init_gene_scale array must be 1d (n_genes) or 2d (n_batches, n_genes)."
            )
        if arr.shape[1] != n_genes:
            raise ValueError(
                f"init_gene_scale array has {arr.shape[1]} genes but adata has {n_genes}."
            )
        return np.clip(arr, 1e-6, 1e6)

    @classmethod
    def from_reference(
        cls,
        reference_model: "scDEF",
        adata: AnnData,
        counts_layer: Optional[str] = None,
        batch_key: Optional[str] = None,
        reference_obs: Optional[str] = None,
        query_obs: Optional[str] = None,
        copy_cell_z: bool = True,
        init_gene_scale: Union[str, np.ndarray] = "batch",
        **kwargs: Any,
    ) -> "scDEF":
        """Create a new model initialized from a fitted reference hierarchy.

        The new model uses ``adata`` as its data matrix and initializes global
        hierarchy parameters (W, BRD/ARD, alpha-related hyperparameters) from
        ``reference_model``. Cell/gene budgets are initialized from the new data
        so modality/batch-specific scales can be learned.

        Args:
            init_gene_scale: how to initialize per-batch ``gene_scale`` variational
                means before the first :meth:`fit` on this model.

                * ``'batch'`` (default): use per-batch count means from
                  :meth:`load_adata` (``1 / gene_ratio_init``), which pre-separates
                  batches when ``batch_key`` is confounded with condition.
                * ``'reference'``: broadcast the reference model's fitted
                  ``pmeans['gene_scale']`` to every batch (geometric mean across
                  reference batches if the reference had more than one). Pass-2
                  optimization then learns batch-specific *deviations* from the
                  pass-1 pooled profile instead of starting from separated batch
                  means, which helps preserve factor loadings from pass 1.
                * array: explicit ``(n_genes,)`` or ``(n_batches, n_genes)`` means.
        """
        if reference_model.adata.n_vars != adata.n_vars or not np.array_equal(
            np.asarray(reference_model.adata.var_names), np.asarray(adata.var_names)
        ):
            if set(reference_model.adata.var_names).issubset(set(adata.var_names)):
                adata = adata[:, reference_model.adata.var_names].copy()
            else:
                raise ValueError(
                    "adata must contain all reference genes; pass matching genes or "
                    "pre-align adata.var_names to reference_model.adata.var_names."
                )
        if batch_key is not None:
            if batch_key not in adata.obs:
                raise KeyError(f"batch_key {batch_key!r} not found in adata.obs.")
            values = set(map(str, adata.obs[batch_key].astype(str).unique()))
            for label, value in {
                "reference_obs": reference_obs,
                "query_obs": query_obs,
            }.items():
                if value is not None and str(value) not in values:
                    raise ValueError(
                        f"{label}={value!r} is not present in adata.obs[{batch_key!r}]."
                    )

        factor_lists = [np.asarray(f, dtype=int) for f in reference_model.factor_lists]
        layer_sizes = [len(f) for f in factor_lists]
        init_w = []
        for layer_idx, keep in enumerate(factor_lists):
            w = np.asarray(
                reference_model.pmeans[f"{reference_model.layer_names[layer_idx]}W"],
                dtype=np.float32,
            )
            if layer_idx == 0:
                init_w.append(w[keep])
            else:
                parent_keep = factor_lists[layer_idx - 1]
                init_w.append(w[np.ix_(keep, parent_keep)])

        init_brd = np.asarray(reference_model.pmeans["brd"], dtype=np.float32)[
            factor_lists[0]
        ]
        init_ard = np.asarray(reference_model.pmeans["factor_means"], dtype=np.float32)[
            factor_lists[0]
        ]

        init_z = None
        if copy_cell_z:
            init_z = [
                np.ones((adata.n_obs, size), dtype=np.float32) for size in layer_sizes
            ]
            ref_pos = {
                str(name): i for i, name in enumerate(reference_model.adata.obs_names)
            }
            matches = [
                (new_i, ref_pos[str(name)])
                for new_i, name in enumerate(adata.obs_names)
                if str(name) in ref_pos
            ]
            if len(matches) > 0:
                new_idx = np.asarray([m[0] for m in matches], dtype=int)
                ref_idx = np.asarray([m[1] for m in matches], dtype=int)
                for layer_idx, keep in enumerate(factor_lists):
                    z = np.asarray(
                        reference_model.pmeans[
                            f"{reference_model.layer_names[layer_idx]}z"
                        ],
                        dtype=np.float32,
                    )[np.ix_(ref_idx, keep)]
                    init_z[layer_idx][new_idx] = z

        model_kwargs = cls._reference_model_kwargs(reference_model, layer_sizes)
        model_kwargs.update(kwargs)
        model = cls(
            adata,
            counts_layer=counts_layer,
            batch_key=batch_key,
            **model_kwargs,
        )
        model.alpha = float(reference_model.alpha)
        model.top_alpha = reference_model.top_alpha
        model.update_model_priors(update_alpha_from_cov=False)
        init_gene_scale_arr = None
        if init_gene_scale != "batch":
            init_gene_scale_arr = cls._resolve_init_gene_scale_array(
                reference_model,
                init_gene_scale,
                int(model.n_batches),
                int(model.adata.n_vars),
            )
        model._pending_reference_init = {
            "init_w": init_w,
            "init_brd": init_brd,
            "init_ard": init_ard,
            "init_z": init_z,
            "init_gene_scale": init_gene_scale_arr,
            "reference_obs": reference_obs,
            "query_obs": query_obs,
        }
        return model

    @classmethod
    def add_batch_correction(
        cls,
        reference_model: "scDEF",
        batch_key: str,
        *,
        adata: Optional[AnnData] = None,
        counts_layer: Optional[str] = None,
        copy_cell_z: bool = True,
        freeze_w: bool = False,
        learn_budgets: bool = True,
        n_epoch: int = 400,
        lr: float = 0.05,
        tolerance: float = 1e-4,
        from_reference_kwargs: Optional[Mapping[str, Any]] = None,
        **fit_kwargs: Any,
    ) -> "scDEF":
        """Warm-start a batch-corrected model from a fitted hierarchy.

        Designed for the workflow:
        1. Fit ``reference_model`` without a ``batch_key`` to learn the factor
           hierarchy on the unbatched signal (optionally followed by
           ``filter_factors``).
        2. Call this method to construct a new model that shares the same
           hierarchy (``factor_lists``, layer sizes, ``W``, ``BRD``, ``ARD``)
           and re-fits it under per-batch gene-scale priors so batch effects
           are absorbed by ``gene_scale``, not by the hierarchy.

        Internally calls :meth:`from_reference` to build the new model with the
        new ``batch_key`` and warm-started initial values, then runs
        :meth:`fit` with ``freeze_w`` (defaulting to ``True``) so the carried
        hierarchy cannot drift during the second pass.

        Args:
            reference_model: a fitted ``scDEF`` providing the hierarchy.
            batch_key: column in ``adata.obs`` to use as the new batch
                annotation.
            adata: AnnData for the second pass. Defaults to
                ``reference_model.adata``. Pass a different one if you want to
                re-fit on a superset of cells; see :meth:`from_reference` for
                gene-alignment requirements.
            counts_layer: counts layer for the new ``adata``; passed through
                to :meth:`from_reference`.
            copy_cell_z: whether to copy per-cell ``z`` warm starts for cells
                shared with ``reference_model.adata`` (passed through to
                :meth:`from_reference`).
            freeze_w: hold every per-layer ``W`` fixed at the warm-started
                value during the second fit. Default ``False``; set ``True``
                when you explicitly want to keep the hierarchy stable while
                gene scales adapt.
            learn_budgets: passed to :meth:`fit` as
                ``learn_budgets_on_refit``. Default ``True``: per-batch
                gene-scale and per-cell budgets must move for batch correction
                to work.
            n_epoch: epochs for the second-pass fit. Default 400 (shorter than
                a fresh fit because the global structure is already in place).
            lr: learning rate for the second-pass fit. Default 0.05 (gentler
                than a fresh fit).
            tolerance: early-stopping tolerance for the second-pass fit.
            from_reference_kwargs: extra kwargs forwarded verbatim to
                :meth:`from_reference` (e.g. ``reference_obs``,
                ``query_obs``, ``init_gene_scale``). Defaults to
                ``init_gene_scale='reference'`` so pass 2 starts from the
                pass-1 pooled gene-scale profile rather than per-batch means.
            **fit_kwargs: additional kwargs forwarded to :meth:`fit`. Override
                any of the defaults above (``n_epoch``, ``lr``, etc.) by
                passing them here -- they take precedence.

        Returns:
            The new fitted model with batch correction applied.
        """
        target_adata = adata if adata is not None else reference_model.adata
        from_reference_kwargs = dict(from_reference_kwargs or {})
        from_reference_kwargs.setdefault("counts_layer", counts_layer)
        from_reference_kwargs.setdefault("copy_cell_z", copy_cell_z)
        from_reference_kwargs.setdefault("init_gene_scale", "reference")

        model = cls.from_reference(
            reference_model,
            target_adata,
            batch_key=batch_key,
            **from_reference_kwargs,
        )

        merged_fit_kwargs: Dict[str, Any] = dict(
            n_epoch=n_epoch,
            lr=lr,
            tolerance=tolerance,
            learn_budgets_on_refit=learn_budgets,
            freeze_w=freeze_w,
        )
        merged_fit_kwargs.update(fit_kwargs)
        model.fit(**merged_fit_kwargs)
        return model

    @classmethod
    def decompose_batch_effects(
        cls,
        reference_model: "scDEF",
        *,
        adata: Optional[AnnData] = None,
        counts_layer: Optional[str] = None,
        top_layer: int = 1,
        n_epoch: int = 400,
        lr: float = 0.05,
        tolerance: float = 1e-4,
        nmf_init: bool = False,
        **fit_kwargs: Any,
    ) -> "scDEF":
        """Re-learn lower layers under a frozen upper hierarchy to discover batch programs.

        Two-stage workflow:

        1. ``reference_model`` was fitted **with** a ``batch_key``, producing a
           batch-corrected hierarchy where per-batch ``gene_scale`` absorbed
           technical variance.
        2. This method creates a new model **without** ``batch_key``, resets
           ``W^{L0}`` gene loadings, and re-learns all layers up to
           ``top_layer``.  At the boundary (``top_layer``), only ``W`` is
           re-learned while ``z`` stays fixed — preserving the cell-to-group
           assignments as the structural constraint.  Layers below
           ``top_layer`` are fully re-learned (both ``W`` and ``z``).
           Layers above ``top_layer`` remain completely fixed.

        With ``top_layer=1`` (default):
            - L0: W reset and re-learned, z re-learned
            - L1: W warm-started and re-learned, z frozen
            - L2+: fully frozen

        With ``top_layer=2``:
            - L0: W reset and re-learned, z re-learned
            - L1: W warm-started and re-learned, z re-learned
            - L2: W warm-started and re-learned, z frozen
            - L3+: fully frozen

        Args:
            reference_model: a fitted ``scDEF`` that was trained with
                ``batch_key``. Its upper hierarchy provides the frozen
                structural constraint.
            adata: AnnData for the second stage.  Defaults to
                ``reference_model.adata``.  Must share the same ``var_names``.
            counts_layer: counts layer for ``adata``; passed to the new model
                constructor.
            top_layer: the highest layer whose ``W`` is re-learned.  Its ``z``
                remains frozen as the structural anchor.  Layers below it are
                fully re-learned; layers above are completely frozen.
                Default ``1``.
            n_epoch: training epochs for the re-learning phase.
            lr: learning rate for the re-learning phase.
            tolerance: early-stopping tolerance.
            nmf_init: if True, initialize the new L0 W via NMF on the data.
                If False (default), use random initialization.
            **fit_kwargs: additional keyword arguments forwarded to
                :meth:`_learn` (e.g. ``batch_size``, ``num_samples``).

        Returns:
            A new fitted model whose lower-layer factors reveal batch-specific
            and shared gene programs under the frozen upper-layer cell
            assignments.
        """
        top_layer = int(top_layer)
        if reference_model.n_layers < top_layer + 1:
            raise ValueError(
                f"reference_model must have at least {top_layer + 1} layers "
                f"for top_layer={top_layer}, but has {reference_model.n_layers}."
            )

        target_adata = adata if adata is not None else reference_model.adata

        factor_lists = [np.asarray(f, dtype=int) for f in reference_model.factor_lists]
        layer_sizes = [len(f) for f in factor_lists]

        # Build init_w: None for L0 (reset), reference values for L1+
        init_w: List[Optional[np.ndarray]] = [None]
        for layer_idx in range(1, reference_model.n_layers):
            keep = factor_lists[layer_idx]
            parent_keep = factor_lists[layer_idx - 1]
            w = np.asarray(
                reference_model.pmeans[f"{reference_model.layer_names[layer_idx]}W"],
                dtype=np.float32,
            )
            init_w.append(w[np.ix_(keep, parent_keep)])

        # Build init_z: None for layers below top_layer (re-learned),
        # reference values for top_layer and above (frozen or stop-gradiented)
        init_z: List[Optional[np.ndarray]] = []
        for layer_idx in range(reference_model.n_layers):
            if layer_idx < top_layer:
                init_z.append(None)
            else:
                keep = factor_lists[layer_idx]
                z = np.asarray(
                    reference_model.pmeans[
                        f"{reference_model.layer_names[layer_idx]}z"
                    ],
                    dtype=np.float32,
                )
                if target_adata is reference_model.adata:
                    init_z.append(z[:, keep])
                else:
                    ref_pos = {
                        str(name): i
                        for i, name in enumerate(reference_model.adata.obs_names)
                    }
                    matches = [
                        (new_i, ref_pos[str(name)])
                        for new_i, name in enumerate(target_adata.obs_names)
                        if str(name) in ref_pos
                    ]
                    z_layer = np.ones((target_adata.n_obs, len(keep)), dtype=np.float32)
                    if len(matches) > 0:
                        new_idx = np.asarray([m[0] for m in matches], dtype=int)
                        ref_idx = np.asarray([m[1] for m in matches], dtype=int)
                        z_layer[new_idx] = z[ref_idx][:, keep]
                    init_z.append(z_layer)

        # BRD and ARD from reference (these apply to L0 factors)
        init_brd = np.asarray(reference_model.pmeans["brd"], dtype=np.float32)[
            factor_lists[0]
        ]
        init_ard = np.asarray(reference_model.pmeans["factor_means"], dtype=np.float32)[
            factor_lists[0]
        ]

        # Create new model WITHOUT batch_key
        model_kwargs = cls._reference_model_kwargs(reference_model, layer_sizes)
        model_kwargs["batch_key"] = None
        model = cls(
            target_adata,
            counts_layer=counts_layer,
            **model_kwargs,
        )
        model.alpha = float(reference_model.alpha)
        model.top_alpha = reference_model.top_alpha
        model.update_model_priors(update_alpha_from_cov=False)

        # Initialize variational parameters
        model.init_var_params(
            init_budgets=True,
            init_alpha=False,
            init_z=init_z,
            init_w=init_w,
            init_brd=init_brd,
            init_ard=init_ard,
            nmf_init=nmf_init,
            z_init_concentration=0.05,
        )

        # Overwrite z at top_layer+ with tight distributions at reference values
        # (init_var_params adds Gamma noise; we want exact values for frozen z)
        z_params = model.local_params[1]
        for layer_idx in range(top_layer, model.n_layers):
            start = int(np.sum(model.layer_sizes[:layer_idx]))
            end = start + int(model.layer_sizes[layer_idx])
            z_ref = init_z[layer_idx]
            m = jnp.clip(jnp.asarray(z_ref, dtype=jnp.float32), 1e-3, 1e6)
            v = m / 1000.0
            mu = jnp.log(m**2 / jnp.sqrt(m**2 + v))
            log_sigma = jnp.log(jnp.sqrt(jnp.log(1 + v / (m**2))))
            z_params = z_params.at[0, :, start:end].set(mu)
            z_params = z_params.at[1, :, start:end].set(log_sigma)
        model.local_params = list(model.local_params)
        model.local_params[1] = z_params

        model._invalidate_cached_diagnostics()
        model.elbos = []
        model.step_sizes = []

        # Layers 0..top_layer get W gradients; only top_layer has z frozen
        learn_kwargs = dict(fit_kwargs)
        learn_kwargs.setdefault("n_epoch", n_epoch)
        learn_kwargs.setdefault("lr", lr)
        learn_kwargs.setdefault("tolerance", tolerance)
        learn_kwargs.setdefault("filter", True)
        learn_kwargs.setdefault("annotate", True)
        optimize = list(range(top_layer + 1))
        freeze_z = [top_layer]
        model._learn(
            optimize_layers=optimize,
            freeze_z_layers=freeze_z,
            **learn_kwargs,
        )

        model.clear_runtime_cache(clear_jax_cache=False)
        model._has_fit = True
        model._fit_revision = getattr(model, "_fit_revision", 0) + 1
        return model

    @classmethod
    def from_hierarchy(
        cls,
        adata: AnnData,
        hierarchy: Union["scDEF", Sequence[np.ndarray]],
        counts_layer: Optional[str] = None,
        batch_key: Optional[str] = None,
        init_brd: Optional[np.ndarray] = None,
        init_ard: Optional[np.ndarray] = None,
        init_z: Optional[Sequence[np.ndarray]] = None,
        **kwargs: Any,
    ) -> "scDEF":
        """Create a model for new data initialized from a learned hierarchy.

        ``hierarchy`` can be either a fitted scDEF model (preferred) or an
        explicit sequence of W matrices. When a model is passed, current
        ``factor_lists`` are respected and the corresponding W submatrices,
        BRD/ARD, and hyperparameters are copied.
        """
        reference_model = hierarchy if hasattr(hierarchy, "pmeans") else None
        if reference_model is not None:
            if reference_model.adata.n_vars != adata.n_vars or not np.array_equal(
                np.asarray(reference_model.adata.var_names), np.asarray(adata.var_names)
            ):
                if set(reference_model.adata.var_names).issubset(set(adata.var_names)):
                    adata = adata[:, reference_model.adata.var_names].copy()
                else:
                    raise ValueError(
                        "adata must contain all hierarchy model genes; pass matching "
                        "genes or pre-align adata.var_names."
                    )
            factor_lists = [
                np.asarray(f, dtype=int) for f in reference_model.factor_lists
            ]
            init_w = []
            for layer_idx, keep in enumerate(factor_lists):
                w = np.asarray(
                    reference_model.pmeans[
                        f"{reference_model.layer_names[layer_idx]}W"
                    ],
                    dtype=np.float32,
                )
                if layer_idx == 0:
                    init_w.append(w[keep])
                else:
                    parent_keep = factor_lists[layer_idx - 1]
                    init_w.append(w[np.ix_(keep, parent_keep)])
            if init_brd is None:
                init_brd = np.asarray(reference_model.pmeans["brd"], dtype=np.float32)[
                    factor_lists[0]
                ]
            if init_ard is None:
                init_ard = np.asarray(
                    reference_model.pmeans["factor_means"], dtype=np.float32
                )[factor_lists[0]]
            model_kwargs = cls._reference_model_kwargs(
                reference_model, [len(f) for f in factor_lists]
            )
            model_kwargs.update(kwargs)
            kwargs = model_kwargs
        else:
            w_matrices = hierarchy
            if len(w_matrices) == 0:
                raise ValueError("hierarchy must contain at least L0W.")
            init_w = [np.asarray(w, dtype=np.float32) for w in w_matrices]
            kwargs = dict(kwargs)
        w_matrices = init_w
        if len(w_matrices) == 0:
            raise ValueError("hierarchy must contain at least L0W.")
        init_w = [np.asarray(w, dtype=np.float32) for w in w_matrices]
        if init_w[0].ndim != 2:
            raise ValueError("Each W matrix must be 2-dimensional.")
        if init_w[0].shape[1] != adata.n_vars:
            raise ValueError(
                f"L0W has {init_w[0].shape[1]} genes, but adata has {adata.n_vars}."
            )
        layer_sizes = [int(init_w[0].shape[0])]
        for layer_idx in range(1, len(init_w)):
            if init_w[layer_idx].ndim != 2:
                raise ValueError("Each W matrix must be 2-dimensional.")
            expected_parent = layer_sizes[layer_idx - 1]
            if int(init_w[layer_idx].shape[1]) != expected_parent:
                raise ValueError(
                    f"W matrix {layer_idx} has {init_w[layer_idx].shape[1]} columns; "
                    f"expected {expected_parent}."
                )
            layer_sizes.append(int(init_w[layer_idx].shape[0]))

        model_kwargs = dict(kwargs)
        model_kwargs.setdefault("layer_sizes", layer_sizes)
        model_kwargs.setdefault("n_factors", layer_sizes[0])
        top_idx = (
            len(layer_sizes) - 2
            if len(layer_sizes) > 1 and layer_sizes[-1] == 1
            else len(layer_sizes) - 1
        )
        model_kwargs.setdefault("top_factors", layer_sizes[top_idx])
        model = cls(
            adata,
            counts_layer=counts_layer,
            batch_key=batch_key,
            **model_kwargs,
        )
        model._pending_reference_init = {
            "init_w": init_w,
            "init_brd": init_brd,
            "init_ard": init_ard,
            "init_z": init_z,
            "reference_obs": None,
            "query_obs": None,
        }
        return model

    @staticmethod
    def _reference_model_kwargs(
        reference_model: "scDEF", layer_sizes: Sequence[int]
    ) -> Dict[str, Any]:
        top_idx = (
            len(layer_sizes) - 2
            if len(layer_sizes) > 1 and int(layer_sizes[-1]) == 1
            else len(layer_sizes) - 1
        )
        return {
            "n_factors": int(layer_sizes[0]),
            "top_factors": int(layer_sizes[top_idx]),
            "n_layers": reference_model.n_layers_schedule,
            "layer_sizes": [int(x) for x in layer_sizes],
            "alpha": reference_model.alpha,
            "top_alpha": reference_model.top_alpha,
            "shrinkage_shape": reference_model.shrinkage_shape,
            "shrinkage_rate": reference_model.shrinkage_rate,
            "shrinkage_mean": reference_model.shrinkage_mean,
            "factor_shape": reference_model.factor_shape,
            "brd_strength": reference_model.brd,
            "brd_mean": reference_model.brd_mean,
            "use_brd": reference_model.use_brd,
            "cell_scale_shape": reference_model.cell_scale_shape,
            "gene_scale_shape": reference_model.gene_scale_shape,
            "batch_cpal": reference_model.batch_cpal,
            "layer_cpal": reference_model.layer_cpal,
            "lightness_mult": reference_model.lightness_mult,
            "set_alpha_from_cov": reference_model.set_alpha_from_cov,
            "hierarchy_weight": reference_model.hierarchy_weight,
            "marginalize_alpha": reference_model.marginalize_alpha,
            "seed": reference_model.seed,
        }

    def save(
        self,
        dir_path: Union[str, Path],
        overwrite: bool = False,
        save_anndata: bool = False,
    ) -> None:
        """Save model state to disk, similarly to scvi-tools.

        This writes a model state pickle plus metadata to ``dir_path``. AnnData
        is saved separately as ``adata.h5ad`` only when ``save_anndata=True``.

        Args:
            dir_path: output directory path
            overwrite: whether to overwrite an existing non-empty directory
            save_anndata: whether to save ``model.adata`` as ``adata.h5ad``
        """
        out_dir = Path(dir_path).expanduser().resolve()
        if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
            raise FileExistsError(
                f"Directory '{out_dir}' already exists and is not empty. "
                "Set overwrite=True to overwrite."
            )
        out_dir.mkdir(parents=True, exist_ok=True)

        if save_anndata:
            self.adata.write(out_dir / "adata.h5ad")

        state = dict(self.__dict__)
        # Graphviz graph objects are environment-dependent and not required for
        # restoring model state.
        if "graph" in state:
            state["graph"] = None
        # JIT-compiled callables are runtime-only and not serializable.
        if "_learn_jit_cache" in state:
            state["_learn_jit_cache"] = None
        logger_level = (
            self.logger.level
            if hasattr(self, "logger") and self.logger is not None
            else None
        )
        state["logger"] = None

        payload = {
            "state": state,
            "class_name": self.__class__.__name__,
            "module": self.__class__.__module__,
            "save_anndata": bool(save_anndata),
            "logger_level": logger_level,
        }
        with open(out_dir / "model_state.pkl", "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

        metadata = {
            "class_name": payload["class_name"],
            "module": payload["module"],
            "save_anndata": payload["save_anndata"],
            "files": ["model_state.pkl"] + (["adata.h5ad"] if save_anndata else []),
        }
        with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(
        cls,
        dir_path: Union[str, Path],
        adata: Optional[AnnData] = None,
    ) -> "scDEF":
        """Load model from disk.

        Args:
            dir_path: directory created by :meth:`save`
            adata: optional AnnData to attach when ``adata.h5ad`` was not saved.

        Returns:
            Loaded model instance.
        """
        in_dir = Path(dir_path).expanduser().resolve()
        state_path = in_dir / "model_state.pkl"
        if not state_path.exists():
            raise FileNotFoundError(f"Model state file not found: '{state_path}'.")

        with open(state_path, "rb") as f:
            try:
                payload = pickle.load(f)
            except ModuleNotFoundError as e:
                # Compatibility path for artifacts pickled under a different
                # NumPy internal module layout (e.g. numpy._core vs numpy.core).
                if "numpy._core" not in str(e):
                    raise
                f.seek(0)

                class _GraphPlaceholder:
                    """Fallback placeholder for optional Graphviz graph classes."""

                    def __init__(self, *args, **kwargs):
                        self.args = args
                        self.kwargs = kwargs

                class _CompatUnpickler(pickle.Unpickler):
                    def find_class(self, module, name):
                        if module.startswith("numpy._core"):
                            module = module.replace("numpy._core", "numpy.core", 1)
                        if module == "numpy.rec":
                            module = "numpy.core.records"
                        if module == "graphviz.graphs" and name in {"Graph", "Digraph"}:
                            return _GraphPlaceholder
                        return super().find_class(module, name)

                payload = _CompatUnpickler(f).load()

        obj = cls.__new__(cls)
        obj.__dict__.update(payload["state"])
        obj._learn_jit_cache = None
        if not hasattr(obj, "marginalize_alpha"):
            obj.marginalize_alpha = False
        if not hasattr(obj, "_marginalize_alpha_init"):
            obj._marginalize_alpha_init = bool(obj.marginalize_alpha)
        if not hasattr(obj, "hierarchy_weight"):
            obj.hierarchy_weight = 1.0
        obj.logger = logging.getLogger(payload.get("class_name", cls.__name__))
        if payload.get("logger_level") is not None:
            obj.logger.setLevel(payload["logger_level"])

        adata_path = in_dir / "adata.h5ad"
        if adata is not None:
            obj.adata = adata
        elif getattr(obj, "adata", None) is not None:
            # Prefer embedded AnnData saved in the model state.
            pass
        elif payload.get("save_anndata", False) and adata_path.exists():
            obj.adata = ad.read_h5ad(adata_path)
        else:
            raise ValueError(
                "No AnnData available. Provide `adata` to load(...), or save with "
                "`save_anndata=True` so `adata.h5ad` is available."
            )

        obj.n_cells, obj.n_genes = obj.adata.shape
        return obj

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

        self.X = self.X.astype(float)

        self.batch_indices_onehot = np.ones((self.adata.shape[0], 1))
        self.batch_lib_sizes = np.sum(self.X, axis=1)
        self.batch_lib_ratio = (
            np.ones((self.X.shape[0], 1))
            * np.mean(self.batch_lib_sizes)
            / np.var(self.batch_lib_sizes)
        )
        self.exposure = jnp.array(self.batch_lib_sizes / np.mean(self.batch_lib_sizes))

        eps = 1e-12

        # ---- NEW: gene_ratio is per-gene and encodes baseline abundance ----
        # Choose a baseline definition. For raw UMI counts, using mean counts per cell is reasonable.
        # If you prefer library-normalized baseline, replace mu = X.mean(0) by mu = (X/lib).mean(0).
        mu = self.X.mean(axis=0) + eps
        mu_bar = float(mu.mean())
        # Prior mean of gene_scale_g will be ~ mu_g / mu_bar
        gene_ratio = mu_bar / mu  # vector length G
        gene_ratio = np.array(gene_ratio)[None, :]  # shape (1, G)
        self.gene_ratio_init = (
            gene_ratio  # will be overridden per batch if batch_key has >1 values
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
                # If multiple batches: compute gene_ratio per batch (still per gene)
                if self.n_batches > 1:
                    self.gene_ratio = np.ones((self.n_batches, self.adata.shape[1]))
                    self.gene_ratio_init = np.ones(
                        (self.n_batches, self.adata.shape[1])
                    )
                    # self.gene_ratio_init = np.tile(
                    #     (mu_bar / mu)[None, :], (self.n_batches, 1)
                    # )
                    for i, b in enumerate(batches):
                        cells = np.where(self.adata.obs[batch_key] == b)[0]
                        self.batch_indices_onehot[cells, i] = 1
                        self.batch_lib_sizes[cells] = np.sum(self.X, axis=1)[cells]
                        self.batch_lib_ratio[cells] = np.mean(
                            self.batch_lib_sizes[cells]
                        ) / (np.var(self.batch_lib_sizes[cells]) + eps)
                        mu_b = self.X[cells].mean(axis=0) + eps
                        mu_b_bar = float(mu_b.mean())
                        self.gene_ratio_init[i] = mu_b_bar / mu_b
                        gene_size = np.sum(self.X[cells], axis=0)
                        gene_means = np.mean(gene_size)
                        gene_vars = np.var(gene_size)
                        self.gene_ratio[i] = (
                            self.gene_ratio[i] * 0 + gene_means / gene_vars
                        )

        self.batch_indices_onehot = jnp.array(self.batch_indices_onehot)
        self.batch_lib_sizes = jnp.array(self.batch_lib_sizes)
        self.batch_lib_ratio = jnp.array(self.batch_lib_ratio)
        self.gene_ratio = jnp.array(self.gene_ratio)

    def _geometric_layer_sizes(self, n_layers: int) -> List[int]:
        """Layer counts from ``n_factors`` (L0) through the ``top_factors`` layer, optional root.

        ``n_layers`` counts only layers from K0 up to and including the ``top_factors``
        width; a final root of width ``1`` (when ``top_factors > 1``) is appended and does
        not count toward ``n_layers``.

        For ``n_layers >= 3``, uses
        ``K_l = K_0 * (K_top / K_0) ** (l / (n_layers - 1))`` for ``l = 0, ..., n_layers - 2``,
        with the last geometric layer set to ``top_factors`` exactly, then appends ``1`` if
        ``top_factors > 1``.

        For ``n_layers == 2``: ``[n_factors, top_factors]``, then appends ``1`` if
        ``top_factors > 1``.
        """
        n_layers = max(2, int(n_layers))
        min_k = float(self.n_factors)
        top_k = float(self.top_factors)
        ratio = top_k / max(min_k, 1.0)
        add_root = int(self.top_factors) > 1
        if n_layers == 2:
            out = [max(1, int(round(min_k))), int(self.top_factors)]
            if out[0] < out[1]:
                out[0] = out[1]
            if add_root:
                out.append(1)
            return out

        denom = float(n_layers - 1)
        out: List[int] = []
        prev: Optional[int] = None
        for l in range(n_layers):
            r = l / denom
            if l == n_layers - 1:
                k_i = int(self.top_factors)
            else:
                k_f = min_k * (ratio**r)
                k_i = max(1, int(round(k_f)))
                if prev is not None and k_i >= prev:
                    k_i = max(1, prev - 1)
            out.append(k_i)
            prev = k_i
        out[-1] = int(self.top_factors)
        for i in range(len(out) - 2, -1, -1):
            if out[i] < out[i + 1]:
                out[i] = out[i + 1]
        if add_root:
            out.append(1)
        return out

    @staticmethod
    def _geometric_layer_sizes_for(
        n_factors: int, top_factors: int, n_layers: int
    ) -> List[int]:
        n_layers = max(2, int(n_layers))
        min_k = float(int(n_factors))
        top_k = float(int(top_factors))
        ratio = top_k / max(min_k, 1.0)
        add_root = int(top_factors) > 1
        if n_layers == 2:
            out = [max(1, int(round(min_k))), int(top_factors)]
            if out[0] < out[1]:
                out[0] = out[1]
            if add_root:
                out.append(1)
            return out

        denom = float(n_layers - 1)
        out: List[int] = []
        prev: Optional[int] = None
        for l in range(n_layers):
            r = l / denom
            if l == n_layers - 1:
                k_i = int(top_factors)
            else:
                k_f = min_k * (ratio**r)
                k_i = max(1, int(round(k_f)))
                if prev is not None and k_i >= prev:
                    k_i = max(1, prev - 1)
            out.append(k_i)
            prev = k_i
        out[-1] = int(top_factors)
        for i in range(len(out) - 2, -1, -1):
            if out[i] < out[i + 1]:
                out[i] = out[i + 1]
        if add_root:
            out.append(1)
        return out

    def _should_collapse_adjacent_layer_sizes(
        self,
        layer_sizes: Sequence[int],
        parent_idx: int,
        fraction: float,
        min_l0_factors: int = 10,
        min_upper_factors: int = 5,
    ) -> bool:
        """True when the layer above is nearly as wide as the layer below (redundant step).

        ``parent_idx`` indexes the lower layer in ``layer_sizes`` (L0 when 0). The
        hierarchy grows upward: L0 is the child of L1, L1 of L2, etc.
        """
        layer_sizes = [int(x) for x in layer_sizes]
        if parent_idx < 0 or parent_idx + 1 >= len(layer_sizes):
            return False
        n_lower = layer_sizes[parent_idx]
        n_upper = layer_sizes[parent_idx + 1]
        min_lower = int(min_l0_factors) if parent_idx == 0 else int(min_upper_factors)
        if n_lower <= min_lower:
            return False
        return n_upper >= float(fraction) * float(n_lower)

    def _should_collapse_redundant_l1(
        self,
        fraction: float,
        min_l0_factors: int = 10,
    ) -> bool:
        """Backward-compatible L0/L1 redundancy check."""
        layer_sizes = [len(f) for f in self.factor_lists]
        return self._should_collapse_adjacent_layer_sizes(
            layer_sizes, 0, fraction, min_l0_factors
        )

    def _sanitize_layer_sizes(
        self,
        layer_sizes: Sequence[int],
        old_keep: Optional[Sequence[int]] = None,
    ) -> Tuple[List[int], List[int]]:
        """Clip non-increasing widths and drop consecutive duplicate layer sizes.

        Returns sanitized sizes and ``old_keep[j]`` = original layer index kept at
        new position ``j``. Used on refit so warm-start can compose ``W`` across
        removed layers.
        """
        sizes = [int(x) for x in layer_sizes]
        if old_keep is None:
            keep = list(range(len(sizes)))
        else:
            keep = [int(i) for i in old_keep]
            if len(keep) != len(sizes):
                raise ValueError(
                    "old_keep must have the same length as layer_sizes "
                    f"({len(keep)} != {len(sizes)})."
                )
        for i in range(len(sizes) - 1):
            if sizes[i + 1] > sizes[i]:
                sizes[i + 1] = sizes[i]
        new_sizes: List[int] = []
        new_keep: List[int] = []
        for j, size in enumerate(sizes):
            if len(new_sizes) == 0 or size != new_sizes[-1]:
                new_sizes.append(size)
                new_keep.append(keep[j])
        return new_sizes, new_keep

    def _collapse_redundant_adjacent_layer_sizes(
        self,
        layer_sizes: Sequence[int],
        fraction: float,
        min_l0_factors: int = 10,
        min_upper_factors: int = 5,
    ) -> Tuple[List[int], List[int], int]:
        """Drop redundant intermediate layers (bottom-up scan along layer_sizes)."""
        sizes = [int(x) for x in layer_sizes]
        old_keep = list(range(len(sizes)))
        n_dropped = 0
        i = 0
        while i < len(sizes) - 1 and len(sizes) >= 3:
            if not self._should_collapse_adjacent_layer_sizes(
                sizes,
                i,
                fraction,
                min_l0_factors,
                min_upper_factors,
            ):
                i += 1
                continue
            dropped_old = old_keep[i + 1]
            promote_old = old_keep[i + 2]
            lower_name = (
                self.layer_names[old_keep[i]]
                if old_keep[i] < len(self.layer_names)
                else f"L{old_keep[i]}"
            )
            dropped_name = (
                self.layer_names[dropped_old]
                if dropped_old < len(self.layer_names)
                else f"L{dropped_old}"
            )
            promote_name = (
                self.layer_names[promote_old]
                if promote_old < len(self.layer_names)
                else f"L{promote_old}"
            )
            self.logger.info(
                "Redundant hierarchy on refit (%s %s, %s %s factors; threshold %.2f). "
                "Dropping %s; %s becomes the new parent of %s.",
                sizes[i],
                lower_name,
                sizes[i + 1],
                dropped_name,
                float(fraction),
                dropped_name,
                promote_name,
                lower_name,
            )
            sizes = sizes[: i + 1] + sizes[i + 2 :]
            old_keep = old_keep[: i + 1] + old_keep[i + 2 :]
            n_dropped += 1
            if hasattr(self, "n_layers_schedule"):
                self.n_layers_schedule = max(2, int(self.n_layers_schedule) - 1)
        return sizes, old_keep, n_dropped

    def _compose_w_to_parent_layer(
        self,
        layer_names: Sequence[str],
        factor_lists: Sequence[np.ndarray],
        child_old_idx: int,
        parent_old_idx: int,
    ) -> np.ndarray:
        """Compose ``W`` from ``child_old_idx`` down to ``parent_old_idx`` (inclusive)."""
        if child_old_idx <= parent_old_idx:
            raise ValueError("child_old_idx must be greater than parent_old_idx.")
        fl = [np.asarray(f, dtype=int) for f in factor_lists]
        w = np.eye(len(fl[parent_old_idx]), dtype=np.float32)
        for k in range(parent_old_idx + 1, child_old_idx + 1):
            w = (
                np.asarray(self.pmeans[f"{layer_names[k]}W"], dtype=np.float32)[
                    np.ix_(fl[k], fl[k - 1])
                ]
                @ w
            )
        return np.clip(w, 1e-8, None).astype(np.float32)

    def _rescale_relevance_init(
        self,
        values: np.ndarray,
        target_mean: float,
        *,
        max_ratio: Optional[float] = 50.0,
        eps: float = 1e-8,
    ) -> np.ndarray:
        """Rescale positive per-factor BRD/ARD warm starts preserving ratios.

        Centers the geometric mean on ``target_mean`` (``brd_mean`` / ``shrinkage_mean``).
        Optionally limits the max/min ratio across factors via ``max_ratio``.
        """
        arr = np.asarray(values, dtype=np.float64)
        flat = arr.ravel()
        target = max(float(target_mean), eps)
        if flat.size == 0:
            return arr.astype(np.float32)
        positive = flat > eps
        if not np.any(positive):
            return np.full(arr.shape, target, dtype=np.float32)

        log_v = np.log(np.clip(flat[positive], eps, None))
        log_target = np.log(target)
        log_scaled = log_v - np.mean(log_v) + log_target
        if max_ratio is not None and float(max_ratio) > 1.0:
            half_span = 0.5 * np.log(float(max_ratio))
            log_scaled = np.clip(
                log_scaled, log_target - half_span, log_target + half_span
            )
        out = flat.copy()
        out[positive] = np.exp(log_scaled)
        out[~positive] = target
        return out.reshape(arr.shape).astype(np.float32)

    def _prepare_refit_relevance_inits(
        self,
        init_brd,
        init_ard,
        *,
        refit_rescale_relevance: bool = True,
        refit_relevance_max_ratio: Optional[float] = 50.0,
    ):
        """Rescale BRD/ARD warm starts on refit (see ``_rescale_relevance_init``)."""
        if not refit_rescale_relevance:
            return init_brd, init_ard
        max_ratio = (
            None
            if refit_relevance_max_ratio is None
            else float(refit_relevance_max_ratio)
        )
        if init_brd is not None and self.use_brd:
            before = np.asarray(init_brd, dtype=np.float64)
            init_brd = self._rescale_relevance_init(
                init_brd, self.brd_mean, max_ratio=max_ratio
            )
            self.logger.info(
                "Refit: rescaled BRD warm start to geometric mean %.3g "
                "(range %.3g–%.3g -> %.3g–%.3g).",
                float(self.brd_mean),
                float(np.min(before)),
                float(np.max(before)),
                float(np.min(init_brd)),
                float(np.max(init_brd)),
            )
        if init_ard is not None:
            before = np.asarray(init_ard, dtype=np.float64)
            init_ard = self._rescale_relevance_init(
                init_ard, self.shrinkage_mean, max_ratio=max_ratio
            )
            self.logger.info(
                "Refit: rescaled ARD warm start to geometric mean %.3g "
                "(range %.3g–%.3g -> %.3g–%.3g).",
                float(self.shrinkage_mean),
                float(np.min(before)),
                float(np.max(before)),
                float(np.min(init_ard)),
                float(np.max(init_ard)),
            )
        return init_brd, init_ard

    def _refit_old_layer_sizes_for_w(
        self,
        old_layer_sizes_full: Sequence[int],
        new_layer_sizes: Sequence[int],
        init_w: Sequence[np.ndarray],
        old_keep: Optional[Sequence[int]],
        n_original: int,
    ) -> List[int]:
        """Map each warm-started ``W`` layer to its pre-refit width."""
        n_w = len(init_w)
        if old_keep is not None and len(old_keep) > 0 and len(old_keep) != n_original:
            if len(old_keep) != n_w:
                raise ValueError(
                    f"init_w has {n_w} layers but old_keep has {len(old_keep)} entries."
                )
            return [int(old_layer_sizes_full[int(i)]) for i in old_keep]
        n_old = len(old_layer_sizes_full)
        if n_w <= n_old:
            return [int(old_layer_sizes_full[j]) for j in range(n_w)]
        out = [int(old_layer_sizes_full[j]) for j in range(n_old)]
        out.extend(int(new_layer_sizes[j]) for j in range(n_old, n_w))
        return out

    def _rescale_w_inits_for_layer_sizes(
        self,
        old_layer_sizes: Sequence[int],
        new_layer_sizes: Sequence[int],
        init_w: Optional[Sequence[np.ndarray]],
    ) -> Optional[List[np.ndarray]]:
        """Scale warm-started ``W`` when refit layer widths change.

        Matches the cold-start convention ``W ∝ 1 / K`` by multiplying each layer's
        init ``W`` by ``old_K / new_K`` (see ``init_var_params`` prior means).
        """
        if init_w is None:
            return None
        old_sizes = [max(int(x), 1) for x in old_layer_sizes]
        new_sizes = [max(int(x), 1) for x in new_layer_sizes]
        if len(old_sizes) != len(new_sizes):
            raise ValueError(
                f"old_layer_sizes length ({len(old_sizes)}) must match "
                f"new_layer_sizes ({len(new_sizes)}) for W rescaling."
            )
        if len(init_w) != len(old_sizes):
            raise ValueError(
                f"init_w has {len(init_w)} layers but expected {len(old_sizes)}."
            )
        scaled: List[np.ndarray] = []
        for layer_idx, w in enumerate(init_w):
            factor = old_sizes[layer_idx] / new_sizes[layer_idx]
            w_arr = (np.asarray(w, dtype=np.float32) * np.float32(factor)).astype(
                np.float32
            )
            scaled.append(w_arr)
            if factor != 1.0:
                layer_label = (
                    self.layer_names[layer_idx]
                    if layer_idx < len(self.layer_names)
                    else f"L{layer_idx}"
                )
                self.logger.info(
                    "Refit: scaled %s W warm start by %.4g (layer size %s -> %s).",
                    layer_label,
                    factor,
                    old_sizes[layer_idx],
                    new_sizes[layer_idx],
                )
        return scaled

    def _build_collapsed_refit_init(
        self,
        old_keep: Sequence[int],
        factor_lists: Optional[Sequence[np.ndarray]] = None,
        layer_names: Optional[Sequence[str]] = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
        """Warm-start inits after dropping one or more redundant intermediate layers."""
        old_keep = [int(i) for i in old_keep]
        if len(old_keep) < 2:
            raise ValueError("Collapsed hierarchy must keep at least two layers.")

        fl = [
            np.asarray(f, dtype=int)
            for f in (factor_lists if factor_lists is not None else self.factor_lists)
        ]
        names = list(layer_names if layer_names is not None else self.layer_names)

        init_z: List[np.ndarray] = [
            np.asarray(self.pmeans[f"{names[old_keep[0]]}z"], dtype=np.float32)[
                :, fl[old_keep[0]]
            ]
        ]
        init_w: List[np.ndarray] = [
            np.asarray(self.pmeans[f"{names[old_keep[0]]}W"], dtype=np.float32)[
                fl[old_keep[0]]
            ]
        ]

        for new_j in range(1, len(old_keep)):
            child_old = old_keep[new_j]
            parent_old = old_keep[new_j - 1]
            init_w.append(
                self._compose_w_to_parent_layer(names, fl, child_old, parent_old)
            )
            init_z.append(
                np.asarray(self.pmeans[f"{names[child_old]}z"], dtype=np.float32)[
                    :, fl[child_old]
                ]
            )

        init_brd = np.asarray(self.pmeans["brd"], dtype=np.float32)[fl[old_keep[0]]]
        init_ard = np.asarray(self.pmeans["factor_means"], dtype=np.float32)[
            fl[old_keep[0]]
        ]
        return init_z, init_w, init_brd, init_ard

    def _resolve_factor_name_in_old_layers(
        self,
        factor_name: str,
        factor_lists: Sequence[np.ndarray],
        factor_names: Sequence[Sequence[str]],
    ) -> Tuple[int, int, int]:
        """Return ``(layer_idx, slot_idx, original_idx)`` for an old factor name."""
        for layer_idx, names in enumerate(factor_names):
            if factor_name in names:
                slot_idx = int(list(names).index(factor_name))
                return (
                    int(layer_idx),
                    slot_idx,
                    int(np.asarray(factor_lists[layer_idx], dtype=int)[slot_idx]),
                )
        raise ValueError(f"Unknown refit_top_factor {factor_name!r}.")

    def _allocate_groups_by_top(
        self, child_to_top: np.ndarray, n_groups: int, n_top: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Assign child groups to a smaller parent layer while preserving top labels."""
        child_to_top = np.asarray(child_to_top, dtype=int)
        n_groups = int(n_groups)
        if n_groups < n_top:
            raise ValueError(
                f"Intermediate layer size {n_groups} cannot be smaller than "
                f"the number of sensible top factors ({n_top})."
            )
        counts = np.bincount(child_to_top, minlength=n_top).astype(float)
        active = np.where(counts > 0)[0]
        group_counts = np.zeros(n_top, dtype=int)
        group_counts[active] = 1
        remaining = n_groups - int(group_counts.sum())
        if remaining > 0 and counts.sum() > 0:
            quotas = counts / counts.sum() * remaining
            extra = np.floor(quotas).astype(int)
            group_counts += extra
            remaining = n_groups - int(group_counts.sum())
            order = np.argsort(-(quotas - extra))
            for top_idx in order[:remaining]:
                group_counts[int(top_idx)] += 1
        while int(group_counts.sum()) > n_groups:
            candidates = np.where(group_counts > 1)[0]
            if len(candidates) == 0:
                break
            group_counts[candidates[np.argmin(counts[candidates])]] -= 1

        group_to_top = np.concatenate(
            [np.full(group_counts[t], t, dtype=int) for t in range(n_top)]
        )
        if len(group_to_top) != n_groups:
            raise RuntimeError("Failed to allocate requested number of groups.")
        child_to_group = np.zeros(len(child_to_top), dtype=int)
        for top_idx in range(n_top):
            children = np.where(child_to_top == top_idx)[0]
            groups = np.where(group_to_top == top_idx)[0]
            if len(children) == 0 or len(groups) == 0:
                continue
            for pos, child in enumerate(children):
                child_to_group[int(child)] = int(groups[pos % len(groups)])
        return child_to_group, group_to_top

    def _build_frontier_refit_init(
        self,
        top_factors: Sequence[str],
        factor_lists: Optional[Sequence[np.ndarray]] = None,
        factor_names: Optional[Sequence[Sequence[str]]] = None,
        layer_names: Optional[Sequence[str]] = None,
    ) -> Tuple[List[int], List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
        """Warm-start a geometric hierarchy whose top layer is a mixed-depth frontier.

        The rebuilt hierarchy preserves the current fitted depth (non-root
        layer count, and root presence when width-1 root exists), instead of
        using ``n_layers_schedule`` directly. This keeps the selected frontier
        factors on the same top non-root layer index as before refit.
        """
        fl = [
            np.asarray(f, dtype=int)
            for f in (factor_lists if factor_lists is not None else self.factor_lists)
        ]
        names = list(layer_names if layer_names is not None else self.layer_names)
        old_factor_names = (
            [list(x) for x in factor_names]
            if factor_names is not None
            else [list(x) for x in self.factor_names]
        )
        top_factors = [str(x) for x in top_factors]
        if len(top_factors) == 0:
            raise ValueError("refit_top_factors must contain at least one factor.")
        resolved = [
            self._resolve_factor_name_in_old_layers(f, fl, old_factor_names)
            for f in top_factors
        ]
        n_l0 = len(fl[0])
        n_top = len(resolved)
        has_old_root = len(fl) > 1 and int(len(fl[-1])) == 1
        n_layers_target = len(fl) - 1 if has_old_root else len(fl)
        layer_sizes, _ = self._sanitize_layer_sizes(
            self._geometric_layer_sizes_for(n_l0, n_top, int(n_layers_target))
        )
        has_root = len(layer_sizes) > 1 and int(layer_sizes[-1]) == 1
        top_layer_idx = len(layer_sizes) - 2 if has_root else len(layer_sizes) - 1
        if layer_sizes[top_layer_idx] != n_top:
            raise RuntimeError("Frontier top layer size was not preserved.")

        init_w: List[np.ndarray] = [
            np.asarray(self.pmeans[f"{names[0]}W"], dtype=np.float32)[fl[0]]
        ]
        init_z: List[np.ndarray] = [
            np.asarray(self.pmeans[f"{names[0]}z"], dtype=np.float32)[:, fl[0]]
        ]

        top_to_l0 = np.zeros((n_top, n_l0), dtype=np.float32)
        top_z = np.zeros((self.n_cells, n_top), dtype=np.float32)
        for top_slot, (old_layer, old_slot, _) in enumerate(resolved):
            if old_layer == 0:
                top_to_l0[top_slot, old_slot] = 1.0
            else:
                composed = self._compose_w_to_parent_layer(names, fl, old_layer, 0)
                top_to_l0[top_slot] = composed[old_slot]
            top_z[:, top_slot] = np.asarray(
                self.pmeans[f"{names[old_layer]}z"], dtype=np.float32
            )[:, fl[old_layer][old_slot]]
        top_to_l0 = np.clip(top_to_l0, 1e-8, None)

        current_z = init_z[0]
        current_to_l0 = np.eye(n_l0, dtype=np.float32)
        child_to_top = np.argmax(top_to_l0, axis=0).astype(int)
        for layer_idx in range(1, top_layer_idx):
            next_size = int(layer_sizes[layer_idx])
            child_to_group, group_to_top = self._allocate_groups_by_top(
                child_to_top, next_size, n_top
            )
            w = np.zeros((next_size, current_z.shape[1]), dtype=np.float32)
            for child_idx, group_idx in enumerate(child_to_group):
                w[int(group_idx), int(child_idx)] = 1.0
            group_sizes = np.maximum(w.sum(axis=1, keepdims=True), 1.0)
            init_w.append(np.clip(w / group_sizes, 1e-8, None).astype(np.float32))
            current_z = current_z @ w.T
            current_to_l0 = w @ current_to_l0
            child_to_top = group_to_top
            init_z.append(np.clip(current_z, 1e-8, None).astype(np.float32))

        w_top = top_to_l0 @ current_to_l0.T
        w_top = w_top / np.maximum(w_top.sum(axis=0, keepdims=True), 1e-8)
        init_w.append(np.clip(w_top, 1e-8, None).astype(np.float32))
        init_z.append(np.clip(top_z, 1e-8, None).astype(np.float32))

        if has_root:
            init_w.append(
                np.ones((1, n_top), dtype=np.float32) / max(float(n_top), 1.0)
            )
            old_root_idx = len(fl) - 1
            if len(fl[old_root_idx]) == 1:
                root_z = np.asarray(
                    self.pmeans[f"{names[old_root_idx]}z"], dtype=np.float32
                )[:, fl[old_root_idx]]
            else:
                root_z = np.ones((self.n_cells, 1), dtype=np.float32)
            init_z.append(np.clip(root_z, 1e-8, None).astype(np.float32))

        init_brd = np.asarray(self.pmeans["brd"], dtype=np.float32)[fl[0]]
        init_ard = np.asarray(self.pmeans["factor_means"], dtype=np.float32)[fl[0]]
        return layer_sizes, init_z, init_w, init_brd, init_ard

    def update_model_size(
        self,
        max_n_factors=None,
        n_layers=None,
        layer_sizes=None,
        use_decay_factor_schedule: bool = False,
    ):
        """Update latent hierarchy dimensions.

        Args:
            max_n_factors: bottom-layer factor count when ``use_decay_factor_schedule``
                is True (``iscDEF`` marker layer 0 path).
            n_layers: target number of geometric layers from K0 through ``top_factors`` (a
                final root of size ``1`` when ``top_factors > 1`` is not counted), or maximum
                layers for the decay schedule.
            layer_sizes: explicit per-layer sizes. If provided, sizes are
                sanitized to be non-increasing and consecutive duplicates are
                collapsed.
            use_decay_factor_schedule: if True, use ``decay_factor``-based halving
                (``iscDEF`` only). If False, use ``n_factors``, ``top_factors``, and
                ``n_layers_schedule`` on ``self`` for a geometric ladder (``scDEF``).
        """
        if layer_sizes is not None:
            layer_sizes, _ = self._sanitize_layer_sizes(layer_sizes)
            self.layer_sizes = layer_sizes
            self.n_layers = len(self.layer_sizes)
            self.layer_names = [f"L{i}" for i in range(self.n_layers)]
            self.factor_lists = [np.arange(size) for size in self.layer_sizes]
            self.set_factor_names()
            return

        if use_decay_factor_schedule:
            if n_layers is None:
                n_layers = self.n_layers_schedule
            if max_n_factors is None:
                raise ValueError(
                    "max_n_factors is required when use_decay_factor_schedule=True."
                )
            df = getattr(self, "decay_factor", None)
            if df is None or float(df) <= 0:
                raise ValueError(
                    "decay_factor must be set on the model when use_decay_factor_schedule=True."
                )
            n_factors = int(max_n_factors)
            self.layer_sizes = []
            self.layer_sizes.append(n_factors)
            if int(n_layers) > 1:
                while len(self.layer_sizes) < int(n_layers):
                    n_factors = int(np.floor(n_factors / float(df)))
                    self.layer_sizes.append(n_factors)
                    if n_factors == 1:
                        break
                if self.layer_sizes[-1] > 1:
                    self.layer_sizes[-1] = 1

            self.n_layers = len(self.layer_sizes)
            self.layer_names = [f"L{i}" for i in range(self.n_layers)]
            self.factor_lists = [np.arange(size) for size in self.layer_sizes]
            self.set_factor_names()
            return

        # scDEF geometric default
        if n_layers is None:
            n_layers = self.n_layers_schedule
        self.layer_sizes = self._geometric_layer_sizes(int(n_layers))
        self.n_layers = len(self.layer_sizes)
        self.n_factors = int(self.layer_sizes[0])
        self.layer_names = [f"L{i}" for i in range(self.n_layers)]
        self.factor_lists = [np.arange(size) for size in self.layer_sizes]
        self.set_factor_names()

    def update_model_priors(self, update_alpha_from_cov: bool = True):
        if self.use_brd:
            self.factor_shapes = [1.0] + [
                self.factor_shape  # * self.layer_sizes[layer_idx] / self.layer_sizes[0]
                for layer_idx in range(1, self.n_layers)
            ]
        else:
            self.factor_shapes = [
                self.factor_shape  # * self.layer_sizes[layer_idx] / self.layer_sizes[0]
                for layer_idx in range(self.n_layers)
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
                    * 1.0
                )
                prior_shapes = jnp.clip(jnp.array(prior_shapes), 1e-12, 1e12)
                prior_rates = jnp.clip(jnp.array(prior_rates), 1e-12, 1e12)

            self.w_priors.append([prior_shapes, prior_rates])

        self.cell_alpha_factor = self.batch_lib_sizes / float(
            np.median(self.batch_lib_sizes)
        )
        if self.set_alpha_from_cov and update_alpha_from_cov:
            self.alpha = (
                float(np.median(self.batch_lib_sizes))
                / float(self.layer_sizes[0])
                * self.hierarchy_weight
            )

    def _get_factor_obs_l0(self, required_cols: Optional[set] = None):
        """Layer-0 slice of ``factor_obs`` / ``factor_obs_full`` for filtering."""
        required = {"original_factor_idx"}
        if required_cols is not None:
            required = required | set(required_cols)

        def _has_required(key):
            return key in self.adata.uns and required.issubset(
                set(self.adata.uns[key].columns)
            )

        if not _has_required("factor_obs_full") and not _has_required("factor_obs"):
            from scdef.tools.factor import factor_diagnostics

            factor_diagnostics(self)

        source_key = (
            "factor_obs_full" if _has_required("factor_obs_full") else "factor_obs"
        )
        factor_obs = self.adata.uns[source_key]
        if "child_layer" in factor_obs.columns:
            return factor_obs[factor_obs["child_layer"] == self.layer_names[0]]
        return factor_obs

    def _batch_purity_keep_mask(
        self,
        factor_obs_l0,
        batch_purity_max: Optional[float] = None,
        batch_purity_soft_max: Optional[float] = None,
    ) -> np.ndarray:
        """Boolean mask over layer-0 factor indices that pass batch purity caps."""
        n_factors = int(self.layer_sizes[0])
        keep_mask = np.zeros(n_factors, dtype=bool)
        if batch_purity_max is None and batch_purity_soft_max is None:
            return np.ones(n_factors, dtype=bool)

        original_idx = factor_obs_l0["original_factor_idx"].to_numpy(dtype=int)
        valid = (original_idx >= 0) & (original_idx < n_factors)
        oidx = original_idx[valid]
        if oidx.size == 0:
            return keep_mask

        keep_mask[oidx] = True

        if batch_purity_max is not None:
            if "batch_purity" not in factor_obs_l0.columns:
                raise KeyError(
                    "batch_purity is missing from factor_obs. Run "
                    "scdef.tools.factor_diagnostics(model, batch_key=...) first."
                )
            vals = factor_obs_l0["batch_purity"].to_numpy(dtype=float)[valid]
            purity = np.full(n_factors, np.nan, dtype=float)
            purity[oidx] = vals
            keep_mask &= np.isfinite(purity) & (purity <= float(batch_purity_max))

        if batch_purity_soft_max is not None:
            if "batch_purity_soft" not in factor_obs_l0.columns:
                raise KeyError(
                    "batch_purity_soft is missing from factor_obs. Run "
                    "scdef.tools.factor_diagnostics(model, batch_key=...) first."
                )
            vals = factor_obs_l0["batch_purity_soft"].to_numpy(dtype=float)[valid]
            purity_soft = np.full(n_factors, np.nan, dtype=float)
            purity_soft[oidx] = vals
            keep_mask &= np.isfinite(purity_soft) & (
                purity_soft <= float(batch_purity_soft_max)
            )

        return keep_mask

    def get_effective_factors(
        self,
        brd_min: Optional[float] = 1.0,
        ard_min: Optional[float] = 0.001,
        clarity_min: Optional[float] = 0.5,
        n_eff_parents_max: float = 1.5,
        local_l0_scores: bool = False,
        min_cells: Optional[float] = 0.001,
        batch_purity_max: Optional[float] = None,
        batch_purity_soft_max: Optional[float] = None,
    ):
        layer_name = self.layer_names[0]
        if min_cells != 0:
            min_cells = (
                np.maximum(10, min_cells * self.n_cells)
                if min_cells < 1.0
                else min_cells
            )

        normed = (
            self.pmeans[f"{layer_name}z"]
            / np.sum(self.pmeans[f"{layer_name}z"], axis=1)[:, None]
        )
        assignments = np.argmax(normed, axis=1)
        counts = np.array(
            [np.count_nonzero(assignments == a) for a in range(self.layer_sizes[0])]
        )

        keep = np.array(range(self.layer_sizes[0]))[np.where(counts >= min_cells)[0]]
        use_batch_purity = (
            batch_purity_max is not None or batch_purity_soft_max is not None
        )
        if self.use_brd or use_batch_purity:
            required_cols: set = set()
            if self.use_brd:
                required_cols |= {"BRD", "ARD"}
            if batch_purity_max is not None:
                required_cols.add("batch_purity")
            if batch_purity_soft_max is not None:
                required_cols.add("batch_purity_soft")
            factor_obs_l0 = self._get_factor_obs_l0(required_cols=required_cols)

        if self.use_brd:
            original_idx = factor_obs_l0["original_factor_idx"].to_numpy(dtype=int)
            brd_vals = factor_obs_l0["BRD"].to_numpy(dtype=float)
            ard_vals = factor_obs_l0["ARD"].to_numpy(dtype=float)

            valid_idx = (original_idx >= 0) & (original_idx < self.layer_sizes[0])
            original_idx_f = original_idx[valid_idx]
            brd_vals = brd_vals[valid_idx]
            ard_vals = ard_vals[valid_idx]

            brd = np.full(self.layer_sizes[0], np.nan, dtype=float)
            ard = np.full(self.layer_sizes[0], np.nan, dtype=float)
            brd[original_idx_f] = brd_vals
            ard[original_idx_f] = ard_vals
            ard_sum = np.nansum(ard)

            if local_l0_scores:
                clarity_vals = factor_obs_l0["clarity_score_01"].to_numpy(dtype=float)[
                    valid_idx
                ]
                clarity = np.full(self.layer_sizes[0], np.nan, dtype=float)
                clarity[original_idx_f] = clarity_vals
                valid = np.isfinite(brd) & np.isfinite(ard) & np.isfinite(clarity)
                tree_ok = clarity >= float(clarity_min)
            elif "avg_n_eff_parents" in factor_obs_l0.columns:
                avg_vals = factor_obs_l0["avg_n_eff_parents"].to_numpy(dtype=float)[
                    valid_idx
                ]
                avg_neff = np.full(self.layer_sizes[0], np.nan, dtype=float)
                avg_neff[original_idx_f] = avg_vals
                valid = np.isfinite(brd) & np.isfinite(ard) & np.isfinite(avg_neff)
                tree_ok = avg_neff < float(n_eff_parents_max)
            else:
                clarity_vals = factor_obs_l0["clarity_score_01"].to_numpy(dtype=float)[
                    valid_idx
                ]
                clarity = np.full(self.layer_sizes[0], np.nan, dtype=float)
                clarity[original_idx_f] = clarity_vals
                valid = np.isfinite(brd) & np.isfinite(ard) & np.isfinite(clarity)
                tree_ok = clarity >= float(clarity_min)

            pass_mask = valid & (brd >= brd_min) & tree_ok & (ard >= ard_min * ard_sum)
            if use_batch_purity:
                pass_mask &= self._batch_purity_keep_mask(
                    factor_obs_l0,
                    batch_purity_max=batch_purity_max,
                    batch_purity_soft_max=batch_purity_soft_max,
                )
            brd_keep = np.where(pass_mask)[0]
            keep = np.unique(list(set(brd_keep).intersection(keep)))
        elif use_batch_purity:
            batch_keep = np.where(
                self._batch_purity_keep_mask(
                    factor_obs_l0,
                    batch_purity_max=batch_purity_max,
                    batch_purity_soft_max=batch_purity_soft_max,
                )
            )[0]
            keep = np.unique(list(set(batch_keep).intersection(keep)))

        return keep

    def init_var_params(
        self,
        init_budgets=True,
        init_alpha=True,
        init_z=None,
        init_w=None,
        init_brd=None,
        init_ard=None,
        init_gene_scale: Optional[np.ndarray] = None,
        nmf_init=False,
        z_init_concentration=0.05,
        **kwargs,
    ):
        rngs = random.split(random.PRNGKey(self.seed), self.n_layers)
        init_z_provided = init_z is not None

        def _get_layer_init(init_value, layer_idx):
            if init_value is None:
                return None
            if isinstance(init_value, (list, tuple)):
                if layer_idx >= len(init_value) or init_value[layer_idx] is None:
                    return None
                return np.asarray(init_value[layer_idx], dtype=np.float32)
            if layer_idx == 0:
                return np.asarray(init_value, dtype=np.float32)
            return None

        if init_budgets:
            m = np.array(self.batch_lib_sizes / np.mean(self.batch_lib_sizes))[
                :, None
            ]  # (N,1)
            m = np.clip(m, 1e-3, 1e2)
            v = m / 10.0
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
            # initialize gene scales at the prior mean (per gene)
            # prior mean under Gamma(shape, rate=shape*gene_ratio) is 1/gene_ratio
            if init_gene_scale is not None:
                m = np.asarray(init_gene_scale, dtype=np.float32)
                if m.ndim == 1:
                    m = np.tile(m[None, :], (max(int(self.n_batches), 1), m.shape[0]))
                elif m.shape[0] == 1 and int(self.n_batches) > 1:
                    m = np.tile(m, (int(self.n_batches), 1))
                m = np.clip(m, 1e-6, 1e6)
            else:
                m = 1.0 / np.array(self.gene_ratio_init)  # shape: (G,) or (B,G)
                m = np.clip(m, 1e-6, 1e6)
            v = m / 10.0

            m = jnp.array(m, dtype=jnp.float32)
            v = jnp.array(v, dtype=jnp.float32)

            self.global_params = [
                jnp.array(
                    (
                        jnp.log(m**2 / jnp.sqrt(m**2 + v)),
                        jnp.log(jnp.sqrt(jnp.log(1 + v / (m**2)))),
                    )
                ),
            ]
        else:
            self.local_params = [self.local_params[0]]
            self.global_params = [self.global_params[0]]
        # BRD
        m_brd = tfd.Gamma(100.0, 100.0 / (self.brd_mean)).sample(
            seed=rngs[0],
            sample_shape=[self.layer_sizes[0], 1],
        )  # self.brd_mean
        v_brd = m_brd**2 / 100.0
        if init_brd is not None:
            m_brd = init_brd
            v_brd = m_brd**2 / 100.0
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

        if nmf_init and init_w is None:
            self.logger.info("Initializing the factor weights with NMF.")
            init_z, init_w = self.get_nmf_init(**kwargs)

        z_shapes = []
        z_rates = []
        rng_cnt = 0

        for layer_idx in range(
            self.n_layers
        ):  # we go from layer 0 (bottom) to layer L (top)
            # Init z
            a = z_init_concentration
            clip = 1e-3  # alpha=1 and clip=1e-6 allows for many factors to be learned, but leads to BRD becoming kind of big

            if (
                layer_idx > 0
            ):  # If the top layer inits to very small numbers, the loss goes crazy when MC=10...
                clip = 1e-3
                a = 1.0

            if layer_idx == self.n_layers - 1:
                a = 100.0
                clip = 1e-3

            z_init_layer = _get_layer_init(init_z, layer_idx)
            if z_init_layer is not None and not nmf_init:
                m = jnp.clip(jnp.asarray(z_init_layer, dtype=jnp.float32), clip, 1e6)
                m = jnp.clip(
                    tfd.Gamma(z_init_concentration, z_init_concentration / m).sample(
                        seed=rngs[rng_cnt],
                    ),
                    clip,
                    1e6,
                )
                v = m**2 / 100.0
            elif (
                nmf_init
                and (not init_z_provided)
                and layer_idx == 0
                and z_init_layer is not None
                and z_init_layer.shape[1] == self.layer_sizes[layer_idx]
                and z_init_layer.shape[0] == self.n_cells
            ):
                # Use NMF L0 z pattern, but keep balanced per-cell mass and prior scale.
                iz = z_init_layer.astype(np.float32)
                # Prevent single-factor domination by normalizing each cell across factors.
                iz = iz / np.maximum(np.max(iz, axis=1, keepdims=True), 1e-8)
                iz = iz + 1.0  # / self.layer_sizes[layer_idx]
                # iz = iz * float(self.layer_sizes[layer_idx])  # prior mean ~1 per factor
                m = jnp.asarray(np.clip(iz, max(1e-3, clip), 1e1), dtype=jnp.float32)
                m = jnp.clip(
                    tfd.Gamma(1.0, 1.0 / m).sample(
                        seed=rngs[rng_cnt],
                    ),
                    clip,
                    1e1,
                )
                rng_cnt += 1
                v = m / 100.0

                if layer_idx > 0:
                    v = m / 10.0
            else:
                m = jnp.clip(
                    tfd.Gamma(a, a / 1.0).sample(
                        seed=rngs[rng_cnt],
                        sample_shape=[self.n_cells, self.layer_sizes[layer_idx]],
                    ),
                    clip,
                    4.0,
                )
                v = m / 100.0

                if layer_idx > 0:
                    v = m / 10.0
                # sd = 2.0  # tunable
                # Lognormal with mean = 1 (corrected via -sd²/2 shift in log space)
                # m = tfd.LogNormal(loc=-sd**2 / 2, scale=sd).sample(
                #     seed=rngs[rng_cnt],
                #     sample_shape=[self.n_cells, self.layer_sizes[layer_idx]],
                # )
                # m = jnp.clip(m, clip, 1e1)
                # rng_cnt += 1

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

            a = 10.0

            w_init_layer = _get_layer_init(init_w, layer_idx)
            if w_init_layer is not None and not nmf_init:
                m = jnp.clip(jnp.asarray(w_init_layer, dtype=jnp.float32), 1e-3, 1e8)
                v = m**2 / 100.0
            else:
                m = 1.0 / self.layer_sizes[layer_idx] * jnp.ones((in_layer, out_layer))
                m = m * self.w_priors[layer_idx][0] / self.w_priors[layer_idx][1]
                if layer_idx < self.n_layers - 1:
                    m = tfd.Gamma(a, a / (m)).sample(seed=rngs[rng_cnt])
                rng_cnt += 1
                v = m / 100.0
                if layer_idx > 0:
                    v = m / 10.0
            # if nmf_init and layer_idx == 0:
            #     iw = init_w[layer_idx].astype(jnp.float32)
            #     # Rescale
            #     iw = 1.0 * iw + 1.0
            #     # iw = 10. * 100.**iw/100. + 1.
            #     iw = iw
            #     iw = iw  # * self.w_priors[layer_idx][0] / self.w_priors[layer_idx][1]
            #     iw = (
            #         iw / np.sum(iw, axis=1, keepdims=True)
            #         + 1.0 / self.layer_sizes[layer_idx]
            #     )
            #     iw = jnp.clip(iw, 1e-6, 1e2)  # need to set scales to avoid huge BRDs...
            #     m = tfd.Gamma(100.0, 100.0 / (iw)).sample(seed=rngs[rng_cnt])
            #     rng_cnt += 1
            #     v = m / 100.0
            w_shape = jnp.log(m**2 / jnp.sqrt(m**2 + v)) * jnp.ones(
                (in_layer, out_layer)
            )
            w_rate = jnp.log(jnp.sqrt(jnp.log(1 + v / (m**2)))) * jnp.ones(
                (in_layer, out_layer)
            )

            self.global_params.append(jnp.array((w_shape, w_rate)))

        # Gamma(1e3, 1e3/m)
        m = (
            self.shrinkage_shape / self.shrinkage_rate
        )  # self.shrinkage_mean * self.shrinkage_shape  / self.shrinkage_rate
        v = m**2 / 10.0
        if init_ard is not None:
            m = init_ard
            v = m**2 / 100.0

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

        if init_alpha:
            m = self.alpha  # + 0. * np.array(self.gene_ratio_init)
        else:
            m = self.pmeans["alpha"]  # + 0. * np.array(self.gene_ratio_init)
        v = m**2 / 10.0

        self.global_params.append(
            jnp.array(
                (
                    jnp.log(m**2 / jnp.sqrt(m**2 + v)) * jnp.ones((1, 1)),
                    jnp.log(jnp.sqrt(jnp.log(1 + v / (m**2)))) * jnp.ones((1, 1)),
                )
            ),
        )

    def get_nmf_init(self, max_cells=None):
        from sklearn.decomposition import NMF

        X_full = self.X
        X_norm = (
            X_full / np.maximum(np.sum(X_full, axis=1, keepdims=True), 1e-8) * 10_000
        )

        if max_cells is not None and max_cells < X_full.shape[0]:
            key = jax.random.PRNGKey(self.seed)
            idx = np.array(
                jax.random.choice(
                    key, X_full.shape[0], replace=False, shape=(max_cells,)
                )
            )
        else:
            idx = np.arange(X_full.shape[0])

        init_z = []
        init_W = []

        X_fit = X_norm[idx]  # subset for NMF
        X_proj = X_norm  # full data for projection

        for layer_idx in range(self.n_layers):
            model = NMF(
                n_components=self.layer_sizes[layer_idx],
                random_state=self.seed,
                beta_loss="kullback-leibler",
                solver="mu",
            )
            z_subset = model.fit_transform(X_fit)
            W = model.components_

            # Project full data through learned W
            WtW_inv = np.linalg.pinv(W @ W.T)
            z_full = np.clip(X_proj @ W.T @ WtW_inv, 1e-8, None)

            init_z.append(z_full)
            init_W.append(W)

            # For next layer: subset and full-data versions of z
            X_fit = z_subset
            X_proj = z_full

        return init_z, init_W

    def identify_mixture_factors(
        self, max_n_genes: int = 20, thres: float = 0.5
    ) -> np.ndarray:
        """Identify factors that might be better if broken apart.

        Args:
            max_n_genes: maximum number of genes per factor
            thres: threshold for identifying mixture factors

        Returns:
            array of factor indices that are mixture factors
        """
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
        alpha,
        min_shape=jnp.log(1e-10),
        max_shape=jnp.log(1e6),
        min_rate=jnp.log(1e-10),
        max_rate=jnp.log(1e10),
    ):
        # Only anneal the entropy of factor-related variables
        annealing_z = annealing_parameter  # for z
        annealing_w = annealing_parameter  # for W
        annealing_brd = 1.0  # fixed
        annealing_ard = 1.0  # fixed
        annealing_scales = 1.0  # fixed

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

        s_shape = jnp.clip(
            global_params[2 + self.n_layers + 1][0], min_shape, max_shape
        )
        s_rate = jnp.exp(
            jnp.clip(global_params[2 + self.n_layers + 1][1], min_rate, max_rate)
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
            brd_samples = fscale_samples  # / jnp.maximum(_wm_sample, 1.0)
            # log_brd = jnp.log(fscale_samples + 1e-8)
            # log_brd = log_brd - jnp.mean(log_brd) + jnp.log(self.brd_mean)
            # brd_samples = jnp.exp(log_brd)
        # w will be sampled in a loop below because it cannot be vectorized

        # Compute ELBO
        global_pl = gamma_logpdf(
            gene_budget_sample,
            self.gene_scale_shape,
            self.gene_scale_shape * self.gene_ratio,
        )
        global_en = annealing_scales * lognormal_entropy(
            gene_budget_shape, gene_budget_rate
        )
        # exposure = (self.batch_lib_sizes)[:, None]
        # local_pl = 0.0
        # local_en = 0.0
        # local_pl = gamma_logpdf(cell_budget_sample,
        #                         self.cell_scale_shape,
        #                         self.cell_scale_shape / exposure[indices])
        local_pl = gamma_logpdf(
            cell_budget_sample,
            self.cell_scale_shape,
            self.cell_scale_shape * self.batch_lib_ratio[indices],
        )
        local_en = annealing_scales * lognormal_entropy(
            cell_budget_shape, cell_budget_rate
        )

        # Optional alpha marginalization via variational posterior q(alpha).
        alpha_value = alpha
        if self.marginalize_alpha:
            s_sample = lognormal_sample(rng, s_shape, s_rate)
            alpha_value = jnp.squeeze(s_sample)
            global_pl += gamma_logpdf(
                s_sample,
                2.0,
                2.0 / jnp.maximum(jnp.asarray(alpha), 1e-8),
            )
            global_en += lognormal_entropy(s_shape, s_rate)
        if self.use_brd:
            global_pl += gamma_logpdf(
                fscale_samples, self.brd, self.brd / self.brd_mean
            )
            global_en += annealing_brd * lognormal_entropy(fscale_shapes, fscale_rates)

            global_pl += gamma_logpdf(
                _wm_sample, self.shrinkage_shape, self.shrinkage_rate
            )

            global_en += annealing_ard * lognormal_entropy(wm_shape, wm_rate)

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
                    self.w_priors[idx][0] * (1.0 / brd_samples),
                    self.w_priors[idx][1]
                    * (1.0 / brd_samples)
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
                        self.w_priors[idx][1]
                        * 1.0
                        / brd_samples.T
                        * self.layer_sizes[idx],
                    )
                else:
                    global_pl += gamma_logpdf(
                        _w_sample,
                        self.w_priors[idx][0],
                        self.w_priors[idx][1] * self.layer_sizes[idx],
                    )
            global_en += annealing_w * lognormal_entropy(_w_shape, _w_rate)

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

            # if idx == 0 and self.use_brd:
            #     z_mean = z_mean * _wm_sample.T

            alpha_layer = alpha_value  # * (self.layer_sizes[0] / self.layer_sizes[idx])
            alpha_layer = jnp.maximum(
                alpha_layer * self.cell_alpha_factor[indices][:, None], 0.1
            )
            # Top-layer prior: ``top_alpha`` on the layer that is currently the "active top"
            # in the optimization schedule (via ``stop_gradients`` on the root).
            # - Root unfrozen: root ``z`` uses ``top_alpha``; all other layers ``alpha_layer``.
            # - Root frozen: penultimate ``z`` uses ``top_alpha``; root and other layers
            #   ``alpha_layer``.
            root_frozen = stop_gradients[self.n_layers - 1] > 0.5
            is_root = jnp.asarray(idx == self.n_layers - 1, dtype=jnp.bool_)
            is_penultimate = jnp.asarray(idx == self.n_layers - 2, dtype=jnp.bool_)
            use_top_prior = jnp.logical_or(
                jnp.logical_and(is_root, jnp.logical_not(root_frozen)),
                jnp.logical_and(is_penultimate, root_frozen),
            )
            local_pl += jax.lax.cond(
                use_top_prior,
                lambda: gamma_logpdf(_z_sample, self.top_alpha, self.top_alpha),
                lambda: gamma_logpdf(
                    _z_sample,
                    alpha_layer,
                    alpha_layer / (z_mean),
                ),
            )

            local_en += annealing_z * lognormal_entropy(_z_shape, _z_rate)

            z_mean = jnp.einsum("nk,kp->np", _z_sample, _w_sample)

        n_w_sample = _w_sample  # / jnp.sum(_w_sample, axis=1, keepdims=True)
        n_z_sample = _z_sample  # / jnp.sum(_z_sample, axis=1, keepdims=True)
        if self.use_brd:
            n_w_sample = n_w_sample  # * _wm_sample
        mean_bottom = (
            jnp.einsum("nk,kg->ng", n_z_sample, n_w_sample) * cell_budget_sample
        )  # self.exposures[indices]
        mean_bottom = mean_bottom * (batch_indices_onehot.dot(gene_budget_sample))

        # x = jnp.array(batch)
        # lam = mean_bottom  # same shape

        # mask = x > 0
        # x_nz = jnp.where(mask, x, 0.0)

        # # X log λ  -  log(X!)
        # ll_nz = jnp.sum(x_nz * jnp.log(lam + 1e-8))
        # ll_nz -= jnp.sum(jax.scipy.special.gammaln(x_nz + 1.0))

        # # -∑ λ over *all* entries
        # ll_zero = -jnp.sum(lam)

        # ll = ll_nz + ll_zero
        ll = jnp.sum(vmap(poisson.logpmf)(jnp.array(batch), mean_bottom))
        # x = jnp.asarray(batch)
        # x = jnp.maximum(x, 0.0)
        # x = jnp.rint(x).astype(jnp.int32)   # counts should be ints
        # mu = jnp.clip(mean_bottom, 1e-8, 1e8)

        # # phi_sample should be shape (1,G) or (B,G)->(N,G); make sure it broadcasts to mu
        # phi = jnp.clip(s_sample, 1e-3, 1e6)          # IMPORTANT: keep total_count away from 0

        # logits = jnp.log(mu + 1e-8) - jnp.log(phi + 1e-8)

        # nb = tfd.NegativeBinomial(total_count=phi, logits=logits)
        # ll = jnp.sum(nb.log_prob(x))
        ll = jax.lax.cond(
            stop_gradients[0],
            lambda: ll * 0.0,
            lambda: ll,
        )

        # Anneal the entropy
        # global_en *= annealing_parameter
        # local_en *= annealing_parameter

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
        alpha,
    ):
        # Average over a batch of random samples.
        rngs = random.split(rng, num_samples)
        vectorized_elbo = vmap(
            self.elbo,
            in_axes=(0, None, None, None, None, None, None, None, None, None),
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
                alpha,
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
        patience=5,
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
                        (
                            loss,
                            global_params,
                            global_opt_state,
                            global_grads,
                        ) = global_update_func(
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

                epoch_time = time.time() - start_time  # noqa: F841
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
        self,
        nmf_init: bool = False,
        max_cells_init: int = 5000,
        z_init_concentration: float = 0.05,
        n_rounds: int = 1,
        pretraining: bool = False,
        force_decay_factor: bool = False,
        root_epochs: int = 0,
        collapse_l1_fraction: Optional[float] = 0.8,
        collapse_l0_min_factors: int = 10,
        collapse_upper_min_factors: int = 5,
        learn_budgets_on_refit: bool = False,
        refit_top_layer: Optional[Union[int, str]] = None,
        refit_top_factors: Optional[Sequence[str]] = None,
        refit_rescale_relevance: bool = True,
        refit_relevance_max_ratio: float = 50.0,
        refit_rescale_w_by_layer_sizes: bool = True,
        freeze_w: bool = False,
        **kwargs: Any,
    ) -> None:
        """Fit scDEF, warm-starting from a previous fit when available.
        Args:
            nmf_init: whether to initialize the model with NMF.
            max_cells_init: maximum number of cells to use for initialization.
            z_init_concentration: concentration parameter of a Gamma distribution to sample the initial z values from. If high coverage, prefer higher values to avoid overfitting early.
            n_rounds: number of rounds to run the optimization.
            pretraining: whether to run ``pretrain`` before standard fit.
            force_decay_factor: on refit, whether to clip upper-layer sizes to the
                geometric template implied by ``n_factors``, ``top_factors``, and
                ``n_layers_schedule``. If False, preserves learned per-layer dimensions and
                initializes all layers from previous posterior means.
            root_epochs: if ``> 0`` (default ``0``), run a first phase with the root layer frozen, then
                a root-only refinement phase. In ``elbo``, ``top_alpha`` applies to the
                penultimate layer ``z`` while the root is frozen, and to the root ``z`` when
                the root is optimized; other layers use ``alpha_layer`` (see
                ``stop_gradients`` on the root). Factor ``filter`` and ``annotate`` (see
                ``_learn``) are skipped during the frozen-root phase and run only after the
                root step, using the ``filter`` / ``annotate`` values passed to ``fit``.
            collapse_l1_fraction: on **refit only** (second and later ``fit()`` calls),
                scan adjacent pairs along the hierarchy (L0/L1, L1/L2, …; L0 is the
                bottom child layer). When the upper layer is nearly as wide as the layer
                below (at least this fraction of its factor count) and the lower layer
                exceeds ``collapse_l0_min_factors`` (L0) or ``collapse_upper_min_factors``
                (L1+), drop the upper redundant layer, promote the next layer up as its
                replacement parent, and warm-start through composed ``W`` matrices.
                Ignored on the first ``fit()``. Set to ``None`` to disable. Default ``0.8``.
            collapse_l0_min_factors: minimum L0 width required to collapse a redundant L1.
            collapse_upper_min_factors: minimum lower-layer width for L1/L2, L2/L3, … pairs.
            learn_budgets_on_refit: on refit only, keep existing cell/gene budget
                parameters (no re-initialization) but allow them to be optimized when
                ``True``. When ``False`` (default), budgets are warm-started and frozen
                during the main learning phase.
            refit_top_layer: on refit only, truncate the hierarchy so this layer becomes
                the top non-root layer, preserving a final width-1 root when present and
                warm-starting through composed ``W`` matrices. Accepts a layer index or
                layer name (for example the ``recommended_layer_idx`` returned by
                ``scd.tl.find_sensible_top_layer``).
            refit_top_factors: on refit only, rebuild a geometric hierarchy using
                these mixed-depth factors as the top non-root layer. Intermediate
                layer sizes follow the existing ``n_layers_schedule``.
            refit_rescale_relevance: on refit only, rescale ``init_ard`` / ``init_brd``
                warm starts so their geometric mean matches ``shrinkage_mean`` /
                ``brd_mean`` while preserving relative differences across factors
                (optionally capped by ``refit_relevance_max_ratio``).
            refit_relevance_max_ratio: max/min ratio across factors after relevance
                rescaling on refit. Set to ``None`` to disable the ratio cap.
            refit_rescale_w_by_layer_sizes: on refit only, multiply each warm-started
                ``W`` layer by ``old_K / new_K`` so loadings match the ``1 / K`` cold-start
                convention when layer widths change (for example after ``filter_factors``).
            freeze_w: if True, hold every per-layer ``W`` variational parameter
                fixed at its current value during the entire fit. The lower-bottom
                gene loadings (``L0W``) and all parent-child layer matrices stay
                exactly where they are, while ``z``, cell budgets, gene budgets,
                ``BRD``, ``ARD``, and ``alpha`` continue to learn freely. Useful
                for warm-starting from a fitted hierarchy (e.g. the second pass
                of :meth:`add_batch_correction`) when you want batch-specific
                gene scales to absorb new variance without letting the hierarchy
                drift. Default ``False``. Not applied during pretraining.
            **kwargs: additional keyword arguments.

            On the first call, parameters are initialized from priors (or NMF if enabled).
            On subsequent calls, the model is re-initialized from the current posterior
            quantities and the current `factor_lists`, enabling a fit -> filter -> fit
            workflow. During refit, upper-layer sizes are clipped to the geometric
            template when ``force_decay_factor`` is True before rebuilding the hierarchy.
        """
        marginalize_alpha_for_main_fit = bool(self.marginalize_alpha)
        self.root_epochs = int(root_epochs)
        is_refit = bool(getattr(self, "_has_fit", False))
        refit_old_keep: Optional[List[int]] = None
        pending_reference_init: Optional[Mapping[str, Any]] = None
        init_gene_scale = None
        if is_refit:
            # Keep original kept indices before resizing; update_model_size resets factor_lists.
            old_layer_sizes = np.array(self.layer_sizes).copy()
            old_factor_lists = [np.array(f).copy() for f in self.factor_lists]
            layer_sizes = [len(factors) for factors in old_factor_lists]
            old_layer_names = list(self.layer_names)
            old_factor_names = [list(names) for names in self.factor_names]
            n_original = len(layer_sizes)
            old_keep = list(range(n_original))
            refit_old_keep = old_keep
            if refit_top_factors is not None:
                if refit_top_layer is not None:
                    raise ValueError(
                        "Pass only one of refit_top_factors or refit_top_layer."
                    )
                (
                    layer_sizes,
                    init_z,
                    init_w,
                    init_brd,
                    init_ard,
                ) = self._build_frontier_refit_init(
                    refit_top_factors,
                    factor_lists=old_factor_lists,
                    factor_names=old_factor_names,
                    layer_names=old_layer_names,
                )
                old_keep = []
                refit_old_keep = []
                self.logger.info(
                    "Refit: using %s sensible top factors as the top non-root layer.",
                    len(refit_top_factors),
                )
            elif refit_top_layer is not None:
                if isinstance(refit_top_layer, str):
                    if refit_top_layer not in old_layer_names:
                        raise ValueError(
                            f"Unknown refit_top_layer {refit_top_layer!r}; expected one of "
                            f"{old_layer_names}."
                        )
                    top_layer_idx = old_layer_names.index(refit_top_layer)
                else:
                    top_layer_idx = int(refit_top_layer)
                if top_layer_idx < 0 or top_layer_idx >= n_original:
                    raise ValueError(
                        f"refit_top_layer must be within [0, {n_original - 1}]."
                    )
                has_root = n_original > 1 and int(layer_sizes[-1]) == 1
                if has_root and top_layer_idx == n_original - 1:
                    raise ValueError(
                        "refit_top_layer should name the top non-root layer, not the root."
                    )
                old_keep = list(range(top_layer_idx + 1))
                if has_root and (n_original - 1) not in old_keep:
                    old_keep.append(n_original - 1)
                layer_sizes = [layer_sizes[i] for i in old_keep]
                refit_old_keep = old_keep
                self.logger.info(
                    "Refit: using %s as the top non-root layer (old_keep=%s).",
                    old_layer_names[top_layer_idx],
                    old_keep,
                )
            elif collapse_l1_fraction is not None:
                (
                    layer_sizes,
                    old_keep,
                    n_dropped,
                ) = self._collapse_redundant_adjacent_layer_sizes(
                    layer_sizes,
                    float(collapse_l1_fraction),
                    int(collapse_l0_min_factors),
                    int(collapse_upper_min_factors),
                )
            if refit_top_factors is not None:
                hierarchy_changed = True
                warm_start_all_layers = True
            else:
                if force_decay_factor:
                    template = self._geometric_layer_sizes(int(self.n_layers_schedule))
                    for i in range(min(len(layer_sizes), len(template))):
                        if layer_sizes[i] > template[i]:
                            layer_sizes[i] = template[i]
                n_before_sanitize = len(old_keep)
                layer_sizes, old_keep = self._sanitize_layer_sizes(
                    layer_sizes, old_keep
                )
                if len(old_keep) < n_before_sanitize:
                    self.logger.info(
                        "Refit: dropped %s layer(s) with duplicate adjacent widths; "
                        "warm-start via composed W (old_keep=%s).",
                        n_before_sanitize - len(old_keep),
                        old_keep,
                    )
                refit_old_keep = old_keep
                hierarchy_changed = len(old_keep) != n_original or old_keep != list(
                    range(n_original)
                )
                warm_start_all_layers = hierarchy_changed or not force_decay_factor
            if refit_top_factors is None and warm_start_all_layers:
                init_z, init_w, init_brd, init_ard = self._build_collapsed_refit_init(
                    old_keep,
                    factor_lists=old_factor_lists,
                    layer_names=old_layer_names,
                )
            self.update_model_size(
                max_n_factors=max(layer_sizes), layer_sizes=layer_sizes
            )
            self.update_model_priors(update_alpha_from_cov=False)
            self.logger.info(
                f"Continuing scDEF from previous fit with layer sizes {self.layer_sizes}."
            )
            nmf_init = False
            init_budgets = False
            init_alpha = True
            z_init_concentration = 100.0  # on re-fit, use high concentration to avoid ignoring initial state
            if not warm_start_all_layers and force_decay_factor:
                l0_keep = np.array(old_factor_lists[0], dtype=int)
                init_z = np.array(self.pmeans[f"{self.layer_names[0]}z"])[:, l0_keep]
                init_w = np.array(self.pmeans["L0W"])[l0_keep]
                init_brd = np.array(self.pmeans["brd"])[l0_keep]
                init_ard = np.array(self.pmeans["factor_means"])[l0_keep]
        else:
            init_budgets = True
            init_alpha = True
            init_z = None
            init_w = None
            init_brd = None
            init_ard = None
            pending_reference_init = getattr(self, "_pending_reference_init", None)
            init_gene_scale = None
            if pending_reference_init is not None:
                init_z = pending_reference_init.get("init_z")
                init_w = pending_reference_init.get("init_w")
                init_brd = pending_reference_init.get("init_brd")
                init_ard = pending_reference_init.get("init_ard")
                init_gene_scale = pending_reference_init.get("init_gene_scale")
                if init_gene_scale is not None:
                    init_gene_scale = np.asarray(init_gene_scale, dtype=np.float32)
                    self.gene_ratio_init = 1.0 / np.clip(init_gene_scale, 1e-6, 1e6)
                nmf_init = False
                z_init_concentration = 100.0
        if is_refit:
            init_brd, init_ard = self._prepare_refit_relevance_inits(
                init_brd,
                init_ard,
                refit_rescale_relevance=refit_rescale_relevance,
                refit_relevance_max_ratio=refit_relevance_max_ratio,
            )
        elif pending_reference_init is not None:
            # Warm-starts coming from ``from_reference`` carry the reference
            # model's BRD/ARD posteriors verbatim. Those posteriors can have
            # extreme dynamic ranges (e.g. ARD values spanning 6+ orders of
            # magnitude after aggressive shrinkage). Without rescaling, the
            # implied layer-0 W prior rate ``(1/brd)*(1/ard)*K`` becomes huge
            # for low-ARD factors, while ``init_var_params`` floors warm-started
            # W at 1e-3, breaking the equilibrium learned by the reference
            # and producing extreme step-1 gradients that NaN out the fit.
            # Reusing the refit rescaling caps ``max(brd)/min(brd)`` (and ARD)
            # at ``refit_relevance_max_ratio`` while preserving the geometric
            # mean, putting the prior rates back into a numerically safe range.
            init_brd, init_ard = self._prepare_refit_relevance_inits(
                init_brd,
                init_ard,
                refit_rescale_relevance=refit_rescale_relevance,
                refit_relevance_max_ratio=refit_relevance_max_ratio,
            )
        if is_refit:
            if (
                refit_rescale_w_by_layer_sizes
                and init_w is not None
                and len(init_w) > 0
            ):
                old_sizes_for_w = self._refit_old_layer_sizes_for_w(
                    old_layer_sizes,
                    layer_sizes,
                    init_w,
                    refit_old_keep,
                    n_original,
                )
                init_w = self._rescale_w_inits_for_layer_sizes(
                    old_sizes_for_w,
                    layer_sizes,
                    init_w,
                )
        self.init_var_params(
            init_budgets=init_budgets,
            init_alpha=init_alpha,
            init_z=init_z,
            init_w=init_w,
            init_brd=init_brd,
            init_ard=init_ard,
            init_gene_scale=init_gene_scale,
            nmf_init=nmf_init,
            max_cells=max_cells_init,
            z_init_concentration=z_init_concentration,
        )
        if not is_refit and getattr(self, "_pending_reference_init", None) is not None:
            self._pending_reference_init = None
        self._invalidate_cached_diagnostics()
        self.elbos = []
        self.step_sizes = []
        if pretraining:
            pretraining_n_epoch = int(
                kwargs.pop(
                    "pretraining_n_epoch",
                    max(10, int(kwargs.get("n_epoch", 1000)) // 5),
                )
            )
            pretraining_prune_alpha = kwargs.pop("pretraining_prune_alpha", None)
            if pretraining_prune_alpha is not None:
                pretraining_prune_alpha = float(pretraining_prune_alpha)
            pretrain_kwargs = dict(kwargs)
            pretrain_kwargs.pop("n_epoch", None)
            pretrain_kwargs.pop("n_epochs", None)
            self.pretrain(
                n_epoch=pretraining_n_epoch,
                prune_alpha=pretraining_prune_alpha,
                **pretrain_kwargs,
            )

        self.marginalize_alpha = marginalize_alpha_for_main_fit
        optimize_layers = list(range(self.n_layers))
        if root_epochs > 0:
            # Learn with frozen root
            optimize_layers = list(range(self.n_layers - 1))

        if is_refit:
            stop_gene_budgets = not learn_budgets_on_refit
            stop_cell_budgets = not learn_budgets_on_refit
        else:
            stop_gene_budgets = False
            stop_cell_budgets = False
        main_learn_kwargs = dict(kwargs)
        if int(root_epochs) > 0:
            # Defer factor filtering and adata annotation until after root refinement;
            # the frozen-root phase leaves root z unsettled for downstream summaries.
            main_learn_kwargs["filter"] = False
            main_learn_kwargs["annotate"] = False
        self._learn(
            n_rounds=n_rounds,
            optimize_layers=optimize_layers,
            stop_gene_budgets=stop_gene_budgets,
            stop_cell_budgets=stop_cell_budgets,
            freeze_w=freeze_w,
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

        if root_epochs > 0:
            # Update with unfrozen root
            root_kwargs = dict(kwargs)
            root_kwargs["n_epoch"] = root_epochs
            # Root-only refinement should always run with baseline entropy
            # temperature, independent of the main fit entropy adaptation.
            root_kwargs["annealing"] = 1.0
            root_kwargs["entropy_anneal"] = False
            self._learn(
                n_rounds=1,
                optimize_layers=[self.n_layers - 1],
                freeze_w=freeze_w,
                **root_kwargs,
            )
            # Keep QC focused on the main optimization pass, not root-only updates.
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
            if len(self.entropy_annealing_trace) > 0:
                self.adata.uns[
                    "entropy_annealing_trace"
                ] = self.entropy_annealing_trace.copy()
                self.adata.uns[
                    "entropy_annealing_trace_epochs"
                ] = self.entropy_annealing_trace_epochs.copy()
            else:
                self.adata.uns.pop("entropy_annealing_trace", None)
                self.adata.uns.pop("entropy_annealing_trace_epochs", None)

        self.clear_runtime_cache(clear_jax_cache=False)
        self._has_fit = True
        self._fit_revision += 1

    def pretrain(
        self,
        n_epoch: int = 200,
        prune_alpha: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Run two-pass alpha pretraining without the final full fit pass.

        Schedule:
        1) Full model pass with low alpha (=1)
        2) Full model pass with high alpha (pruning pressure)
        """
        if not hasattr(self, "elbos"):
            self.elbos = []
        if not hasattr(self, "step_sizes"):
            self.step_sizes = []

        base_alpha = float(self.alpha)
        if prune_alpha is None:
            prune_alpha = max(10.0, base_alpha * 5.0)

        pretrain_kwargs = dict(kwargs)
        pretrain_kwargs["n_rounds"] = 1
        pretrain_kwargs["n_epoch"] = int(n_epoch)
        pretrain_kwargs["filter"] = False
        pretrain_kwargs["annotate"] = False
        self.logger.info(
            "Starting two-pass pretraining (n_epoch=%s).",
            int(n_epoch),
        )
        prev_marginalize_alpha = bool(self.marginalize_alpha)
        self.marginalize_alpha = False
        try:
            self.alpha = 1.0
            self._learn(**pretrain_kwargs)

            self.alpha = float(prune_alpha)
            self._learn(**pretrain_kwargs)
        finally:
            self.alpha = base_alpha
            self.marginalize_alpha = prev_marginalize_alpha
            self.clear_runtime_cache(clear_jax_cache=False)

    def clear_runtime_cache(self, clear_jax_cache: bool = False) -> None:
        """Clear runtime-only compiled caches to reduce notebook memory pressure."""
        self._learn_jit_cache = None
        if clear_jax_cache:
            jax.clear_caches()

    def _invalidate_cached_diagnostics(self) -> None:
        """Drop cached diagnostics/signatures that become stale after refit."""
        if not hasattr(self, "adata") or self.adata is None:
            return
        if not hasattr(self.adata, "uns"):
            return
        for key in [
            "_factor_obs_upper_lists_fixed",
            "_factor_obs_fit_revision",
            "factor_obs",
            "factor_obs_full",
            "factor_signatures",
            "confident_signatures",
            "obs_scores",
            "within_group_pairwise_dissimilarity",
            "technical_hierarchy",
            "global_hierarchy",
        ]:
            self.adata.uns.pop(key, None)

    def _get_or_build_learn_grad_fns(self, num_samples: int):
        """Return cached JIT-compiled local/global gradient callables for _learn."""
        marginalize_alpha = bool(self.marginalize_alpha)
        cache = getattr(self, "_learn_jit_cache", None)
        cache_hit = (
            isinstance(cache, dict)
            and cache.get("num_samples", None) == int(num_samples)
            and cache.get("marginalize_alpha", None) == marginalize_alpha
            and "local_loss_grad" in cache
            and "global_loss_grad" in cache
        )
        if cache_hit:
            return cache["local_loss_grad"], cache["global_loss_grad"]

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
            alpha,
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
                alpha,
            )

        local_loss_grad = jit(value_and_grad(objective, argnums=2))
        global_loss_grad = jit(value_and_grad(objective, argnums=3))
        self._learn_jit_cache = {
            "num_samples": int(num_samples),
            "marginalize_alpha": marginalize_alpha,
            "local_loss_grad": local_loss_grad,
            "global_loss_grad": global_loss_grad,
        }
        return local_loss_grad, global_loss_grad

    def _compute_median_parents(
        self,
        local_params=None,
        global_params=None,
        return_active_l0_count: bool = False,
    ):
        """Estimate median effective parent count using variational means."""
        if self.n_layers < 2:
            if return_active_l0_count:
                return 1.0, 0
            return 1.0

        if local_params is None:
            local_params = self.local_params
        if global_params is None:
            global_params = self.global_params

        z_params = local_params[1]
        # Active L0 factors: at least 1 cell assigned.
        start0 = 0
        end0 = int(self.layer_sizes[0])
        z0 = np.array(
            jnp.exp(
                z_params[0][:, start0:end0]
                + 0.5 * jnp.exp(z_params[1][:, start0:end0]) ** 2
            )
        )
        assignments0 = np.argmax(z0, axis=1)
        counts0 = np.array(
            [np.sum(assignments0 == j) for j in range(self.layer_sizes[0])]
        )
        active_0 = np.where(counts0 >= 1)[0]

        start1 = int(np.sum(self.layer_sizes[:1]))
        end1 = start1 + int(self.layer_sizes[1])
        z1 = np.array(
            jnp.exp(
                z_params[0][:, start1:end1]
                + 0.5 * jnp.exp(z_params[1][:, start1:end1]) ** 2
            )
        )
        assignments = np.argmax(z1, axis=1)
        counts = np.array(
            [np.sum(assignments == j) for j in range(self.layer_sizes[1])]
        )
        active_1 = np.where(counts >= 1)[0]

        if len(active_0) == 0 or len(active_1) == 0:
            if return_active_l0_count:
                return 1.0, int(len(active_0))
            return 1.0

        w1_params = global_params[3]
        W1 = np.array(jnp.exp(w1_params[0] + 0.5 * jnp.exp(w1_params[1]) ** 2))[
            np.ix_(active_1, active_0)
        ]
        p = W1 / np.clip(W1.sum(axis=0, keepdims=True), 1e-12, None)
        H = -np.sum(p * np.log(p + 1e-12), axis=0)
        n_parents = np.exp(H)

        median_parents = np.median(n_parents)
        if return_active_l0_count:
            return median_parents, int(len(active_0))
        return median_parents

    def _learn(
        self,
        n_rounds=1,
        n_epoch=1000,
        lr=1e-1,
        local_lr=1e-2,
        annealing=1.0,
        num_samples=100,
        batch_size=256,
        layerwise=False,
        min_epochs=50,
        tolerance=1e-5,
        patience=50,
        update_locals=True,
        update_globals=True,
        stop_cell_budgets=0,
        stop_gene_budgets=0,
        stop_gene_budgets_after_burnin=False,
        opt_layer=None,
        optimize_layers=None,
        filter=True,
        annotate=True,
        entropy_anneal=False,
        entropy_window=50,
        entropy_check_every=10,
        entropy_rel_change_low=1e-3,
        entropy_rel_change_high=1e-2,
        entropy_increase_factor=1.2,
        entropy_decrease_factor=0.9,
        entropy_min_annealing=1.0,
        entropy_max_annealing=5.0,
        entropy_optimizer_reset_threshold=0.25,
        freeze_w=False,
        freeze_z_layers=None,
        **kwargs,
    ):
        """Fit the model."""
        if "n_epochs" in kwargs:
            n_epoch = kwargs.pop("n_epochs")
        if len(kwargs) > 0:
            unknown = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unexpected keyword arguments for _learn: {unknown}")

        if freeze_z_layers is None:
            freeze_z_layers = []
        freeze_z_layers = sorted({int(i) for i in freeze_z_layers})

        if int(n_rounds) > 1:
            total_rounds = int(n_rounds)
            for r in range(total_rounds):
                round_lr = lr if layerwise else lr * (0.5**r)
                round_local_lr = local_lr if layerwise else local_lr * (0.5**r)
                is_last_round = r == (total_rounds - 1)
                round_filter = bool(filter) and is_last_round
                round_annotate = bool(annotate) and is_last_round
                self.logger.info(
                    "Starting learning round %s/%s (lr=%.5g, local_lr=%.5g, filter=%s, annotate=%s).",
                    r + 1,
                    total_rounds,
                    round_lr,
                    round_local_lr,
                    round_filter,
                    round_annotate,
                )
                self._learn(
                    n_rounds=1,
                    n_epoch=n_epoch,
                    lr=round_lr,
                    local_lr=round_local_lr,
                    annealing=annealing,
                    num_samples=num_samples,
                    batch_size=batch_size,
                    layerwise=layerwise,
                    min_epochs=min_epochs,
                    tolerance=tolerance,
                    patience=patience,
                    update_locals=update_locals,
                    update_globals=update_globals,
                    stop_cell_budgets=stop_cell_budgets,
                    stop_gene_budgets=stop_gene_budgets,
                    stop_gene_budgets_after_burnin=stop_gene_budgets_after_burnin,
                    opt_layer=opt_layer,
                    optimize_layers=optimize_layers,
                    filter=round_filter,
                    annotate=round_annotate,
                    entropy_anneal=entropy_anneal,
                    entropy_window=entropy_window,
                    entropy_check_every=entropy_check_every,
                    entropy_rel_change_low=entropy_rel_change_low,
                    entropy_rel_change_high=entropy_rel_change_high,
                    entropy_increase_factor=entropy_increase_factor,
                    entropy_decrease_factor=entropy_decrease_factor,
                    entropy_min_annealing=entropy_min_annealing,
                    entropy_max_annealing=entropy_max_annealing,
                    entropy_optimizer_reset_threshold=entropy_optimizer_reset_threshold,
                    freeze_w=freeze_w,
                    freeze_z_layers=freeze_z_layers,
                    **kwargs,
                )
            return

        if batch_size is None:
            batch_size = self.n_cells

        num_complete_batches, leftover = divmod(self.n_cells, batch_size)
        num_batches = num_complete_batches + bool(leftover)
        self.logger.info(
            f"Each epoch contains {num_batches} batches of size "
            f"{int(min(batch_size, self.n_cells))}"
        )

        def clip_params(
            params, min_mu=-1e10, max_mu=1e4, min_logstd=-1e10, max_logstd=1e1
        ):
            for i in range(len(params))[:-2]:
                params[i] = params[i].at[0].set(jnp.clip(params[i][0], min_mu, max_mu))
                params[i] = (
                    params[i].at[1].set(jnp.clip(params[i][1], min_logstd, max_logstd))
                )
            return params

        local_loss_grad, global_loss_grad = self._get_or_build_learn_grad_fns(
            num_samples=num_samples
        )

        # --- Set up layer-wise stop gradients ---

        if optimize_layers is None:
            if opt_layer is not None:
                optimize_layers = [int(opt_layer)]
            else:
                optimize_layers = list(range(self.n_layers))

        layers_to_optimize = sorted({int(i) for i in optimize_layers})
        if len(layers_to_optimize) == 0:
            raise ValueError("optimize_layers must contain at least one layer index.")
        if min(layers_to_optimize) < 0 or max(layers_to_optimize) >= self.n_layers:
            raise ValueError(
                f"optimize_layers must be within [0, {self.n_layers - 1}]."
            )

        stop_gradients = jnp.ones((self.n_layers,))
        layers_to_optimize = jnp.array(layers_to_optimize, dtype=jnp.int32)
        stop_gradients = stop_gradients.at[layers_to_optimize].set(0.0)
        stop_cell_budgets = jnp.array(stop_cell_budgets)
        stop_gene_budgets = jnp.array(stop_gene_budgets)

        # Column ranges for z layers that should be snapped back after updates
        _freeze_z_slices: List[tuple] = []
        for lidx in freeze_z_layers:
            start = int(np.sum(self.layer_sizes[:lidx]))
            end = start + int(self.layer_sizes[lidx])
            _freeze_z_slices.append((start, end))

        # --- Initialize optimizers ---

        local_optimizer = optax.adam(local_lr)
        global_optimizer = optax.adam(lr)
        local_opt_state = local_optimizer.init(self.local_params)
        global_opt_state = global_optimizer.init(self.global_params)

        local_params = self.local_params
        global_params = self.global_params

        def local_update(
            X,
            indices,
            key,
            local_params,
            global_params,
            local_opt_state,
            annealing_parameter,
            stop_gradients,
            stop_cell_budgets,
            stop_gene_budgets,
            alpha,
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
                alpha,
            )
            updates, local_opt_state_new = local_optimizer.update(
                gradient, local_opt_state, local_params
            )
            local_params_new = optax.apply_updates(local_params, updates)
            local_params_new = clip_params(local_params_new)
            return value, local_params_new, local_opt_state_new

        # Indices in `global_params` corresponding to the per-layer W matrices.
        # Layout: [gene_scale, BRD, W_layer_0, ..., W_layer_{n-1}, ARD/wm, alpha].
        w_param_start = 2
        w_param_end = 2 + self.n_layers

        def global_update(
            X,
            indices,
            key,
            local_params,
            global_params,
            global_opt_state,
            annealing_parameter,
            stop_gradients,
            stop_cell_budgets,
            stop_gene_budgets,
            alpha,
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
                alpha,
            )
            updates, global_opt_state_new = global_optimizer.update(
                gradient, global_opt_state, global_params
            )
            global_params_new = optax.apply_updates(global_params, updates)
            if freeze_w:
                # Restore per-layer W variational params to their pre-update
                # values. The Adam moments still accumulate from the non-zero
                # W gradients, but they never affect the parameters because we
                # snap each W slot back every step. Optimizer state is reset
                # at the start of each `_learn` call, so this is contained.
                global_params_new = list(global_params_new)
                for i in range(w_param_start, w_param_end):
                    global_params_new[i] = global_params[i]
            global_params_new = clip_params(global_params_new)
            return value, global_params_new, global_opt_state_new

        # --- Data stream ---

        def data_stream():
            rng = np.random.RandomState(0)
            while True:
                perm = rng.permutation(self.n_cells)
                for i in range(num_batches):
                    batch_idx = perm[i * batch_size : (i + 1) * batch_size]
                    yield jnp.array(self.X[batch_idx]), jnp.array(batch_idx)

        batches = data_stream()

        # --- Main training loop ---

        annealing_parameter = jnp.array(annealing)
        alpha_jnp = jnp.asarray(self.alpha, dtype=jnp.float32)
        rng = random.PRNGKey(self.seed)

        all_losses = []
        total_epochs = 0
        min_loss = np.inf
        early_stop_counter = 0
        stop_message = None
        interrupted = False
        if int(entropy_window) <= 0:
            raise ValueError("entropy_window must be > 0.")
        if int(entropy_check_every) <= 0:
            raise ValueError("entropy_check_every must be > 0.")
        if float(entropy_rel_change_low) < 0.0 or float(entropy_rel_change_high) < 0.0:
            raise ValueError("entropy relative-change thresholds must be >= 0.")
        if float(entropy_rel_change_low) > float(entropy_rel_change_high):
            raise ValueError(
                "entropy_rel_change_low must be <= entropy_rel_change_high."
            )
        if float(entropy_increase_factor) <= 1.0:
            raise ValueError("entropy_increase_factor must be > 1.0.")
        if not (0.0 < float(entropy_decrease_factor) < 1.0):
            raise ValueError("entropy_decrease_factor must be in (0, 1).")
        if float(entropy_min_annealing) <= 0.0:
            raise ValueError("entropy_min_annealing must be > 0.")
        if float(entropy_max_annealing) < float(entropy_min_annealing):
            raise ValueError("entropy_max_annealing must be >= entropy_min_annealing.")
        if float(entropy_optimizer_reset_threshold) < 0.0:
            raise ValueError("entropy_optimizer_reset_threshold must be >= 0.")
        entropy_annealing_trace: List[float] = []
        entropy_annealing_trace_epochs: List[int] = []
        gene_budgets_frozen = bool(float(stop_gene_budgets) >= 1.0)

        pbar = tqdm(range(n_epoch))
        try:
            for epoch in pbar:
                epoch_losses = []

                if (
                    stop_gene_budgets_after_burnin
                    and not gene_budgets_frozen
                    and total_epochs >= min_epochs
                ):
                    stop_gene_budgets = jnp.array(1.0)
                    gene_budgets_frozen = True
                    self.logger.info(
                        f"Freezing gene budgets at epoch {total_epochs} "
                        "(post burn-in)."
                    )

                for it in range(num_batches):
                    rng, rng_input = random.split(rng)
                    X, indices = next(batches)

                    if update_locals:
                        loss, local_params, local_opt_state = local_update(
                            X,
                            indices,
                            rng_input,
                            local_params,
                            global_params,
                            local_opt_state,
                            annealing_parameter,
                            stop_gradients,
                            stop_cell_budgets,
                            stop_gene_budgets,
                            alpha_jnp,
                        )
                        if _freeze_z_slices:
                            z_snap = local_params[1]
                            z_orig = self.local_params[1]
                            for s, e in _freeze_z_slices:
                                z_snap = z_snap.at[0, :, s:e].set(z_orig[0, :, s:e])
                                z_snap = z_snap.at[1, :, s:e].set(z_orig[1, :, s:e])
                            local_params = list(local_params)
                            local_params[1] = z_snap
                    if update_globals:
                        loss, global_params, global_opt_state = global_update(
                            X,
                            indices,
                            rng_input,
                            local_params,
                            global_params,
                            global_opt_state,
                            annealing_parameter,
                            stop_gradients,
                            stop_cell_budgets,
                            stop_gene_budgets,
                            alpha_jnp,
                        )

                    epoch_losses.append(loss)

                current_loss = np.mean(epoch_losses)
                all_losses.append(current_loss)
                total_epochs += 1
                entropy_annealing_trace.append(float(annealing_parameter))
                entropy_annealing_trace_epochs.append(int(total_epochs))

                if bool(entropy_anneal) and (
                    int(total_epochs) % int(entropy_check_every) == 0
                ):
                    entropy_loss_buffer = all_losses
                    if len(entropy_loss_buffer) < int(entropy_window):
                        entropy_loss_buffer = []
                    if len(entropy_loss_buffer) == 0:
                        recent = None
                    else:
                        recent = np.asarray(
                            entropy_loss_buffer[-int(entropy_window) :], dtype=float
                        )
                else:
                    recent = None

                if recent is not None:
                    recent_mean = float(np.mean(recent))
                    denom = max(abs(recent_mean), 1e-12)
                    relative_change = float((np.max(recent) - np.min(recent)) / denom)
                    old_annealing = float(annealing_parameter)
                    new_annealing = old_annealing
                    if relative_change < float(entropy_rel_change_low):
                        new_annealing = min(
                            old_annealing * float(entropy_increase_factor),
                            float(entropy_max_annealing),
                        )
                    elif relative_change > float(entropy_rel_change_high):
                        new_annealing = max(
                            old_annealing * float(entropy_decrease_factor),
                            float(entropy_min_annealing),
                        )
                    if abs(new_annealing - old_annealing) > 0.0:
                        annealing_parameter = jnp.asarray(
                            new_annealing, dtype=annealing_parameter.dtype
                        )
                        rel_step = abs(new_annealing - old_annealing) / max(
                            abs(old_annealing), 1e-12
                        )
                        if rel_step >= float(entropy_optimizer_reset_threshold):
                            local_opt_state = local_optimizer.init(local_params)
                            global_opt_state = global_optimizer.init(global_params)
                        self.logger.info(
                            "Entropy annealing update at epoch %s: %.4f -> %.4f "
                            "(relative_change=%.6g).",
                            int(total_epochs),
                            float(old_annealing),
                            float(new_annealing),
                            float(relative_change),
                        )

                relative_improvement = np.nan
                if total_epochs >= min_epochs:
                    if min_loss == np.inf:
                        min_loss = current_loss
                    relative_improvement = (min_loss - current_loss) / np.abs(min_loss)
                    min_loss = min(min_loss, current_loss)
                    if relative_improvement < tolerance:
                        early_stop_counter += 1
                    else:
                        early_stop_counter = 0

                postfix = {"Loss": current_loss}
                if bool(entropy_anneal):
                    postfix["anneal"] = float(annealing_parameter)
                if total_epochs >= min_epochs:
                    postfix["Rel. impr."] = relative_improvement
                pbar.set_postfix(postfix)
                if total_epochs >= min_epochs and early_stop_counter >= patience:
                    stop_message = f"Converged at epoch {total_epochs}."
                    break

        except KeyboardInterrupt:
            interrupted = True
            self.logger.info("Interrupted. Exiting safely...")
        finally:
            pbar.close()

        if stop_message is None and not interrupted:
            stop_message = (
                f"Stopping learning: reached max epochs (n_epoch={int(n_epoch)})."
            )
        if stop_message is not None:
            self.logger.info(stop_message)

        self.local_params = local_params
        self.global_params = global_params
        self.elbos.append(all_losses)
        self.step_sizes.append(lr)
        self.adata.uns.pop("alpha_trace", None)
        self.adata.uns.pop("alpha_trace_epochs", None)
        self.adata.uns.pop("n_eff_parents_trace", None)
        self.adata.uns.pop("n_eff_parents_trace_epochs", None)
        self.adata.uns.pop("active_l0_factor_counts_trace", None)
        self.adata.uns.pop("alpha_schedule_alphas", None)
        self.adata.uns.pop("alpha_schedule_losses", None)
        self.adata.uns.pop("alpha_schedule_epochs", None)
        self.entropy_annealing_trace = np.asarray(entropy_annealing_trace, dtype=float)
        self.entropy_annealing_trace_epochs = np.asarray(
            entropy_annealing_trace_epochs, dtype=int
        )
        self.adata.uns["entropy_annealing_trace"] = self.entropy_annealing_trace.copy()
        self.adata.uns[
            "entropy_annealing_trace_epochs"
        ] = self.entropy_annealing_trace_epochs.copy()

        self.set_posterior_means()
        self.set_posterior_variances()
        if self.use_brd and filter:
            self.filter_factors(upper_only=True)
        elif annotate:
            self.make_layercolors(
                layer_cpal=self.layer_cpal, lightness_mult=self.lightness_mult
            )
            self.annotate_adata()

    def set_posterior_means(self):
        cell_budget_params = self.local_params[0]
        gene_budget_params = self.global_params[0]
        fscale_params = self.global_params[1]
        wm_params = self.global_params[-2]
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
        self.pmeans["brd"] = self.pmeans["factor_concentrations"]  # / jnp.maximum(
        #  self.pmeans["factor_means"], 1.0
        #  )
        self.pmeans["wm"] = self.pmeans["factor_means"]
        self.pmeans["ard"] = self.pmeans["wm"]
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

        self.pmeans["alpha"] = np.exp(
            self.global_params[-1][0] + 0.5 * np.exp(self.global_params[-1][1]) ** 2
        )

    def set_posterior_variances(self):
        cell_budget_params = self.local_params[0]
        gene_budget_params = self.global_params[0]
        fscale_params = self.global_params[1]
        wm_params = self.global_params[-2]
        z_params = self.local_params[1]

        def _lognormal_var(mu, log_sigma):
            sigma2 = np.exp(log_sigma) ** 2
            return (np.exp(sigma2) - 1.0) * np.exp(2.0 * mu + sigma2)

        self.pvars = {
            "cell_scale": np.array(
                _lognormal_var(cell_budget_params[0], cell_budget_params[1])
            ),
            "gene_scale": np.array(
                _lognormal_var(gene_budget_params[0], gene_budget_params[1])
            ),
            "factor_concentrations": np.array(
                _lognormal_var(fscale_params[0], fscale_params[1])
            ),
            "factor_means": np.array(_lognormal_var(wm_params[0], wm_params[1])),
            "alpha": np.array(
                _lognormal_var(self.global_params[-1][0], self.global_params[-1][1])
            ),
        }

        for idx in range(self.n_layers):
            start = sum(self.layer_sizes[:idx])
            end = start + self.layer_sizes[idx]
            self.pvars[f"{self.layer_names[idx]}z"] = np.array(
                _lognormal_var(z_params[0][:, start:end], z_params[1][:, start:end])
            )
            _w_shape = self.global_params[2 + idx][0]
            _w_rate = self.global_params[2 + idx][1]
            self.pvars[f"{self.layer_names[idx]}W"] = np.array(
                _lognormal_var(_w_shape, _w_rate)
            )

    def filter_factors(
        self,
        brd_min: Optional[float] = 1.0,
        ard_min: Optional[float] = 0.001,
        clarity_min: Optional[float] = 0.5,
        n_eff_parents_max: float = 1.5,
        local_l0_scores: bool = False,
        batch_purity_max: Optional[float] = None,
        batch_purity_soft_max: Optional[float] = None,
        min_cells_upper: Optional[float] = 0.001,
        min_cells_lower: Optional[float] = 0.0,
        filter_up: Optional[bool] = True,
        annotate: Optional[bool] = True,
        upper_only: Optional[bool] = False,
    ):
        """Filter irrelevant factors using BRD/ARD and hierarchy diagnostics.

        Args:
            brd_min: minimum factor BRD value for layer 0 when ``use_brd``.
            ard_min: minimum ARD fraction of total ARD for layer 0.
            clarity_min: minimum L0 ``clarity_score_01`` when not using lineage
                ``avg_n_eff_parents`` (``local_l0_scores`` or missing lineage columns).
            n_eff_parents_max: only when ``avg_n_eff_parents`` is present and
                ``local_l0_scores`` is False: cutoff ``avg_n_eff_parents < n_eff_parents_max``
                (default ``1.5``). Ignored for local-L0 / clarity-only filtering.
            local_l0_scores: if True, filter by ``clarity_score_01 >= clarity_min``; if
                False and ``avg_n_eff_parents`` exists, filter by ``n_eff_parents_max``.
            batch_purity_max: if set, keep layer-0 factors with hard-assignment
                ``batch_purity <= batch_purity_max`` (requires
                ``factor_diagnostics(..., batch_key=...)``).
            batch_purity_soft_max: if set, keep factors with
                ``batch_purity_soft <= batch_purity_soft_max``.
            min_cells_upper: minimum cells attached to upper-layer factors (fraction if <1).
            min_cells_lower: minimum cells attached to layer-0 factors (fraction if <1).
            filter_up: whether to prune upper layers via inter-layer attachments.
            annotate: whether to run ``annotate_adata`` after filtering.
            upper_only: if True, only adjust upper layers (layer 0 unchanged).
        """
        if min_cells_upper != 0:
            if min_cells_upper < 1.0:
                min_cells_upper = max(min_cells_upper * self.adata.shape[0], 10)
        if min_cells_lower != 0:
            if min_cells_lower < 1.0:
                min_cells_lower = max(min_cells_lower * self.adata.shape[0], 10)

        new_factor_lists = []
        for i, layer_name in enumerate(self.layer_names):
            if i == 0:
                if upper_only:
                    keep = np.arange(self.layer_sizes[i])
                else:
                    keep = self.get_effective_factors(
                        brd_min=brd_min,
                        ard_min=ard_min,
                        clarity_min=clarity_min,
                        n_eff_parents_max=n_eff_parents_max,
                        local_l0_scores=local_l0_scores,
                        batch_purity_max=batch_purity_max,
                        batch_purity_soft_max=batch_purity_soft_max,
                        min_cells=min_cells_lower,
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
                    np.where(counts >= min_cells_upper)[0]
                ]
                if filter_up and len(keep) > 0 and len(new_factor_lists[i - 1]) > 0:
                    mat = self.pmeans[f"{layer_name}W"][keep]
                    assignments = []
                    for factor in new_factor_lists[i - 1]:
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
            new_factor_lists.append(keep)

        self.factor_lists = new_factor_lists
        self.set_factor_names()
        if "factor_obs" in self.adata.uns:
            self._sync_factor_obs_with_filter()

        self.make_layercolors(
            layer_cpal=self.layer_cpal, lightness_mult=self.lightness_mult
        )
        if annotate:
            self.annotate_adata()

    def _sync_factor_obs_with_filter(self):
        """Rebuild ``adata.uns['factor_obs']`` so it reflects the currently
        kept factors, with rows renamed to match ``self.factor_names``.

        The rebuild is sourced from ``adata.uns['factor_obs_full']`` when
        available (the complete snapshot written by
        ``scdef.tools.factor_diagnostics``), so that re-filtering with
        looser thresholds can restore rows for factors that had been
        previously dropped from the live view. Falls back to the current
        ``factor_obs`` if the full snapshot is not present.

        Rows for filtered-out factors are omitted, kept rows are renamed
        (index and ``child_factor`` column) to the current factor names, and
        ``technical`` is reset to False. Name-valued columns that reference
        other factors (e.g. ``best_parent``) retain their diagnostic-time
        values and may become stale if upper-layer factors were filtered;
        re-run ``scdef.tools.factor_diagnostics(model, recompute=True)`` to
        refresh those.
        """
        if "factor_obs_full" in self.adata.uns:
            source = self.adata.uns["factor_obs_full"]
        else:
            source = self.adata.uns["factor_obs"]

        if not (
            "child_layer" in source.columns and "original_factor_idx" in source.columns
        ):
            self.adata.uns["factor_obs"]["technical"] = False
            if "global" in self.adata.uns["factor_obs"].columns:
                self.adata.uns["factor_obs"]["global"] = False
            return

        layer_lookup = {}
        for layer_idx, layer_name in enumerate(self.layer_names):
            layer_lookup[layer_name] = {
                int(orig): (slot, self.factor_names[layer_idx][slot])
                for slot, orig in enumerate(self.factor_lists[layer_idx])
            }

        rows = []
        for pos in range(len(source)):
            row = source.iloc[pos]
            mapping = layer_lookup.get(row["child_layer"], {}).get(
                int(row["original_factor_idx"])
            )
            if mapping is None:
                continue
            slot, current_name = mapping
            rows.append((row["child_layer"], slot, pos, current_name))

        rows.sort(key=lambda t: (self.layer_names.index(t[0]), t[1]))
        positions = [t[2] for t in rows]
        new_index = [t[3] for t in rows]

        new_factor_obs = source.iloc[positions].copy()
        new_factor_obs.index = new_index
        if "child_factor" in new_factor_obs.columns:
            new_factor_obs["child_factor"] = new_index
        new_factor_obs["technical"] = False
        new_factor_obs["global"] = False
        self.adata.uns["factor_obs"] = new_factor_obs

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

        if not getattr(self, "_preserve_factor_names_on_annotate", False):
            self.set_factor_names()
        from scdef.tools.factor import _resolve_signature_drop_factors

        ranked_genes, ranked_scores = self.get_signatures_dict(
            scores=True,
            sorted_scores=True,
            drop_factors=_resolve_signature_drop_factors(self, None),
        )

        for idx in range(self.n_layers):
            layer_name = self.layer_names[idx]
            self.adata.obsm[f"X_{layer_name}"] = np.array(
                self.pmeans[f"{layer_name}z"][:, self.factor_lists[idx]]
            )
            assignments = np.argmax(self.adata.obsm[f"X_{layer_name}"], axis=1)
            factor_names = list(self.factor_names[idx])
            self.adata.obs[f"{layer_name}"] = pd.Categorical(
                [factor_names[a] for a in assignments],
                categories=factor_names,
            )
            self.adata.uns[f"{layer_name}_colors"] = [
                matplotlib.colors.to_hex(self.layer_colorpalettes[idx][i])
                for i in range(len(factor_names))
            ]

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

        self.normalize_cellscores()

    def normalize_cellscores(self):
        for idx in range(self.n_layers):
            layer_name = self.layer_names[idx]
            scores = np.asarray(self.adata.obsm[f"X_{layer_name}"], dtype=float)
            den = np.clip(np.sum(scores, axis=1, keepdims=True), 1e-12, None)
            self.adata.obsm[f"X_{layer_name}_probs"] = scores / den
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

    def get_annotations(
        self,
        marker_reference: Mapping[str, Sequence[str]],
        gene_rankings: Optional[List[List[str]]] = None,
    ) -> List[List[str]]:
        """Get annotations for factors based on marker gene reference.

        Args:
            marker_reference: dictionary mapping annotation names to gene lists
            gene_rankings: gene rankings for each factor, if None will be computed

        Returns:
            list of annotation lists, one per factor
        """
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
        layer_idx: int = 0,
        top_genes: Optional[int] = None,
        genes: bool = True,
        return_scores: bool = False,
        sorted_scores: bool = True,
        drop_factors: Optional[List[str]] = None,
    ) -> Union[List[List[str]], Tuple[List[List[str]], List[List[float]]]]:
        """Get gene or factor rankings for each factor in a layer.

        Args:
            layer_idx: layer index to get rankings for
            top_genes: number of top genes/factors to return
            genes: whether to return gene rankings (True) or factor rankings (False).
                Gene rankings use cached confidence+mean combined scores when
                ``sorted_scores=True`` and ``drop_factors`` is not provided.
            return_scores: whether to return scores along with rankings
            sorted_scores: whether to return scores sorted by ranking
            drop_factors: list of factors to drop from rankings

        Returns:
            list of rankings per factor, or tuple of (rankings, scores) if return_scores is True
        """
        top_genes_provided = top_genes is not None
        if top_genes is None:
            top_genes = len(self.adata.var_names)

        from scdef.tools.factor import _resolve_signature_drop_factors

        drop_factors = _resolve_signature_drop_factors(self, drop_factors)

        if genes and sorted_scores and top_genes_provided and len(drop_factors) == 0:
            from scdef.tools.factor import (
                get_stored_confident_signatures,
                set_confident_signatures,
            )

            try:
                top_terms_dict, top_scores_dict = get_stored_confident_signatures(
                    self,
                    layer_idx=layer_idx,
                    max_genes=int(top_genes),
                    return_combined_scores=True,
                )
            except KeyError:
                set_confident_signatures(self)
                top_terms_dict, top_scores_dict = get_stored_confident_signatures(
                    self,
                    layer_idx=layer_idx,
                    max_genes=int(top_genes),
                    return_combined_scores=True,
                )

            top_terms = []
            top_scores = []
            for factor_name in self.factor_names[layer_idx]:
                genes_k = list(top_terms_dict.get(factor_name, []))
                scores_k = np.asarray(top_scores_dict.get(factor_name, []), dtype=float)
                top_terms.append(genes_k)
                top_scores.append(scores_k.tolist())

            if return_scores:
                return top_terms, top_scores
            return top_terms

        term_names = np.array(self.adata.var_names)
        factor_names_0 = np.array(self.factor_names[0])
        from scdef.tools.factor import _l0_keep_indices

        keep_indices_0 = _l0_keep_indices(self, drop_factors)

        term_scores = self.pmeans[f"{self.layer_names[0]}W"][self.factor_lists[0]]
        n_factors = len(self.factor_lists[layer_idx])

        if layer_idx > 0:
            if genes:
                # Compute factor-to-gene mapping for this upper layer
                term_scores = self.pmeans[f"{self.layer_names[layer_idx]}W"][
                    self.factor_lists[layer_idx]
                ][:, self.factor_lists[layer_idx - 1]]
                for layer in range(layer_idx - 1, 0, -1):
                    lower_mat = self.pmeans[f"{self.layer_names[layer]}W"][
                        self.factor_lists[layer]
                    ][:, self.factor_lists[layer - 1]]
                    term_scores = term_scores.dot(lower_mat)
                # For mapping to genes, drop columns of w0 corresponding to factors in drop_factors
                w0 = self.pmeans[f"{self.layer_names[0]}W"][self.factor_lists[0]]
                if len(keep_indices_0) != len(factor_names_0):
                    w0 = w0[keep_indices_0, :]
                term_scores = (
                    term_scores[:, keep_indices_0]
                    if term_scores.shape[1] == len(factor_names_0)
                    else term_scores
                )
                term_scores = term_scores.dot(w0)
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
        self,
        rng: Any,
        factor_idx: int,
        layer_idx: int,
        top_genes: int = 10,
        return_scores: bool = False,
    ) -> Union[List[str], Tuple[List[str], np.ndarray]]:
        """Get a single signature sample from the posterior for a factor.

        Args:
            rng: JAX random number generator key
            factor_idx: index of the factor
            layer_idx: layer index of the factor
            top_genes: number of top genes to return
            return_scores: whether to return scores along with gene names

        Returns:
            list of gene names, or tuple of (gene_names, scores) if return_scores is True
        """
        term_names = np.array(self.adata.var_names)

        if layer_idx == 0:
            l0_rows = np.asarray(self.factor_lists[0], dtype=int)
            w0_shape = self.global_params[2 + 0][0][l0_rows]
            w0_rate = np.exp(self.global_params[2 + 0][1][l0_rows])
            rng, sample_rng = random.split(rng)
            term_scores_sample = lognormal_sample(sample_rng, w0_shape, w0_rate)
        else:
            from scdef.tools.factor import _hierarchy_gene_scores_draw

            scores, rng = _hierarchy_gene_scores_draw(
                self, rng, max_layer_idx=layer_idx
            )
            term_scores_sample = scores[layer_idx]

        top_terms_idx = (term_scores_sample[factor_idx, :]).argsort()[::-1][:top_genes]
        top_terms = term_names[top_terms_idx].tolist()
        if return_scores:
            top_scores = np.asarray(term_scores_sample[factor_idx, :], dtype=float)
            return top_terms, top_scores
        return top_terms

    def get_signature_confidence(
        self,
        factor_idx: int,
        layer_idx: int,
        mc_samples: int = 100,
        top_genes: int = 10,
        pairwise: bool = False,
    ) -> float:
        """Get confidence score for a factor signature using Monte Carlo sampling.

        Args:
            factor_idx: index of the factor
            layer_idx: layer index of the factor
            mc_samples: number of Monte Carlo samples to take
            top_genes: number of top genes to consider in each sample
            pairwise: whether to compute pairwise Jaccard similarities

        Returns:
            confidence score as Jaccard similarity
        """
        signatures = []
        base_rng = random.PRNGKey(0)
        for i in range(mc_samples):
            rng = random.fold_in(base_rng, i)
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

    def get_relevances_dict(self) -> Dict[str, float]:
        """Get dictionary of factor relevance scores.

        Returns:
            dictionary mapping factor names to relevance scores
        """
        relevance_dict = {}
        for layer_idx in range(self.n_layers):
            for factor_idx, factor_name in enumerate(self.factor_names[layer_idx]):
                relevance_dict[factor_name] = self.pmeans["factor_means"][
                    self.factor_lists[layer_idx][factor_idx]
                ]
        return relevance_dict

    def get_sizes_dict(self) -> Dict[str, float]:
        """Get dictionary of factor sizes (number of cells per factor).

        Returns:
            dictionary mapping factor names to cell counts
        """
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
        self,
        top_genes: Optional[int] = None,
        scores: bool = False,
        sorted_scores: bool = False,
        layer_normalize: bool = False,
        drop_factors: Optional[List[str]] = None,
    ) -> Union[
        Dict[str, List[str]], Tuple[Dict[str, List[str]], Dict[str, np.ndarray]]
    ]:
        """Get dictionary of gene signatures for all factors across all layers.

        Args:
            top_genes: number of top genes per signature
            scores: whether to return scores along with signatures
            sorted_scores: whether to return scores sorted by ranking
            layer_normalize: whether to normalize scores within each layer
            drop_factors: list of factors to exclude

        Returns:
            dictionary mapping factor names to gene lists, or tuple of (signatures, scores) if scores is True
        """
        signatures_dict = {}
        scores_dict = {}
        for layer_idx in range(self.n_layers):
            layer_signatures, layer_scores = self.get_rankings(
                layer_idx=layer_idx,
                top_genes=top_genes,
                return_scores=True,
                sorted_scores=sorted_scores,
                drop_factors=drop_factors,
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

    def get_summary(self, top_genes: int = 10, reindex: bool = True) -> str:
        """Get a text summary of the model factors and their top genes.

        Args:
            top_genes: number of top genes to show per factor
            reindex: whether to reindex factors

        Returns:
            string summary of the model
        """
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
        factor_order = np.argsort(np.array(assignments))  # noqa: F841

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

    def get_layer_factor_orders(self) -> List[np.ndarray]:
        """Get the ordering of factors in each layer for plotting.

        Returns:
            list of arrays, one per layer, containing factor indices in plotting order
        """
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

    def attach_factors_to_obs(self, obs_key: str) -> List[List[str]]:
        """Attach factors to observation categories.

        Args:
            obs_key: key in model.adata.obs to use for attachment

        Returns:
            list of attachment lists, one per layer
        """
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
                        np.count_nonzero(self.adata.obs[obs_key].iloc[cells] == b)
                        / len(np.where(self.adata.obs[obs_key] == b)[0])
                        for b in self.adata.obs[obs_key].cat.categories
                    ]
                    obs_idx = np.argmax(prevs)  # obs attachment
                    layer_attachments.append(
                        self.adata.obs[obs_key].cat.categories[obs_idx]
                    )
            attachments.append(layer_attachments)
        return attachments

    def compute_weight(self, upper_factor_name: str, lower_factor_name: str) -> float:
        """Compute the weight between two factors across any number of layers.

        Args:
            upper_factor_name: name of the upper factor
            lower_factor_name: name of the lower factor

        Returns:
            weight value between the two factors
        """
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
        self, layer_idx: int, factor_name: str, obs_key: str, obs_val: str
    ) -> float:
        """Compute association score between a factor and an observation category.

        Args:
            layer_idx: layer index of the factor
            factor_name: name of the factor
            obs_key: key in model.adata.obs
            obs_val: value in obs_key to compute association with

        Returns:
            association score value
        """
        layer_name = self.layer_names[layer_idx]

        # Cells attached to factor
        adata_cells_in_factor = self.adata[
            np.where(self.adata.obs[f"{layer_name}"] == factor_name)[0]
        ]

        # Cells from obs_val
        adata_cells_from_obs = self.adata[
            np.where(self.adata.obs[obs_key] == obs_val)[0]
        ]

        cells_from_obs = float(adata_cells_from_obs.shape[0])  # noqa: F841

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
        self, layer_idx: int, factor_name: str, obs_key: str, obs_val: str
    ) -> float:
        """Compute assignment fraction between a factor and an observation category.

        Args:
            layer_idx: layer index of the factor
            factor_name: name of the factor
            obs_key: key in model.adata.obs
            obs_val: value in obs_key to compute fraction with

        Returns:
            assignment fraction value
        """
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

    def compute_factor_obs_weight_score(
        self, layer_idx: int, factor_name: str, obs_key: str, obs_val: str
    ) -> float:
        """Compute weight score between a factor and an observation category.

        Args:
            layer_idx: layer index of the factor
            factor_name: name of the factor
            obs_key: key in model.adata.obs
            obs_val: value in obs_key to compute weight with

        Returns:
            weight score value
        """
        layer_name = self.layer_names[layer_idx]  # noqa: F841

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
