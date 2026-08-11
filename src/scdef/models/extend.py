"""Model-derivation functions for creating new scDEF models from existing ones."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Optional,
    Union,
    Sequence,
    Mapping,
    Dict,
    List,
    Any,
)

import numpy as np
import jax.numpy as jnp
from anndata import AnnData

if TYPE_CHECKING:  # circular at runtime; imported lazily in the bodies below
    from scdef.models._scdef import scDEF

# Bounds applied to any warm-started ``gene_scale`` mean. The pooled-marginal MLE
# reaches ~1e7 for the most abundant genes in real data, so the historical 1e6
# upper bound silently truncated exactly the genes these warm starts exist to get
# right.
_GENE_SCALE_MIN = 1e-6
_GENE_SCALE_MAX = 1e7

# Denominator floor for the pooled-marginal MLE. Genes whose expected
# reconstruction is at or below this have no factor support to divide by and fall
# back to the geometric-mean profile instead of producing inf/nan.
_POOLED_U_FLOOR = 1e-12


def _resolve_init_gene_scale_array(
    reference_model,
    init_gene_scale: Union[str, np.ndarray],
    n_batches: int,
    n_genes: int,
) -> np.ndarray:
    """Build ``(n_batches, n_genes)`` gene-scale means for warm-starting pass 2.

    With ``init_gene_scale='reference'`` the reference's per-batch rows are pooled
    with a geometric mean (a straight copy when there is only one row). This is the
    warm start used by [`from_reference`][scdef.from_reference], and the fallback used by
    [`decompose_batch_effects`][scdef.decompose_batch_effects] when the pooled-marginal MLE is unavailable; see
    `_pooled_marginal_gene_scale` for the difference between the two.

    Returns:
        Per-batch gene-scale means to pass to ``init_var_params``.

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
        profile = np.clip(profile, _GENE_SCALE_MIN, _GENE_SCALE_MAX)
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


def _pooled_marginal_gene_scale(
    model,
    init_z: Sequence[np.ndarray],
    init_w: Sequence[np.ndarray],
    fallback_profile: np.ndarray,
    logger=None,
) -> np.ndarray:
    """Shared ``gene_scale`` profile that reproduces the pooled counts at init.

    The likelihood mean is ``(z @ W) * cell_scale * gene_scale``. Summing over cells
    at the model's own initialization gives, for gene ``g``,

    .. math::
        U_g = \\big((c^\\top Z_0)\\, W_0\\big)_g, \\qquad s_g = S_g / U_g,

    where ``c`` is the ``cell_scale`` warm start that ``init_var_params`` builds when
    ``init_budgets=True`` (``clip(lib / mean(lib), 1e-3, 1e2)``, ``lib = X.sum(1)``),
    ``Z_0``/``W_0`` are the layer-0 warm starts after the same clipping
    ``init_var_params`` applies to them, and ``S_g = X.sum(0)`` are the observed
    pooled counts. ``s_g`` is therefore the maximum-likelihood shared scale for the
    pooled marginal of gene ``g`` under the model's actual initialization.

    Why not the geometric mean across the reference's batch rows: that profile was
    fitted against the *reference's* ``cell_scale``, which the decomposed model
    discards, and a geometric middle under-predicts the pooled arithmetic level of
    any gene whose batch levels differ. Both defects are level errors in the warm
    start, and both disappear here because ``s_g`` is derived from the pooled counts
    directly rather than transplanted.

    This does not read the reference's per-batch rows at all, so it applies equally
    to one-row and multi-row references, and it carries no batch information: the
    resulting profile depends only on the pooled totals.

    Args:
        model: the freshly constructed (unfitted) target model. Supplies ``X`` and
            ``batch_lib_sizes``.
        init_z: per-layer ``z`` warm starts; only layer 0 is used.
        init_w: per-layer ``W`` warm starts; only layer 0 is used.
        fallback_profile: ``(n_genes,)`` profile used for genes with no factor
            support, i.e. where ``U_g`` underflows.
        logger: optional logger used to report the fallback count.

    Returns:
        A ``(n_genes,)`` array of gene-scale means, clipped to
        ``[_GENE_SCALE_MIN, _GENE_SCALE_MAX]``.
    """
    # numpy raises spurious divide/overflow/invalid flags from the Accelerate BLAS
    # matmul even for wholly finite operands, so the flags are muted here and the
    # result is validated explicitly by the `isfinite` check below instead.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        # Mirror `init_var_params(init_budgets=True)`: cell_scale means are the
        # library sizes divided by their mean, clipped.
        lib = np.asarray(model.batch_lib_sizes, dtype=np.float64).reshape(-1)
        cell_scale = np.clip(lib / np.mean(lib), 1e-3, 1e2)

        # Mirror the clipping `init_var_params` applies to the layer-0 warm starts
        # before they become variational means.
        z0 = np.clip(np.asarray(init_z[0], dtype=np.float64), 1e-3, 1e6)
        w0 = np.clip(np.asarray(init_w[0], dtype=np.float64), 1e-3, 1e8)

        # ((c^T Z) W)_g -- O(NK + KG), never forms the (N, G) reconstruction.
        u = (cell_scale @ z0) @ w0

    pooled_counts = np.asarray(model.X, dtype=np.float64).sum(axis=0)

    unsupported = ~np.isfinite(u) | (u <= _POOLED_U_FLOOR)
    n_unsupported = int(np.count_nonzero(unsupported))
    if n_unsupported and logger is not None:
        logger.info(
            f"Pooled-marginal gene_scale warm start: {n_unsupported} of {u.size} "
            "genes have no factor support at initialization; falling back to the "
            "geometric mean of the reference gene_scale for those genes."
        )

    safe_u = np.where(unsupported, 1.0, u)
    profile = np.where(
        unsupported,
        np.asarray(fallback_profile, dtype=np.float64).reshape(-1),
        pooled_counts / safe_u,
    )
    return np.clip(profile, _GENE_SCALE_MIN, _GENE_SCALE_MAX).astype(np.float32)


def _reference_model_kwargs(
    reference_model, layer_sizes: Sequence[int]
) -> Dict[str, Any]:
    """Extract constructor kwargs from a fitted reference model."""
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


def from_reference(
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
        reference_model: a fitted ``scDEF`` providing the hierarchy.
        adata: AnnData for the new model.
        counts_layer: counts layer key in ``adata``.
        batch_key: batch annotation column in ``adata.obs``.
        reference_obs: reference batch label (for gene-scale init).
        query_obs: query batch label (for gene-scale init).
        copy_cell_z: copy per-cell z warm starts for shared cells.
        init_gene_scale: how to initialize per-batch ``gene_scale`` variational
            means before the first fit.

            * ``'batch'`` (default): use per-batch count means from
              ``load_adata`` (``1 / gene_ratio_init``).
            * ``'reference'``: broadcast the reference model's fitted
              ``pmeans['gene_scale']`` to every batch.
            * array: explicit ``(n_genes,)`` or ``(n_batches, n_genes)`` means.
        **kwargs: additional keyword arguments forwarded to the model constructor.

    Returns:
        A new (unfitted) ``scDEF`` model with hierarchy warm-started from the reference.
    """
    from scdef.models._scdef import scDEF

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

    model_kwargs = _reference_model_kwargs(reference_model, layer_sizes)
    model_kwargs.update(kwargs)
    model = scDEF(
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
        init_gene_scale_arr = _resolve_init_gene_scale_array(
            reference_model,
            init_gene_scale,
            int(model.n_gene_scale_batches),
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


def add_batch_correction(
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
    2. Call this function to construct a new model that shares the same
       hierarchy (``factor_lists``, layer sizes, ``W``, ``BRD``, ``ARD``)
       and re-fits it under per-batch gene-scale priors so batch effects
       are absorbed by ``gene_scale``, not by the hierarchy.

    Args:
        reference_model: a fitted ``scDEF`` providing the hierarchy.
        batch_key: column in ``adata.obs`` to use as the new batch annotation.
        adata: AnnData for the second pass. Defaults to ``reference_model.adata``.
        counts_layer: counts layer for the new ``adata``.
        copy_cell_z: whether to copy per-cell ``z`` warm starts for shared cells.
        freeze_w: hold every per-layer ``W`` fixed during the second fit.
        learn_budgets: allow per-batch gene-scale and per-cell budgets to move.
        n_epoch: epochs for the second-pass fit.
        lr: learning rate for the second-pass fit.
        tolerance: early-stopping tolerance for the second-pass fit.
        from_reference_kwargs: extra kwargs forwarded to [`from_reference`][scdef.from_reference].
        **fit_kwargs: additional kwargs forwarded to ``model.fit()``.

    Returns:
        The new fitted model with batch correction applied.
    """
    if counts_layer is None:
        counts_layer = getattr(reference_model, "counts_layer", None)

    target_adata = adata if adata is not None else reference_model.adata
    from_reference_kwargs = dict(from_reference_kwargs or {})
    from_reference_kwargs.setdefault("counts_layer", counts_layer)
    from_reference_kwargs.setdefault("copy_cell_z", copy_cell_z)
    from_reference_kwargs.setdefault("init_gene_scale", "reference")

    model = from_reference(
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


def _resolve_decompose_batch_kwargs(
    reference_model,
    target_adata: AnnData,
    batch_cell_scale: bool,
    logger=None,
) -> Dict[str, Any]:
    """Resolve the batch-related constructor kwargs for ``decompose_batch_effects``.

    ``batch_key`` in ``scDEF`` controls two independent things. The **cell side** is
    the per-batch Gamma prior that ``cell_scale`` shrinks toward (``batch_lib_ratio``);
    it is gene-independent and can therefore only carry sequencing depth. The **gene
    side** gives ``gene_scale`` one row per batch; it is gene-specific and can absorb
    gene programmes. Decomposition wants the gene side *off* -- that is the whole point
    of pushing structure into the L0 factors -- but there is no reason to also discard
    the depth prior.

    Args:
        reference_model: the fitted model being decomposed.
        target_adata: the AnnData the decomposed model will be built on.
        batch_cell_scale: whether to keep the reference's per-batch ``cell_scale`` prior.
        logger: optional logger used to report graceful degradation.

    Returns:
        Constructor kwargs: ``{"batch_key": None}`` to reproduce the historical
        construction, or ``{"batch_key": <key>, "batch_gene_scale": False}`` to keep
        only the cell side.
    """

    def _log(msg: str) -> None:
        if logger is not None:
            logger.info(msg)

    if not batch_cell_scale:
        return {"batch_key": None}

    batch_key = getattr(reference_model, "batch_key", None)
    if batch_key is None:
        _log(
            "batch_cell_scale=True but the reference model has no batch_key; "
            "building the decomposed model without a batch key."
        )
        return {"batch_key": None}

    if batch_key not in target_adata.obs.columns:
        _log(
            f"batch_cell_scale=True but `{batch_key}` is not a column of the target "
            "adata.obs; building the decomposed model without a batch key."
        )
        return {"batch_key": None}

    n_present = int(len(np.unique(np.asarray(target_adata.obs[batch_key].values))))
    if n_present < 2:
        _log(
            f"batch_cell_scale=True but `{batch_key}` has only {n_present} distinct "
            "value(s) in the cells being fitted; building the decomposed model "
            "without a batch key."
        )
        return {"batch_key": None}

    _log(
        f"batch_cell_scale=True: keeping the per-batch cell_scale prior from "
        f"`{batch_key}` ({n_present} batches) while gene_scale stays a single "
        "shared row."
    )
    return {"batch_key": batch_key, "batch_gene_scale": False}


def _resolve_decompose_gene_scale(
    reference_model,
    model,
    init_z: Sequence[np.ndarray],
    init_w: Sequence[np.ndarray],
    init_gene_scale: Union[str, np.ndarray],
    nmf_init: bool,
) -> Optional[np.ndarray]:
    """Resolve the shared ``gene_scale`` warm start for ``decompose_batch_effects``.

    ``gene_scale`` is a single shared row in every decomposition, because the
    decomposed model is always built with ``batch_gene_scale=False`` (or no batch
    key), so this returns one row tiled to ``model.n_gene_scale_batches``.

    Args:
        reference_model: the fitted model being decomposed.
        model: the freshly constructed (unfitted) decomposed model.
        init_z: per-layer ``z`` warm starts passed to ``init_var_params``.
        init_w: per-layer ``W`` warm starts passed to ``init_var_params``.
        init_gene_scale: ``'reference'``, ``'prior'``, or an explicit array.
        nmf_init: whether layer-0 ``W`` will be set by NMF instead of ``init_w``.

    Returns:
        The gene-scale means to pass to ``init_var_params``, or ``None`` for
        ``'prior'`` (which leaves the count-derived prior mean in place).

    Raises:
        ValueError: if ``init_gene_scale`` is an unrecognized string.
    """
    n_rows = int(model.n_gene_scale_batches)
    n_genes = int(model.adata.n_vars)
    logger = getattr(model, "logger", None)

    if not isinstance(init_gene_scale, str):
        return _resolve_init_gene_scale_array(
            reference_model,
            np.asarray(init_gene_scale, dtype=np.float32),
            n_batches=n_rows,
            n_genes=n_genes,
        )

    if init_gene_scale == "prior":
        return None
    if init_gene_scale != "reference":
        raise ValueError(
            "init_gene_scale must be 'reference', 'prior', or an array; "
            f"got {init_gene_scale!r}."
        )

    geometric_arr = _resolve_init_gene_scale_array(
        reference_model,
        "reference",
        n_batches=n_rows,
        n_genes=n_genes,
    )
    if nmf_init:
        # `init_var_params` ignores `init_w` under `nmf_init`, so a profile derived
        # from the reference W would not describe this model's actual initialization.
        if logger is not None:
            logger.info(
                "nmf_init=True: layer-0 W is set by NMF rather than by the reference, "
                "so the pooled-marginal gene_scale warm start does not describe this "
                "model's initialization; falling back to the geometric mean of the "
                "reference gene_scale."
            )
        return geometric_arr

    profile = _pooled_marginal_gene_scale(
        model,
        init_z,
        init_w,
        geometric_arr[0],
        logger=logger,
    )
    return np.tile(profile[None, :], (n_rows, 1))


def decompose_batch_effects(
    reference_model: "scDEF",
    *,
    adata: Optional[AnnData] = None,
    counts_layer: Optional[str] = None,
    batch_cell_scale: bool = True,
    top_layer: int = 1,
    n_epoch: int = 400,
    lr: float = 0.05,
    tolerance: float = 1e-4,
    nmf_init: bool = False,
    init_gene_scale: Union[str, np.ndarray] = "reference",
    **fit_kwargs: Any,
) -> "scDEF":
    """Re-learn lower layers under a frozen upper hierarchy to discover batch programs.

    Two-stage workflow:

    1. ``reference_model`` was fitted **with** a ``batch_key``, producing a
       hierarchy where per-batch ``gene_scale`` absorbed between-batch variance.
    2. This function creates a new model with the per-batch ``gene_scale``
       **switched off**, warm-starts all ``W`` from the reference, and re-learns
       all layers up to ``top_layer``.  At the boundary (``top_layer``), only
       ``W`` is re-learned while ``z`` stays fixed — preserving the cell-to-group
       assignments as the structural constraint.  Layers below
       ``top_layer`` are fully re-learned (both ``W`` and ``z``).
       Layers above ``top_layer`` remain completely fixed.

    **Which half of ``batch_key`` is discarded.** ``batch_key`` in ``scDEF``
    controls two independent quantities:

    * the **gene side** — ``gene_scale`` with one row per batch, i.e. a
      gene-*specific* per-batch multiplier;
    * the **cell side** — ``batch_lib_sizes`` / ``batch_lib_ratio``, the
      per-batch Gamma prior that ``cell_scale`` shrinks toward, i.e. a
      gene-*independent* per-batch multiplier.

    Only the gene side must go. A gene-specific term can express a gene
    programme, so leaving it in place would let it re-absorb exactly the
    structure this function is trying to surface in the L0 factors. A
    gene-independent term cannot express a programme at all — it is one number
    per batch, so the most it can represent is sequencing depth. Historically
    both were discarded together; ``batch_cell_scale=True`` (the default) now
    keeps the cell side.

    Keeping the cell side is *expected* to reduce the depth component that would
    otherwise have to land in ``z``: with ``batch_key=None`` the model shrinks
    every ``cell_scale`` toward a single Gamma fitted to the pooled library-size
    mean and variance, which is misspecified for both batches when they differ
    in depth. This expectation has **not** been validated by refitting; no claim
    is made here about the effect on the resulting factors.

    L0 factor BRD and ARD are re-initialized from model priors rather than
    copied from the reference, so factor relevance can be re-estimated during
    decomposition.

    **How the shared ``gene_scale`` is warm-started.** The count-derived prior alone
    leaves a large reconstruction gap here, because the batch-key model often learns
    per-batch scales orders of magnitude above it. The default
    (``init_gene_scale='reference'``) instead solves for the shared scale directly:
    with the likelihood mean ``(z @ W) * cell_scale * gene_scale``, summing over
    cells at this model's own initialization gives
    ``U_g = ((cell_scale^T z_0) W_0)_g``, and ``s_g = X.sum(0)_g / U_g`` is the
    maximum-likelihood shared scale for the pooled marginal of gene ``g``. It is
    computed from the target data and this model's warm starts, so it needs no
    transplant of the reference's fitted scale, and it applies whether the reference
    had one ``gene_scale`` row or several.

    This is exact **for the pooled marginal only**. A single shared row still cannot
    fit two batches whose levels for a gene differ — no shared scale can. That
    residual is left deliberately unabsorbed: pushing it into ``z @ W`` is what makes
    per-batch structure visible in the re-learned lower layers, which is the point of
    the decomposition. What the pooled MLE removes is the part that is *not*
    structure, namely a systematic level offset in the warm start.

    With ``top_layer=1`` (default):
        - L0: W warm-started and re-learned, z re-learned
        - L1: W warm-started and re-learned, z frozen
        - L2+: fully frozen

    With ``top_layer=2``:
        - L0: W warm-started and re-learned, z re-learned
        - L1: W warm-started and re-learned, z re-learned
        - L2: W warm-started and re-learned, z frozen
        - L3+: fully frozen

    Args:
        reference_model: a fitted ``scDEF`` that was trained with
            ``batch_key``.
        adata: AnnData for the second stage. Defaults to
            ``reference_model.adata``.
        counts_layer: counts layer for ``adata``.
        batch_cell_scale: if True (default), carry the reference model's
            ``batch_key`` into the decomposed model with ``batch_gene_scale=False``,
            so the per-batch ``cell_scale`` prior is kept while ``gene_scale``
            stays a single shared row. Degrades gracefully (with a log message) to
            no batch key when the reference has no ``batch_key``, when that key is
            absent from the target ``adata.obs``, or when fewer than two batches
            are present in the cells being fitted. Set to False to reproduce the
            historical construction, which discarded both sides.
        top_layer: the highest layer whose ``W`` is re-learned. Its ``z``
            remains frozen as the structural anchor. Default ``1``.
        n_epoch: training epochs for the re-learning phase.
        lr: learning rate for the re-learning phase.
        tolerance: early-stopping tolerance.
        nmf_init: if True, initialize L0 W via NMF on the data instead of
            warm-starting from the reference. Default False.
        init_gene_scale: warm start for the shared ``gene_scale`` in the
            decomposed model.

            * ``\"reference\"`` (default): the pooled-marginal MLE
              ``X.sum(0)_g / ((cell_scale^T z_0) W_0)_g``, which reproduces the
              observed pooled counts exactly at initialization (see above). Falls
              back to the geometric mean of
              ``reference_model.pmeans['gene_scale']`` across batches when
              ``nmf_init=True`` — layer-0 ``W`` is then set by NMF, so a profile
              derived from the reference ``W`` would not describe the model's actual
              initialization — and, per gene, for genes with no factor support.
            * ``\"prior\"``: only the count-derived prior mean
              (``1 / gene_ratio_init``).
            * an explicit ``(n_genes,)`` array.
        **fit_kwargs: additional keyword arguments forwarded to ``_learn``.

    Returns:
        A new fitted model whose lower-layer factors reveal batch-specific
        and shared gene programs under the frozen upper-layer cell assignments.
    """
    from scdef.models._scdef import scDEF

    top_layer = int(top_layer)
    if reference_model.n_layers < top_layer + 1:
        raise ValueError(
            f"reference_model must have at least {top_layer + 1} layers "
            f"for top_layer={top_layer}, but has {reference_model.n_layers}."
        )

    if counts_layer is None:
        counts_layer = getattr(reference_model, "counts_layer", None)

    target_adata = adata if adata is not None else reference_model.adata

    factor_lists = [np.asarray(f, dtype=int) for f in reference_model.factor_lists]
    layer_sizes = [len(f) for f in factor_lists]

    # Build init_w: warm-start all layers from reference
    init_w: List[Optional[np.ndarray]] = []
    for layer_idx in range(reference_model.n_layers):
        keep = factor_lists[layer_idx]
        w = np.asarray(
            reference_model.pmeans[f"{reference_model.layer_names[layer_idx]}W"],
            dtype=np.float32,
        )
        if layer_idx == 0:
            init_w.append(w[keep])
        else:
            parent_keep = factor_lists[layer_idx - 1]
            init_w.append(w[np.ix_(keep, parent_keep)])

    # Build init_z: warm-start ALL layers from reference.
    # Layers below top_layer are free to re-learn; top_layer+ will be frozen
    # after init via the tight-distribution overwrite below.
    init_z: List[Optional[np.ndarray]] = []
    for layer_idx in range(reference_model.n_layers):
        keep = factor_lists[layer_idx]
        z = np.asarray(
            reference_model.pmeans[f"{reference_model.layer_names[layer_idx]}z"],
            dtype=np.float32,
        )
        if target_adata is reference_model.adata:
            init_z.append(z[:, keep])
        else:
            ref_pos = {
                str(name): i for i, name in enumerate(reference_model.adata.obs_names)
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

    # Create the new model with the per-batch gene side switched off. The cell side
    # (per-batch cell_scale prior) is kept when `batch_cell_scale` is True.
    model_kwargs = _reference_model_kwargs(reference_model, layer_sizes)
    model_kwargs.update(
        _resolve_decompose_batch_kwargs(
            reference_model,
            target_adata,
            bool(batch_cell_scale),
            logger=getattr(reference_model, "logger", None),
        )
    )
    model = scDEF(
        target_adata,
        counts_layer=counts_layer,
        **model_kwargs,
    )
    model.alpha = float(reference_model.alpha)
    model.top_alpha = reference_model.top_alpha
    model.update_model_priors(update_alpha_from_cov=False)

    init_gene_scale_arr = _resolve_decompose_gene_scale(
        reference_model,
        model,
        init_z,
        init_w,
        init_gene_scale,
        bool(nmf_init),
    )

    # Initialize variational parameters with high concentration to stay
    # close to reference warm-starts (avoids NaN from large deviations)
    model.init_var_params(
        init_budgets=True,
        init_alpha=False,
        init_z=init_z,
        init_w=init_w,
        init_brd=None,
        init_ard=None,
        init_gene_scale=init_gene_scale_arr,
        nmf_init=nmf_init,
        z_init_concentration=100.0,
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
    # Layers below `top_layer` are the ones re-learned without a batch key, so
    # that is where per-batch splits can appear; `top_layer` itself still holds
    # the batch-corrected signal and is the roll-up target for those splits.
    model.adata.uns["batch_technical_top_layer"] = int(top_layer)

    # Record the reference fit's per-batch `gene_scale` -- the term this model
    # deliberately does without. It is what the decomposed L0 factors have to
    # re-absorb, so keeping it here lets `batch_structure_report` and
    # `get_factor_batch_gene_scale_affinity` score them against it without the
    # reference model being kept around. A (n_batches, n_genes) profile is tens
    # of KB, and the gene axes match by construction.
    ref_pmeans = getattr(reference_model, "pmeans", None)
    ref_gene_scale = (
        ref_pmeans.get("gene_scale") if isinstance(ref_pmeans, dict) else None
    )
    if ref_gene_scale is not None:
        gs = np.asarray(ref_gene_scale, dtype=float)
        # Only a genuinely per-batch profile carries a contrast worth storing: a
        # single shared row has nothing to score factors against.
        if gs.ndim == 2 and gs.shape[0] > 1 and gs.shape[1] == int(model.adata.n_vars):
            model.adata.uns["reference_gene_scale"] = gs
            model.adata.uns["reference_gene_scale_batches"] = [
                str(b) for b in getattr(reference_model, "batches", [])
            ]
    return model


def from_hierarchy(
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

    Args:
        adata: AnnData for the new model.
        hierarchy: a fitted ``scDEF`` model or a sequence of W matrices.
        counts_layer: counts layer key in ``adata``.
        batch_key: batch annotation column in ``adata.obs``.
        init_brd: explicit BRD initialization (overrides reference).
        init_ard: explicit ARD initialization (overrides reference).
        init_z: explicit per-layer z initialization.
        **kwargs: additional keyword arguments forwarded to the model constructor.

    Returns:
        A new (unfitted) ``scDEF`` model initialized from the hierarchy.
    """
    from scdef.models._scdef import scDEF

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
        factor_lists = [np.asarray(f, dtype=int) for f in reference_model.factor_lists]
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
        if init_brd is None:
            init_brd = np.asarray(reference_model.pmeans["brd"], dtype=np.float32)[
                factor_lists[0]
            ]
        if init_ard is None:
            init_ard = np.asarray(
                reference_model.pmeans["factor_means"], dtype=np.float32
            )[factor_lists[0]]
        model_kwargs = _reference_model_kwargs(
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
    model = scDEF(
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
