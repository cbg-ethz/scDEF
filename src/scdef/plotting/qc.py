"""QC plotting functions for scDEF.

This module provides QC-related plotting functions for scDEF models.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import Optional, Tuple, Literal, Any, TYPE_CHECKING

FactorDiagQuantity = Literal[
    "ARD",
    "BRD",
    "n_eff_parents",
    "avg_n_eff_parents",
    "batch_purity",
    "batch_purity_soft",
    "signature_confidence",
    "n_cells",
]

_FACTOR_DIAG_LABELS: dict[str, str] = {
    "ARD": "ARD",
    "BRD": "BRD",
    "n_eff_parents": "Effective number of parents (L0)",
    "avg_n_eff_parents": "Avg. effective parents (lineage)",
    "batch_purity": "Batch purity",
    "batch_purity_soft": "Batch purity (soft)",
    "signature_confidence": "Signature confidence",
    "n_cells": "Number of cells",
}

from scdef.tools.hierarchy import effective_parents_from_clarity

if TYPE_CHECKING:
    from scdef.models._iscdef import iscDEF
    from scdef.models._scdef import scDEF


def scales(
    model: "scDEF",
    figsize: Tuple[float, float] = (8, 4),
    alpha: float = 0.6,
    fontsize: int = 12,
    legend_fontsize: int = 10,
    show: bool = True,
) -> Optional[Figure]:
    """Plot both cell and gene scales.

    Args:
        model: scDEF model instance
        figsize: figure size
        alpha: transparency level
        fontsize: font size for labels
        legend_fontsize: font size for legend
        show: whether to show the plot

    Returns:
        Figure object if show is False, None otherwise
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    scale(
        model,
        "cell",
        figsize,
        alpha,
        fontsize,
        legend_fontsize,
        axes[0],
        False,
    )
    scale(
        model,
        "gene",
        figsize,
        alpha,
        fontsize,
        legend_fontsize,
        axes[1],
        False,
    )
    if show:
        fig.tight_layout()
        plt.show()
    else:
        return fig


def scale(
    model: "scDEF",
    scale_type: Literal["cell", "gene"],
    figsize: Tuple[float, float] = (4, 4),
    alpha: float = 0.6,
    fontsize: int = 12,
    legend_fontsize: int = 10,
    ax: Optional[Axes] = None,
    show: bool = True,
) -> Optional[Axes]:
    """Plot learned scale factors vs observed scales.

    Args:
        model: scDEF model instance
        scale_type: type of scale to plot, either "cell" or "gene"
        figsize: figure size
        alpha: transparency level
        fontsize: font size for labels
        legend_fontsize: font size for legend
        ax: matplotlib axes to plot on
        show: whether to show the plot

    Returns:
        Axes object if show is False, None otherwise
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    if scale_type == "cell":
        x_data = model.batch_lib_sizes
        x_label = "Observed library size"

        def get_x_data_batch(b_cells):
            return model.batch_lib_sizes[np.where(b_cells)[0]]

        def get_y_data_batch(_, b_cells):
            return model.pmeans["cell_scale"].ravel()[np.where(b_cells)[0]]

    else:
        x_data = np.sum(model.X, axis=0)
        x_label = "Observed gene scale"

        def get_x_data_batch(b_cells):
            return np.sum(model.X[b_cells], axis=0)

        def get_y_data_batch(b_id, _):
            return model.pmeans["gene_scale"][b_id].ravel()

    if len(model.batches) > 1:
        for b_id, b in enumerate(model.batches):
            b_cells = model.adata.obs[model.batch_key] == b
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
            model.pmeans[f"{scale_type}_scale"].ravel(),
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


def relevance(
    model: "scDEF",
    mode: Literal["brd", "ard"] = "brd",
    thres: Optional[float] = None,
    iqr_mult: Optional[float] = None,
    show_yticks: bool = False,
    scale: Literal["linear", "log"] = "linear",
    normalize: bool = False,
    fontsize: int = 14,
    legend_fontsize: int = 12,
    xlabel: str = "Factor",
    ylabel: str = "Relevance",
    color: bool = False,
    show: bool = True,
    ax: Optional[Axes] = None,
    **kwargs: Any,
) -> Optional[Axes]:
    """Plot relevance determination scores.

    Args:
        model: scDEF model instance
        mode: mode to plot, either "brd" or "ard"
        thres: threshold value for relevance cutoff
        iqr_mult: multiplier for IQR-based threshold
        show_yticks: whether to show y-axis ticks
        scale: scale for y-axis, either "linear" or "log"
        normalize: whether to normalize relevance scores
        fontsize: font size for labels
        legend_fontsize: font size for legend
        xlabel: label for x-axis
        ylabel: label for y-axis
        color: whether to color bars by factor type
        show: whether to show the plot
        ax: matplotlib axes to plot on
        **kwargs: additional plotting keyword arguments

    Returns:
        Axes object if show is False, None otherwise
    """
    if not model.use_brd:
        raise ValueError("This model instance doesn't use the relevance prior.")

    ard = []
    if thres is not None:
        ard = thres
    else:
        ard = iqr_mult

    scales = model.pmeans["brd"].ravel()
    if mode == "ard":
        scales = model.pmeans["ard"].ravel()
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
        fig = ax.get_figure()  # noqa: F841

    layer_size = len(scales)
    if thres is not None or iqr_mult is not None:
        plt.axhline(cutoff, color="red", ls="--")

    if color:
        ax.bar(np.arange(layer_size), scales, color=model.layer_colorpalettes[0][0])
    else:
        ax.bar(np.arange(layer_size), scales)

    if len(scales) > 15:
        tick_pos = np.arange(0, layer_size, 2)
    else:
        tick_pos = np.arange(layer_size)
    ax.set_xticks(tick_pos)

    from scdef.models._iscdef import iscDEF

    if (
        isinstance(model, iscDEF)
        and hasattr(model, "factor_names")
        and len(model.factor_names) > 0
        and len(model.factor_names[0]) >= layer_size
    ):
        tick_labels = [str(name) for name in model.factor_names[0][:layer_size]]
        ax.set_xticklabels(
            [tick_labels[i] for i in tick_pos],
            rotation=45,
            ha="right",
            fontsize=max(6, fontsize - 4),
        )
    if not show_yticks:
        ax.set_yticks([])
    if mode == "brd":
        ax.set_title("Biological relevance determination", fontsize=fontsize)
    elif mode == "ard":
        ax.set_title("Automatic relevance determination", fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_yscale(scale)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    # Intentionally do not distinguish kept vs removed factors in QC plots.

    if show:
        plt.show()
    else:
        return ax


def gini_brd(
    model: "scDEF",
    normalize: bool = False,
    figsize: Tuple[float, float] = (4, 4),
    alpha: float = 0.6,
    fontsize: int = 12,
    legend_fontsize: int = 10,
    show: bool = True,
    ax: Optional[Axes] = None,
) -> Optional[Axes]:
    """Plot Gini coefficient vs BRD scores.

    Args:
        model: scDEF model instance
        normalize: whether to normalize BRD scores
        figsize: figure size
        alpha: transparency level
        fontsize: font size for labels
        legend_fontsize: font size for legend
        show: whether to show the plot
        ax: matplotlib axes to plot on

    Returns:
        Axes object if show is False, None otherwise
    """
    from ..utils import score_utils

    brds = model.pmeans["brd"].ravel()
    if normalize:
        brds = brds - np.min(brds)
        brds = brds / np.max(brds)
    ginis = np.array(
        [
            score_utils.gini(model.pmeans[f"{model.layer_names[0]}W"][k])
            for k in range(model.layer_sizes[0])
        ]
    )
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(ginis, brds, alpha=alpha)
    ax.set_xlabel("Gini index", fontsize=fontsize)
    ax.set_ylabel("BRD posterior mean", fontsize=fontsize)
    ax.set_xlim(0, 1)

    if show:
        plt.show()
    else:
        return ax


def loss(
    model: "scDEF",
    figsize: Tuple[float, float] = (4, 4),
    fontsize: int = 12,
    ax: Optional[Axes] = None,
    show: bool = True,
) -> Optional[Axes]:
    """Plot training loss over epochs.

    Args:
        model: scDEF model instance
        figsize: figure size
        fontsize: font size for labels
        ax: matplotlib axes to plot on
        show: whether to show the plot

    Returns:
        Axes object if show is False, None otherwise
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    elbos_for_qc = getattr(model, "qc_elbos", None)
    if elbos_for_qc is None:
        elbos_for_qc = model.elbos
        root_epochs = int(getattr(model, "root_epochs", 0))
    else:
        root_epochs = 0

    y = np.concatenate(elbos_for_qc)[:]
    if root_epochs > 0 and root_epochs < len(y):
        y = y[:-root_epochs]
    x = np.arange(1, len(y) + 1)
    ax.plot(x, y)
    ax.set_xlabel("Epoch", fontsize=fontsize)
    ax.set_yscale("log")
    ax.set_ylabel("Loss [log]", fontsize=fontsize)

    if show:
        plt.show()
    else:
        return ax


def _trace_plot(
    values: np.ndarray,
    ax: Axes,
    ylabel: str,
    xlabel: str = "Epoch",
    fontsize: int = 12,
    x_values: Optional[np.ndarray] = None,
) -> Axes:
    if x_values is None:
        x_values = np.arange(1, len(values) + 1)
    ax.plot(x_values, values)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    return ax


def ard_brd(
    model: "scDEF",
    figsize: Tuple[float, float] = (4, 4),
    show: bool = True,
    ax: Optional[Axes] = None,
    annotate_threshold: Optional[float] = None,
    legend_fontsize: int = 10,
    fontsize: int = 12,
) -> Optional[Axes]:
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()  # noqa: F841
    x = model.pmeans["factor_concentrations"].ravel()
    y = model.pmeans["factor_means"].ravel()
    ax.scatter(x, y, c="C0")
    ax.set_xlabel("BRD", fontsize=fontsize)
    ax.set_ylabel("ARD", fontsize=fontsize)
    if annotate_threshold is not None:
        for i in np.where(x > annotate_threshold)[0]:
            ax.annotate(
                str(i),  # label (factor index)
                (x[i], y[i]),  # point to annotate
                xytext=(3, 3),  # offset in points
                textcoords="offset points",
            )
    if show:
        plt.show()
    else:
        return ax


def qc(
    model: "scDEF",
    figsize: Tuple[float, float] = (8, 12),
    show: bool = True,
) -> Optional[Figure]:
    """Plot QC metrics for scDEF run.

    Plots include: loss over epochs, BRD vs Gini coefficient, learned vs observed
    cell scales, learned vs observed gene scales, and biological relevance determination.
    If trace diagnostics are available (e.g. ``n_eff_parents_trace``), a
    trace-oriented layout is used.

    Args:
        model: scDEF model instance
        figsize: figure size in inches
        show: whether to show the plot
    Returns:
        Figure object if show is False, None otherwise
    """

    has_neff_trace = (
        "n_eff_parents_trace" in model.adata.uns
        and len(model.adata.uns["n_eff_parents_trace"]) > 0
    )
    has_trace_epochs = (
        "n_eff_parents_trace_epochs" in model.adata.uns
        and len(model.adata.uns["n_eff_parents_trace_epochs"]) > 0
    )
    use_trace_layout = bool(has_neff_trace)

    if model.use_brd and use_trace_layout:
        fig = plt.figure(figsize=figsize)
        outer = fig.add_gridspec(
            4, 1, height_ratios=[1.0, 1.0, 0.85, 0.85], hspace=0.35
        )
        n_top_cols = 1 + int(has_neff_trace)
        top = outer[0].subgridspec(1, n_top_cols, wspace=0.35)
        middle = outer[1].subgridspec(1, 3, wspace=0.35)

        # First row: ELBO + available traces.
        col = 0
        loss(model, ax=fig.add_subplot(top[0, col]), show=False)
        col += 1
        if has_neff_trace:
            neff = np.asarray(model.adata.uns["n_eff_parents_trace"], dtype=float)
            neff_epochs = (
                np.asarray(model.adata.uns["n_eff_parents_trace_epochs"], dtype=int)
                if has_trace_epochs
                else None
            )
            _trace_plot(
                neff,
                ax=fig.add_subplot(top[0, col]),
                ylabel="n_eff_parents",
                x_values=neff_epochs,
            )
            col += 1
        # Second row: BRD vs Gini, cell scale, gene scale
        gini_brd(model, ax=fig.add_subplot(middle[0, 0]), show=False)
        scale(model, "cell", ax=fig.add_subplot(middle[0, 1]), show=False)
        scale(model, "gene", ax=fig.add_subplot(middle[0, 2]), show=False)
        # Third/Fourth rows: BRD and ARD as full-width panels
        relevance(model, mode="brd", ax=fig.add_subplot(outer[2]), show=False)
        relevance(model, mode="ard", ax=fig.add_subplot(outer[3]), show=False)
    elif model.use_brd:
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(4, 2)
        # First row
        loss(model, ax=fig.add_subplot(gs[0, 0]), show=False)
        gini_brd(model, ax=fig.add_subplot(gs[0, 1]), show=False)
        # Second row
        scale(model, "cell", ax=fig.add_subplot(gs[1, 0]), show=False)
        scale(model, "gene", ax=fig.add_subplot(gs[1, 1]), show=False)
        # Third row
        relevance(
            model,
            mode="brd",
            ax=fig.add_subplot(gs[2, 0:2]),
            show=False,
        )
        # Fourth row
        relevance(
            model,
            mode="ard",
            ax=fig.add_subplot(gs[3, 0:2]),
            show=False,
        )
    else:
        fig = plt.figure(figsize=(figsize[0], int(figsize[1] * 2 / 3)))
        gs = GridSpec(2, 2)
        # First row
        loss(model, ax=fig.add_subplot(gs[0, 0:2]), show=False)
        # Second row
        scale(model, "cell", ax=fig.add_subplot(gs[1, 0]), show=False)
        scale(model, "gene", ax=fig.add_subplot(gs[1, 1]), show=False)

    fig.tight_layout()
    if show:
        plt.show()
    else:
        return fig


def _factor_diag_quantity_label(quantity: str) -> str:
    return _FACTOR_DIAG_LABELS.get(quantity, quantity)


def _resolve_factor_diag_quantity(
    model: "scDEF",
    factor_obs_l0: Any,
    labels: np.ndarray,
    quantity: str,
) -> np.ndarray:
    """Return per-factor values for one diagnostic quantity."""
    if quantity == "signature_confidence":
        from scdef.tools.factor import get_stored_confident_signatures

        try:
            _, sig_conf = get_stored_confident_signatures(
                model,
                layer_idx=0,
                return_signature_confidences=True,
            )
        except KeyError as exc:
            raise KeyError(
                "signature_confidence requires confident signatures. Run "
                "`scd.tl.set_confident_signatures(model)` first."
            ) from exc
        return np.array(
            [float(sig_conf.get(str(label), np.nan)) for label in labels],
            dtype=float,
        )

    if quantity in ("batch_purity", "batch_purity_soft"):
        if quantity not in factor_obs_l0.columns:
            raise KeyError(
                f"{quantity} is missing from factor diagnostics. Run "
                "`scdef.tools.factor_diagnostics(model, batch_key=...)` first."
            )
        return factor_obs_l0[quantity].to_numpy(dtype=float)

    if quantity not in factor_obs_l0.columns:
        raise KeyError(f"Quantity '{quantity}' is missing from factor diagnostics.")
    return factor_obs_l0[quantity].to_numpy(dtype=float)


def _scale_marker_sizes(
    values: np.ndarray,
    min_size: float = 20.0,
    max_size: float = 300.0,
    fallback_size: float = 50.0,
) -> np.ndarray:
    """Map values to matplotlib scatter marker areas."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.full_like(values, fallback_size, dtype=float)
    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))
    if vmax <= vmin:
        return np.full_like(values, fallback_size, dtype=float)
    sizes = min_size + (max_size - min_size) * (values - vmin) / (vmax - vmin)
    sizes = np.where(np.isfinite(sizes), sizes, min_size)
    return sizes


def _scatter_with_optional_color(
    ax: Axes,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    color_vals: Optional[np.ndarray],
    sizes: Optional[np.ndarray],
    cmap: str = "viridis",
) -> Any:
    """Scatter with optional colormap; NaN colors are drawn as white."""
    min_marker_size = 3.0
    if sizes is not None:
        sizes = np.asarray(sizes, dtype=float)
        sizes = np.where(np.isfinite(sizes) & (sizes > 0), sizes, min_marker_size)

    if color_vals is None:
        scatter_kwargs: dict[str, Any] = {}
        if sizes is not None:
            scatter_kwargs["s"] = sizes
        return ax.scatter(x_vals, y_vals, **scatter_kwargs)

    finite_color = np.isfinite(color_vals)
    if np.any(finite_color):
        cmin = float(np.min(color_vals[finite_color]))
        cmax = float(np.max(color_vals[finite_color]))
    else:
        cmin, cmax = 0.0, 1.0
    if cmax <= cmin:
        cmax = cmin + 1.0
    color_cmap = plt.get_cmap(cmap).copy()
    color_cmap.set_under(color="white")
    under_val = cmin - (cmax - cmin) - 1e-6
    color_plot = np.where(finite_color, color_vals, under_val)
    return ax.scatter(
        x_vals,
        y_vals,
        c=color_plot,
        s=sizes,
        cmap=color_cmap,
        vmin=cmin,
        vmax=cmax,
    )


def factor_diagnostics(
    model: "scDEF",
    brd_min: float = 1.0,
    ard_min: float = 0.001,
    clarity_min: float = 0.5,
    batch_purity_max: Optional[float] = None,
    batch_purity_soft_max: Optional[float] = None,
    n_eff_parents_max: float = 1.5,
    figsize: tuple = (6, 4),
    ax: Optional[Axes] = None,
    annotate_factors: bool = False,
    annotation_fontsize: int = 8,
    annotation_alpha: float = 0.8,
    all_factors: bool = False,
    local_l0_scores: bool = False,
    x: FactorDiagQuantity = "BRD",
    y: Optional[FactorDiagQuantity] = None,
    color: Optional[FactorDiagQuantity] = None,
    size: Optional[FactorDiagQuantity] = "ARD",
    show: bool = True,
) -> Optional[Axes]:
    """
    Diagnostic scatter plot of layer-0 factors with flexible axis/color/size mapping.

    By default plots BRD vs effective parents with marker size scaled by ARD.
    When ``batch_purity_max`` or ``batch_purity_soft_max`` is set and ``color``
    is not overridden, points are colored by the corresponding batch purity
    (sizes still default to ARD).

    Args:
        model: scDEF model instance
        brd_min: minimum BRD filter threshold
        ard_min: minimum ARD filter threshold (fraction of total ARD)
        clarity_min: used for the horizontal cutoff when filtering on local
            ``n_eff_parents`` (not lineage ``avg_n_eff_parents``): cutoff is
            ``effective_parents_from_clarity(clarity_min, K_parents)``.
        batch_purity_max: optional upper bound on hard-assignment batch purity.
            Factors pass when ``batch_purity <= batch_purity_max``. If provided
            and ``color`` is not set, the scatter is colored by ``batch_purity``
            with the colorbar threshold at this value.
        batch_purity_soft_max: optional upper bound on soft batch purity (from
            ``X_<layer>_probs``). Factors pass when
            ``batch_purity_soft <= batch_purity_soft_max``. If provided and
            ``color`` is not set (and ``batch_purity_max`` is None), color
            defaults to ``batch_purity_soft``. Requires
            ``scdef.tools.factor_diagnostics(..., batch_key=...)``.
        n_eff_parents_max: used when filtering on lineage
            ``avg_n_eff_parents`` (``local_l0_scores=False`` and column present):
            dashed line at this value and pass rule ``y < n_eff_parents_max``
            (default ``1.5``).
        figsize: Figure size (if ax is None)
        ax: matplotlib Axes to plot on
        annotate_factors: whether to annotate each point with its factor label
        annotation_fontsize: fontsize for factor text annotations
        annotation_alpha: alpha value for factor text annotations
        all_factors: if True, plot diagnostics for all layer-0 factors from the
            complete snapshot ``model.adata.uns['factor_obs_full']`` (including
            factors that were filtered out). Default (False) plots the current
            view ``model.adata.uns['factor_obs']``, which after
            ``model.filter_factors()`` contains only kept factors.
        local_l0_scores: when ``y`` is not set, use layer-0 ``n_eff_parents``
            on the y-axis instead of lineage ``avg_n_eff_parents``.
        x: quantity for the x-axis (``ARD``, ``BRD``, ``n_eff_parents``,
            ``avg_n_eff_parents``, ``batch_purity``, ``batch_purity_soft``,
            ``signature_confidence``, ``n_cells``).
        y: quantity for the y-axis; default follows ``local_l0_scores`` /
            ``avg_n_eff_parents`` availability.
        color: quantity for marker color. Default ``None``: use
            ``batch_purity`` when ``batch_purity_max`` is set, else
            ``batch_purity_soft`` when ``batch_purity_soft_max`` is set, else
            uncolored markers.
        size: quantity for marker size. Default ``ARD``. Pass ``None`` for
            fixed marker size.
        show: whether to show the plot

    Returns:
        Axes object if show is False, None otherwise.
    """
    source_key = "factor_obs_full" if all_factors else "factor_obs"
    if source_key not in model.adata.uns:
        if all_factors and "factor_obs" in model.adata.uns:
            source_key = "factor_obs"
        else:
            raise KeyError(
                f"model.adata.uns['{source_key}'] not found. Run "
                "scdef.tools.factor_diagnostics(model) first."
            )

    factor_obs = model.adata.uns[source_key]
    l0_name = model.layer_names[0]
    if "child_layer" in factor_obs.columns:
        factor_obs_l0 = factor_obs[factor_obs["child_layer"] == l0_name].copy()
    else:
        factor_obs_l0 = factor_obs.copy()
        l0_prefix = f"{l0_name}_"
        factor_obs_l0 = factor_obs_l0[
            [
                isinstance(idx, str) and idx.startswith(l0_prefix)
                for idx in factor_obs_l0.index
            ]
        ]

    if "original_factor_idx" in factor_obs_l0.columns:
        factor_obs_l0 = factor_obs_l0.sort_values("original_factor_idx")

    labels = factor_obs_l0.index.to_numpy()
    if (
        "original_factor_idx" in factor_obs_l0.columns
        and hasattr(model, "factor_names")
        and len(model.factor_names) > 0
    ):
        original_idx = factor_obs_l0["original_factor_idx"].to_numpy(dtype=int)
        kept = np.asarray(model.factor_lists[0], dtype=int)
        orig_to_slot = {int(o): i for i, o in enumerate(kept)}
        labels = labels.astype(object, copy=True)
        current_names_l0 = model.factor_names[0]
        for i, oidx in enumerate(original_idx):
            slot = orig_to_slot.get(int(oidx))
            if slot is not None:
                labels[i] = current_names_l0[slot]

    brd_vals = factor_obs_l0["BRD"].to_numpy(dtype=float)
    ard_vals = factor_obs_l0["ARD"].to_numpy(dtype=float)
    batch_purity = None
    batch_purity_soft = None
    if batch_purity_max is not None or color == "batch_purity":
        batch_purity = _resolve_factor_diag_quantity(
            model, factor_obs_l0, labels, "batch_purity"
        )
    if batch_purity_soft_max is not None or color == "batch_purity_soft":
        batch_purity_soft = _resolve_factor_diag_quantity(
            model, factor_obs_l0, labels, "batch_purity_soft"
        )

    if y is None:
        if local_l0_scores:
            y_quantity = "n_eff_parents"
        elif "avg_n_eff_parents" in factor_obs_l0.columns:
            y_quantity = "avg_n_eff_parents"
        else:
            y_quantity = "n_eff_parents"
    else:
        y_quantity = y

    lineage_filter = (not local_l0_scores) and (
        "avg_n_eff_parents" in factor_obs_l0.columns
    )
    if lineage_filter:
        neffective_parents_max = float(n_eff_parents_max)
        filter_y_vals = _resolve_factor_diag_quantity(
            model, factor_obs_l0, labels, "avg_n_eff_parents"
        )
    else:
        filter_y_vals = _resolve_factor_diag_quantity(
            model, factor_obs_l0, labels, "n_eff_parents"
        )
        k_parents = factor_obs_l0["K_parents"].to_numpy(dtype=float)
        finite_k = k_parents[np.isfinite(k_parents)]
        if len(finite_k) == 0:
            raise ValueError(
                "No valid K_parents values found in factor_obs for layer 0."
            )
        k_for_threshold = int(finite_k[0])
        neffective_parents_max = float(
            effective_parents_from_clarity(clarity_min, k_for_threshold)
        )

    if color is None:
        if batch_purity_max is not None:
            color_quantity = "batch_purity"
        elif batch_purity_soft_max is not None:
            color_quantity = "batch_purity_soft"
        else:
            color_quantity = None
    else:
        color_quantity = color

    size_quantity = size

    x_vals = _resolve_factor_diag_quantity(model, factor_obs_l0, labels, x)
    y_vals = _resolve_factor_diag_quantity(model, factor_obs_l0, labels, y_quantity)
    color_vals = (
        None
        if color_quantity is None
        else _resolve_factor_diag_quantity(model, factor_obs_l0, labels, color_quantity)
    )
    size_source_vals = (
        None
        if size_quantity is None
        else _resolve_factor_diag_quantity(model, factor_obs_l0, labels, size_quantity)
    )

    ard_total = np.nansum(ard_vals)
    ard_thresh = ard_min * ard_total
    pass_mask = (
        (brd_vals > brd_min)
        & (filter_y_vals < neffective_parents_max)
        & (ard_vals > ard_thresh)
    )
    if batch_purity_max is not None:
        pass_mask &= batch_purity <= float(batch_purity_max)
    if batch_purity_soft_max is not None:
        pass_mask &= batch_purity_soft <= float(batch_purity_soft_max)
    factors_pass = np.where(pass_mask)[0]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    sizes = None if size_source_vals is None else _scale_marker_sizes(size_source_vals)
    im = _scatter_with_optional_color(ax, x_vals, y_vals, color_vals, sizes)

    if annotate_factors:
        for i in range(len(labels)):
            if np.isfinite(x_vals[i]) and np.isfinite(y_vals[i]):
                ax.text(
                    x_vals[i],
                    y_vals[i],
                    str(labels[i]),
                    fontsize=annotation_fontsize,
                    alpha=annotation_alpha,
                )

    if len(factors_pass) > 0:
        keep_sizes = 100 if sizes is None else sizes[factors_pass] + 40
        ax.scatter(
            x_vals[factors_pass],
            y_vals[factors_pass],
            s=keep_sizes,
            facecolors="none",
            edgecolors=plt.rcParams["axes.prop_cycle"].by_key()["color"][0],
            marker="o",
            label="Keep",
        )

    ax.set_xlabel(_factor_diag_quantity_label(x))
    ax.set_ylabel(_factor_diag_quantity_label(y_quantity))
    ax.axvline(
        brd_min,
        linestyle="--",
        color=plt.rcParams["axes.prop_cycle"].by_key()["color"][0],
    )
    ax.axhline(
        neffective_parents_max,
        linestyle="--",
        color=plt.rcParams["axes.prop_cycle"].by_key()["color"][0],
    )

    cbar = None
    cbar_thresh: Optional[float] = None
    if color_vals is not None:
        cbar_label = _factor_diag_quantity_label(color_quantity)
        cbar = plt.colorbar(im, ax=ax, label=cbar_label)
        if color_quantity == "ARD":
            cbar_thresh = ard_thresh
        elif color_quantity == "batch_purity" and batch_purity_max is not None:
            cbar_thresh = float(batch_purity_max)
        elif (
            color_quantity == "batch_purity_soft" and batch_purity_soft_max is not None
        ):
            cbar_thresh = float(batch_purity_soft_max)

        if cbar_thresh is not None:
            norm = im.norm
            cbar_min, cbar_max = norm.vmin, norm.vmax
            if cbar_min == cbar_max:
                finite_vals = color_vals[np.isfinite(color_vals)]
                if finite_vals.size > 0:
                    cbar_min, cbar_max = (
                        float(np.min(finite_vals)),
                        float(np.max(finite_vals)),
                    )
            if cbar_min < cbar_thresh < cbar_max:
                cbar.ax.axhline(
                    cbar_thresh,
                    color=plt.rcParams["axes.prop_cycle"].by_key()["color"][0],
                    linestyle="--",
                    linewidth=5,
                )

    if sizes is not None and size_source_vals is not None:
        size_finite = size_source_vals[np.isfinite(size_source_vals)]
        if size_finite.size > 0:
            s_min = float(np.nanmin(size_finite))
            s_max = float(np.nanmax(size_finite))
            if s_max > s_min:
                from matplotlib.lines import Line2D

                ref_vals = np.nanpercentile(size_finite, [10, 50, 90])
                ref_labels = ["10%", "50%", "90%"]
                handles = []
                for val, label in zip(ref_vals, ref_labels):
                    s_val = 20.0 + 280.0 * (val - s_min) / (s_max - s_min)
                    handles.append(
                        Line2D(
                            [0],
                            [0],
                            marker="o",
                            linestyle="",
                            markerfacecolor="lightgray",
                            markeredgecolor="gray",
                            markersize=float(np.sqrt(s_val)),
                            label=label,
                        )
                    )
                size_legend = ax.legend(
                    handles=handles,
                    title=f"Size ({_factor_diag_quantity_label(size_quantity)})",
                    loc="upper right",
                    fontsize=8,
                    title_fontsize=8,
                    labelspacing=1.2,
                    borderpad=1.0,
                    handletextpad=1.2,
                    frameon=True,
                )
                ax.add_artist(size_legend)

    ax.set_title(f"{len(factors_pass)} factors pass filters")
    if len(factors_pass) > 0:
        from matplotlib.lines import Line2D

        keep_handle = Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="none",
            markeredgecolor=plt.rcParams["axes.prop_cycle"].by_key()["color"][0],
            markersize=10,
            label="Keep",
        )
        ax.legend(handles=[keep_handle], loc="best")

    if show:
        plt.show()
    else:
        return ax
