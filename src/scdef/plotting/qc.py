"""QC plotting functions for scDEF.

This module provides QC-related plotting functions for scDEF models.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import Optional, Tuple, Literal, Any, TYPE_CHECKING

from scdef.tools.hierarchy import effective_parents_from_clarity

if TYPE_CHECKING:
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
        ax.set_xticks(np.arange(0, layer_size, 2))
    else:
        ax.set_xticks(np.arange(layer_size))
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


def factor_diagnostics(
    model: "scDEF",
    brd_min: float = 1.0,
    ard_min: float = 0.001,
    clarity_min: float = 0.5,
    batch_purity_min: Optional[float] = None,
    n_eff_parents_max: float = 1.5,
    figsize: tuple = (6, 4),
    ax: Optional[Axes] = None,
    annotate_factors: bool = False,
    annotation_fontsize: int = 8,
    annotation_alpha: float = 0.8,
    all_factors: bool = False,
    local_l0_scores: bool = False,
    show: bool = True,
) -> Optional[Axes]:
    """
    Diagnostic scatter plot of factors: BRD vs effective parents colored by ARD.

    Args:
        model: scDEF model instance
        brd_min: minimum BRD filter threshold
        ard_min: minimum ARD filter threshold (fraction of total ARD)
        clarity_min: used for the horizontal cutoff when **not** plotting lineage
            ``avg_n_eff_parents`` (local-L0 mode or fallback to L0 ``n_eff_parents``):
            cutoff is ``effective_parents_from_clarity(clarity_min, K_parents)``.
        batch_purity_min: optional threshold for batch purity. If provided,
            an additional panel is shown with ``batch_purity`` on the x-axis and
            the same y-axis as the default panel. Requires
            ``scdef.tools.factor_diagnostics(..., batch_key=...)``.
        n_eff_parents_max: used **only** when the y-axis is lineage
            ``avg_n_eff_parents`` (``local_l0_scores=False`` and column present): dashed
            line at this value and pass rule ``y < n_eff_parents_max`` (default ``1.5``).
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
        local_l0_scores: if True, plot layer-0-only ``n_eff_parents`` on the y-axis.
            If False (default), plot lineage-averaged ``avg_n_eff_parents`` when
            present in ``factor_obs``, otherwise fall back to ``n_eff_parents``.
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
    # Remap labels of currently kept factors to their current model names
    # (useful when plotting from factor_obs_full, whose index reflects the
    # diagnostic-time naming and may differ from the post-filter names).
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
    x = factor_obs_l0["BRD"].to_numpy(dtype=float)
    if local_l0_scores:
        y_col = "n_eff_parents"
        y_label = "Effective number of parents (L0)"
    else:
        if "avg_n_eff_parents" in factor_obs_l0.columns:
            y_col = "avg_n_eff_parents"
            y_label = "Avg. effective parents (lineage)"
        else:
            y_col = "n_eff_parents"
            y_label = "Effective number of parents (L0)"
    y = factor_obs_l0[y_col].to_numpy(dtype=float)
    z = factor_obs_l0["ARD"].to_numpy(dtype=float)
    batch_purity = None
    if batch_purity_min is not None:
        if "batch_purity" not in factor_obs_l0.columns:
            raise KeyError(
                "batch_purity is missing from factor diagnostics. Run "
                "`scdef.tools.factor_diagnostics(model, batch_key=...)` first."
            )
        batch_purity = factor_obs_l0["batch_purity"].to_numpy(dtype=float)
    lineage_plot = (not local_l0_scores) and (
        "avg_n_eff_parents" in factor_obs_l0.columns
    )
    if lineage_plot:
        neffective_parents_max = float(n_eff_parents_max)
    else:
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

    ard_total = np.nansum(z)
    factors_pass_base = np.where(
        (x > brd_min) & (y < neffective_parents_max) & (z > ard_min * ard_total)
    )[0]
    if batch_purity is None:
        factors_pass = factors_pass_base
    else:
        factors_pass = np.where(
            (x > brd_min)
            & (y < neffective_parents_max)
            & (z > ard_min * ard_total)
            & (batch_purity >= float(batch_purity_min))
        )[0]

    if batch_purity is None:
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        im = ax.scatter(x, y, c=z, cmap="viridis")
        axes = [ax]
    else:
        if ax is not None:
            raise ValueError(
                "When batch_purity_min is provided, pass ax=None to allow a "
                "2-panel layout."
            )
        fig, axes_arr = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]))
        axes = list(np.atleast_1d(axes_arr))
        ax = axes[0]
        im = ax.scatter(x, y, c=z, cmap="viridis")
    if annotate_factors:
        for i in range(len(labels)):
            if np.isfinite(x[i]) and np.isfinite(y[i]):
                ax.text(
                    x[i],
                    y[i],
                    str(labels[i]),
                    fontsize=annotation_fontsize,
                    alpha=annotation_alpha,
                )
    # Draw blue circle around dots that pass filters
    if len(factors_pass) > 0:
        ax.scatter(
            x[factors_pass],
            y[factors_pass],
            s=80,
            facecolors="none",
            edgecolors=plt.rcParams["axes.prop_cycle"].by_key()["color"][
                0
            ],  # default blue
            marker="o",
            label="Keep",
        )
    ax.set_xlabel("BRD")
    ax.set_ylabel(y_label)
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
    # Add ARD colorbar with threshold line
    cbar = plt.colorbar(im, ax=ax, label="ARD")
    ard_thresh = ard_min * ard_total
    # ARD color range
    norm = im.norm
    cbar_min, cbar_max = norm.vmin, norm.vmax
    if cbar_min == cbar_max:
        cbar_min, cbar_max = np.min(z), np.max(z)
    # Only draw threshold line if in colorbar range
    if cbar_min < ard_thresh < cbar_max:
        cb_ax = cbar.ax
        rel_pos = (ard_thresh - cbar_min) / (cbar_max - cbar_min)
        cb_ax.axhline(
            rel_pos,
            color=plt.rcParams["axes.prop_cycle"].by_key()["color"][0],
            linestyle="--",
            linewidth=5,
            # label='ARD threshold'
        )
    ax.set_title(f"{len(factors_pass)} factors pass filters")
    if len(factors_pass) > 0:
        ax.legend()

    if batch_purity is not None:
        ax_bp = axes[1]
        im_bp = ax_bp.scatter(batch_purity, y, c=z, cmap="viridis")
        if annotate_factors:
            for i in range(len(labels)):
                if np.isfinite(batch_purity[i]) and np.isfinite(y[i]):
                    ax_bp.text(
                        batch_purity[i],
                        y[i],
                        str(labels[i]),
                        fontsize=annotation_fontsize,
                        alpha=annotation_alpha,
                    )
        if len(factors_pass) > 0:
            ax_bp.scatter(
                batch_purity[factors_pass],
                y[factors_pass],
                s=80,
                facecolors="none",
                edgecolors=plt.rcParams["axes.prop_cycle"].by_key()["color"][0],
                marker="o",
                label="Keep",
            )
        ax_bp.set_xlabel("Batch purity")
        ax_bp.set_ylabel(y_label)
        ax_bp.set_xlim(-0.02, 1.02)
        ax_bp.axvline(
            float(batch_purity_min),
            linestyle="--",
            color=plt.rcParams["axes.prop_cycle"].by_key()["color"][0],
        )
        ax_bp.axhline(
            neffective_parents_max,
            linestyle="--",
            color=plt.rcParams["axes.prop_cycle"].by_key()["color"][0],
        )
        ax_bp.set_title("Batch purity diagnostics")
        if len(factors_pass) > 0:
            ax_bp.legend()
        plt.colorbar(im_bp, ax=ax_bp, label="ARD")

    if show:
        plt.show()
    else:
        return ax
