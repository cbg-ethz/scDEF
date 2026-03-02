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

    scales = model.pmeans[f"brd"].ravel()
    if mode == "ard":
        scales = model.pmeans[f"ard"].ravel()
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
        fig = ax.get_figure()

    layer_size = len(scales)
    below = []
    if thres is None and iqr_mult is None:
        l = np.arange(layer_size)
        above = model.factor_lists[0]
        below = np.array([f for f in l if f not in above])
    else:
        plt.axhline(cutoff, color="red", ls="--")
        above = np.where(scales >= cutoff)[0]
        below = np.where(scales < cutoff)[0]

    if color:
        colors = []
        f_idx = 0
        for i in range(layer_size):
            if i in above:
                colors.append(model.layer_colorpalettes[0][f_idx])
                f_idx += 1
            else:
                colors.append("grey")
        ax.bar(np.arange(layer_size), scales, color=colors)
    else:
        ax.bar(np.arange(layer_size)[above], scales[above], label="Kept")
        if len(below) > 0:
            ax.bar(
                np.arange(layer_size)[below],
                scales[below],
                alpha=0.6,
                color="gray",
                label="Removed",
            )
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
    if not color:
        ax.legend(fontsize=legend_fontsize)

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
    is_kept = np.zeros((model.layer_sizes[0]))
    is_kept[model.factor_lists[0]] = 1

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    for c in [1, 0]:
        label = "Kept" if c == 1 else "Removed"
        color = "C0" if c == 1 else "gray"
        _alpha = 1 if c == 1 else alpha
        ax.scatter(
            ginis[np.where(is_kept == c)[0]],
            brds[np.where(is_kept == c)[0]],
            label=label,
            color=color,
            alpha=_alpha,
        )
    ax.set_xlabel("Gini index", fontsize=fontsize)
    ax.set_ylabel("BRD posterior mean", fontsize=fontsize)
    ax.legend(fontsize=legend_fontsize)
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
    ax.plot(np.concatenate(model.elbos)[:])
    ax.set_xlabel("Epoch", fontsize=fontsize)
    ax.set_yscale("log")
    ax.set_ylabel("Loss [log]", fontsize=fontsize)

    if show:
        plt.show()
    else:
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
        fig = ax.get_figure()
    x = model.pmeans["factor_concentrations"].ravel()
    y = model.pmeans["factor_means"].ravel()
    is_kept = np.zeros((model.layer_sizes[0]))
    is_kept[model.factor_lists[0]] = 1
    ax.scatter(x[np.where(is_kept == 1)[0]], y[np.where(is_kept == 1)[0]], c="C0")
    ax.scatter(x[np.where(is_kept == 0)[0]], y[np.where(is_kept == 0)[0]], c="gray")
    ax.legend(["Kept", "Removed"], fontsize=legend_fontsize)
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

    Args:
        model: scDEF model instance
        figsize: figure size in inches
        show: whether to show the plot
    Returns:
        Figure object if show is False, None otherwise
    """

    if model.use_brd:
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
    figsize: tuple = (6, 4),
    ax: Optional[Axes] = None,
    annotate_factors: bool = False,
    annotation_fontsize: int = 8,
    annotation_alpha: float = 0.8,
    show: bool = True,
) -> Optional[Axes]:
    """
    Diagnostic scatter plot of factors: BRD vs Effective parents colored by ARD.

    Args:
        model: scDEF model instance
        brd_min: minimum BRD filter threshold
        ard_min: minimum ARD filter threshold (fraction of total ARD)
        clarity: clarity threshold for effective parents calculation
        figsize: Figure size (if ax is None)
        ax: matplotlib Axes to plot on
        annotate_factors: whether to annotate each point with its factor label
        annotation_fontsize: fontsize for factor text annotations
        annotation_alpha: alpha value for factor text annotations
        show: whether to show the plot

    Returns:
        Axes object if show is False, None otherwise.
    """
    if "factor_obs" not in model.adata.uns:
        raise KeyError(
            "model.adata.uns['factor_obs'] not found. Run scdef.tools.factor_diagnostics(model) first."
        )

    factor_obs = model.adata.uns["factor_obs"]
    l0_name = model.layer_names[0]
    if "child_layer" in factor_obs.columns:
        factor_obs_l0 = factor_obs[factor_obs["child_layer"] == l0_name].copy()
    else:
        factor_obs_l0 = factor_obs.copy()
        l0_prefix = f"{l0_name}_"
        factor_obs_l0 = factor_obs_l0[
            [isinstance(idx, str) and idx.startswith(l0_prefix) for idx in factor_obs_l0.index]
        ]

    if "original_factor_idx" in factor_obs_l0.columns:
        factor_obs_l0 = factor_obs_l0.sort_values("original_factor_idx")

    labels = factor_obs_l0.index.to_numpy()
    x = factor_obs_l0["BRD"].to_numpy(dtype=float)
    y = factor_obs_l0["n_eff_parents"].to_numpy(dtype=float)
    z = factor_obs_l0["ARD"].to_numpy(dtype=float)
    k_parents = factor_obs_l0["K_parents"].to_numpy(dtype=float)
    finite_k = k_parents[np.isfinite(k_parents)]
    if len(finite_k) == 0:
        raise ValueError("No valid K_parents values found in factor_obs for layer 0.")
    k_for_threshold = int(finite_k[0])
    neffective_parents_max = float(
        effective_parents_from_clarity(clarity_min, k_for_threshold)
    )

    ard_total = np.nansum(z)
    factors_pass = np.where(
        (x > brd_min) & (y < neffective_parents_max) & (z > ard_min * ard_total)
    )[0]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
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
    ax.set_ylabel("Effective number of parents")
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
    plt.legend()
    if show:
        plt.show()
    else:
        return ax
