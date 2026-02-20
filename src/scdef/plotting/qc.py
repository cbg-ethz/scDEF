"""QC plotting functions for scDEF.

This module provides QC-related plotting functions for scDEF models.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import Optional, Tuple, Literal, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from scdef.models._scdef import scDEF


def scales(
    model: "scDEF",
    figsize: Tuple[float, float] = (8, 4),
    alpha: float = 0.6,
    fontsize: int = 12,
    legend_fontsize: int = 10,
    pretrain: bool = False,
    show: bool = True,
) -> Optional[Figure]:
    """Plot both cell and gene scales.

    Args:
        model: scDEF model instance
        figsize: figure size
        alpha: transparency level
        fontsize: font size for labels
        legend_fontsize: font size for legend
        pretrain: whether to plot pretrain data
        show: whether to show the plot

    Returns:
        Figure object if show is False, None otherwise
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    scale(model, "cell", figsize, alpha, fontsize, legend_fontsize, axes[0], False, pretrain=pretrain)
    scale(model, "gene", figsize, alpha, fontsize, legend_fontsize, axes[1], False, pretrain=pretrain)
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
    pretrain: bool = False,
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
            if pretrain:
                return model.pretrain_cell_scales.ravel()[np.where(b_cells)[0]]
            else:
                return model.pmeans["cell_scale"].ravel()[np.where(b_cells)[0]]

    else:
        x_data = np.sum(model.X, axis=0)
        x_label = "Observed gene scale"

        def get_x_data_batch(b_cells):
            return np.sum(model.X[b_cells], axis=0)

        def get_y_data_batch(b_id, _):
            if pretrain:
                return model.pretrain_gene_scales[b_id].ravel()
            else:
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
    pretrain: bool = False,
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
        pretrain: whether to plot pretrain data
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

    if pretrain:
        scales = model.pretrain_brd.ravel()
        if mode == "ard":
            scales = model.pretrain_ard.ravel()
    else:
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
        if pretrain:
            above = model.pretrain_factors
        else:
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
    pretrain: bool = False,
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
        pretrain: whether to plot pretrain data
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

    if pretrain:
        brds = model.pretrain_brd.ravel()
    else:
        brds = model.pmeans["brd"].ravel()
    if normalize:
        brds = brds - np.min(brds)
        brds = brds / np.max(brds)
    if pretrain:
        ginis = np.array(
            [score_utils.gini(model.pretrain_w[k]) for k in range(len(brds))]
        )
        is_kept = np.zeros((len(model.pretrain_brd)))
        is_kept[model.pretrain_factors] = 1
    else:
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
    pretrain: bool = False,
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
    if pretrain:
        ax.plot(np.concatenate(model.pretrain_elbos)[:])
    else:
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
    pretrain: bool = False,
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
    if pretrain:
        x = model.pretrain_ard.ravel()
        y = model.pretrain_brd.ravel()
    else:
        x = model.pmeans["factor_concentrations"].ravel()
        y = model.pmeans["factor_means"].ravel()
    if pretrain:
        is_kept = np.zeros((len(model.pretrain_brd)))
        is_kept[model.pretrain_factors] = 1
    else:
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
    pretrain: bool = False,
) -> Optional[Figure]:
    """Plot QC metrics for scDEF run.

    Plots include: loss over epochs, BRD vs Gini coefficient, learned vs observed
    cell scales, learned vs observed gene scales, and biological relevance determination.

    Args:
        model: scDEF model instance
        figsize: figure size in inches
        show: whether to show the plot
        pretrain: whether to plot pretrain data
    Returns:
        Figure object if show is False, None otherwise
    """

    if model.use_brd:
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(4, 2)
        # First row
        loss(model, pretrain=pretrain, ax=fig.add_subplot(gs[0, 0]), show=False)
        gini_brd(
            model, pretrain=pretrain, ax=fig.add_subplot(gs[0, 1]), show=False
        )
        # Second row
        scale(
            model, "cell", pretrain=pretrain, ax=fig.add_subplot(gs[1, 0]), show=False
        )
        scale(
            model, "gene", pretrain=pretrain, ax=fig.add_subplot(gs[1, 1]), show=False
        )
        # Third row
        relevance(
            model,
            pretrain=pretrain,
            mode="brd",
            ax=fig.add_subplot(gs[2, 0:2]),
            show=False,
        )
        # Fourth row
        relevance(
            model,
            pretrain=pretrain,
            mode="ard",
            ax=fig.add_subplot(gs[3, 0:2]),
            show=False,
        )
    else:
        fig = plt.figure(figsize=(figsize[0], int(figsize[1] * 2 / 3)))
        gs = GridSpec(2, 2)
        # First row
        loss(model, pretrain=pretrain, ax=fig.add_subplot(gs[0, 0:2]), show=False)
        # Second row
        scale(
            model, "cell", pretrain=pretrain, ax=fig.add_subplot(gs[1, 0]), show=False
        )
        scale(
            model, "gene", pretrain=pretrain, ax=fig.add_subplot(gs[1, 1]), show=False
        )

    fig.tight_layout()
    if show:
        plt.show()
    else:
        return fig
