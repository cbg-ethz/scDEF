"""QC plotting functions for scDEF.

This module provides QC-related plotting functions for scDEF models.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import Optional, Union, Sequence, Literal, Mapping
import pandas as pd


def plot_scales(
    model, figsize=(8, 4), alpha=0.6, fontsize=12, legend_fontsize=10, show=True
):
    """Plot both cell and gene scales."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    plot_scale(model, "cell", figsize, alpha, fontsize, legend_fontsize, axes[0], False)
    plot_scale(model, "gene", figsize, alpha, fontsize, legend_fontsize, axes[1], False)
    if show:
        fig.tight_layout()
        plt.show()
    else:
        return fig


def plot_scale(
    model,
    scale_type,
    figsize=(4, 4),
    alpha=0.6,
    fontsize=12,
    legend_fontsize=10,
    ax=None,
    show=True,
):
    """Plot learned scale factors vs observed scales."""
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


def plot_brd(
    model,
    thres=None,
    iqr_mult=None,
    show_yticks=False,
    scale="linear",
    normalize=False,
    fontsize=14,
    legend_fontsize=12,
    xlabel="Factor",
    ylabel="Relevance",
    title="Biological relevance determination",
    color=False,
    show=True,
    ax=None,
    **kwargs,
):
    """Plot biological relevance determination (BRD) scores."""
    if not model.use_brd:
        raise ValueError("This model instance doesn't use the BRD prior.")

    ard = []
    if thres is not None:
        ard = thres
    else:
        ard = iqr_mult

    layer_size = model.layer_sizes[0]
    scales = model.pmeans[f"brd"].ravel()
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

    below = []
    if thres is None and iqr_mult is None:
        l = np.arange(model.layer_sizes[0])
        above = model.factor_lists[0]
        below = np.array([f for f in l if f not in above])
    else:
        plt.axhline(cutoff, color="red", ls="--")
        above = np.where(scales >= cutoff)[0]
        below = np.where(scales < cutoff)[0]

    if color:
        colors = []
        f_idx = 0
        for i in range(model.layer_sizes[0]):
            if i in model.factor_lists[0]:
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
    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_yscale(scale)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    if not color:
        ax.legend(fontsize=legend_fontsize)

    if show:
        plt.show()
    else:
        return ax


def plot_gini_brd(
    model,
    normalize=False,
    figsize=(4, 4),
    alpha=0.6,
    fontsize=12,
    legend_fontsize=10,
    show=True,
    ax=None,
):
    """Plot Gini coefficient vs BRD scores."""
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

    if show:
        plt.show()
    else:
        return ax


def plot_loss(model, figsize=(4, 4), fontsize=12, ax=None, show=True):
    """Plot training loss over epochs."""
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


def plot_qc(model, figsize=(8, 12), show=True):
    """Plot QC metrics for scDEF run:
        top left: Loss (-1xELBO [log]) vs. Epoch
        top right: Biological relevance det. (BRD) vs. Gini coefficient
        middle left: Learned cell scale vs. Observed library size
        middle right: Learned gene scale vs. Observed gene scale
        bottom:  Biological relevance determination

    Args:
        model: scDEF model instance
        figsize (tuple(float, float)): Figure size in inches
        show (bool): whether to show the plot

    Returns:
        fig object if show is False and None otherwise
    """

    if model.use_brd:
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(3, 2)
        # First row
        plot_loss(model, ax=fig.add_subplot(gs[0, 0]), show=False)
        plot_gini_brd(model, ax=fig.add_subplot(gs[0, 1]), show=False)
        # Second row
        plot_scale(model, "cell", ax=fig.add_subplot(gs[1, 0]), show=False)
        plot_scale(model, "gene", ax=fig.add_subplot(gs[1, 1]), show=False)
        # Third row
        plot_brd(model, ax=fig.add_subplot(gs[2, 0:2]), show=False)
    else:
        fig = plt.figure(figsize=(figsize[0], int(figsize[1] * 2 / 3)))
        gs = GridSpec(2, 2)
        # First row
        plot_loss(model, ax=fig.add_subplot(gs[0, 0:2]), show=False)
        # Second row
        plot_scale(model, "cell", ax=fig.add_subplot(gs[1, 0]), show=False)
        plot_scale(model, "gene", ax=fig.add_subplot(gs[1, 1]), show=False)

    fig.tight_layout()
    if show:
        plt.show()
    else:
        return fig
