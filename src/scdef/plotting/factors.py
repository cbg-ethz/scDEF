"""Factor plotting functions for scDEF.

This module provides factor-related plotting functions for scDEF models.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import scanpy as sc
from scipy.cluster.hierarchy import ward, leaves_list
from scipy.spatial.distance import pdist
from typing import Optional, Union, Sequence, Literal, Mapping
import pandas as pd
from ..utils import data_utils, score_utils, hierarchy_utils


def plot_obs_factor_dotplot(
    model,
    obs_key,
    layer_idx,
    cluster_rows=True,
    cluster_cols=True,
    figsize=(8, 2),
    s_min=100,
    s_max=500,
    titlesize=12,
    labelsize=12,
    legend_fontsize=12,
    legend_titlesize=12,
    cmap="viridis",
    logged=False,
    width_ratios=[5, 1, 1],
    show_ylabel=True,
    show=True,
):
    """Plot dotplot showing factor assignments for observations."""
    # For each obs, compute the average cell score on each factor among the cells that attach to that obs, use as color
    # And compute the fraction of cells in the obs that attach to each factor, use as circle size
    layer_name = model.layer_names[layer_idx]

    obs = model.adata.obs[obs_key].unique()
    n_obs = len(obs)
    n_factors = len(model.factor_lists[layer_idx])

    df_rows = []
    c = np.zeros((n_obs, n_factors))
    s = np.zeros((n_obs, n_factors))
    for i, obs_val in enumerate(obs):
        cells_from_obs = model.adata.obs.index[
            np.where(model.adata.obs[obs_key] == obs_val)[0]
        ]
        n_cells_obs = len(cells_from_obs)
        for factor in range(n_factors):
            factor_name = model.factor_names[layer_idx][factor]
            cells_attached = model.adata.obs.index[
                np.where(
                    model.adata.obs.loc[cells_from_obs][f"{layer_name}"] == factor_name
                )[0]
            ]
            if len(cells_attached) == 0:
                average_weight = 0  # np.nan
                fraction_attached = 0  # np.nan
            else:
                average_weight = np.mean(
                    model.adata.obs.loc[cells_from_obs][f"{factor_name}_score"]
                )
                fraction_attached = len(cells_attached) / n_cells_obs
            c[i, factor] = average_weight
            s[i, factor] = fraction_attached

    ylabels = obs
    xlabels = model.factor_names[layer_idx]

    if cluster_rows:
        Z = ward(pdist(s))
        hclust_index = leaves_list(Z)
        s = s[hclust_index]
        c = c[hclust_index]
        ylabels = ylabels[hclust_index]

    if cluster_cols:
        Z = ward(pdist(s.T))
        hclust_index = leaves_list(Z)
        s = s[:, hclust_index]
        c = c[:, hclust_index]
        xlabels = xlabels[hclust_index]

    x, y = np.meshgrid(np.arange(len(xlabels)), np.arange(len(ylabels)))

    fig, axes = plt.subplots(1, 3, figsize=figsize, width_ratios=width_ratios)
    plt.sca(axes[0])
    ax = plt.gca()
    s = s / np.max(s)
    s *= s_max
    s += s_max / s_min
    if logged:
        c = np.log(c)
    plt.scatter(x, y, c=c, s=s, cmap=cmap)

    ax.set(
        xticks=np.arange(n_factors),
        yticks=np.arange(n_obs),
        xticklabels=xlabels,
        yticklabels=ylabels,
    )
    ax.tick_params(axis="both", which="major", labelsize=labelsize)
    ax.set_xticks(np.arange(n_factors + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_obs + 1) - 0.5, minor=True)
    ax.grid(which="minor")

    if show_ylabel:
        ax.set_ylabel(obs_key, rotation=270, labelpad=20.0, fontsize=labelsize)
    plt.xlabel("Factor", fontsize=labelsize)
    plt.title(f"Layer {layer_idx}\n", fontsize=titlesize)

    # Make legend
    map_f_to_s = {"0": 5, "25": 7, "50": 9, "75": 11, "100": 13}
    plt.sca(axes[1])
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    circles = [
        Line2D(
            [],
            [],
            color="white",
            marker="o",
            markersize=map_f_to_s[f],
            markerfacecolor="gray",
        )
        for f in map_f_to_s.keys()
    ]
    lg = plt.legend(
        circles[::-1],
        list(map_f_to_s.keys())[::-1],
        numpoints=1,
        loc=2,
        frameon=False,
        fontsize=legend_fontsize,
    )
    plt.title("Fraction of \ncells in group (%)", fontsize=legend_titlesize)

    plt.sca(axes[2])
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    cb = plt.colorbar(ax=axes[0], cmap=cmap)
    cb.ax.set_title("Average\ncell score", fontsize=legend_titlesize)
    cb.ax.tick_params(labelsize=legend_fontsize)
    plt.grid("off")
    if show:
        plt.show()


def plot_multilevel_paga(
    model,
    neighbors_rep: Optional[str] = "X_L0",
    layers: Optional[list] = None,
    figsize: Optional[tuple] = (16, 4),
    reuse_pos: Optional[bool] = True,
    fontsize: Optional[int] = 12,
    show: Optional[bool] = True,
    **paga_kwargs,
):
    """Plot a PAGA graph from each scDEF layer.

    Args:
        model: scDEF model instance
        neighbors_rep: the model.obsm key to use to compute the PAGA graphs
        layers: which layers to plot
        figsize: figure size
        reuse_pos: whether to initialize each PAGA graph with the graph from the layer above
        show: whether to show the plot
        **paga_kwargs: keyword arguments to adjust the PAGA layouts
    """

    if layers is None:
        layers = [
            i
            for i in range(model.n_layers - 1, -1, -1)
            if len(model.factor_lists[i]) > 1
        ]

    if len(layers) == 0:
        print("Cannot run PAGA on 0 layers.")
        return

    n_layers = len(layers)

    fig, axes = plt.subplots(1, n_layers, figsize=figsize)
    sc.pp.neighbors(model.adata, use_rep=neighbors_rep)
    pos = None
    for i, layer_idx in enumerate(layers):
        ax = axes[i]
        new_layer_name = f"{model.layer_names[layer_idx]}"

        print(f"Computing PAGA graph of layer {layer_idx}")

        # Use previous PAGA as initial positions for new PAGA
        if layer_idx != layers[0] and reuse_pos:
            print(
                f"Re-using PAGA positions from layer {layer_idx+1} to init {layer_idx}"
            )
            matches = sc._utils.identify_groups(
                model.adata.obs[new_layer_name], model.adata.obs[old_layer_name]
            )
            pos = []
            np.random.seed(0)
            for c in model.adata.obs[new_layer_name].cat.categories:
                pos_coarse = model.adata.uns["paga"]["pos"]  # previous PAGA
                coarse_categories = model.adata.obs[old_layer_name].cat.categories
                idx = coarse_categories.get_loc(matches[c][0])
                pos_i = pos_coarse[idx] + np.random.random(2)
                pos.append(pos_i)
            pos = np.array(pos)

        sc.tl.paga(model.adata, groups=new_layer_name)
        sc.pl.paga(
            model.adata,
            init_pos=pos,
            layout="fa",
            ax=ax,
            show=False,
            **paga_kwargs,
        )
        ax.set_title(f"Layer {layer_idx} PAGA", fontsize=fontsize)

        old_layer_name = new_layer_name
    if show:
        plt.show()


def plot_layers_obs(
    model,
    obs_keys,
    obs_mats,
    obs_clusters,
    obs_vals_dict,
    sort_layer_factors=True,
    orders=None,
    layers=None,
    vmax=None,
    vmin=None,
    cb_title="",
    cb_title_fontsize=10,
    fontsize=12,
    title_fontsize=12,
    pad=0.1,
    shrink=0.7,
    figsize=(10, 4),
    xticks_rotation=90.0,
    cmap=None,
    show=True,
    rasterized=False,
):
    """Plot observation matrices across layers."""
    if not isinstance(obs_keys, list):
        obs_keys = [obs_keys]

    if layers is None:
        layers = [i for i in range(0, model.n_layers) if len(model.factor_lists[i]) > 1]

    n_layers = len(layers)

    if sort_layer_factors:
        layer_factor_orders = model.get_layer_factor_orders()
    else:
        if orders is not None:
            layer_factor_orders = orders
        else:
            layer_factor_orders = [
                np.arange(len(model.factor_lists[i])) for i in range(model.n_layers)
            ]

    n_factors = [len(model.factor_lists[idx]) for idx in layers]
    n_obs = [len(obs_clusters[obs_key]) for obs_key in obs_keys]
    fig, axs = plt.subplots(
        len(obs_keys),
        n_layers,
        figsize=figsize,
        gridspec_kw={"width_ratios": n_factors, "height_ratios": n_obs},
    )
    axs = axs.reshape((len(obs_keys), n_layers))
    for i in layers:
        axs[0][i].set_title(f"Layer {i}", fontsize=title_fontsize)
        for j, obs_key in enumerate(obs_keys):
            ax = axs[j][i]
            mat = obs_mats[obs_key][i]
            mat = mat[obs_clusters[obs_key]][:, layer_factor_orders[i]]
            axplt = ax.pcolormesh(
                mat, vmax=vmax, vmin=vmin, cmap=cmap, rasterized=rasterized
            )

            if j == len(obs_keys) - 1:
                xlabels = model.factor_names[i]
                xlabels = np.array(xlabels)[layer_factor_orders[i]]
                ax.set_xticks(
                    np.arange(len(xlabels)) + 0.5,
                    xlabels,
                    rotation=xticks_rotation,
                    fontsize=fontsize,
                )
            else:
                ax.set(xticks=[])

            if i == 0:
                ylabels = np.array(obs_vals_dict[obs_keys[j]])[
                    obs_clusters[obs_keys[j]]
                ]
                ax.set_yticks(
                    np.arange(len(ylabels)) + 0.5,
                    ylabels,
                )
            else:
                ax.set(yticks=[])

            if i == n_layers - 1:
                ax.yaxis.set_label_position("right")
                ax.set_ylabel(obs_key, rotation=270, labelpad=20.0, fontsize=fontsize)

    plt.subplots_adjust(wspace=0.05)
    plt.subplots_adjust(hspace=0.05)

    cb = fig.colorbar(axplt, ax=axs.ravel().tolist(), pad=pad, shrink=shrink)
    cb.ax.set_title(cb_title, fontsize=cb_title_fontsize)
    if show:
        plt.show()


def plot_pathway_scores(
    model,
    pathways: pd.DataFrame,
    top_genes: Optional[int] = 20,
    **kwargs,
):
    """Plot the association between a set of cell annotations and a set of gene signatures.

    Args:
        model: scDEF model instance
        pathways: a pandas DataFrame containing PROGENy pathways
        top_genes: number of top genes to consider
        **kwargs: plotting keyword arguments
    """
    (
        obs_mats,
        obs_clusters,
        obs_vals_dict,
        joined_mats,
    ) = data_utils.prepare_pathway_factor_scores(
        model,
        pathways,
        top_genes=top_genes,
    )

    vmax = joined_mats.max()
    vmin = joined_mats.min()
    plot_layers_obs(
        model,
        ["Pathway"],
        obs_mats,
        obs_clusters,
        obs_vals_dict,
        vmax=vmax,
        vmin=vmin,
        **kwargs,
    )


def plot_signatures_scores(
    model,
    obs_keys: Sequence[str],
    markers: Mapping[str, Sequence[str]],
    top_genes: Optional[int] = 10,
    hierarchy: Optional[dict] = None,
    **kwargs,
):
    """Plot the association between a set of cell annotations and a set of gene signatures.

    Args:
        model: scDEF model instance
        obs_keys: the keys in model.adata.obs to use
        markers: a dictionary with keys corresponding to model.adata.obs[obs_keys] and values to gene lists
        top_genes: number of genes to consider in the score computations
        hierarchy: the polytree to restrict the associations to
        **kwargs: plotting keyword arguments
    """
    obs_mats, obs_clusters, obs_vals_dict = data_utils.prepare_obs_factor_scores(
        model,
        obs_keys,
        data_utils.get_signature_scores,
        markers=markers,
        top_genes=top_genes,
        hierarchy=hierarchy,
    )
    plot_layers_obs(model, obs_keys, obs_mats, obs_clusters, obs_vals_dict, **kwargs)


def plot_obs_scores(
    model,
    obs_keys: Sequence[str],
    hierarchy: Optional[dict] = None,
    mode: Literal["f1", "fracs", "weights"] = "fracs",
    **kwargs,
):
    """Plot the association between a set of cell annotations and factors.

    Args:
        model: scDEF model instance
        obs_keys: the keys in model.adata.obs to use
        hierarchy: the polytree to restrict the associations to
        mode: whether to compute scores based on assignments or weights
        **kwargs: plotting keyword arguments
    """
    if mode == "f1":
        f = data_utils.get_assignment_scores
    elif mode == "fracs":
        f = data_utils.get_assignment_fracs
    elif mode == "weights":
        f = data_utils.get_weight_scores
    else:
        raise ValueError("`mode` must be one of ['f1', 'fracs', 'weights']")

    obs_mats, obs_clusters, obs_vals_dict = data_utils.prepare_obs_factor_scores(
        model,
        obs_keys,
        f,
        hierarchy=hierarchy,
    )
    vmax = None
    vmin = None
    if mode == "f1" or mode == "fracs":
        vmax = 1.0
        vmin = 0.0

    plot_layers_obs(
        model,
        obs_keys,
        obs_mats,
        obs_clusters,
        obs_vals_dict,
        vmax=vmax,
        vmin=vmin,
        **kwargs,
    )


def plot_umaps(
    model,
    color=[],
    layers=None,
    figsize=(16, 4),
    fontsize=12,
    legend_fontsize=10,
    use_log=False,
    metric="euclidean",
    rasterized=True,
    n_legend_cols=1,
    show=True,
):
    """Plot UMAPs for different layers."""
    if layers is None:
        layers = [
            i
            for i in range(model.n_layers - 1, -1, -1)
            if len(model.factor_lists[i]) > 1
        ]

    n_layers = len(layers)

    if "X_umap" in model.adata.obsm:
        model.adata.obsm["X_umap_original"] = model.adata.obsm["X_umap"].copy()

    if not isinstance(color, list):
        color = [color]

    n_rows = len(color)
    if n_rows == 0:
        n_rows = 1

    fig, axes = plt.subplots(n_rows, n_layers, figsize=figsize)
    for layer in layers:
        # Compute UMAP
        model.adata.obsm[f"X_{model.layer_names[layer]}_log"] = np.log(
            model.adata.obsm[f"X_{model.layer_names[layer]}"]
        )
        if use_log:
            sc.pp.neighbors(model.adata, use_rep=f"X_{model.layer_names[layer]}_log")
        else:
            sc.pp.neighbors(
                model.adata,
                use_rep=f"X_{model.layer_names[layer]}",
                metric=metric,
            )
        sc.tl.umap(model.adata)

        for row in range(len(color)):
            if n_rows > 1:
                ax = axes[row, layer]
            else:
                ax = axes[layer]
            legend_loc = None
            if layer == n_layers - 1:
                legend_loc = "right margin"
            ax = sc.pl.umap(
                model.adata,
                color=[color[row]],
                frameon=False,
                show=False,
                ax=ax,
                legend_loc=legend_loc,
            )
            if row == 0:
                ax.set_title(f"Layer {layer}", fontsize=fontsize)
            else:
                ax.set_title("")

            if layer == n_layers - 1:
                leg = ax.legend(
                    loc="center left",
                    bbox_to_anchor=(1, 0.5),
                    frameon=False,
                    title_fontsize=legend_fontsize,
                    fontsize=legend_fontsize,
                    title=color[row],
                    ncols=n_legend_cols,
                )
                leg._legend_box.align = "left"

    # Put the original one back
    if "X_umap_original" in model.adata.obsm:
        model.adata.obsm["X_umap"] = model.adata.obsm["X_umap_original"].copy()

    if show:
        plt.show()


def plot_factors_bars(
    model,
    obs_keys,
    sort_layer_factors=True,
    orders=None,
    sharey=True,
    layers=None,
    vmax=None,
    vmin=None,
    fontsize=12,
    title_fontsize=12,
    legend_fontsize=8,
    figsize=(10, 4),
    show=True,
):
    """Plot factor scores as bar charts."""
    if not isinstance(obs_keys, list):
        obs_keys = [obs_keys]

    if layers is None:
        layers = [i for i in range(0, model.n_layers) if len(model.factor_lists[i]) > 1]

    n_layers = len(layers)

    if sort_layer_factors:
        layer_factor_orders = model.get_layer_factor_orders()
    else:
        if orders is not None:
            layer_factor_orders = orders
        else:
            layer_factor_orders = [
                np.arange(len(model.factor_lists[i])) for i in range(model.n_layers)
            ]

    n_factors = [len(model.factor_lists[idx]) for idx in layers]
    fig, axs = plt.subplots(
        len(obs_keys),
        n_layers,
        figsize=figsize,
        gridspec_kw={"width_ratios": n_factors},
        sharey=sharey,
    )
    axs = axs.reshape((len(obs_keys), n_layers))
    # Get data using the new utility functions
    obs_mats, obs_clusters, obs_vals_dict = data_utils.prepare_obs_factor_scores(
        model,
        obs_keys,
        data_utils.get_assignment_fracs,
    )

    for i in layers:
        axs[0][i].set_title(f"Layer {i}", fontsize=title_fontsize)
        for j, obs_key in enumerate(obs_keys):
            ax = axs[j][i]
            mat = obs_mats[obs_key][i]
            mat = mat[:, layer_factor_orders[i]]

            # Plot bars for each observation
            for idx, obs in enumerate(obs_vals_dict[obs_key]):
                obs_idx = np.where(model.adata.obs[obs_key].cat.categories == obs)[0][0]
                color = model.adata.uns[f"{obs_key}_colors"][obs_idx]
                y = mat[idx]
                if idx == 0:
                    ax.bar(np.arange(len(y)), y, color=color, label=obs)
                else:
                    ax.bar(
                        np.arange(len(y)),
                        y,
                        bottom=mat[:idx].sum(axis=0),
                        color=color,
                        label=obs,
                    )

            ax.set_xticks(np.arange(len(model.factor_names[i])))
            ax.set_xticklabels(
                np.array(model.factor_names[i])[layer_factor_orders[i]],
                rotation=90,
                fontsize=fontsize,
            )
            if i == 0:
                ax.set_ylabel(obs_key, fontsize=fontsize)
            if vmax is not None:
                ax.set_ylim(0, vmax)
            if vmin is not None:
                ax.set_ylim(vmin, ax.get_ylim()[1])

    plt.tight_layout()
    if show:
        plt.show()


def plot_cell_entropies(model, thres=0.9, show=True):
    """Plot cell entropies and factor numbers across layers.

    Args:
        model: scDEF model instance
        thres: Threshold for cumulative sum calculation
        show: Whether to show the plot
    """
    entropies = []
    factor_ns = []
    for layer_idx in range(4):
        layer_name = model.layer_names[layer_idx]
        nf = len(model.factor_lists[layer_idx])
        a = (
            model.adata.obsm[f"X_{layer_name}factors"]
            / np.sum(model.adata.obsm[f"X_{layer_name}factors"], axis=1)[:, None]
        )
        a_sorted = np.vstack(
            [a[i, np.argsort(a, axis=1)[:, ::-1][i]] for i in range(a.shape[0])]
        )
        a_cumsums = np.cumsum(a_sorted, axis=1)
        n_factors = np.array(
            [(np.where(a_cumsums[i] > thres)[0][0] + 1) for i in range(a.shape[0])]
        )
        factor_ns.append(n_factors)
        entropy = np.array(
            [np.sum(-np.log(a[i]) * a[i]) / np.log(nf) for i in range(a.shape[0])]
        )
        entropies.append(entropy)
    fig, axes = plt.subplots(1, 2)
    plt.sca(axes[0])
    plt.boxplot(entropies)
    plt.sca(axes[1])
    plt.boxplot(factor_ns)
    if show:
        plt.show()
    else:
        return fig


def plot_factor_genes(model, thres=0.9, show=True):
    """Plot number of genes in factors across layers.

    Args:
        model: scDEF model instance
        thres: Threshold for cumulative sum calculation
        show: Whether to show the plot
    """
    layer_kept_n_genes = []
    layer_removed_n_genes = []
    factor_n_kept_genes = dict()
    for layer_idx in range(len(model.layer_sizes)):
        layer_name = f"{model.layer_names[layer_idx]}"

        term_scores = model.pmeans[f"{model.layer_names[0]}W"]

        if layer_idx > 0:
            term_scores = model.pmeans[f"{model.layer_names[layer_idx]}W"]
            for layer in range(layer_idx - 1, 0, -1):
                lower_mat = model.pmeans[f"{model.layer_names[layer]}W"]
                term_scores = term_scores.dot(lower_mat)
            term_scores = term_scores.dot(model.pmeans[f"{model.layer_names[0]}W"])

        kept_factors_n_genes = []
        removed_factors_n_genes = []
        f_idx = 0
        for factor in range(model.layer_sizes[layer_idx]):
            vals = term_scores[factor] / np.sum(term_scores[factor])
            # sort
            vals_sorted = vals[np.argsort(vals)[::-1]]
            if factor in model.factor_lists[layer_idx]:
                kept_factors_n_genes.append(
                    np.where(np.cumsum(vals_sorted) > thres)[0][0]
                )
                factor_n_kept_genes[f"{layer_name}{f_idx}"] = kept_factors_n_genes[-1]
                f_idx += 1
            else:
                removed_factors_n_genes.append(
                    np.where(np.cumsum(vals_sorted) > thres)[0][0]
                )

        layer_kept_n_genes.append(kept_factors_n_genes)
        layer_removed_n_genes.append(removed_factors_n_genes)

    m = np.max(list(factor_n_kept_genes.values()))
    for f in factor_n_kept_genes:
        factor_n_kept_genes[f] = factor_n_kept_genes[f] / m

    fig, axes = plt.subplots(1, 4, figsize=(6, 3), sharey=True)
    for i in range(4):
        plt.sca(axes[i])
        plt.boxplot(
            [layer_kept_n_genes[i], layer_removed_n_genes[i]],
            tick_labels=["Kept", "Removed"],
        )
    plt.suptitle("Number of genes in factors")
    if show:
        plt.show()
    else:
        return fig


def plot_factor_gini(model, idx, thres=0.9, show=True):
    """Plot Gini coefficient for a specific factor.

    Args:
        model: scDEF model instance
        idx: Factor index to plot
        thres: Threshold for cumulative sum calculation
        show: Whether to show the plot
    """
    a = model.pmeans[f"{model.layer_names[0]}W"][idx]
    norm_a = a / np.sum(a)
    vals_sorted = norm_a[np.argsort(norm_a)[::-1]]
    plt.plot(np.cumsum(vals_sorted))
    plt.axvline(np.where(np.cumsum(vals_sorted) > thres)[0][0], color="gray")
    plt.title(model.pmeans["brd"][idx])
    if show:
        plt.show()
    else:
        return plt.gca()
