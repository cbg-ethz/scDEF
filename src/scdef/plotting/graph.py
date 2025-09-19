import numpy as np
import pandas as pd
import matplotlib
from graphviz import Graph
from typing import Optional
from ..utils import hierarchy_utils


def make_graph(
    model,
    hierarchy: Optional[dict] = None,
    show_all: Optional[bool] = False,
    factor_annotations: Optional[dict] = None,
    top_factor: Optional[str] = None,
    show_signatures: Optional[bool] = True,
    enrichments: Optional[pd.DataFrame] = None,
    top_genes: Optional[int] = None,
    show_batch_counts: Optional[bool] = False,
    filled: Optional[str] = None,
    wedged: Optional[str] = None,
    color_edges: Optional[bool] = True,
    show_confidences: Optional[bool] = False,
    mc_samples: Optional[int] = 100,
    n_cells_label: Optional[bool] = False,
    n_cells: Optional[bool] = False,
    node_size_max: Optional[int] = 2.0,
    node_size_min: Optional[int] = 0.05,
    scale_level: Optional[bool] = False,
    show_label: Optional[bool] = True,
    gene_score: Optional[str] = None,
    gene_cmap: Optional[str] = "viridis",
    shell: Optional[bool] = False,
    r: Optional[float] = 2.0,
    r_decay: Optional[float] = 0.8,
    **fontsize_kwargs,
):
    """Make Graphviz-formatted scDEF graph.

    Args:
        model: scDEF model instance
        hierarchy: a dictionary containing the polytree to draw instead of the whole graph
        show_all: whether to show all factors even post filtering
        factor_annotations: factor annotations to include in the node labels
        top_factor: only include factors below this factor
        show_signatures: whether to show the ranked gene signatures in the node labels
        enrichments: enrichment results from gseapy to include in the node labels
        top_genes: number of genes from each signature to be shown in the node labels
        show_batch_counts: whether to show the number of cells from each batch that attach to each factor
        filled: key from model.adata.obs to use to fill the nodes with, or dictionary of factor scores
        wedged: key from model.adata.obs to use to wedge the nodes with
        color_edges: whether to color the graph edges according to the upper factors
        show_confidences: whether to show the confidence score for each signature
        mc_samples: number of Monte Carlo samples to take from the posterior to compute signature confidences
        n_cells_label: wether to show the number of cells that attach to the factor
        n_cells: wether to scale the node sizes by the number of cells that attach to the factor
        node_size_max: maximum node size when scaled by cell numbers
        node_size_min: minimum node size when scaled by cell numbers
        scale_level: wether to scale node sizes per level instead of across all levels
        show_label: wether to show labels on nodes
        gene_score: color the nodes by the score they attribute to a gene, normalized by layer. Overrides filled and wedged
        gene_cmap: colormap to use for gene_score
        **fontsize_kwargs: keyword arguments to adjust the fontsizes according to the gene scores
    """
    if top_genes is None:
        top_genes = [10] * model.n_layers
    elif isinstance(top_genes, float):
        top_genes = [top_genes] * model.n_layers
    elif len(top_genes) != model.n_layers:
        raise IndexError("top_genes list must be of size scDEF.n_layers")

    gene_cmap = matplotlib.colormaps[gene_cmap]
    gene_scores = dict()
    if gene_score is not None:
        if gene_score not in model.adata.var_names:
            raise ValueError("gene_score must be a gene name in model.adata")
        else:
            style = "filled"
            gene_loc = np.where(model.adata.var_names == gene_score)[0][0]
            scores_dict = model.get_signatures_dict(scores=True, layer_normalize=True)[
                1
            ]
            for n in scores_dict:
                gene_scores[n] = scores_dict[n][gene_loc]
    else:
        if filled is None:
            style = None
        elif filled == "factor":
            style = "filled"
        else:
            if isinstance(filled, str):
                if filled not in model.adata.obs:
                    raise ValueError(
                        "filled must be factor or any `obs` in model.adata"
                    )
            else:
                style = "filled"

        if style is None:
            if wedged is None:
                style = None
            else:
                if wedged not in model.adata.obs:
                    raise ValueError("wedged must be any `obs` in model.adata")
                else:
                    style = "wedged"
        else:
            if wedged is not None:
                model.logger.info("Filled style takes precedence over wedged")

    hierarchy_nodes = None
    if hierarchy is not None:
        if top_factor is None:
            hierarchy_nodes = hierarchy_utils.get_nodes_from_hierarchy(hierarchy)
        else:
            flattened_hierarchy = hierarchy_utils.flatten_hierarchy(hierarchy)
            hierarchy_nodes = flattened_hierarchy[top_factor] + [top_factor]

    layer_factor_orders = []
    for layer_idx in np.arange(0, model.n_layers)[::-1]:  # Go top down
        if show_all:
            factors = np.arange(model.layer_sizes[layer_idx])
        else:
            factors = model.factor_lists[layer_idx]
        n_factors = len(factors)
        if not show_all and layer_idx < model.n_layers - 1:
            # Assign factors to upper factors to set the plotting order
            if show_all:
                mat = model.pmeans[f"{model.layer_names[layer_idx+1]}W"]
            else:
                mat = model.pmeans[f"{model.layer_names[layer_idx+1]}W"][
                    model.factor_lists[layer_idx + 1]
                ][:, model.factor_lists[layer_idx]]
            normalized_factor_weights = mat / np.sum(mat, axis=1).reshape(-1, 1)
            assignments = []
            for factor_idx in range(n_factors):
                assignments.append(np.argmax(normalized_factor_weights[:, factor_idx]))
            assignments = np.array(assignments)

            factor_order = []
            for upper_factor_idx in layer_factor_orders[-1]:
                factor_order.append(np.where(assignments == upper_factor_idx)[0])
            factor_order = np.concatenate(factor_order).astype(int)
            layer_factor_orders.append(factor_order)
        else:
            layer_factor_orders.append(np.arange(n_factors))
    layer_factor_orders = layer_factor_orders[::-1]

    def map_scores_to_fontsizes(scores, max_fontsize=11, min_fontsize=5):
        scores = scores - np.min(scores)
        scores = scores / np.max(scores)
        fontsizes = min_fontsize + scores * (max_fontsize - min_fontsize)
        return fontsizes

    g = Graph()
    ordering = "out"
    if shell:
        g.engine = "neato"
    else:
        g.engine = "dot"
    # g.node('root', style = 'invis')
    angle_dict = dict()
    for layer_idx in range(model.n_layers):
        layer_name = model.layer_names[layer_idx]
        if show_all:
            factors = np.arange(model.layer_sizes[layer_idx])
            layer_colors = []
            f_idx = 0
            for i in range(model.layer_sizes[layer_idx]):
                if i in model.factor_lists[layer_idx]:
                    layer_colors.append(model.layer_colorpalettes[layer_idx][f_idx])
                    f_idx += 1
                else:
                    layer_colors.append("grey")
        else:
            factors = model.factor_lists[layer_idx]
            layer_colors = model.layer_colorpalettes[layer_idx][: len(factors)]
        n_factors = len(factors)

        if show_signatures:
            gene_rankings, gene_scores = model.get_rankings(
                layer_idx=layer_idx,
                genes=True,
                return_scores=True,
            )

        factor_order = layer_factor_orders[layer_idx]
        for ii, factor_idx in enumerate(factor_order):
            factor_idx = int(factor_idx)
            alpha = "FF"
            color = None
            if show_all:
                factor_name = f"{model.layer_names[layer_idx]}{int(factor_idx)}"
            else:
                factor_name = f"{model.factor_names[layer_idx][int(factor_idx)]}"

            if hierarchy is not None and factor_name not in hierarchy_nodes:
                continue

            label = factor_name
            if factor_annotations is not None:
                if factor_name in factor_annotations:
                    label = factor_annotations[factor_name]

            cells = np.where(model.adata.obs[f"{layer_name}"] == factor_name)[0]
            node_num_cells = len(cells)

            if n_cells_label:
                label = f"{label}<br/>({node_num_cells} cells)"

            if color_edges:
                color = matplotlib.colors.to_hex(layer_colors[factor_idx])
            fillcolor = "#FFFFFF"
            if style == "filled":
                if filled == "factor":
                    fillcolor = matplotlib.colors.to_hex(layer_colors[factor_idx])
                elif gene_score is not None:
                    # Color by gene score
                    rgba = gene_cmap(gene_scores[factor_name])
                    fillcolor = matplotlib.colors.rgb2hex(rgba)
                elif isinstance(filled, str):
                    # cells attached to this factor
                    original_factor_index = model.factor_lists[layer_idx][factor_idx]
                    if len(cells) > 0:
                        # cells in this factor that belong to each obs
                        prevs = [
                            np.count_nonzero(model.adata.obs[filled][cells] == b)
                            / len(np.where(model.adata.obs[filled] == b)[0])
                            for b in model.adata.obs[filled].cat.categories
                        ]
                        obs_idx = np.argmax(prevs)  # obs attachment
                        label = f"{label}<br/>{model.adata.obs[filled].cat.categories[obs_idx]}"
                        alpha = prevs[obs_idx] / np.sum(
                            prevs
                        )  # confidence on obs_idx attachment -- should I account for the number of cells in each batch in total?
                        alpha = matplotlib.colors.rgb2hex(
                            (0, 0, 0, alpha), keep_alpha=True
                        )[-2:].upper()
                        fillcolor = model.adata.uns[f"{filled}_colors"][obs_idx]
                elif isinstance(filled, dict):
                    # Color by dictionary of signature values with gene_cmap
                    rgba = gene_cmap(filled[factor_name])
                    fillcolor = matplotlib.colors.rgb2hex(rgba)
                fillcolor = fillcolor + alpha
                color = fillcolor + alpha
            elif style == "wedged":
                # cells attached to this factor
                original_factor_index = model.factor_lists[layer_idx][factor_idx]
                if len(cells) > 0:
                    # cells in this factor that belong to each obs
                    # normalized by total num of cells in each obs
                    prevs = [
                        np.count_nonzero(model.adata.obs[wedged][cells] == b)
                        / len(np.where(model.adata.obs[wedged] == b)[0])
                        for b in model.adata.obs[wedged].cat.categories
                    ]
                    fracs = prevs / np.sum(prevs)
                    # make color string for pie chart
                    fillcolor = ":".join(
                        [
                            f"{model.adata.uns[f'{wedged}_colors'][obs_idx]};{frac}"
                            for obs_idx, frac in enumerate(fracs)
                        ]
                    )

            if enrichments is not None:
                label += "<br/><br/>" + "<br/>".join(
                    [
                        enrichments[factor_idx].results["Term"].values[i]
                        + f" ({enrichments[factor_idx].results['Adjusted P-value'][i]:.3f})"
                        for i in range(top_genes[layer_idx])
                    ]
                )
            elif show_signatures:

                def print_signature(i):
                    factor_gene_rankings = gene_rankings[i][: top_genes[layer_idx]]
                    factor_gene_scores = gene_scores[i][: top_genes[layer_idx]]
                    fontsizes = map_scores_to_fontsizes(
                        gene_scores[i], **fontsize_kwargs
                    )[: top_genes[layer_idx]]
                    gene_labels = []
                    for j, gene in enumerate(factor_gene_rankings):
                        gene_labels.append(
                            f'<FONT POINT-SIZE="{fontsizes[j]}">{gene}</FONT>'
                        )
                    return "<br/><br/>" + "<br/>".join(gene_labels)

                idx = factor_idx
                if show_all:
                    if factor_idx in model.factor_lists[layer_idx]:
                        idx = np.where(
                            factor_idx == np.array(model.factor_lists[layer_idx])
                        )[0][0]
                        label += print_signature(idx)
                        if show_confidences:
                            confidence_score = model.get_signature_confidence(
                                idx,
                                layer_idx,
                                top_genes=top_genes[layer_idx],
                                mc_samples=mc_samples,
                            )
                            label += f"<br/><br/>({confidence_score:.3f})"
                else:
                    label += print_signature(idx)
                    if show_confidences:
                        confidence_score = model.get_signature_confidence(
                            idx,
                            layer_idx,
                            top_genes=top_genes[layer_idx],
                            mc_samples=mc_samples,
                        )
                        label += f"<br/><br/>({confidence_score:.3f})"

            elif isinstance(filled, str) and filled != "factor":
                label += "<br/><br/>" + ""

            label = "<" + label + ">"
            size = node_size_min
            fixedsize = "false"
            if n_cells:
                max_cells = model.n_cells
                if scale_level:
                    max_cells = model.adata.obs[f"{layer_name}"].value_counts().max()
                size = np.maximum(
                    node_size_max * np.sqrt((node_num_cells / max_cells)),
                    node_size_min,
                )
                if len(model.factor_lists[layer_idx]) == 1:
                    size = node_size_min
                fixedsize = "true"
            elif show_all:
                if (
                    factor_idx not in model.factor_lists[layer_idx]
                    or len(model.factor_lists[layer_idx]) == 1
                ):
                    size = node_size_min
                    fixedsize = "true"
                    color = "gray"
                    fillcolor = "gray"
                    if len(model.factor_lists[layer_idx]) == 1:
                        label = ""

            if not show_label:
                label = ""

            if shell:
                radius = r * (layer_idx + 1) ** r_decay  # distance from root
                if layer_idx == 0:
                    angle_dict[factor_name] = (
                        ii * 2 * np.pi / len(model.factor_lists[0])
                    )
                else:
                    children_angles = [angle_dict[f] for f in hierarchy[factor_name]]
                    angle_dict[factor_name] = np.mean(children_angles)
                x = radius * np.cos(angle_dict[factor_name])
                y = radius * np.sin(angle_dict[factor_name])
                g.node(
                    factor_name,
                    label=label,
                    fillcolor=fillcolor,
                    color=color,
                    ordering=ordering,
                    style=style,
                    width=str(size),
                    height=str(size),
                    fixedsize=fixedsize,
                    pos=f"{x},{y}!",
                    pin="true",
                )
            else:
                g.node(
                    factor_name,
                    label=label,
                    fillcolor=fillcolor,
                    color=color,
                    ordering=ordering,
                    style=style,
                    width=str(size),
                    height=str(size),
                    fixedsize=fixedsize,
                )

            if not color_edges:
                color = None
            if layer_idx > 0:
                if hierarchy is not None:
                    if factor_name in hierarchy:
                        lower_factor_names = hierarchy[factor_name]
                        mat = np.array(
                            [
                                model.compute_weight(factor_name, lower_factor_name)
                                for lower_factor_name in lower_factor_names
                            ]
                        )
                        normalized_factor_weights = mat / np.sum(mat)
                        for lower_factor_idx, lower_factor_name in enumerate(
                            lower_factor_names
                        ):
                            normalized_weight = normalized_factor_weights[
                                lower_factor_idx
                            ]
                            g.edge(
                                factor_name,
                                lower_factor_name,
                                penwidth=str(4 * normalized_weight),
                                color=color,
                            )
                else:
                    if show_all:
                        mat = model.pmeans[f"{model.layer_names[layer_idx]}W"]
                    else:
                        mat = model.pmeans[f"{model.layer_names[layer_idx]}W"][
                            model.factor_lists[layer_idx]
                        ][:, model.factor_lists[layer_idx - 1]]
                    normalized_factor_weights = mat / np.sum(mat, axis=1).reshape(-1, 1)
                    for lower_factor_idx in layer_factor_orders[layer_idx - 1]:
                        if show_all:
                            lower_factor_name = f"{model.layer_names[layer_idx-1]}{int(lower_factor_idx)}"
                        else:
                            lower_factor_name = model.factor_names[layer_idx - 1][
                                lower_factor_idx
                            ]

                        normalized_weight = normalized_factor_weights[
                            factor_idx, lower_factor_idx
                        ]

                        if factor_idx not in model.factor_lists[layer_idx] or (
                            len(model.factor_lists[layer_idx]) == 1 and show_all
                        ):
                            normalized_weight = normalized_weight / 5.0

                        g.edge(
                            factor_name,
                            lower_factor_name,
                            penwidth=str(4 * normalized_weight),
                            color=color,
                        )
    model.graph = g

    model.logger.info(f"Updated scDEF graph")

    return g
