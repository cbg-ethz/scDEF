import numpy as np
import pandas as pd
import matplotlib
from graphviz import Graph
from typing import Optional, Dict, List, Sequence, Union, Any, TYPE_CHECKING
from ..tools import get_technical_signature
from ..utils import hierarchy_utils

if TYPE_CHECKING:
    from scdef.models._scdef import scDEF


def _validate_top_genes(top_genes, n_layers):
    """Validate and normalize top_genes parameter."""
    if top_genes is None:
        return [10] * n_layers
    elif isinstance(top_genes, (int, float)):
        return [int(top_genes)] * n_layers
    elif len(top_genes) != n_layers:
        raise IndexError("top_genes list must be of size scDEF.n_layers")
    return top_genes


def _determine_style(filled, wedged, model):
    """Determine the node style (filled/wedged/None) based on inputs."""
    if filled is None:
        if wedged is None:
            return None
        else:
            if wedged not in model.adata.obs:
                raise ValueError("wedged must be any `obs` in model.adata")
            return "wedged"
    elif filled == "factor":
        if wedged is not None:
            model.logger.info("Filled style takes precedence over wedged")
        return "filled"
    else:
        if isinstance(filled, str):
            if filled not in model.adata.obs:
                raise ValueError("filled must be factor or any `obs` in model.adata")
        if wedged is not None:
            model.logger.info("Filled style takes precedence over wedged")
        return "filled"


def _get_gene_scores(model, gene_score, drop_factors):
    """Get gene scores if gene_score is provided."""
    if gene_score is None:
        return {}

    if gene_score not in model.adata.var_names:
        raise ValueError("gene_score must be a gene name in model.adata")

    gene_loc = np.where(model.adata.var_names == gene_score)[0][0]
    scores_dict = model.get_signatures_dict(
        scores=True, layer_normalize=True, drop_factors=drop_factors
    )[1]

    return {n: scores_dict[n][gene_loc] for n in scores_dict}


def _get_hierarchy_nodes(hierarchy, top_factor):
    """Get hierarchy nodes based on hierarchy and top_factor."""
    if hierarchy is None:
        return None

    if top_factor is None:
        return hierarchy_utils.get_nodes_from_hierarchy(hierarchy)
    else:
        flattened_hierarchy = hierarchy_utils.flatten_hierarchy(hierarchy)
        return flattened_hierarchy[top_factor] + [top_factor]


def _compute_layer_factor_orders(model, show_all):
    """Compute the order of factors in each layer."""
    layer_factor_orders = []
    for layer_idx in np.arange(0, model.n_layers)[::-1]:  # Go top down
        if show_all:
            factors = np.arange(model.layer_sizes[layer_idx])
        else:
            factors = model.factor_lists[layer_idx]
        n_factors = len(factors)

        if not show_all and layer_idx < model.n_layers - 1:
            # Assign factors to upper factors to set the plotting order
            mat = model.pmeans[f"{model.layer_names[layer_idx+1]}W"][
                model.factor_lists[layer_idx + 1]
            ][:, model.factor_lists[layer_idx]]
            normalized_factor_weights = mat / np.sum(mat, axis=1).reshape(-1, 1)
            assignments = np.array(
                [
                    np.argmax(normalized_factor_weights[:, factor_idx])
                    for factor_idx in range(n_factors)
                ]
            )

            factor_order = []
            for upper_factor_idx in layer_factor_orders[-1]:
                factor_order.append(np.where(assignments == upper_factor_idx)[0])
            factor_order = np.concatenate(factor_order).astype(int)
            layer_factor_orders.append(factor_order)
        else:
            layer_factor_orders.append(np.arange(n_factors))

    return layer_factor_orders[::-1]


def _get_layer_colors(model, layer_idx, show_all, factors):
    """Get colors for a layer."""
    if show_all:
        layer_colors = []
        f_idx = 0
        for i in range(model.layer_sizes[layer_idx]):
            if i in model.factor_lists[layer_idx]:
                layer_colors.append(model.layer_colorpalettes[layer_idx][f_idx])
                f_idx += 1
            else:
                layer_colors.append("grey")
        return layer_colors
    else:
        return model.layer_colorpalettes[layer_idx][: len(factors)]


def _map_scores_to_fontsizes(
    scores, max_fontsize=11, min_fontsize=5, **fontsize_kwargs
):
    """Map scores to font sizes."""
    max_fontsize = fontsize_kwargs.get("max_fontsize", max_fontsize)
    min_fontsize = fontsize_kwargs.get("min_fontsize", min_fontsize)
    scores = scores - np.min(scores)
    scores = scores / np.max(scores) if np.max(scores) > 0 else scores
    return min_fontsize + scores * (max_fontsize - min_fontsize)


def _compute_filled_fillcolor(
    model,
    layer_idx,
    factor_idx,
    factor_name,
    filled,
    gene_score,
    gene_scores,
    gene_cmap,
    cells,
    layer_colors,
    label,
):
    """Compute fillcolor for filled style nodes."""
    alpha = "FF"
    fillcolor = "#FFFFFF"

    if filled == "factor":
        fillcolor = matplotlib.colors.to_hex(layer_colors[factor_idx])
    elif gene_score is not None:
        # Color by gene score
        rgba = gene_cmap(gene_scores[factor_name])
        fillcolor = matplotlib.colors.rgb2hex(rgba)
    elif isinstance(filled, str):
        # cells attached to this factor
        if len(cells) > 0:
            # cells in this factor that belong to each obs
            prevs = [
                np.count_nonzero(model.adata.obs[filled][cells] == b)
                / len(np.where(model.adata.obs[filled] == b)[0])
                for b in model.adata.obs[filled].cat.categories
            ]
            obs_idx = np.argmax(prevs)  # obs attachment
            label = f"{label}<br/>{model.adata.obs[filled].cat.categories[obs_idx]}"
            alpha = prevs[obs_idx] / np.sum(prevs)
            alpha = matplotlib.colors.rgb2hex((0, 0, 0, alpha), keep_alpha=True)[
                -2:
            ].upper()
            fillcolor = model.adata.uns[f"{filled}_colors"][obs_idx]
    elif isinstance(filled, dict):
        # Color by dictionary of signature values with gene_cmap
        rgba = gene_cmap(filled[factor_name])
        fillcolor = matplotlib.colors.rgb2hex(rgba)

    fillcolor = fillcolor + alpha
    return fillcolor, alpha, label


def _compute_wedged_fillcolor(model, wedged, cells):
    """Compute wedged fillcolor for pie chart style."""
    if len(cells) == 0:
        return "#FFFFFF"

    # cells in this factor that belong to each obs
    # normalized by total num of cells in each obs
    prevs = [
        np.count_nonzero(model.adata.obs[wedged][cells] == b)
        / len(np.where(model.adata.obs[wedged] == b)[0])
        for b in model.adata.obs[wedged].cat.categories
    ]
    fracs = prevs / np.sum(prevs)
    # make color string for pie chart
    return ":".join(
        [
            f"{model.adata.uns[f'{wedged}_colors'][obs_idx]};{frac}"
            for obs_idx, frac in enumerate(fracs)
        ]
    )


def _compute_wedged_fillcolor_with_scores(model, wedged, factor_idx):
    scores = model.adata.obs[f"{model.layer_names[0]}_{factor_idx}_score"].values
    prevs = []
    for b in model.adata.obs[wedged].cat.categories:
        group_mask = (model.adata.obs[wedged] == b).values
        score = np.mean(scores[group_mask]) if len(group_mask) > 0 else 0.0
        prevs.append(score)
    fracs = prevs / np.sum(prevs)
    return ":".join(
        [
            f"{model.adata.uns[f'{wedged}_colors'][obs_idx]};{frac}"
            for obs_idx, frac in enumerate(fracs)
        ]
    )


def _get_base_label(factor_name, factor_annotations, n_cells_label, node_num_cells):
    """Get base label for a node."""
    label = factor_name
    if factor_annotations is not None and factor_name in factor_annotations:
        label = factor_annotations[factor_name]

    if n_cells_label:
        label = f"{label}<br/>({node_num_cells} cells)"

    return label


def _add_enrichments_to_label(label, enrichments, factor_idx, top_genes_layer):
    """Add enrichment information to label."""
    label += "<br/><br/>" + "<br/>".join(
        [
            enrichments[factor_idx].results["Term"].values[i]
            + f" ({enrichments[factor_idx].results['Adjusted P-value'][i]:.3f})"
            for i in range(top_genes_layer)
        ]
    )
    return label


def _add_signature_to_label(
    label,
    model,
    layer_idx,
    factor_idx,
    show_all,
    gene_rankings,
    gene_scores_layer,
    top_genes_layer,
    show_confidences,
    mc_samples,
    fontsize_kwargs,
):
    """Add signature information to label."""

    def print_signature(i):
        factor_gene_rankings = gene_rankings[i][:top_genes_layer]
        factor_gene_scores = gene_scores_layer[i][:top_genes_layer]
        fontsizes = _map_scores_to_fontsizes(gene_scores_layer[i], **fontsize_kwargs)[
            :top_genes_layer
        ]
        gene_labels = [
            f'<FONT POINT-SIZE="{fontsizes[j]}">{gene}</FONT>'
            for j, gene in enumerate(factor_gene_rankings)
        ]
        return "<br/><br/>" + "<br/>".join(gene_labels)

    idx = factor_idx
    if show_all:
        if factor_idx in model.factor_lists[layer_idx]:
            idx = np.where(factor_idx == np.array(model.factor_lists[layer_idx]))[0][0]
            label += print_signature(idx)
            if show_confidences:
                confidence_score = model.get_signature_confidence(
                    idx, layer_idx, top_genes=top_genes_layer, mc_samples=mc_samples
                )
                label += f"<br/><br/>({confidence_score:.3f})"
    else:
        label += print_signature(idx)
        if show_confidences:
            confidence_score = model.get_signature_confidence(
                idx, layer_idx, top_genes=top_genes_layer, mc_samples=mc_samples
            )
            label += f"<br/><br/>({confidence_score:.3f})"

    return label


def _compute_node_size(
    model,
    layer_idx,
    factor_idx,
    node_num_cells,
    n_cells,
    node_size_max,
    node_size_min,
    scale_level,
    show_all,
    layer_name,
):
    """Compute node size and fixedsize flag."""
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

    return size, fixedsize


def _compute_shell_position(
    layer_idx, ii, factor_name, hierarchy, r, r_decay, angle_dict, model
):
    """Compute position for shell layout."""
    radius = r * (layer_idx + 1) ** r_decay  # distance from root
    if layer_idx == 0:
        angle_dict[factor_name] = ii * 2 * np.pi / len(model.factor_lists[0])
    else:
        children_angles = [angle_dict[f] for f in hierarchy.get(factor_name, [])]
        angle_dict[factor_name] = np.mean(children_angles) if children_angles else 0
    x = radius * np.cos(angle_dict[factor_name])
    y = radius * np.sin(angle_dict[factor_name])
    return f"{x},{y}!", angle_dict


def _add_node_to_graph(
    g,
    factor_name,
    label,
    fillcolor,
    color,
    style,
    size,
    fixedsize,
    shell,
    pos,
    ordering,
):
    """Add a node to the graph."""
    node_kwargs = {
        "label": label,
        "fillcolor": fillcolor,
        "color": color,
        "ordering": ordering,
        "style": style,
        "width": str(size),
        "height": str(size),
        "fixedsize": fixedsize,
    }

    if shell:
        node_kwargs["pos"] = pos
        node_kwargs["pin"] = "true"

    g.node(factor_name, **node_kwargs)


def _add_edges_from_hierarchy(g, model, factor_name, hierarchy, color):
    """Add edges from hierarchy."""
    if factor_name not in hierarchy:
        return

    lower_factor_names = hierarchy[factor_name]
    mat = np.array(
        [
            model.compute_weight(factor_name, lower_factor_name)
            for lower_factor_name in lower_factor_names
        ]
    )
    normalized_factor_weights = mat / np.sum(mat)

    for lower_factor_idx, lower_factor_name in enumerate(lower_factor_names):
        normalized_weight = normalized_factor_weights[lower_factor_idx]
        g.edge(
            factor_name,
            lower_factor_name,
            penwidth=str(4 * normalized_weight),
            color=color,
        )


def _add_technical_edges(g, model, hierarchy, color, factor_name="tech_top"):
    """Add edges from hierarchy."""
    if factor_name not in hierarchy:
        return

    lower_factor_names = hierarchy[factor_name]
    relevances = model.get_relevances_dict()
    mat = np.array(
        [relevances[lower_factor_name] for lower_factor_name in lower_factor_names]
    )
    normalized_factor_weights = mat / np.sum(mat)

    for lower_factor_idx, lower_factor_name in enumerate(lower_factor_names):
        normalized_weight = normalized_factor_weights[lower_factor_idx]
        g.edge(
            factor_name,
            lower_factor_name,
            penwidth=str(4 * normalized_weight),
            color=color,
        )


def _add_edges_from_weights(
    g, model, factor_name, layer_idx, factor_idx, color, layer_factor_orders, show_all
):
    """Add edges from weight matrices."""
    if show_all:
        mat = model.pmeans[f"{model.layer_names[layer_idx]}W"]
    else:
        mat = model.pmeans[f"{model.layer_names[layer_idx]}W"][
            model.factor_lists[layer_idx]
        ][:, model.factor_lists[layer_idx - 1]]

    normalized_factor_weights = mat / np.sum(mat, axis=1).reshape(-1, 1)

    for lower_factor_idx in layer_factor_orders[layer_idx - 1]:
        if show_all:
            lower_factor_name = (
                f"{model.layer_names[layer_idx-1]}{int(lower_factor_idx)}"
            )
        else:
            lower_factor_name = model.factor_names[layer_idx - 1][lower_factor_idx]

        normalized_weight = normalized_factor_weights[factor_idx, lower_factor_idx]

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


def make_graph(
    model: "scDEF",
    hierarchy: Optional[Dict[str, Sequence[str]]] = None,
    show_all: Optional[bool] = False,
    factor_annotations: Optional[Dict[str, str]] = None,
    top_factor: Optional[str] = None,
    show_signatures: Optional[bool] = True,
    drop_factors: Optional[List[str]] = None,
    root_signature: Optional[List[str]] = None,
    root_ranking: Optional[List[str]] = None,
    enrichments: Optional[pd.DataFrame] = None,
    top_genes: Optional[Union[int, List[int]]] = None,
    show_batch_counts: Optional[bool] = False,
    filled: Optional[Union[str, Dict[str, float]]] = None,
    wedged: Optional[str] = None,
    color_edges: Optional[bool] = True,
    show_confidences: Optional[bool] = False,
    mc_samples: Optional[int] = 100,
    n_cells_label: Optional[bool] = False,
    n_cells: Optional[bool] = False,
    node_size_max: Optional[float] = 2.0,
    node_size_min: Optional[float] = 0.05,
    scale_level: Optional[bool] = False,
    show_label: Optional[bool] = True,
    gene_score: Optional[str] = None,
    gene_cmap: Optional[str] = "viridis",
    shell: Optional[bool] = False,
    r: Optional[float] = 2.0,
    r_decay: Optional[float] = 0.8,
    **fontsize_kwargs: Any,
) -> Graph:
    """Make Graphviz-formatted scDEF graph.

    Args:
        model: scDEF model instance
        hierarchy: dictionary containing the polytree to draw instead of the whole graph
        show_all: whether to show all factors even post filtering
        factor_annotations: factor annotations to include in the node labels
        top_factor: only include factors below this factor
        show_signatures: whether to show the ranked gene signatures in the node labels
        drop_factors: list of factors to drop from the graph
        root_signature: root signature to display
        root_ranking: root ranking to display
        enrichments: enrichment results from gseapy to include in the node labels
        top_genes: number of genes from each signature to be shown in the node labels
        show_batch_counts: whether to show the number of cells from each batch that attach to each factor
        filled: key from model.adata.obs to use to fill the nodes with, or dictionary of factor scores
        wedged: key from model.adata.obs to use to wedge the nodes with
        color_edges: whether to color the graph edges according to the upper factors
        show_confidences: whether to show the confidence score for each signature
        mc_samples: number of Monte Carlo samples to take from the posterior to compute signature confidences
        n_cells_label: whether to show the number of cells that attach to the factor
        n_cells: whether to scale the node sizes by the number of cells that attach to the factor
        node_size_max: maximum node size when scaled by cell numbers
        node_size_min: minimum node size when scaled by cell numbers
        scale_level: whether to scale node sizes per level instead of across all levels
        show_label: whether to show labels on nodes
        gene_score: color the nodes by the score they attribute to a gene, normalized by layer. Overrides filled and wedged
        gene_cmap: colormap to use for gene_score
        shell: whether to use shell layout
        r: radius parameter for shell layout
        r_decay: radius decay parameter for shell layout
        **fontsize_kwargs: keyword arguments to adjust the fontsizes according to the gene scores

    Returns:
        Graphviz Graph object
    """
    # Validate and normalize inputs
    top_genes = _validate_top_genes(top_genes, model.n_layers)
    gene_cmap = matplotlib.colormaps[gene_cmap]
    gene_scores_dict = _get_gene_scores(model, gene_score, drop_factors)
    style = (
        "filled" if gene_score is not None else _determine_style(filled, wedged, model)
    )
    hierarchy_nodes = _get_hierarchy_nodes(hierarchy, top_factor)
    layer_factor_orders = _compute_layer_factor_orders(model, show_all)

    # Initialize graph
    g = Graph()
    g.engine = "neato" if shell else "dot"
    ordering = "out"
    angle_dict = {}

    # Process each layer
    for layer_idx in range(model.n_layers):
        layer_name = model.layer_names[layer_idx]

        # Get factors and colors for this layer
        if show_all:
            factors = np.arange(model.layer_sizes[layer_idx])
        else:
            factors = model.factor_lists[layer_idx]
        layer_colors = _get_layer_colors(model, layer_idx, show_all, factors)

        # Get gene rankings if showing signatures
        gene_rankings_layer = None
        gene_scores_layer = None
        if show_signatures:
            gene_rankings_layer, gene_scores_layer = model.get_rankings(
                layer_idx=layer_idx,
                genes=True,
                return_scores=True,
                drop_factors=drop_factors,
            )

        # Process each factor in the layer
        factor_order = layer_factor_orders[layer_idx]
        for ii, factor_idx in enumerate(factor_order):
            factor_idx = int(factor_idx)

            # Get factor name
            if show_all:
                factor_name = f"{model.layer_names[layer_idx]}{int(factor_idx)}"
            else:
                factor_name = f"{model.factor_names[layer_idx][int(factor_idx)]}"

            # Skip if not in hierarchy
            if hierarchy is not None and factor_name not in hierarchy_nodes:
                continue

            # Get cells attached to this factor
            cells = np.where(model.adata.obs[f"{layer_name}"] == factor_name)[0]
            node_num_cells = len(cells)

            # Build label
            label = _get_base_label(
                factor_name, factor_annotations, n_cells_label, node_num_cells
            )

            # Determine colors
            color = None
            if color_edges:
                color = matplotlib.colors.to_hex(layer_colors[factor_idx])

            fillcolor = "#FFFFFF"
            alpha = "FF"

            # Compute fillcolor based on style
            if style == "filled":
                fillcolor, alpha, label = _compute_filled_fillcolor(
                    model,
                    layer_idx,
                    factor_idx,
                    factor_name,
                    filled,
                    gene_score,
                    gene_scores_dict,
                    gene_cmap,
                    cells,
                    layer_colors,
                    label,
                )
                color = fillcolor + alpha
            elif style == "wedged":
                if assignments:
                    fillcolor = _compute_wedged_fillcolor(model, wedged, cells)
                else:
                    fillcolor = _compute_wedged_fillcolor_with_scores(
                        model, wedged, factor_idx
                    )

            # Add enrichments or signatures to label
            if enrichments is not None:
                label = _add_enrichments_to_label(
                    label, enrichments, factor_idx, top_genes[layer_idx]
                )
            elif show_signatures:
                label = _add_signature_to_label(
                    label,
                    model,
                    layer_idx,
                    factor_idx,
                    show_all,
                    gene_rankings_layer,
                    gene_scores_layer,
                    top_genes[layer_idx],
                    show_confidences,
                    mc_samples,
                    fontsize_kwargs,
                )
            elif isinstance(filled, str) and filled != "factor":
                label += "<br/><br/>"

            # Compute node size
            size, fixedsize = _compute_node_size(
                model,
                layer_idx,
                factor_idx,
                node_num_cells,
                n_cells,
                node_size_max,
                node_size_min,
                scale_level,
                show_all,
                layer_name,
            )

            # Handle special cases for show_all
            if show_all and (
                factor_idx not in model.factor_lists[layer_idx]
                or len(model.factor_lists[layer_idx]) == 1
            ):
                if len(model.factor_lists[layer_idx]) == 1:
                    label = ""
                color = "gray"
                fillcolor = "gray"

            # Finalize label
            if not show_label:
                label = ""
            label = "<" + label + ">"

            # Compute position for shell layout
            pos = None
            if shell:
                pos, angle_dict = _compute_shell_position(
                    layer_idx,
                    ii,
                    factor_name,
                    hierarchy or {},
                    r,
                    r_decay,
                    angle_dict,
                    model,
                )

            # Add node to graph
            _add_node_to_graph(
                g,
                factor_name,
                label,
                fillcolor,
                color,
                style,
                size,
                fixedsize,
                shell,
                pos,
                ordering,
            )

            # Add edges
            if not color_edges:
                color = None

            if layer_idx > 0:
                if hierarchy is not None:
                    _add_edges_from_hierarchy(g, model, factor_name, hierarchy, color)
                else:
                    _add_edges_from_weights(
                        g,
                        model,
                        factor_name,
                        layer_idx,
                        factor_idx,
                        color,
                        layer_factor_orders,
                        show_all,
                    )

    model.graph = g
    model.logger.info(f"Updated scDEF graph")
    return g


def make_technical_hierarchy_graph(
    model,
    hierarchy: Optional[dict] = None,
    factor_annotations: Optional[dict] = None,
    top_factor: Optional[str] = None,
    show_signatures: Optional[bool] = True,
    drop_factors: Optional[list] = None,
    root_gene_rankings: Optional[list] = None,
    root_gene_scores: Optional[list] = None,
    root_name: Optional[str] = "tech_top",
    enrichments: Optional[pd.DataFrame] = None,
    top_genes: Optional[int] = None,
    filled: Optional[str] = None,
    wedged: Optional[str] = None,
    assignments: Optional[bool] = False,
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
    **fontsize_kwargs: Any,
) -> Graph:
    """Make Graphviz-formatted scDEF graph for technical hierarchy.

    Args:
        model: scDEF model instance
        hierarchy: dictionary containing the polytree to draw instead of the whole graph
        factor_annotations: factor annotations to include in the node labels
        top_factor: only include factors below this factor
        show_signatures: whether to show the ranked gene signatures in the node labels
        drop_factors: list of factors to drop from the graph
        root_gene_rankings: gene rankings for the root node
        root_gene_scores: gene scores for the root node
        root_name: name of the root node
        enrichments: enrichment results from gseapy to include in the node labels
        top_genes: number of genes from each signature to be shown in the node labels
        filled: key from model.adata.obs to use to fill the nodes with, or dictionary of factor scores
        wedged: key from model.adata.obs to use to wedge the nodes with
        assignments: whether to use the assignments of cells to factors to wedge the nodes, rather than the scores
        color_edges: whether to color the graph edges according to the upper factors
        show_confidences: whether to show the confidence score for each signature
        mc_samples: number of Monte Carlo samples to take from the posterior to compute signature confidences
        n_cells_label: whether to show the number of cells that attach to the factor
        n_cells: whether to scale the node sizes by the number of cells that attach to the factor
        node_size_max: maximum node size when scaled by cell numbers
        node_size_min: minimum node size when scaled by cell numbers
        scale_level: whether to scale node sizes per level instead of across all levels
        show_label: whether to show labels on nodes
        gene_score: color the nodes by the score they attribute to a gene, normalized by layer. Overrides filled and wedged
        gene_cmap: colormap to use for gene_score
        shell: whether to use shell layout
        r: radius parameter for shell layout
        r_decay: radius decay parameter for shell layout
        **fontsize_kwargs: keyword arguments to adjust the fontsizes according to the gene scores

    Returns:
        Graphviz Graph object
    """
    # Validate and normalize inputs
    top_genes = _validate_top_genes(top_genes, model.n_layers)
    gene_cmap = matplotlib.colormaps[gene_cmap]
    gene_scores_dict = _get_gene_scores(model, gene_score, drop_factors)
    style = (
        "filled" if gene_score is not None else _determine_style(filled, wedged, model)
    )
    hierarchy_nodes = _get_hierarchy_nodes(hierarchy, top_factor)
    layer_factor_orders = _compute_layer_factor_orders(model, False)

    # Initialize graph
    g = Graph()
    g.engine = "neato" if shell else "dot"
    ordering = "out"
    angle_dict = {}

    # Process each layer
    for layer_idx in [0, model.n_layers - 1]:
        if layer_idx == model.n_layers - 1:
            layer_name = root_name
        else:
            layer_name = model.layer_names[layer_idx]

        # Get factors and colors for this layer
        factors = model.factor_lists[layer_idx]
        layer_colors = _get_layer_colors(model, layer_idx, False, factors)

        # Get gene rankings if showing signatures
        gene_rankings_layer = None
        gene_scores_layer = None
        if show_signatures:
            if layer_idx == model.n_layers - 1:
                gene_rankings_layer, gene_scores_layer = (
                    root_gene_rankings,
                    root_gene_scores,
                )
            else:
                gene_rankings_layer, gene_scores_layer = model.get_rankings(
                    layer_idx=layer_idx,
                    genes=True,
                    return_scores=True,
                    drop_factors=drop_factors,
                )

        # Process each factor in the layer
        factor_order = layer_factor_orders[layer_idx]
        for ii, factor_idx in enumerate(factor_order):
            factor_idx = int(factor_idx)

            if layer_idx == model.n_layers - 1:
                factor_name = root_name
            else:
                # Get factor name
                factor_name = f"{model.factor_names[layer_idx][int(factor_idx)]}"

            # Skip if not in hierarchy
            if hierarchy is not None and factor_name not in hierarchy_nodes:
                continue

            if layer_idx == model.n_layers - 1:
                cells = np.array([])
            else:
                # Get cells attached to this factor
                cells = np.where(model.adata.obs[f"{layer_name}"] == factor_name)[0]
            node_num_cells = len(cells)

            # Build label
            label = _get_base_label(
                factor_name, factor_annotations, n_cells_label, node_num_cells
            )

            # Determine colors
            color = None
            if color_edges:
                color = matplotlib.colors.to_hex(layer_colors[factor_idx])

            fillcolor = "#FFFFFF"
            alpha = "FF"

            # Compute fillcolor based on style
            if style == "filled":
                fillcolor, alpha, label = _compute_filled_fillcolor(
                    model,
                    layer_idx,
                    factor_idx,
                    factor_name,
                    filled,
                    gene_score,
                    gene_scores_dict,
                    gene_cmap,
                    cells,
                    layer_colors,
                    label,
                )
                color = fillcolor + alpha
            elif style == "wedged":
                if assignments:
                    fillcolor = _compute_wedged_fillcolor(model, wedged, cells)
                else:
                    fillcolor = _compute_wedged_fillcolor_with_scores(
                        model, wedged, factor_idx
                    )

            # Add enrichments or signatures to label
            if enrichments is not None:
                label = _add_enrichments_to_label(
                    label, enrichments, factor_idx, top_genes[layer_idx]
                )
            elif show_signatures:
                print(layer_idx, factor_idx, gene_rankings_layer, gene_scores_layer)
                label = _add_signature_to_label(
                    label,
                    model,
                    layer_idx,
                    factor_idx,
                    False,
                    gene_rankings_layer,
                    gene_scores_layer,
                    top_genes[layer_idx],
                    show_confidences,
                    mc_samples,
                    fontsize_kwargs,
                )
            elif isinstance(filled, str) and filled != "factor":
                label += "<br/><br/>"

            # Compute node size
            size, fixedsize = _compute_node_size(
                model,
                layer_idx,
                factor_idx,
                node_num_cells,
                n_cells,
                node_size_max,
                node_size_min,
                scale_level,
                False,
                layer_name,
            )

            # Finalize label
            if not show_label:
                label = ""
            label = "<" + label + ">"

            # Compute position for shell layout
            pos = None
            if shell:
                pos, angle_dict = _compute_shell_position(
                    layer_idx,
                    ii,
                    factor_name,
                    hierarchy or {},
                    r,
                    r_decay,
                    angle_dict,
                    model,
                )

            # Add node to graph
            _add_node_to_graph(
                g,
                factor_name,
                label,
                fillcolor,
                color,
                style,
                size,
                fixedsize,
                shell,
                pos,
                ordering,
            )

            # Add edges
            if not color_edges:
                color = None

            if layer_idx > 0:
                _add_technical_edges(g, model, hierarchy, color, factor_name=root_name)

    model.graph = g
    model.logger.info(f"Updated scDEF graph")
    return g


def plot_biological_hierarchy(model: "scDEF", **kwargs: Any) -> Graph:
    """Plot the biological hierarchy of the model.

    Args:
        model: scDEF model instance
        **kwargs: keyword arguments passed to make_graph

    Returns:
        Graphviz Graph object
    """
    # Get the top signature
    technical_factors = model.adata.uns["factor_obs"][
        model.adata.uns["factor_obs"]["technical"] == True
    ].index.tolist()
    g = make_graph(
        model,
        hierarchy=model.adata.uns["biological_hierarchy"],
        drop_factors=technical_factors,
        **kwargs,
    )
    return g


def plot_technical_hierarchy(
    model: "scDEF", show_signatures: bool = True, **kwargs: Any
) -> Graph:
    """Plot the technical hierarchy of the model.

    Args:
        model: scDEF model instance
        show_signatures: whether to show gene signatures
        **kwargs: keyword arguments passed to make_graph

    Returns:
        Graphviz Graph object
    """
    technical_signature = None
    technical_scores = None
    if show_signatures:
        technical_signature, technical_scores = get_technical_signature(
            model,
            return_scores=True,
            top_genes=None,
        )
    g = make_technical_hierarchy_graph(
        model,
        hierarchy=model.adata.uns["technical_hierarchy"],
        root_gene_rankings=[technical_signature],
        root_gene_scores=[technical_scores],
        root_name="tech_top",
        show_signatures=show_signatures,
        **kwargs,
    )
    return g
