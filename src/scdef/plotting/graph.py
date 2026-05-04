import numpy as np
import pandas as pd
import matplotlib
from graphviz import Graph
from typing import Optional, Dict, List, Sequence, Union, Any, TYPE_CHECKING, Set, Tuple
from ..tools import get_technical_signature, get_stored_confident_signatures
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


def _all_graph_factor_names(model: "scDEF", show_all: bool) -> Set[str]:
    """All factor node names that can appear in ``make_graph``."""
    names: Set[str] = set()
    for layer_idx in range(model.n_layers):
        if show_all:
            for i in range(model.layer_sizes[layer_idx]):
                names.add(f"{model.layer_names[layer_idx]}{int(i)}")
        else:
            for fn in model.factor_names[layer_idx]:
                names.add(str(fn))
    return names


def _normalize_path_argument(path: Union[Sequence[str], Dict[str, Any]]) -> List[str]:
    """Normalize ``path`` to an ordered list of factor names."""
    if isinstance(path, dict):
        nodes = path.get("nodes")
        if not nodes:
            raise ValueError("path dict must include a non-empty 'nodes' sequence.")
        return [str(n) for n in nodes]
    return [str(n) for n in path]


def _layer_index_for_factor(
    model: "scDEF", factor_name: str, show_all: bool
) -> Optional[int]:
    """Layer index for a plotted factor name, or ``None`` if unknown."""
    if not show_all:
        for li in range(model.n_layers):
            if factor_name in model.factor_names[li]:
                return int(li)
        return None
    for li in range(model.n_layers):
        ln = str(model.layer_names[li])
        if not factor_name.startswith(ln):
            continue
        rest = factor_name[len(ln) :]
        if rest.isdigit() and int(rest) < int(model.layer_sizes[li]):
            return int(li)
    return None


def _path_edge_pairs_along_path(
    model: "scDEF",
    path_list: Sequence[str],
    hierarchy: Optional[dict],
    show_all: bool,
) -> Set[Tuple[str, str]]:
    """Map consecutive path factors to edges actually drawn in ``make_graph``.

    Weight edges are always emitted as **(coarser, finer)**: layer index
    ``k`` → ``k-1`` (``layer_idx`` decreasing toward L0). Differentiation paths
    are coarse→fine in that sense; **transition** paths may alternate
    (e.g. L0→L1→L0) — each hop must be between **adjacent** layers; we map
    ``(a, b)`` to the single drawn directed edge ``(upper, lower)``.

    When ``hierarchy`` is set, a pair is first accepted if it is a **tree**
    edge ``a → b`` or ``b → a`` in the dict; otherwise the same layer rule
    applies so transition hops still highlight on hierarchy plots when the
    hop is an adjacent-layer link in the model.
    """
    pairs: Set[Tuple[str, str]] = set()

    def _hierarchy_children(parent: str) -> List[str]:
        kids = hierarchy.get(parent, []) if hierarchy is not None else []
        if isinstance(kids, (list, tuple, set)):
            return list(kids)
        if kids is None:
            return []
        return list(kids)

    for i in range(len(path_list) - 1):
        a, b = path_list[i], path_list[i + 1]
        if hierarchy is not None:
            if b in _hierarchy_children(a):
                pairs.add((a, b))
                continue
            if a in _hierarchy_children(b):
                pairs.add((b, a))
                continue

        la = _layer_index_for_factor(model, a, show_all)
        lb = _layer_index_for_factor(model, b, show_all)
        if la is None or lb is None:
            continue
        if abs(int(la) - int(lb)) != 1:
            continue
        if la > lb:
            pairs.add((a, b))
        else:
            pairs.add((b, a))

    return pairs


def _edge_color_with_path(
    base_color: Optional[str],
    parent: str,
    child: str,
    path_edge_set: Optional[Set[Tuple[str, str]]],
    path_color: Optional[str],
) -> Optional[str]:
    if path_edge_set and path_color and (parent, child) in path_edge_set:
        return path_color
    return base_color


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


def _get_confident_signature_rankings_layer(model, layer_idx, top_genes_layer):
    """Get confidence-filtered signatures and combined scores for one layer."""
    if top_genes_layer is None or int(top_genes_layer) <= 0:
        n = len(model.factor_names[layer_idx])
        return (
            [[] for _ in range(n)],
            [np.array([]) for _ in range(n)],
            [np.array([]) for _ in range(n)],
        )

    sigs, confs, combined = get_stored_confident_signatures(
        model,
        layer_idx=layer_idx,
        max_genes=int(top_genes_layer),
        return_confidences=True,
        return_combined_scores=True,
    )
    rankings = []
    scores = []
    confidences = []
    for factor_name in model.factor_names[layer_idx]:
        genes = list(sigs.get(factor_name, []))[: int(top_genes_layer)]
        combined_scores = np.asarray(
            combined.get(factor_name, np.array([])), dtype=float
        )[: int(top_genes_layer)]
        conf_scores = np.asarray(confs.get(factor_name, np.array([])), dtype=float)[
            : int(top_genes_layer)
        ]
        n = min(len(genes), len(combined_scores), len(conf_scores))
        genes = genes[:n]
        combined_scores = combined_scores[:n]
        conf_scores = conf_scores[:n]
        rankings.append(genes)
        scores.append(combined_scores)
        confidences.append(conf_scores)
    return rankings, scores, confidences


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
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return np.asarray([], dtype=float)
    if scores.size == 1:
        return np.asarray([max_fontsize], dtype=float)
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
                np.count_nonzero(model.adata.obs[filled].iloc[cells] == b)
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
    prevs = np.array(
        [
            np.count_nonzero(model.adata.obs[wedged].iloc[cells] == b)
            / len(np.where(model.adata.obs[wedged] == b)[0])
            for b in model.adata.obs[wedged].cat.categories
        ],
        dtype=float,
    )
    total = float(np.sum(prevs))
    if not np.isfinite(total) or total <= 0:
        return "#FFFFFF"
    fracs = prevs / total
    # make color string for pie chart
    return ":".join(
        [
            f"{model.adata.uns[f'{wedged}_colors'][obs_idx]};{frac}"
            for obs_idx, frac in enumerate(fracs)
        ]
    )


def _compute_wedged_fillcolor_with_scores(model, wedged, factor_name):
    scores = model.adata.obs[f"{factor_name}_prob"].values
    prevs = []
    for b in model.adata.obs[wedged].cat.categories:
        group_mask = (model.adata.obs[wedged] == b).values
        score = np.mean(scores[group_mask]) if len(group_mask) > 0 else 0.0
        prevs.append(score)
    prevs = np.asarray(prevs, dtype=float)
    total = float(np.sum(prevs))
    if not np.isfinite(total) or total <= 0:
        return "#FFFFFF"
    fracs = prevs / total
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


def _add_enrichments_to_label(
    label, enrichments_df, factor_name, top_genes_layer, fontsize_kwargs
):
    """Add top significant enrichment terms to node label."""
    if enrichments_df is None or not isinstance(enrichments_df, pd.DataFrame):
        return label
    if "factor" not in enrichments_df.columns:
        return label
    if top_genes_layer is None or int(top_genes_layer) <= 0:
        return label

    df = enrichments_df[enrichments_df["factor"] == factor_name].copy()
    if len(df) == 0:
        return label

    cols_ci = {c.lower(): c for c in df.columns}
    term_col = cols_ci.get("term", None)
    padj_col = cols_ci.get("adjusted p-value", None)
    combined_col = None
    for candidate in ["combined score", "combined_score", "combinedscore"]:
        if candidate in cols_ci:
            combined_col = cols_ci[candidate]
            break
    if term_col is None or combined_col is None:
        return label

    if padj_col is not None:
        df = df[np.isfinite(df[padj_col])].copy()
        df = df[df[padj_col] <= 0.05].copy()
    df = df[np.isfinite(df[combined_col])].copy()
    df = df.sort_values(combined_col, ascending=False).head(int(top_genes_layer))
    if len(df) == 0:
        return label

    term_labels = df[term_col].astype(str).tolist()
    term_scores = np.asarray(df[combined_col].values, dtype=float)
    if len(term_scores) == 0:
        return label
    fontsizes = _map_scores_to_fontsizes(term_scores, **fontsize_kwargs)
    lines = [
        f'<FONT POINT-SIZE="{fontsizes[j]}">{term}</FONT>'
        for j, term in enumerate(term_labels)
    ]
    label += "<br/><br/>" + "<br/>".join(lines)
    return label


def _add_signature_to_label(
    label,
    model,
    layer_idx,
    factor_idx,
    show_all,
    gene_rankings,
    gene_scores_layer,
    gene_confidences_layer,
    top_genes_layer,
    show_confidences,
    mc_samples,
    fontsize_kwargs,
):
    """Add signature information to label."""

    def print_signature(i):
        factor_gene_rankings = gene_rankings[i][:top_genes_layer]
        factor_gene_scores = np.asarray(
            gene_scores_layer[i][:top_genes_layer], dtype=float
        )
        if len(factor_gene_rankings) == 0:
            return "<br/><br/>(no confident genes)"
        if len(factor_gene_scores) == 0:
            fontsizes = np.full(
                len(factor_gene_rankings), fontsize_kwargs.get("min_fontsize", 5)
            )
        else:
            fontsizes = _map_scores_to_fontsizes(factor_gene_scores, **fontsize_kwargs)
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
            if show_confidences and gene_confidences_layer is not None:
                confidence_vals = np.asarray(
                    gene_confidences_layer[idx][:top_genes_layer], dtype=float
                )
                if len(confidence_vals) > 0:
                    label += f"<br/><br/>({float(np.mean(confidence_vals)):.3f})"
    else:
        label += print_signature(idx)
        if show_confidences and gene_confidences_layer is not None:
            confidence_vals = np.asarray(
                gene_confidences_layer[idx][:top_genes_layer], dtype=float
            )
            if len(confidence_vals) > 0:
                label += f"<br/><br/>({float(np.mean(confidence_vals)):.3f})"

    return label


def _cells_for_graph_factor(
    model: "scDEF",
    layer_name: str,
    assignment_factor_name: str,
    confident_assignments: bool,
    confident_key: str,
) -> np.ndarray:
    """Indices of cells assigned to ``assignment_factor_name`` for graph sizing."""
    if confident_assignments:
        col = f"{confident_key}_factor"
    else:
        col = str(layer_name)
    obs_vals = model.adata.obs[col].astype(str).to_numpy()
    return np.where(obs_vals == str(assignment_factor_name))[0]


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
    confident_assignments: bool = False,
    confident_key: str = "confident",
):
    """Compute node size and fixedsize flag."""
    size = node_size_min
    fixedsize = "false"

    if n_cells:
        max_cells = model.n_cells
        if scale_level:
            if confident_assignments:
                cc = f"{confident_key}_factor"
                max_cells = model.adata.obs[cc].astype(str).value_counts().max()
            else:
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
    fontcolor="black",
    penwidth: Optional[float] = None,
):
    """Add a node to the graph."""
    node_kwargs = {
        "fillcolor": fillcolor,
        "color": color,
        "ordering": ordering,
        "style": style,
        "width": str(size),
        "height": str(size),
        "fixedsize": fixedsize,
        "label": label,
        "fontcolor": fontcolor,
    }

    if penwidth is not None and float(penwidth) > 0:
        node_kwargs["penwidth"] = str(float(penwidth))

    if shell:
        node_kwargs["pos"] = pos
        node_kwargs["pin"] = "true"

    g.node(factor_name, **node_kwargs)


def _finalize_node_label(label: str, show_label: bool) -> str:
    """Return a graphviz-safe label string.

    Only wrap labels as HTML-like labels when they actually contain HTML markup.
    """
    if not show_label:
        return ""
    label = str(label)
    html_markers = ("<br", "<FONT", "<TABLE", "<B>", "<I>", "<U>", "<SUB>", "<SUP>")
    if any(marker in label for marker in html_markers):
        return f"<{label}>"
    return label


def _add_edges_from_hierarchy(
    g,
    model,
    factor_name,
    hierarchy,
    color,
    path_edge_set: Optional[Set[Tuple[str, str]]] = None,
    path_color: Optional[str] = None,
):
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
        ec = _edge_color_with_path(
            color, factor_name, lower_factor_name, path_edge_set, path_color
        )
        g.edge(
            factor_name,
            lower_factor_name,
            penwidth=str(4 * normalized_weight),
            color=ec,
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
    g,
    model,
    factor_name,
    layer_idx,
    factor_idx,
    color,
    layer_factor_orders,
    show_all,
    path_edge_set: Optional[Set[Tuple[str, str]]] = None,
    path_color: Optional[str] = None,
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

        ec = _edge_color_with_path(
            color, factor_name, lower_factor_name, path_edge_set, path_color
        )
        g.edge(
            factor_name,
            lower_factor_name,
            penwidth=str(4 * normalized_weight),
            color=ec,
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
    show_enrichments: Optional[bool] = False,
    top_genes: Optional[Union[int, List[int]]] = None,
    show_batch_counts: Optional[bool] = False,
    filled: Optional[Union[str, Dict[str, float]]] = None,
    wedged: Optional[str] = None,
    assignments: Optional[bool] = True,
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
    path: Optional[Union[Sequence[str], Dict[str, Any]]] = None,
    path_color: str = "red",
    path_node_penwidth: float = 2.5,
    confident_assignments: bool = False,
    confident_key: str = "confident",
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
        enrichments: enrichment results dataframe to include in node labels
        show_enrichments: whether to show enrichment terms in node labels
        top_genes: number of genes from each signature to be shown in the node labels
        show_batch_counts: whether to show the number of cells from each batch that attach to each factor
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
        path: ordered factor names along a path, or a dict with key ``"nodes"``
            (e.g. differentiation or transition path ``nodes``). Names outside the
            hierarchy view are dropped when ``hierarchy`` is set. Edge highlights
            map each consecutive pair to the **drawn** weight edge (coarser layer
            → adjacent finer layer), so coarse→fine chains match, and **transition**
            zigzags (e.g. L0→L1→L0→…) work when each hop is between adjacent layers.
            If ``hierarchy`` is set, a pair is highlighted when it is a tree edge in
            either direction; otherwise the same adjacent-layer rule applies.
            Sets Graphviz ``color`` (node border and edge stroke) to ``path_color``;
            ``fillcolor`` is unchanged. With ``gene_score`` and ``color_edges``,
            path edges use ``path_color`` instead of the parent gene-based edge color.
        path_color: stroke color for highlighted path nodes (border) and edges
        path_node_penwidth: Graphviz ``penwidth`` for nodes on ``path`` (border
            thickness). Ignored when ``path`` is not set. Default ``2.5``; use ``1.0``
            for the usual thin border (no emphasis).
        confident_assignments: if True, attach cells using ``adata.obs[f"{confident_key}_factor"]``
            (cross-layer assignment from :func:`~scdef.tools.factor.assign_confident`) instead of
            per-layer ``adata.obs[layer_name]``. Requires that column to exist.
        confident_key: ``key_added`` prefix used with ``assign_confident`` (default ``"confident"``).
        shell: whether to use shell layout
        r: radius parameter for shell layout
        r_decay: radius decay parameter for shell layout
        **fontsize_kwargs: keyword arguments to adjust the fontsizes according to the gene scores

    Returns:
        Graphviz Graph object
    """
    if confident_assignments:
        cf_col = f"{confident_key}_factor"
        if cf_col not in model.adata.obs.columns:
            raise KeyError(
                f"Column {cf_col!r} not found in model.adata.obs; run "
                f"scd.tl.assign_confident(model, key_added={confident_key!r}) first."
            )
    # Validate and normalize inputs
    top_genes = _validate_top_genes(top_genes, model.n_layers)
    gene_cmap = matplotlib.colormaps[gene_cmap]
    gene_scores_dict = _get_gene_scores(model, gene_score, drop_factors)
    style = (
        "filled" if gene_score is not None else _determine_style(filled, wedged, model)
    )
    hierarchy_nodes = _get_hierarchy_nodes(hierarchy, top_factor)
    path_node_set: Optional[Set[str]] = None
    path_edge_set: Optional[Set[Tuple[str, str]]] = None
    path_stroke_hex: Optional[str] = None
    if path is not None:
        raw_path = _normalize_path_argument(path)
        valid_names = _all_graph_factor_names(model, show_all)
        unknown = [n for n in raw_path if n not in valid_names]
        if len(unknown) > 0:
            raise ValueError(
                f"path contains unknown factor names (not in graph): {unknown[:25]}"
            )
        pl = list(raw_path)
        if hierarchy_nodes is not None:
            pl = [n for n in pl if n in set(hierarchy_nodes)]
        if len(pl) == 0:
            model.logger.warning(
                "path has no nodes in the current hierarchy view; ignoring path highlight."
            )
        else:
            path_node_set = set(pl)
            path_edge_set = _path_edge_pairs_along_path(model, pl, hierarchy, show_all)
            path_stroke_hex = matplotlib.colors.to_hex(
                matplotlib.colors.to_rgb(path_color)
            )
    layer_factor_orders = _compute_layer_factor_orders(model, show_all)
    if enrichments is not None:
        show_enrichments = True
    enrichments_df = enrichments
    if show_enrichments and enrichments_df is None:
        stored = model.adata.uns.get("factor_enrichments", {})
        if isinstance(stored, dict):
            rows = []
            current_fit_rev = int(getattr(model, "_fit_revision", 0))
            for _, payload in stored.items():
                if not isinstance(payload, dict):
                    continue
                if int(payload.get("fit_revision", -1)) != current_fit_rev:
                    continue
                rows.extend(payload.get("results", []))
            if len(rows) > 0:
                enrichments_df = pd.DataFrame(rows)

    # Initialize graph
    g = Graph()
    g.engine = "neato" if shell else "dot"
    ordering = "out"
    angle_dict = {}
    technical_factors = set()
    if (
        "factor_obs" in model.adata.uns
        and "technical" in model.adata.uns["factor_obs"].columns
    ):
        technical_factors = set(
            model.adata.uns["factor_obs"]
            .index[model.adata.uns["factor_obs"]["technical"]]
            .tolist()
        )

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
        gene_confidences_layer = None
        if show_signatures:
            (
                gene_rankings_layer,
                gene_scores_layer,
                gene_confidences_layer,
            ) = _get_confident_signature_rankings_layer(
                model,
                layer_idx=layer_idx,
                top_genes_layer=top_genes[layer_idx],
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

            assignment_factor_name = (
                str(model.factor_names[layer_idx][int(factor_idx)])
                if confident_assignments
                else str(factor_name)
            )
            cells = _cells_for_graph_factor(
                model,
                layer_name,
                assignment_factor_name,
                confident_assignments,
                confident_key,
            )
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
                    if len(cells) == 0:
                        fillcolor = _compute_wedged_fillcolor_with_scores(
                            model,
                            wedged,
                            factor_name,
                        )
                else:
                    fillcolor = _compute_wedged_fillcolor_with_scores(
                        model, wedged, factor_name
                    )

            # Add signatures then enrichments.
            if show_signatures:
                label = _add_signature_to_label(
                    label,
                    model,
                    layer_idx,
                    factor_idx,
                    show_all,
                    gene_rankings_layer,
                    gene_scores_layer,
                    gene_confidences_layer,
                    top_genes[layer_idx],
                    show_confidences,
                    mc_samples,
                    fontsize_kwargs,
                )
            if show_enrichments:
                label = _add_enrichments_to_label(
                    label,
                    enrichments_df,
                    factor_name,
                    top_genes[layer_idx],
                    fontsize_kwargs,
                )
            if (
                (not show_signatures)
                and (not show_enrichments)
                and isinstance(filled, str)
                and filled != "factor"
            ):
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
                confident_assignments=confident_assignments,
                confident_key=confident_key,
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

            # Outgoing edge colors follow factor / gene-score convention for children
            # not on the path; path border override below must not change those edges.
            edge_color_for_children = color

            if (
                path_stroke_hex is not None
                and path_node_set is not None
                and factor_name in path_node_set
            ):
                color = path_stroke_hex

            # Finalize label
            label = _finalize_node_label(label, show_label)
            fontcolor = "gray" if factor_name in technical_factors else "black"

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
            path_penw: Optional[float] = None
            if (
                path is not None
                and path_node_set is not None
                and factor_name in path_node_set
                and float(path_node_penwidth) > 0
            ):
                path_penw = float(path_node_penwidth)

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
                fontcolor=fontcolor,
                penwidth=path_penw,
            )

            # Add edges
            edge_default = edge_color_for_children
            if not color_edges:
                edge_default = None

            if layer_idx > 0:
                if hierarchy is not None:
                    _add_edges_from_hierarchy(
                        g,
                        model,
                        factor_name,
                        hierarchy,
                        edge_default,
                        path_edge_set=path_edge_set if path_edge_set else None,
                        path_color=path_stroke_hex,
                    )
                else:
                    _add_edges_from_weights(
                        g,
                        model,
                        factor_name,
                        layer_idx,
                        factor_idx,
                        edge_default,
                        layer_factor_orders,
                        show_all,
                        path_edge_set=path_edge_set if path_edge_set else None,
                        path_color=path_stroke_hex,
                    )

    model.graph = g
    model.logger.info("Updated scDEF graph")
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
    show_enrichments: Optional[bool] = False,
    top_genes: Optional[int] = None,
    filled: Optional[str] = None,
    wedged: Optional[str] = None,
    assignments: Optional[bool] = True,
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
    confident_assignments: bool = False,
    confident_key: str = "confident",
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
        enrichments: enrichment results dataframe to include in node labels
        show_enrichments: whether to show enrichment terms in node labels
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
        confident_assignments: same as :func:`make_graph`.
        confident_key: same as :func:`make_graph`.
        shell: whether to use shell layout
        r: radius parameter for shell layout
        r_decay: radius decay parameter for shell layout
        **fontsize_kwargs: keyword arguments to adjust the fontsizes according to the gene scores

    Returns:
        Graphviz Graph object
    """
    if confident_assignments:
        cf_col = f"{confident_key}_factor"
        if cf_col not in model.adata.obs.columns:
            raise KeyError(
                f"Column {cf_col!r} not found in model.adata.obs; run "
                f"scd.tl.assign_confident(model, key_added={confident_key!r}) first."
            )
    # Validate and normalize inputs
    top_genes = _validate_top_genes(top_genes, model.n_layers)
    gene_cmap = matplotlib.colormaps[gene_cmap]
    gene_scores_dict = _get_gene_scores(model, gene_score, drop_factors)
    style = (
        "filled" if gene_score is not None else _determine_style(filled, wedged, model)
    )
    hierarchy_nodes = _get_hierarchy_nodes(hierarchy, top_factor)
    layer_factor_orders = _compute_layer_factor_orders(model, False)
    if enrichments is not None:
        show_enrichments = True
    enrichments_df = enrichments
    if show_enrichments and enrichments_df is None:
        stored = model.adata.uns.get("factor_enrichments", {})
        if isinstance(stored, dict):
            rows = []
            current_fit_rev = int(getattr(model, "_fit_revision", 0))
            for _, payload in stored.items():
                if not isinstance(payload, dict):
                    continue
                if int(payload.get("fit_revision", -1)) != current_fit_rev:
                    continue
                rows.extend(payload.get("results", []))
            if len(rows) > 0:
                enrichments_df = pd.DataFrame(rows)

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
        if layer_idx == model.n_layers - 1:
            # Technical hierarchy uses a synthetic root node.
            factors = np.array([0], dtype=int)
            layer_colors = ["#000000"]
        else:
            factors = model.factor_lists[layer_idx]
            layer_colors = _get_layer_colors(model, layer_idx, False, factors)

        # Get gene rankings if showing signatures
        gene_rankings_layer = None
        gene_scores_layer = None
        gene_confidences_layer = None
        if show_signatures:
            if layer_idx == model.n_layers - 1:
                gene_rankings_layer, gene_scores_layer = (
                    root_gene_rankings,
                    root_gene_scores,
                )
            else:
                (
                    gene_rankings_layer,
                    gene_scores_layer,
                    gene_confidences_layer,
                ) = _get_confident_signature_rankings_layer(
                    model,
                    layer_idx=layer_idx,
                    top_genes_layer=top_genes[layer_idx],
                )

        # Process each factor in the layer
        if layer_idx == model.n_layers - 1:
            factor_order = np.array([0], dtype=int)
        else:
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
                assignment_factor_name = (
                    str(model.factor_names[layer_idx][int(factor_idx)])
                    if confident_assignments
                    else str(factor_name)
                )
                cells = _cells_for_graph_factor(
                    model,
                    layer_name,
                    assignment_factor_name,
                    confident_assignments,
                    confident_key,
                )
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
                    if len(cells) == 0 and layer_idx == 0:
                        fillcolor = _compute_wedged_fillcolor_with_scores(
                            model,
                            wedged,
                            factor_name,
                        )
                else:
                    if layer_idx == 0:
                        fillcolor = _compute_wedged_fillcolor_with_scores(
                            model,
                            wedged,
                            factor_name,
                        )

            # Add signatures then enrichments.
            if show_signatures:
                label = _add_signature_to_label(
                    label,
                    model,
                    layer_idx,
                    factor_idx,
                    False,
                    gene_rankings_layer,
                    gene_scores_layer,
                    gene_confidences_layer,
                    top_genes[layer_idx],
                    show_confidences,
                    mc_samples,
                    fontsize_kwargs,
                )
            if show_enrichments:
                label = _add_enrichments_to_label(
                    label,
                    enrichments_df,
                    factor_name,
                    top_genes[layer_idx],
                    fontsize_kwargs,
                )
            if (
                (not show_signatures)
                and (not show_enrichments)
                and isinstance(filled, str)
                and filled != "factor"
            ):
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
                confident_assignments=confident_assignments,
                confident_key=confident_key,
            )

            # Finalize label
            label = _finalize_node_label(label, show_label)

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
    model.logger.info("Updated scDEF graph")
    return g


def biological_hierarchy(model: "scDEF", **kwargs: Any) -> Graph:
    """Plot the biological hierarchy of the model.

    Args:
        model: scDEF model instance
        **kwargs: keyword arguments passed to make_graph

    Returns:
        Graphviz Graph object
    """
    # Get the top signature
    technical_factors = model.adata.uns["factor_obs"][
        model.adata.uns["factor_obs"]["technical"]
    ].index.tolist()
    g = make_graph(
        model,
        hierarchy=model.adata.uns["biological_hierarchy"],
        drop_factors=technical_factors,
        **kwargs,
    )
    return g


def technical_hierarchy(
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
