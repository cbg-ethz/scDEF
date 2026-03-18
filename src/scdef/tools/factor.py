import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib as mpl
import copy
from scipy.stats import norm
from scipy.ndimage import uniform_filter1d
from sklearn.preprocessing import minmax_scale
import pandas as pd
from .hierarchy import get_hierarchy, compute_hierarchy_scores
from typing import Optional, Sequence, Dict, List, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from scdef.models._scdef import scDEF


def factor_diagnostics(model: "scDEF", recompute: bool = False) -> None:
    """Compute/store factor diagnostics in ``model.adata.uns['factor_obs']``.

    Args:
        model: scDEF model instance
        recompute: if True, force recomputation of the cached fixed upper-layer
            factor subset used for clarity scores, even if the fit revision
            did not change.
    """
    # Keep layer 0 unfiltered, but use a fixed filtered subset on upper layers.
    # Cache and reuse upper-layer factor lists so diagnostics remain stable across
    # later calls to filter/annotate routines.
    cache_key = "_factor_obs_upper_lists_fixed"
    cache_rev_key = "_factor_obs_fit_revision"
    current_fit_rev = int(getattr(model, "_fit_revision", 0))
    reset_cache = (
        recompute
        or cache_key not in model.adata.uns
        or len(model.adata.uns[cache_key]) != max(model.n_layers - 1, 0)
        or int(model.adata.uns.get(cache_rev_key, -1)) != current_fit_rev
    )
    if not reset_cache:
        # Validate cached indices against current layer sizes.
        for i, idxs in enumerate(model.adata.uns[cache_key], start=1):
            arr = np.asarray(idxs, dtype=int)
            if np.any(arr < 0) or np.any(arr >= model.layer_sizes[i]):
                reset_cache = True
                break
    if reset_cache:
        model.adata.uns[cache_key] = [
            np.asarray(model.factor_lists[i], dtype=int).tolist()
            for i in range(1, model.n_layers)
        ]
        model.adata.uns[cache_rev_key] = current_fit_rev

    fixed_upper_lists = [
        np.asarray(idxs, dtype=int) for idxs in model.adata.uns[cache_key]
    ]
    old_factor_lists = [np.asarray(f, dtype=int).copy() for f in model.factor_lists]
    old_factor_names = (
        [list(names) for names in model.factor_names]
        if hasattr(model, "factor_names")
        else None
    )
    try:
        model.factor_lists = [
            np.arange(model.layer_sizes[0], dtype=int)
        ] + fixed_upper_lists
        if hasattr(model, "set_factor_names"):
            model.set_factor_names()
        res = compute_hierarchy_scores(
            model,
            use_filtered=True,
            filter_upper_layers=True,
        )
    finally:
        model.factor_lists = old_factor_lists
        if old_factor_names is not None:
            model.factor_names = old_factor_names
    model.adata.uns["factor_obs"] = res["per_factor"].set_index("child_factor")
    model.adata.uns["factor_obs"]["ARD"] = np.array(
        [np.nan] * len(model.adata.uns["factor_obs"])
    )
    model.adata.uns["factor_obs"]["BRD"] = np.array(
        [np.nan] * len(model.adata.uns["factor_obs"])
    )
    factor_obs = model.adata.uns["factor_obs"]
    if "original_factor_idx" not in factor_obs.columns:
        raise KeyError(
            "factor_obs is missing 'original_factor_idx'. Recompute diagnostics with updated compute_hierarchy_scores."
        )

    if "child_layer" in factor_obs.columns:
        l0_rows = factor_obs.index[factor_obs["child_layer"] == model.layer_names[0]]
    else:
        l0_rows = factor_obs.index

    original_idx = factor_obs.loc[l0_rows, "original_factor_idx"].to_numpy(dtype=int)
    valid = (original_idx >= 0) & (original_idx < int(model.layer_sizes[0]))
    l0_rows = np.asarray(l0_rows)[valid]
    original_idx = original_idx[valid]

    ard_all = np.asarray(model.pmeans["factor_means"]).ravel()
    brd_all = np.asarray(model.pmeans["factor_concentrations"]).ravel()
    factor_obs.loc[l0_rows, "ARD"] = ard_all[original_idx]
    factor_obs.loc[l0_rows, "BRD"] = brd_all[original_idx]
    # Initialize technical annotation for all factors.
    factor_obs["technical"] = False


def set_factor_signatures(
    model: "scDEF",
    signatures: Optional[Dict[str, List[str]]] = None,
    top_genes: int = 10,
) -> Dict[str, List[str]]:
    if signatures is None:
        signatures = model.get_signatures_dict(top_genes=top_genes)
    model.adata.uns["factor_signatures"] = signatures
    return signatures


def get_confident_signatures(
    model: "scDEF",
    layer_idx: int = 0,
    confidence_threshold: float = 0.95,
    tau_quantile: float = 0.8,
    min_effect: Optional[float] = None,
    max_genes: Optional[int] = None,
    return_confidences: bool = False,
) -> Union[Dict[str, List[str]], Tuple[Dict[str, List[str]], Dict[str, np.ndarray]]]:
    """Get confidence-based signatures per factor using posterior mean/variance.

    For each factor independently, this computes a per-factor threshold
    ``tau = quantile(E[W_k,:], tau_quantile)`` and keeps genes that satisfy
    ``P(W_k,g > tau) >= confidence_threshold`` under a normal approximation
    using the posterior mean and variance of ``W``.

    Args:
        model: scDEF model instance
        layer_idx: layer index to use (currently supports only layer 0)
        confidence_threshold: minimum posterior confidence to keep a gene
        tau_quantile: quantile of factor mean loadings used as threshold tau
        min_effect: optional minimum posterior mean loading ``E[W_k,g]``
        max_genes: optional maximum number of genes to keep per factor
        return_confidences: whether to also return per-gene confidence arrays

    Returns:
        Dictionary mapping factor names to confident gene lists. If
        ``return_confidences`` is True, also returns a dictionary mapping
        factor names to confidence arrays aligned with each gene list.
    """
    if layer_idx != 0:
        raise NotImplementedError(
            "Confidence-based gene signatures are currently implemented for layer 0 only."
        )
    if not (0.0 < confidence_threshold < 1.0):
        raise ValueError("confidence_threshold must be in (0, 1).")
    if not (0.0 < tau_quantile < 1.0):
        raise ValueError("tau_quantile must be in (0, 1).")

    layer_name = model.layer_names[layer_idx]
    kept = np.asarray(model.factor_lists[layer_idx], dtype=int)
    term_means = np.asarray(model.pmeans[f"{layer_name}W"], dtype=float)[kept]
    term_vars = np.asarray(model.pvars[f"{layer_name}W"], dtype=float)[kept]
    term_vars = np.maximum(term_vars, 0.0)
    term_stds = np.sqrt(term_vars + 1e-12)
    term_names = np.asarray(model.adata.var_names)

    signatures: Dict[str, List[str]] = {}
    signature_confidences: Dict[str, np.ndarray] = {}
    for factor_idx, factor_name in enumerate(model.factor_names[layer_idx]):
        mu = term_means[factor_idx]
        sigma = term_stds[factor_idx]
        tau = float(np.quantile(mu, tau_quantile))
        z = (mu - tau) / sigma
        confidences = norm.cdf(z)

        keep_mask = confidences >= confidence_threshold
        if min_effect is not None:
            keep_mask = keep_mask & (mu >= min_effect)
        keep_idx = np.where(keep_mask)[0]

        # Rank by confidence, then mean loading.
        if len(keep_idx) > 0:
            order = np.lexsort((-mu[keep_idx], -confidences[keep_idx]))
            keep_idx = keep_idx[order]
        if max_genes is not None:
            keep_idx = keep_idx[: int(max_genes)]

        signatures[factor_name] = term_names[keep_idx].tolist()
        signature_confidences[factor_name] = confidences[keep_idx]

    if return_confidences:
        return signatures, signature_confidences
    return signatures


def set_technical_factors(
    model: "scDEF", factors: Optional[Sequence[str]] = None
) -> None:
    """Set the technical factors of the model.

    Technical factors must be layer 0 factors.

    Args:
        model: scDEF model instance
        factors: list of factor names to mark as technical
    """
    # in model.adata.uns["factor_obs"], annotate as technical or not.
    if "factor_obs" not in model.adata.uns:
        factor_diagnostics(model)
    if "technical" not in model.adata.uns["factor_obs"].columns:
        model.adata.uns["factor_obs"]["technical"] = False
    if factors is None:
        factors = []
    model.adata.uns["factor_obs"].loc[factors, "technical"] = True

    # Get complete hierarchy
    complete_hierarchy = get_hierarchy(model, simplified=False)
    # Traverse hierarchy. If all the children of a factor are technical, set the factor as technical.
    for factor, children in complete_hierarchy.items():
        if all(
            [
                model.adata.uns["factor_obs"].loc[child, "technical"]
                for child in children
            ]
        ):
            model.adata.uns["factor_obs"].loc[factor, "technical"] = True


def __build_consensus_signature(var_names, gene_scores_array, sizes_array):
    sizes_array = sizes_array / np.sum(sizes_array)
    avg_ranks = np.sum(sizes_array[:, None] * gene_scores_array, axis=0)
    idx_sorted = np.argsort(avg_ranks)[::-1]
    consensus = var_names[idx_sorted].tolist()
    consensus_scores = avg_ranks[idx_sorted]
    return consensus, consensus_scores


def get_technical_signature(
    model: "scDEF", top_genes: int = 10, return_scores: bool = False
) -> Union[List[str], Tuple[List[str], np.ndarray]]:
    hierarchy = model.adata.uns["technical_hierarchy"]
    gene_rankings, gene_scores = model.get_rankings(
        layer_idx=0,
        genes=True,
        return_scores=True,
    )

    # Reorder each gene_rankings and gene_scores by model.adata.var_names
    var_names = np.array(model.adata.var_names)
    n_factors = len(gene_scores)
    gene_scores_ordered = []
    for i in range(n_factors):
        ranking = np.array(gene_rankings[i])
        scores = np.array(gene_scores[i])
        # Map gene ranking to index in model.adata.var_names
        gene_order = np.argsort(
            [np.where(var_names == gene)[0][0] for gene in ranking]
        )  # noqa: F841
        reordered_idx = np.argsort(
            [np.where(ranking == g)[0][0] for g in var_names]
        )  # noqa: F841
        # Pad/truncate scores to fit var_names if necessary
        scores_full = np.full(len(var_names), np.nan)
        mask = np.in1d(var_names, ranking)
        scores_full[mask] = scores[
            [np.where(ranking == g)[0][0] for g in var_names[mask]]
        ]
        # Replace nans with 0 if needed, or keep as nan
        scores_full = np.nan_to_num(scores_full, nan=0)
        gene_scores_ordered.append(scores_full)
    gene_scores = np.array(
        [s / np.max(s) if np.max(s) > 0 else s for s in gene_scores_ordered]
    )

    relevances = model.get_relevances_dict()
    children = hierarchy["tech_top"]
    factors = [
        factor
        for i, factor in enumerate(range(len(gene_scores)))
        if model.factor_names[0][i] in children
    ]
    gene_scores = np.array([gene_scores[f] / np.max(gene_scores[f]) for f in factors])
    children_sizes = np.array([relevances[child] for child in children]).ravel()

    consensus_signature, consensus_scores = __build_consensus_signature(
        model.adata.var_names, gene_scores, children_sizes
    )
    if return_scores:
        return consensus_signature[:top_genes], consensus_scores[:top_genes]
    return consensus_signature[:top_genes]


def get_biological_signature(model: "scDEF", top_genes: int = 10) -> List[str]:
    # Get the top signature
    technical_factors = model.adata.uns["factor_obs"][
        model.adata.uns["factor_obs"]["technical"]
    ].index.tolist()
    signatures_dict = model.get_signatures_dict(
        top_genes=top_genes, drop_factors=technical_factors
    )
    signature = signatures_dict[f"{model.layer_names[model.n_layers - 1]}_0"]
    return signature


def umap(
    model: "scDEF",
    layers: Optional[List[int]] = None,
    use_log: bool = False,
    metric: str = "euclidean",
) -> None:
    """Compute UMAP embeddings for each scDEF layer.

    The resulting embeddings are stored in
    ``model.adata.obsm[f"X_umap_{layer_name}"]`` for each layer.

    Args:
        model: scDEF model instance
        layers: which layers to compute UMAPs for. If None, all layers
            with more than one factor are used (in descending order).
        use_log: whether to use log-transformed cell-factor weights for
            the neighbor graph computation.
        metric: distance metric for neighbors computation.
    """
    if layers is None:
        layers = [
            i
            for i in range(model.n_layers - 1, -1, -1)
            if len(model.factor_lists[i]) > 1
        ]

    for layer in layers:
        layer_name = model.layer_names[layer]
        # Compute log representation
        model.adata.obsm[f"X_{layer_name}_log"] = np.log(
            model.adata.obsm[f"X_{layer_name}"]
        )
        if use_log:
            sc.pp.neighbors(model.adata, use_rep=f"X_{layer_name}_log")
        else:
            sc.pp.neighbors(
                model.adata,
                use_rep=f"X_{layer_name}",
                metric=metric,
            )
        sc.tl.umap(model.adata)
        # Store under a layer-specific key
        model.adata.obsm[f"X_umap_{layer_name}"] = model.adata.obsm["X_umap"].copy()


def multilevel_paga(
    model: "scDEF",
    neighbors_rep: str = "X_L0",
    layers: Optional[List[int]] = None,
    reuse_pos: bool = True,
    layout: str = "fa",
    random_seed: int = 0,
    **paga_kwargs,
) -> None:
    """Compute and cache multilevel PAGA results for plotting.

    This computes PAGA once per requested layer and stores layer-specific
    connectivities and layout positions in ``model.adata.uns['multilevel_paga']``
    so plotting can be repeated without recomputing PAGA.

    Args:
        model: scDEF model instance
        neighbors_rep: ``adata.obsm`` key used for neighbor graph construction
        layers: which layers to compute. If None, all layers with >1 factor
            are used in descending order.
        reuse_pos: whether to initialize each finer layer with positions from
            the previous coarser layer.
        layout: graph layout passed to ``scanpy.pl.paga``
        random_seed: seed used for deterministic jitter when ``reuse_pos`` is True
        **paga_kwargs: extra keyword arguments forwarded to ``scanpy.pl.paga``
            during layout computation
    """
    if layers is None:
        layers = [
            i
            for i in range(model.n_layers - 1, -1, -1)
            if len(model.factor_lists[i]) > 1
        ]
    if len(layers) == 0:
        model.adata.uns["multilevel_paga"] = {
            "neighbors_rep": neighbors_rep,
            "layers": [],
            "reuse_pos": reuse_pos,
            "layout": layout,
            "results": {},
        }
        return

    sc.pp.neighbors(model.adata, use_rep=neighbors_rep)
    results = {}
    pos = None
    old_layer_name = None
    old_paga = copy.deepcopy(model.adata.uns.get("paga", None))

    for layer_idx in layers:
        layer_name = model.layer_names[layer_idx]

        if old_layer_name is not None and reuse_pos:
            matches = sc._utils.identify_groups(
                model.adata.obs[layer_name], model.adata.obs[old_layer_name]
            )
            pos = []
            np.random.seed(random_seed)
            prev_pos = model.adata.uns["paga"]["pos"]
            coarse_categories = model.adata.obs[old_layer_name].cat.categories
            for c in model.adata.obs[layer_name].cat.categories:
                idx = coarse_categories.get_loc(matches[c][0])
                pos_i = prev_pos[idx] + np.random.random(2)
                pos.append(pos_i)
            pos = np.array(pos)

        sc.tl.paga(model.adata, groups=layer_name)
        sc.pl.paga(
            model.adata,
            init_pos=pos,
            layout=layout,
            show=False,
            **paga_kwargs,
        )
        plt.close()
        results[layer_name] = {
            "paga": copy.deepcopy(model.adata.uns["paga"]),
            "pos": np.array(model.adata.uns["paga"]["pos"]),
            "layer_idx": int(layer_idx),
        }
        old_layer_name = layer_name

    if old_paga is None:
        model.adata.uns.pop("paga", None)
    else:
        model.adata.uns["paga"] = old_paga

    model.adata.uns["multilevel_paga"] = {
        "neighbors_rep": neighbors_rep,
        "layers": [int(i) for i in layers],
        "reuse_pos": bool(reuse_pos),
        "layout": layout,
        "results": results,
    }

def plot_trajectory_heatmap(
    model: "scDEF",
    factor_path: Sequence[Union[int, str]],
    layer_idx: int = 0,
    genes_per_factor: Optional[int] = 3,
    confidence_threshold: float = 0.95,
    confidence_tau_quantile: float = 0.8,
    min_effect: Optional[float] = None,
    smoothing: int = 50,
    figwidth: float = 8,
    gene_height: float = 0.28,
    show_celltype: bool = False,
    celltype_col: str = "cell_type",
    heatmap_cmap: str = "RdYlBu_r",
    xlabel: str = "Cells",
    save: Optional[str] = None,
    show: bool = True,
):
    """Plot stacked trajectory heatmap of path factors and their confident genes.

    For each factor in ``factor_path``, this draws one row for the factor score
    followed by rows for its confident genes (computed independently with
    :func:`get_confident_signatures`), all ordered along a path-sorted cell axis.
    """
    layer_name = model.layer_names[layer_idx]
    kept_names = list(model.factor_names[layer_idx])

    path_names: List[str] = []
    for f in factor_path:
        if isinstance(f, str):
            if f not in kept_names:
                raise ValueError(f"Factor '{f}' not found in layer {layer_idx}.")
            path_names.append(f)
        else:
            f_idx = int(f)
            if f_idx < 0 or f_idx >= len(kept_names):
                raise IndexError(
                    f"Factor index {f_idx} out of bounds for layer {layer_idx}."
                )
            path_names.append(kept_names[f_idx])
    if len(path_names) == 0:
        raise ValueError("factor_path must contain at least one factor.")

    path_mask = model.adata.obs[layer_name].isin(path_names).values
    if np.count_nonzero(path_mask) == 0:
        raise ValueError("No cells found for the provided factor_path.")

    X_probs = np.asarray(model.adata.obsm[f"X_{layer_name}_probs"], dtype=float)
    factor_pos = {name: idx for idx, name in enumerate(kept_names)}
    path_cols = np.array([factor_pos[name] for name in path_names], dtype=int)
    path_weights = X_probs[path_mask][:, path_cols]
    denom = np.maximum(np.sum(path_weights, axis=1), 1e-12)
    ranks = np.arange(len(path_names), dtype=float)
    progress = np.sum(path_weights * ranks[None, :], axis=1) / denom
    selected_cells = np.where(path_mask)[0]
    sorted_cells = selected_cells[np.argsort(progress)]
    t_path = model.adata[sorted_cells].copy()

    confident_sigs = get_confident_signatures(
        model,
        layer_idx=layer_idx,
        confidence_threshold=confidence_threshold,
        tau_quantile=confidence_tau_quantile,
        min_effect=min_effect,
        max_genes=genes_per_factor,
    )

    score_cols = [f"{name}_score" for name in path_names]
    missing_scores = [col for col in score_cols if col not in t_path.obs.columns]
    if len(missing_scores) > 0:
        raise KeyError(f"Score columns not found in adata.obs: {missing_scores}")

    row_values: List[np.ndarray] = []
    row_labels: List[str] = []
    block_sizes: List[int] = []
    for factor_name in path_names:
        score_col = f"{factor_name}_score"
        score_vals = uniform_filter1d(
            np.asarray(t_path.obs[score_col].values, dtype=float), size=smoothing
        )
        row_values.append(minmax_scale(score_vals))
        row_labels.append(f"[{factor_name}]")

        genes = [g for g in confident_sigs.get(factor_name, []) if g in t_path.var_names]
        for gene in genes:
            expr = t_path[:, [gene]].X
            if hasattr(expr, "toarray"):
                expr = expr.toarray()
            expr_vals = uniform_filter1d(np.asarray(expr, dtype=float).ravel(), size=smoothing)
            row_values.append(minmax_scale(expr_vals))
            row_labels.append(f"  {gene}")
        block_sizes.append(1 + len(genes))

    if len(row_values) == 0:
        raise ValueError("No rows to plot. Check factor_path and confident signatures.")
    heatmap_matrix = np.vstack(row_values)
    n_rows = heatmap_matrix.shape[0]

    if show_celltype:
        if celltype_col not in t_path.obs.columns:
            raise KeyError(f"{celltype_col} not found in adata.obs.")
        ct_cat = pd.Categorical(t_path.obs[celltype_col])
        categories = ct_cat.categories.tolist()
        uns_key = f"{celltype_col}_colors"
        if uns_key in t_path.uns:
            cat_to_color = dict(
                zip(
                    t_path.uns.get(f"{celltype_col}_categories", categories),
                    t_path.uns[uns_key],
                )
            )
        else:
            cmap_fb = plt.get_cmap("tab10", len(categories))
            cat_to_color = {c: cmap_fb(i) for i, c in enumerate(categories)}
        ct_rgb = np.array(
            [mpl.colors.to_rgb(cat_to_color[c]) for c in t_path.obs[celltype_col]]
        )

    nrows = 1 + int(show_celltype)
    height_ratios = [0.3] if show_celltype else []
    height_ratios.append(n_rows * gene_height)

    total_height = sum(height_ratios) + 1.5
    fig = plt.figure(figsize=(figwidth, total_height))
    gs = fig.add_gridspec(
        nrows=nrows,
        ncols=2,
        height_ratios=height_ratios,
        width_ratios=[figwidth - 0.8, 0.4],
        hspace=0.05,
        wspace=0.05,
    )

    row = 0
    if show_celltype:
        ax_ct = fig.add_subplot(gs[row, 0])
        row += 1
        ax_ct_cb = fig.add_subplot(gs[0, 1])
        ax_ct_cb.axis("off")
        ax_ct.imshow(ct_rgb[np.newaxis, :, :], aspect="auto", interpolation="nearest")
        ax_ct.set_yticks([0])
        ax_ct.set_yticklabels([celltype_col], fontsize=9)
        ax_ct.set_xticks([])
        legend_patches = [
            mpl.patches.Patch(color=cat_to_color[c], label=c) for c in categories
        ]
        ax_ct.legend(
            handles=legend_patches,
            fontsize=8,
            frameon=False,
            loc="lower left",
            bbox_to_anchor=(0.0, 1.05),
            ncol=max(1, min(len(categories), 8)),
            borderaxespad=0,
        )

    ax_hm = fig.add_subplot(gs[row, 0])
    ax_hm_cb = fig.add_subplot(gs[row, 1])
    im = ax_hm.imshow(
        heatmap_matrix, aspect="auto", cmap=heatmap_cmap, interpolation="nearest"
    )
    ax_hm.set_yticks(range(n_rows))
    ax_hm.set_yticklabels(row_labels, fontsize=8)
    ax_hm.set_xlabel(xlabel, fontsize=10)
    ax_hm.set_xticks([])
    plt.colorbar(im, cax=ax_hm_cb, label="Row-scaled signal")
    ax_hm_cb.yaxis.label.set_size(8)
    ax_hm_cb.tick_params(labelsize=7)

    cum = 0
    for bs in block_sizes[:-1]:
        cum += bs
        ax_hm.axhline(cum - 0.5, color="white", linewidth=1.0, alpha=0.8)

    if save:
        plt.savefig(save, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
        return None
    return fig