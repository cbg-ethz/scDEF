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
    confidence_threshold: float = 0.9,
    tau_quantile: float = 0.99,
    min_effect: Optional[float] = None,
    max_genes: Optional[int] = None,
    mc_samples_upper: int = 100,
    random_seed: int = 0,
    return_confidences: bool = False,
) -> Union[Dict[str, List[str]], Tuple[Dict[str, List[str]], Dict[str, np.ndarray]]]:
    """Get confidence-based signatures per factor using posterior mean/variance.

    For each factor independently, this computes a per-factor threshold
    ``tau = quantile(E[W_k,:], tau_quantile)`` and keeps genes that satisfy
    ``P(W_k,g > tau) >= confidence_threshold`` under a normal approximation
    using the posterior mean and variance of ``W``.

    For ``layer_idx > 0``, confidences are estimated with Monte Carlo sampling
    from the variational posterior via ``model.get_signature_sample``.

    Args:
        model: scDEF model instance
        layer_idx: layer index to use
        confidence_threshold: minimum posterior confidence to keep a gene
        tau_quantile: quantile of factor mean loadings used as threshold tau
        min_effect: optional minimum posterior mean loading ``E[W_k,g]``
        max_genes: optional maximum number of genes to keep per factor
        mc_samples_upper: number of Monte Carlo samples used for
            ``layer_idx > 0`` confidence estimation
        random_seed: random seed for Monte Carlo sampling in upper layers
        return_confidences: whether to also return per-gene confidence arrays

    Returns:
        Dictionary mapping factor names to confident gene lists. If
        ``return_confidences`` is True, also returns a dictionary mapping
        factor names to confidence arrays aligned with each gene list.
    """
    if layer_idx < 0 or layer_idx >= model.n_layers:
        raise ValueError(f"layer_idx must be in [0, {model.n_layers - 1}].")
    if not (0.0 < confidence_threshold < 1.0):
        raise ValueError("confidence_threshold must be in (0, 1).")
    if not (0.0 < tau_quantile < 1.0):
        raise ValueError("tau_quantile must be in (0, 1).")
    if mc_samples_upper <= 0:
        raise ValueError("mc_samples_upper must be > 0.")

    layer_name = model.layer_names[layer_idx]
    term_names = np.asarray(model.adata.var_names)
    signatures: Dict[str, List[str]] = {}
    signature_confidences: Dict[str, np.ndarray] = {}

    if layer_idx == 0:
        kept = np.asarray(model.factor_lists[layer_idx], dtype=int)
        term_means = np.asarray(model.pmeans[f"{layer_name}W"], dtype=float)[kept]
        term_vars = np.asarray(model.pvars[f"{layer_name}W"], dtype=float)[kept]
        term_vars = np.maximum(term_vars, 0.0)
        term_stds = np.sqrt(term_vars + 1e-12)

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
    else:
        from jax import random

        if hasattr(model, "logger"):
            model.logger.info(
                "Estimating confident signatures for layer %s with Monte Carlo "
                "(mc_samples_upper=%s). This may be slower than layer 0.",
                layer_idx,
                mc_samples_upper,
            )
        _, mean_scores = model.get_rankings(
            layer_idx=layer_idx,
            top_genes=len(term_names),
            genes=True,
            return_scores=True,
            sorted_scores=False,
        )
        term_means = np.asarray(mean_scores, dtype=float)
        base_rng = random.PRNGKey(int(random_seed))
        n_genes = len(term_names)

        for factor_idx, factor_name in enumerate(model.factor_names[layer_idx]):
            mu = term_means[factor_idx]
            tau = float(np.quantile(mu, tau_quantile))

            samples = []
            for s_idx in range(int(mc_samples_upper)):
                rng = random.fold_in(
                    base_rng, factor_idx * int(mc_samples_upper) + s_idx
                )
                _, sample_scores = model.get_signature_sample(
                    rng,
                    factor_idx=factor_idx,
                    layer_idx=layer_idx,
                    top_genes=n_genes,
                    return_scores=True,
                )
                samples.append(np.asarray(sample_scores, dtype=float))
            sample_arr = np.vstack(samples)
            confidences = np.mean(sample_arr > tau, axis=0)

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
    l0_path: Optional[Sequence[Union[int, str]]] = None,
    genes_per_factor: Optional[int] = 3,
    confidence_threshold: float = 0.95,
    confidence_tau_quantile: float = 0.99,
    confidence_mc_samples_upper: int = 100,
    confidence_random_seed: int = 0,
    min_effect: Optional[float] = None,
    smoothing: int = 50,
    figwidth: float = 8,
    gene_height: float = 0.28,
    block_spacing: int = 1,
    annotation_obs_key: Optional[Union[str, Sequence[str]]] = None,
    subset_obs_key: Optional[str] = None,
    subset_obs: Optional[Union[str, Sequence[str]]] = None,
    heatmap_cmap: str = "RdYlBu_r",
    factor_heatmap_cmap: str = "viridis",
    colorbar_gap: float = 0.16,
    xlabel: str = "Cells",
    save: Optional[str] = None,
    show: bool = True,
):
    """Plot stacked trajectory heatmap of path factors and their confident genes.

    For each factor in ``factor_path``, this draws one row for the factor score
    followed by rows for its confident genes (computed independently with
    :func:`get_confident_signatures`), all ordered along a path-sorted cell axis.
    When ``l0_path`` is provided, cell ordering uses this layer-0 path while rows
    still show factors from ``factor_path`` at ``layer_idx``.
    Optional categorical annotations from ``adata.obs`` can be shown in strips at
    the top via ``annotation_obs_key``. Cells can be pre-filtered via
    ``subset_obs_key`` and ``subset_obs`` before computing path order and plotting.
    ``block_spacing`` controls vertical spacing between factor+genes heatmap
    panels (no blank data rows are inserted).
    ``colorbar_gap`` controls horizontal spacing between the two colorbars.
    """
    layer_name = model.layer_names[layer_idx]
    kept_names = list(model.factor_names[layer_idx])

    def _resolve_path_names(
        path: Sequence[Union[int, str]], layer_names: List[str], layer_idx_resolve: int
    ) -> List[str]:
        resolved: List[str] = []
        for f in path:
            if isinstance(f, str):
                if f not in layer_names:
                    raise ValueError(
                        f"Factor '{f}' not found in layer {layer_idx_resolve}."
                    )
                resolved.append(f)
            else:
                f_idx = int(f)
                if f_idx < 0 or f_idx >= len(layer_names):
                    raise IndexError(
                        f"Factor index {f_idx} out of bounds for layer {layer_idx_resolve}."
                    )
                resolved.append(layer_names[f_idx])
        return resolved

    path_names = _resolve_path_names(factor_path, kept_names, layer_idx)
    if len(path_names) == 0:
        raise ValueError("factor_path must contain at least one factor.")
    if block_spacing < 0:
        raise ValueError("block_spacing must be >= 0.")
    if colorbar_gap < 0.0:
        raise ValueError("colorbar_gap must be >= 0.")

    subset_mask = np.ones(model.adata.n_obs, dtype=bool)
    if subset_obs is not None and subset_obs_key is None:
        raise ValueError("subset_obs_key must be provided when subset_obs is set.")
    if subset_obs_key is not None:
        if subset_obs_key not in model.adata.obs.columns:
            raise KeyError(f"{subset_obs_key} not found in adata.obs.")
        if subset_obs is None:
            raise ValueError("subset_obs must be provided when subset_obs_key is set.")
        if isinstance(subset_obs, str):
            subset_vals = [subset_obs]
        else:
            subset_vals = [str(v) for v in subset_obs]
        if len(subset_vals) == 0:
            raise ValueError("subset_obs must contain at least one value.")
        subset_mask = (
            model.adata.obs[subset_obs_key].astype(str).isin(subset_vals).values
        )

    path_mask = model.adata.obs[layer_name].isin(path_names).values & subset_mask
    if np.count_nonzero(path_mask) == 0:
        raise ValueError(
            "No cells found for the provided factor_path after applying subset filters."
        )

    sort_layer_name = layer_name
    sort_names = path_names
    sort_kept_names = kept_names
    use_l0_sort = layer_idx > 0 and l0_path is not None
    if use_l0_sort:
        l0_kept_names = list(model.factor_names[0])
        sort_layer_name = model.layer_names[0]
        sort_names = _resolve_path_names(l0_path, l0_kept_names, 0)
        if len(sort_names) == 0:
            raise ValueError("l0_path must contain at least one factor.")
        sort_kept_names = l0_kept_names

    X_probs = np.asarray(model.adata.obsm[f"X_{sort_layer_name}_probs"], dtype=float)
    factor_pos = {name: idx for idx, name in enumerate(sort_kept_names)}
    path_cols = np.array([factor_pos[name] for name in sort_names], dtype=int)
    path_weights = X_probs[path_mask][:, path_cols]
    if np.all(np.sum(path_weights, axis=1) <= 1e-12):
        if use_l0_sort and hasattr(model, "logger"):
            model.logger.warning(
                "l0_path provided but selected cells have near-zero weight on this "
                "path. Falling back to layer %s factor_path ordering.",
                layer_idx,
            )
        X_probs_fb = np.asarray(model.adata.obsm[f"X_{layer_name}_probs"], dtype=float)
        factor_pos_fb = {name: idx for idx, name in enumerate(kept_names)}
        path_cols_fb = np.array([factor_pos_fb[name] for name in path_names], dtype=int)
        path_weights = X_probs_fb[path_mask][:, path_cols_fb]
    denom = np.maximum(np.sum(path_weights, axis=1), 1e-12)
    ranks = np.arange(path_weights.shape[1], dtype=float)
    progress = np.sum(path_weights * ranks[None, :], axis=1) / denom
    selected_cells = np.where(path_mask)[0]
    sorted_cells = selected_cells[np.argsort(progress)]
    t_path = model.adata[sorted_cells].copy()

    confident_sigs = get_confident_signatures(
        model,
        layer_idx=layer_idx,
        confidence_threshold=confidence_threshold,
        tau_quantile=confidence_tau_quantile,
        mc_samples_upper=confidence_mc_samples_upper,
        random_seed=confidence_random_seed,
        min_effect=min_effect,
        max_genes=genes_per_factor,
    )

    score_cols = [f"{name}_score" for name in path_names]
    missing_scores = [col for col in score_cols if col not in t_path.obs.columns]
    if len(missing_scores) > 0:
        raise KeyError(f"Score columns not found in adata.obs: {missing_scores}")

    block_matrices: List[Tuple[np.ma.MaskedArray, np.ma.MaskedArray]] = []
    block_labels: List[List[str]] = []
    block_has_genes: List[bool] = []
    for factor_name in path_names:
        block_rows: List[np.ndarray] = []
        labels: List[str] = []
        kinds: List[str] = []
        score_col = f"{factor_name}_score"
        score_vals = uniform_filter1d(
            np.asarray(t_path.obs[score_col].values, dtype=float), size=smoothing
        )
        block_rows.append(minmax_scale(score_vals))
        labels.append(factor_name)
        kinds.append("factor")

        genes = [
            g for g in confident_sigs.get(factor_name, []) if g in t_path.var_names
        ]
        for gene in genes:
            expr = t_path[:, [gene]].X
            if hasattr(expr, "toarray"):
                expr = expr.toarray()
            expr_vals = uniform_filter1d(
                np.asarray(expr, dtype=float).ravel(), size=smoothing
            )
            block_rows.append(minmax_scale(expr_vals))
            labels.append(f"  {gene}")
            kinds.append("gene")

        block_arr = np.vstack(block_rows)
        factor_rows = np.asarray([k == "factor" for k in kinds], dtype=bool)
        gene_rows = np.asarray([k == "gene" for k in kinds], dtype=bool)

        genes_masked = np.ma.array(
            block_arr, mask=np.broadcast_to(~gene_rows[:, None], block_arr.shape)
        )
        factors_masked = np.ma.array(
            block_arr, mask=np.broadcast_to(~factor_rows[:, None], block_arr.shape)
        )

        block_matrices.append((genes_masked, factors_masked))
        block_labels.append(labels)
        block_has_genes.append(bool(np.any(gene_rows)))

    if len(block_matrices) == 0:
        raise ValueError("No rows to plot. Check factor_path and confident signatures.")

    obs_keys: List[str] = []
    if annotation_obs_key is not None:
        if isinstance(annotation_obs_key, str):
            obs_keys = [annotation_obs_key]
        else:
            obs_keys = [str(k) for k in annotation_obs_key]
        if len(obs_keys) == 0:
            raise ValueError(
                "annotation_obs_key must be a non-empty string or sequence of strings."
            )

    obs_tracks: List[Dict[str, object]] = []
    for key in obs_keys:
        if key not in t_path.obs.columns:
            raise KeyError(f"{key} not found in adata.obs.")
        cat = pd.Categorical(t_path.obs[key])
        categories = cat.categories.tolist()
        uns_key = f"{key}_colors"
        if uns_key in t_path.uns:
            cat_to_color = dict(
                zip(
                    t_path.uns.get(f"{key}_categories", categories),
                    t_path.uns[uns_key],
                )
            )
        else:
            cmap_fb = plt.get_cmap("tab10", max(len(categories), 1))
            cat_to_color = {c: cmap_fb(i) for i, c in enumerate(categories)}
        rgb = np.array([mpl.colors.to_rgb(cat_to_color[c]) for c in t_path.obs[key]])
        obs_tracks.append(
            {
                "key": key,
                "categories": categories,
                "cat_to_color": cat_to_color,
                "rgb": rgb,
            }
        )

    nrows = len(obs_tracks) + len(block_matrices)
    height_ratios = [0.3] * len(obs_tracks)
    for labels in block_labels:
        height_ratios.append(len(labels) * gene_height)

    total_height = sum(height_ratios) + 1.5
    fig = plt.figure(figsize=(figwidth, total_height))
    gs = fig.add_gridspec(
        nrows=nrows,
        ncols=4,
        height_ratios=height_ratios,
        width_ratios=[figwidth - 1.6, 0.22, 0.24, 0.22],
        hspace=0.05 + 0.08 * block_spacing,
        wspace=colorbar_gap,
    )

    row = 0
    for obs in obs_tracks:
        ax_obs = fig.add_subplot(gs[row, 0])
        ax_obs_cb1 = fig.add_subplot(gs[row, 1])
        ax_obs_cb_gap = fig.add_subplot(gs[row, 2])
        ax_obs_cb2 = fig.add_subplot(gs[row, 3])
        ax_obs_cb1.axis("off")
        ax_obs_cb_gap.axis("off")
        ax_obs_cb2.axis("off")
        ax_obs.imshow(
            obs["rgb"][np.newaxis, :, :], aspect="auto", interpolation="nearest"
        )
        ax_obs.set_yticks([0])
        ax_obs.set_yticklabels([obs["key"]], fontsize=9)
        ax_obs.set_xticks([])
        legend_patches = [
            mpl.patches.Patch(color=obs["cat_to_color"][c], label=c)
            for c in obs["categories"]
        ]
        ax_obs.legend(
            handles=legend_patches,
            fontsize=8,
            frameon=False,
            loc="lower left",
            bbox_to_anchor=(0.0, 1.05),
            ncol=max(1, min(len(obs["categories"]), 8)),
            borderaxespad=0,
        )
        row += 1

    hm_row_start = row
    im_genes_ref = None
    im_factors_ref = None
    any_gene_rows = False
    for i, labels in enumerate(block_labels):
        ax_hm = fig.add_subplot(gs[hm_row_start + i, 0])
        genes_masked, factors_masked = block_matrices[i]

        im_genes = ax_hm.imshow(
            genes_masked, aspect="auto", cmap=heatmap_cmap, interpolation="nearest"
        )
        im_factors = ax_hm.imshow(
            factors_masked,
            aspect="auto",
            cmap=factor_heatmap_cmap,
            interpolation="nearest",
        )
        ax_hm.set_yticks(range(len(labels)))
        ax_hm.set_yticklabels(labels, fontsize=8)
        if i == len(block_labels) - 1:
            ax_hm.set_xlabel(xlabel, fontsize=10)
        ax_hm.set_xticks([])

        im_factors_ref = im_factors
        if block_has_genes[i]:
            im_genes_ref = im_genes
            any_gene_rows = True

        if i < len(block_labels) - 1:
            ax_hm.axhline(len(labels) - 0.5, color="white", linewidth=1.0, alpha=0.8)

    ax_hm_cb_gene = fig.add_subplot(gs[hm_row_start:, 1])
    ax_hm_cb_factor = fig.add_subplot(gs[hm_row_start:, 3])
    if any_gene_rows and im_genes_ref is not None:
        plt.colorbar(im_genes_ref, cax=ax_hm_cb_gene, label="Gene signal (row-scaled)")
        ax_hm_cb_gene.yaxis.labelpad = 10
        ax_hm_cb_gene.yaxis.label.set_size(8)
        ax_hm_cb_gene.tick_params(labelsize=7)
    else:
        ax_hm_cb_gene.axis("off")
    if im_factors_ref is not None:
        plt.colorbar(
            im_factors_ref, cax=ax_hm_cb_factor, label="Factor score (row-scaled)"
        )
        ax_hm_cb_factor.yaxis.labelpad = 10
        ax_hm_cb_factor.yaxis.label.set_size(8)
        ax_hm_cb_factor.tick_params(labelsize=7)
    else:
        ax_hm_cb_factor.axis("off")

    if save:
        plt.savefig(save, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
        return None
    return fig
