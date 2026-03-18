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
        signatures = {}
        for layer_idx in range(model.n_layers):
            layer_sigs = get_confident_signatures(
                model,
                layer_idx=layer_idx,
                max_genes=top_genes,
            )
            signatures.update(layer_sigs)
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
    top_layer_idx = model.n_layers - 1
    signatures_dict = get_confident_signatures(
        model,
        layer_idx=top_layer_idx,
        max_genes=top_genes,
    )
    for tf in technical_factors:
        signatures_dict.pop(tf, None)
    top_factor = f"{model.layer_names[top_layer_idx]}_0"
    signature = signatures_dict.get(top_factor, [])
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


from .trajectory import multilevel_paga, plot_trajectory_heatmap  # noqa: E402,F401
