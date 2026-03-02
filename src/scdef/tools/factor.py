import numpy as np
import pandas as pd
import scanpy as sc
from .hierarchy import get_hierarchy, compute_hierarchy_scores
from typing import Optional, Sequence, Dict, List, Tuple, Any, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from scdef.models._scdef import scDEF


def factor_diagnostics(model: "scDEF") -> None:
    # Keep layer 0 unfiltered, but use filtered factors on upper layers.
    res = compute_hierarchy_scores(
        model,
        use_filtered=False,
        filter_upper_layers=True,
    )
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


def set_factor_signatures(
    model: "scDEF",
    signatures: Optional[Dict[str, List[str]]] = None,
    top_genes: int = 10,
) -> Dict[str, List[str]]:
    if signatures is None:
        signatures = model.get_signatures_dict(top_genes=top_genes)
    model.adata.uns["factor_signatures"] = signatures
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
    model.adata.uns["factor_obs"]["technical"] = np.array(
        [False] * len(model.adata.uns["factor_obs"])
    )
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
        gene_order = np.argsort([np.where(var_names == gene)[0][0] for gene in ranking])
        reordered_idx = np.argsort([np.where(ranking == g)[0][0] for g in var_names])
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
        model.adata.uns["factor_obs"]["technical"] == True
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
