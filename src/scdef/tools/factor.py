import numpy as np


def set_factor_signatures(model, signatures=None, top_genes=10):
    if signatures is None:
        signatures = model.get_signatures_dict(top_genes=top_genes)
    model.adata.uns["factor_signatures"] = signatures
    return signatures


def set_technical_factors(model, factors=None):
    """Set the technical factors of the model."""
    # in model.adata.uns["factor_obs"], annotate as technical or not.
    model.adata.uns["factor_obs"]["technical"] = False
    model.adata.uns["factor_obs"]["technical"].loc[factors] = True


def __build_consensus_signature(var_names, gene_scores_array, sizes_array):
    sizes_array = sizes_array / np.sum(sizes_array)
    avg_ranks = np.sum(sizes_array[:, None] * gene_scores_array, axis=0)
    idx_sorted = np.argsort(avg_ranks)[::-1]
    consensus = var_names[idx_sorted].tolist()
    consensus_scores = avg_ranks[idx_sorted]
    return consensus, consensus_scores


def get_technical_signature(model, top_genes=10, return_scores=False):
    hierarchy = model.adata.uns["technical_hierarchy"]
    _, gene_scores = model.get_rankings(
        layer_idx=0,
        genes=True,
        return_scores=True,
    )
    relevances = model.get_relevances_dict()
    children = hierarchy["tech_top"]
    gene_scores = np.array(
        [gene_scores[child] / np.max(gene_scores[child]) for child in children]
    )
    children_sizes = np.array([relevances[child] for child in children])
    consensus_signature, consensus_scores = __build_consensus_signature(
        model.adata.var_names, gene_scores, children_sizes
    )
    if return_scores:
        return consensus_signature[:top_genes], consensus_scores[:top_genes]
    return consensus_signature[:top_genes]


def get_biological_signature(model, top_genes=10):
    # Get the top signature
    technical_factors = model.adata.uns["factor_obs"][
        model.adata.uns["factor_obs"]["technical"] == True
    ].index.tolist()
    signatures_dict = model.get_signatures_dict(
        top_genes=top_genes, drop_factors=technical_factors
    )
    signature = signatures_dict[f"{model.layer_names[model.n_layers - 1]}_0"]
    return signature
