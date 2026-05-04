"""Plotting utilities for scDEF.
"""

from .graph import make_graph, technical_hierarchy, biological_hierarchy
from .factors import (
    obs_factor_dotplot,
    layers_obs,
    pathway_scores,
    signatures_scores,
    obs_scores,
    continuous_obs_scores,
    umap,
    factors_bars,
    cell_entropies,
    within_group_pairwise_dissimilarity,
    factor_genes,
    factor_gini,
    factor_gene_uncertainty_boxplot,
)
from .trajectory import (
    multilevel_paga,
    plot_trajectory_heatmap,
    plot_path_trajectory_heatmap,
    path_embedding,
)
from .qc import (
    qc,
    scales,
    scale,
    relevance,
    gini_brd,
    loss,
    factor_diagnostics,
)

__all__ = [
    "make_graph",
    "technical_hierarchy",
    "biological_hierarchy",
    "obs_factor_dotplot",
    "multilevel_paga",
    "plot_trajectory_heatmap",
    "plot_path_trajectory_heatmap",
    "path_embedding",
    "layers_obs",
    "pathway_scores",
    "signatures_scores",
    "obs_scores",
    "continuous_obs_scores",
    "umap",
    "factors_bars",
    "cell_entropies",
    "within_group_pairwise_dissimilarity",
    "factor_genes",
    "factor_gini",
    "factor_gene_uncertainty_boxplot",
    "qc",
    "scales",
    "scale",
    "relevance",
    "gini_brd",
    "loss",
    "factor_diagnostics",
]
