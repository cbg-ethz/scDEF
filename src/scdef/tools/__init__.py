"""Tooling utilities for scDEF.
"""

from .hierarchy import (
    get_hierarchy,
    make_hierarchies,
    make_biological_hierarchy,
    make_technical_hierarchy,
    compute_hierarchy_scores,
)
from .factor import (
    assign_confident,
    compute_within_group_pairwise_dissimilarity,
    factor_diagnostics,
    get_obs_score_rankings,
    get_obs_value_specific_factors,
    set_cell_entropies,
    set_confident_signatures,
    get_stored_confident_signatures,
    set_factor_signatures,
    get_confident_signatures,
    set_technical_factors,
    get_technical_signature,
    get_biological_signature,
    gsea,
    umap,
    multilayer_umap,
)
from .trajectory import (
    multilevel_paga,
    build_differentiation_paths,
    build_transition_paths,
    score_paths,
)

__all__ = [
    "assign_confident",
    "multilayer_umap",
    "get_hierarchy",
    "make_hierarchies",
    "make_biological_hierarchy",
    "make_technical_hierarchy",
    "compute_within_group_pairwise_dissimilarity",
    "factor_diagnostics",
    "get_obs_score_rankings",
    "get_obs_value_specific_factors",
    "set_cell_entropies",
    "set_confident_signatures",
    "get_stored_confident_signatures",
    "set_factor_signatures",
    "get_confident_signatures",
    "set_technical_factors",
    "get_technical_signature",
    "get_biological_signature",
    "gsea",
    "compute_hierarchy_scores",
    "umap",
    "build_differentiation_paths",
    "build_transition_paths",
    "score_paths",
    "multilevel_paga",
]
