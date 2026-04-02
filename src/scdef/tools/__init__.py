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
    compute_within_group_pairwise_dissimilarity,
    factor_diagnostics,
    get_obs_score_rankings,
    set_cell_entropies,
    set_confident_signatures,
    get_stored_confident_signatures,
    set_factor_signatures,
    get_confident_signatures,
    set_technical_factors,
    get_technical_signature,
    get_biological_signature,
    umap,
)
from .trajectory import multilevel_paga

__all__ = [
    "get_hierarchy",
    "make_hierarchies",
    "make_biological_hierarchy",
    "make_technical_hierarchy",
    "compute_within_group_pairwise_dissimilarity",
    "factor_diagnostics",
    "get_obs_score_rankings",
    "set_cell_entropies",
    "set_confident_signatures",
    "get_stored_confident_signatures",
    "set_factor_signatures",
    "get_confident_signatures",
    "set_technical_factors",
    "get_technical_signature",
    "get_biological_signature",
    "compute_hierarchy_scores",
    "umap",
    "multilevel_paga",
]
