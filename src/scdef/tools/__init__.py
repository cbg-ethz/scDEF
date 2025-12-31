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
    set_factor_signatures,
    set_technical_factors,
    get_technical_signature,
    get_biological_signature,
)

__all__ = [
    "get_hierarchy",
    "make_hierarchies",
    "make_biological_hierarchy",
    "make_technical_hierarchy",
    "set_factor_signatures",
    "set_technical_factors",
    "get_technical_signature",
    "get_biological_signature",
    "compute_hierarchy_scores",
]
