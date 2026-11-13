"""Colormaps used across scDEF's plots.

Factor scores (the weight of a cell or spot on a factor) and gene expression are drawn on different colormaps, so
that a reader can tell at a glance which of the two a plot shows.
"""
from typing import Optional

FACTOR_CMAP = "Spectral_r"
"""Colormap for factor scores (cell or spot weights on a factor), as used by Spectra (Kunes et al. 2023)."""

GENE_CMAP = "viridis"
"""Colormap for gene expression."""


def continuous_cmap(model, key: str) -> Optional[str]:
    """The colormap for a continuous colour key of ``model.adata``: :data:`FACTOR_CMAP` for a factor name,
    :data:`GENE_CMAP` for a gene, and None (the plotting library's default) otherwise."""
    factors = {str(f) for names in getattr(model, "factor_names", []) for f in names}
    if key in factors:
        return FACTOR_CMAP
    if key in set(model.adata.var_names):
        return GENE_CMAP
    return None
