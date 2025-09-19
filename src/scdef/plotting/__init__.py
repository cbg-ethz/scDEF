"""Plotting utilities for scDEF.

This module provides direct access to plotting functions, allowing users to call
scdef.pl.plot_qc(model) instead of model.plot.plot_qc().
"""

from .graph import make_graph
from .factors import (
    plot_obs_factor_dotplot,
    plot_multilevel_paga,
    plot_layers_obs,
    plot_pathway_scores,
    plot_signatures_scores,
    plot_obs_scores,
    plot_umaps,
    plot_factors_bars,
    plot_cell_entropies,
    plot_factor_genes,
    plot_factor_gini,
)
from .qc import plot_qc, plot_scales, plot_scale, plot_brd, plot_gini_brd, plot_loss

__all__ = [
    "make_graph",
    "plot_obs_factor_dotplot",
    "plot_multilevel_paga",
    "plot_layers_obs",
    "plot_pathway_scores",
    "plot_signatures_scores",
    "plot_obs_scores",
    "plot_umaps",
    "plot_factors_bars",
    "plot_cell_entropies",
    "plot_factor_genes",
    "plot_factor_gini",
    "plot_qc",
    "plot_scales",
    "plot_scale",
    "plot_brd",
    "plot_gini_brd",
    "plot_loss",
]
