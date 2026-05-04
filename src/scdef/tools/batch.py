"""Batch-related utilities for scDEF."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from scdef.models._scdef import scDEF


def get_batch_specific_genes_from_gene_scale(
    model: "scDEF",
    *,
    eps: float = 1e-12,
    log_base: float = 2.0,
    reference: Literal["mean_other_batches", "global_mean"] = "mean_other_batches",
) -> pd.DataFrame:
    """Per-gene log-ratios of batch-specific ``gene_scale`` vs a reference profile.

    After fitting (and ``annotate_adata``), scDEF stores inferred positive gene
    scaling factors in ``model.pmeans["gene_scale"]`` with shape
    ``(n_batches, n_genes)`` when ``batch_key`` was set with at least two batches.
    Higher scale for a gene in a batch means the model explains more variance /
    signal for that gene in that batch (relative to the Gamma prior mean encoded
    in ``gene_ratio``).

    For each batch ``b``, this computes::

        log_ratio[g, b] = log( scale[b, g] + eps ) - log( ref[g] + eps )

    with ``log`` at the chosen base (default log2), and ``ref`` either the mean
    of the *other* batches at gene ``g`` (default) or the global mean across all
    batches at ``g``.

    Args:
        model: Fitted scDEF model whose ``pmeans['gene_scale']`` has shape
            ``(n_batches, n_genes)`` with ``n_batches >= 2``.
        eps: Small constant for numerical stability.
        log_base: Logarithm base (``2`` for log2, ``np.e`` for natural log).
        reference: ``mean_other_batches`` compares each batch to the mean of the
            remaining batches per gene. ``global_mean`` compares to the mean
            across all batches (same reference column for every batch).

    Returns:
        DataFrame indexed like ``model.adata.var_names``, one column per batch
        label in ``model.batches`` (length must match ``gene_scale`` batch
        dimension). Positive entries indicate relatively higher ``gene_scale`` in
        that batch vs the chosen reference.

    Raises:
        ValueError: If ``gene_scale`` has fewer than two batch rows or is missing.
    """
    if "gene_scale" not in model.pmeans:
        raise ValueError(
            "model.pmeans['gene_scale'] is missing; fit the model before calling "
            "this function."
        )

    G = np.asarray(model.pmeans["gene_scale"], dtype=float)
    if G.ndim == 1:
        G = G.reshape(1, -1)
    n_b, n_genes = G.shape
    if n_b < 2:
        raise ValueError(
            "gene_scale must have at least two batch rows to compute log-ratios. "
            "Use batch_key with at least two distinct batch labels in the training "
            "cells (or check that your subsample still contains multiple batches)."
        )
    if n_genes != int(model.adata.n_vars):
        raise ValueError(
            f"gene_scale second dimension ({n_genes}) does not match "
            f"adata.n_vars ({model.adata.n_vars})."
        )

    log_fn = np.log
    if float(log_base) == float(np.e):
        inv_log_base = 1.0
    else:
        inv_log_base = 1.0 / float(log_fn(float(log_base)))

    batches = [str(b) for b in model.batches]
    if len(batches) != n_b:
        raise ValueError(
            f"len(model.batches)={len(batches)} does not match gene_scale rows ({n_b})."
        )

    out = np.zeros((n_genes, n_b), dtype=float)
    Gp = np.clip(G, 0.0, None) + float(eps)

    if reference == "global_mean":
        ref_mean = np.mean(Gp, axis=0) + float(eps)
        for i in range(n_b):
            out[:, i] = log_fn(Gp[i] / ref_mean) * inv_log_base
    elif reference == "mean_other_batches":
        for i in range(n_b):
            others = np.delete(Gp, i, axis=0)
            ref = np.mean(others, axis=0) + float(eps)
            out[:, i] = log_fn(Gp[i] / ref) * inv_log_base
    else:
        raise ValueError(f"Unknown reference: {reference!r}")

    return pd.DataFrame(out, index=model.adata.var_names.copy(), columns=batches)
