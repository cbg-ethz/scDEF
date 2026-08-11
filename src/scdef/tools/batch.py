"""Batch-related utilities for scDEF."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.stats import rankdata

if TYPE_CHECKING:
    from scdef.models._scdef import scDEF


def _gene_scale_log_ratios(
    gene_scale: np.ndarray,
    batch_labels: Sequence[str],
    gene_index: pd.Index,
    *,
    eps: float = 1e-12,
    log_base: float = 2.0,
    reference: Literal["mean_other_batches", "global_mean"] = "mean_other_batches",
) -> pd.DataFrame:
    """Core per-gene log-ratio computation shared by the public entry points.

    Kept private so :func:`get_batch_specific_genes_from_gene_scale` (which
    reads the array off a model) and
    :func:`get_factor_batch_gene_scale_affinity` (which may be handed the array
    directly) cannot drift apart.
    """
    G = np.asarray(gene_scale, dtype=float)
    if G.ndim == 1:
        G = G.reshape(1, -1)
    if G.ndim != 2:
        raise ValueError(
            f"gene_scale must be 1d or 2d, got array with shape {G.shape}."
        )
    n_b, n_genes = G.shape
    if n_b < 2:
        raise ValueError(
            "gene_scale must have at least two batch rows to compute log-ratios. "
            "Use batch_key with at least two distinct batch labels in the training "
            "cells (or check that your subsample still contains multiple batches). "
            "Note that a model built with batch_gene_scale=False deliberately has a "
            "single shared gene_scale row, so per-batch gene log-ratios do not exist "
            "for it."
        )
    if n_genes != len(gene_index):
        raise ValueError(
            f"gene_scale second dimension ({n_genes}) does not match "
            f"adata.n_vars ({len(gene_index)})."
        )

    batches = [str(b) for b in batch_labels]
    if len(batches) != n_b:
        raise ValueError(
            f"len(model.batches)={len(batches)} does not match gene_scale rows ({n_b})."
        )

    log_fn = np.log
    if float(log_base) == float(np.e):
        inv_log_base = 1.0
    else:
        inv_log_base = 1.0 / float(log_fn(float(log_base)))

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

    return pd.DataFrame(out, index=gene_index.copy(), columns=batches)


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

    return _gene_scale_log_ratios(
        model.pmeans["gene_scale"],
        list(model.batches),
        model.adata.var_names,
        eps=eps,
        log_base=log_base,
        reference=reference,
    )


# Column names added on top of the one-column-per-batch block.
TOP_BATCH_COL = "top_batch"
TOP_SCORE_COL = "top_score"
ABS_TOP_SCORE_COL = "abs_top_score"


def _spearman_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Spearman correlation of every row of ``A`` with every column of ``B``.

    ``A`` is ``(n_factors, n_genes)``, ``B`` is ``(n_genes, n_batches)``; the
    result is ``(n_factors, n_batches)``. Ties get average ranks, matching
    ``scipy.stats.spearmanr``. A constant vector (zero rank variance) yields
    ``NaN`` rather than an error.
    """
    Ar = rankdata(np.asarray(A, dtype=float), axis=1)
    Br = rankdata(np.asarray(B, dtype=float), axis=0)
    Ac = Ar - Ar.mean(axis=1, keepdims=True)
    Bc = Br - Br.mean(axis=0, keepdims=True)
    An = np.linalg.norm(Ac, axis=1, keepdims=True)
    Bn = np.linalg.norm(Bc, axis=0, keepdims=True)
    flat_rows = (An == 0.0).ravel()
    flat_cols = (Bn == 0.0).ravel()
    # Normalise before the product rather than after, so no intermediate is
    # large. `over` is muted because the BLAS matmul path can raise spurious
    # floating-point flags under errstate.
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        out = (Ac / np.where(An == 0.0, 1.0, An)) @ (Bc / np.where(Bn == 0.0, 1.0, Bn))
    out[flat_rows, :] = np.nan
    out[:, flat_cols] = np.nan
    return np.clip(out, -1.0, 1.0)


def _resolve_reference_log_ratios(
    model: "scDEF",
    reference: Union["scDEF", pd.DataFrame, np.ndarray, None],
    *,
    reference_batches: Optional[Sequence[str]],
    eps: float,
    log_base: float,
    reference_mode: Literal["mean_other_batches", "global_mean"],
) -> pd.DataFrame:
    """Turn whatever the caller supplied into a genes-by-batches log-ratio frame."""
    var_names = model.adata.var_names

    if reference is None:
        uns = getattr(model.adata, "uns", {})
        if "reference_gene_scale_log_ratios" in uns:
            reference = uns["reference_gene_scale_log_ratios"]
        elif "reference_gene_scale" in uns:
            reference = np.asarray(uns["reference_gene_scale"])
            if reference_batches is None:
                reference_batches = uns.get("reference_gene_scale_batches")
        else:
            raise ValueError(
                "No reference gene_scale information available. "
                "decompose_batch_effects records the reference fit's per-batch "
                "gene_scale in adata.uns['reference_gene_scale'], so this model "
                "was either decomposed before that was added, or its reference "
                "had a single shared gene_scale row (no batch_key, or "
                "batch_gene_scale=False) and so carries no contrast. Pass the "
                "reference scDEF model, the DataFrame returned by "
                "get_batch_specific_genes_from_gene_scale, or the reference's "
                "pmeans['gene_scale'] array as `reference`."
            )

    # A precomputed genes-by-batches log-ratio table.
    if isinstance(reference, pd.DataFrame):
        ratios = reference
        if len(ratios.index) != len(var_names) or not np.array_equal(
            np.asarray(ratios.index, dtype=object),
            np.asarray(var_names, dtype=object),
        ):
            raise ValueError(
                "Reference log-ratio DataFrame is not on the model's gene axis: "
                f"index has {len(ratios.index)} entries and model.adata.var_names "
                f"has {len(var_names)}; they must be identical and in the same "
                "order. Both must come from the same gene selection."
            )
        if ratios.shape[1] < 2:
            raise ValueError(
                "Reference log-ratio DataFrame has "
                f"{ratios.shape[1]} batch column(s); at least two batches are "
                "needed for a per-batch gene_scale contrast to exist. A model "
                "built with batch_gene_scale=False (or without batch_key) has a "
                "single shared gene_scale row and no contrast to score against."
            )
        return ratios.astype(float)

    # A reference scDEF model: read gene_scale off it, checking the gene axis.
    if not isinstance(reference, np.ndarray):
        ref_model = reference
        if not hasattr(ref_model, "pmeans") or not hasattr(ref_model, "adata"):
            raise TypeError(
                "`reference` must be a fitted scDEF model, a genes-by-batches "
                "DataFrame of log-ratios, or a (n_batches, n_genes) gene_scale "
                f"array; got {type(reference)!r}."
            )
        ref_vars = ref_model.adata.var_names
        if len(ref_vars) != len(var_names) or not np.array_equal(
            np.asarray(ref_vars, dtype=object),
            np.asarray(var_names, dtype=object),
        ):
            raise ValueError(
                "Reference and decomposed models are not on the same gene axis: "
                f"reference adata is {ref_model.adata.shape} with "
                f"{len(ref_vars)} var_names and the decomposed adata is "
                f"{model.adata.shape} with {len(var_names)} var_names. "
                "var_names must be identical and in the same order (a common "
                "cause is re-attaching a differently filtered AnnData with "
                "scDEF.load(path, adata=...); load the saved model without an "
                "`adata` argument so it keeps its own aligned one)."
            )
        if "gene_scale" not in ref_model.pmeans:
            raise ValueError(
                "reference.pmeans['gene_scale'] is missing; the reference model "
                "must be fitted."
            )
        gs = np.asarray(ref_model.pmeans["gene_scale"], dtype=float)
        if gs.ndim == 1:
            gs = gs.reshape(1, -1)
        if gs.shape[1] != len(var_names):
            raise ValueError(
                f"Reference gene_scale has shape {gs.shape} but the decomposed "
                f"model has {len(var_names)} genes; the second dimension must "
                "match."
            )
        if gs.shape[0] < 2:
            raise ValueError(
                f"Reference gene_scale has shape {gs.shape}: a single batch row, "
                "so there is no per-batch contrast to score factors against. "
                "Pass the REFERENCE fit (made with batch_key and "
                "batch_gene_scale=True), not the decomposed one — the decomposed "
                "model deliberately has a single shared gene_scale row."
            )
        return get_batch_specific_genes_from_gene_scale(
            ref_model,
            eps=eps,
            log_base=log_base,
            reference=reference_mode,
        )

    # A raw (n_batches, n_genes) gene_scale array.
    gs = np.asarray(reference, dtype=float)
    if gs.ndim == 1:
        gs = gs.reshape(1, -1)
    if gs.ndim != 2 or gs.shape[1] != len(var_names):
        raise ValueError(
            f"Reference gene_scale array has shape {gs.shape} but the decomposed "
            f"model has {len(var_names)} genes; expected (n_batches, n_genes) "
            f"i.e. (*, {len(var_names)}). Note that a *precomputed log-ratio* "
            "table is genes-by-batches and must be passed as a DataFrame indexed "
            "by var_names."
        )
    if gs.shape[0] < 2:
        raise ValueError(
            f"Reference gene_scale array has shape {gs.shape}: a single batch "
            "row, so there is no per-batch contrast to score factors against. "
            "Pass the reference fit's per-batch gene_scale (n_batches >= 2)."
        )
    labels = (
        [str(b) for b in reference_batches]
        if reference_batches is not None
        else [f"batch_{i}" for i in range(gs.shape[0])]
    )
    return _gene_scale_log_ratios(
        gs,
        labels,
        var_names,
        eps=eps,
        log_base=log_base,
        reference=reference_mode,
    )


def get_factor_batch_gene_scale_affinity(
    model: "scDEF",
    reference: Union["scDEF", pd.DataFrame, np.ndarray, None] = None,
    *,
    reference_batches: Optional[Sequence[str]] = None,
    top_n_genes: Optional[int] = None,
    eps: float = 1e-12,
    log_base: float = 2.0,
    reference_mode: Literal["mean_other_batches", "global_mean"] = "mean_other_batches",
) -> pd.DataFrame:
    r"""Score each layer-0 factor by its affinity for the reference fit's per-batch ``gene_scale`` contrast.

    Named to match :func:`get_batch_specific_genes_from_gene_scale`, which
    produces the contrast this reads: that function answers *which genes* the
    reference fit's per-batch ``gene_scale`` term singled out, this one answers
    *which factors of the decomposed fit look like those genes* — hence
    "factor ... gene_scale affinity".

    **What it measures.** A reference fit made with ``batch_key`` (and
    ``batch_gene_scale=True``) carries a per-batch, per-gene multiplier,
    ``pmeans['gene_scale']`` of shape ``(n_batches, n_genes)``, which soaks up
    the gene-level differences between batches so the hierarchy does not have
    to. :meth:`scDEF.decompose_batch_effects` then freezes the batch-corrected
    upper hierarchy and re-learns layer 0 *without* per-batch gene scales, so
    whatever that term was holding has to reappear in the layer-0 factors. This
    function measures how much of it landed in each factor: for every batch
    ``b`` it takes the per-gene log-ratio
    ``log(gene_scale[b, g] / ref[g])`` from
    :func:`get_batch_specific_genes_from_gene_scale`, and for every kept layer-0
    factor ``k`` computes the **Spearman rank correlation** between that vector
    and the factor's gene loadings ``pmeans['L0W'][k, :]``, over **all** genes.

    Rank correlation over all genes is deliberate. The loadings and the
    log-ratios live on incomparable scales and both have heavy tails, so a
    Pearson correlation would be decided by a handful of genes; and restricting
    to a factor's top genes throws away the (informative) fact that a factor
    *down*-weights the batch-associated genes. ``top_n_genes`` is offered for
    exploration but changes the estimand — it conditions on the factor's own
    top genes, so scores are no longer comparable across factors with
    different loading profiles.

    **This is gene-side evidence.** Every other batch diagnostic in ``scdef``
    is cell-side — ``batch_purity`` / ``batch_purity_soft`` and the
    ``batch_split_*`` columns of :func:`factor_diagnostics`. Those ask *which
    cells* score on a factor and how they are distributed over batches. This one
    never looks
    at a cell: it asks whether the factor's *gene programme* is the one the
    reference fit's per-batch term absorbed. A factor can be perfectly mixed
    across batches at the cell level and still be built out of exactly those
    genes, and vice versa.

    .. warning::

       **This does not separate technical from biological factors, and must not
       be used as if it did.** The per-batch ``gene_scale`` absorbs *whatever*
       differs between batches at the gene level, and what that is depends
       entirely on what the batch key encodes. Two worked cases, both scored
       with this function:

       * **Kang CTRL/STIM (``batch_key='stim'``)** — the reference
         ``gene_scale`` learned the interferon response: a 16-gene ISG panel
         sits at median rank 48 of 4000 in the STIM log-ratio (chance is 2000),
         with *IFIT1* #8, *ISG15* #16 and *CXCL10* #18 at a 62.3x per-batch
         ratio. So here the top-scoring factor is the ISG factor — which is
         **biology**, the entire point of the experiment, and must be kept.
       * **pbmcs2b (``batch_key='Experiment'``, two runs of one donor)** — the
         same term captured platelet/ambient genes, lncRNA and a
         dissociation/immediate-early block, so the top-scoring factor is a
         stress-response factor — an **artefact** to remove.

       Identical statistic, opposite verdicts. Read the score as "how strongly
       this factor matches the gene programme the reference fit's per-batch
       scale absorbed" — a pointer to the batch-associated programme. Inspect
       the flagged factor's genes (``model.adata.uns['L0_signatures']``,
       :func:`get_confident_signatures`) and decide from the experimental
       design. A high score on a condition-like batch key is expected and is
       not grounds for dropping the factor.

    Args:
        model: the **decomposed** scDEF model (after
            :meth:`scDEF.decompose_batch_effects`) whose layer-0 factors are
            scored. Its ``pmeans['L0W']`` is full width; only the factors in
            ``model.factor_lists[0]`` are scored, and they are labelled with
            ``model.factor_names[0]``.
        reference: where the per-batch ``gene_scale`` contrast comes from. One
            of

            * a fitted **reference scDEF model** (``batch_key`` set, ``gene_scale``
              of shape ``(n_batches, n_genes)``) — the log-ratios are computed
              from it with :func:`get_batch_specific_genes_from_gene_scale`;
            * a **DataFrame** of precomputed log-ratios, genes-by-batches,
              indexed by ``var_names`` — exactly what that function returns;
            * a raw **array** of ``gene_scale``, shape ``(n_batches, n_genes)``,
              matching ``pmeans['gene_scale']``.

            ``None`` (default) looks for ``adata.uns['reference_gene_scale_log_ratios']``
            or ``adata.uns['reference_gene_scale']`` on the decomposed model,
            which :meth:`scDEF.decompose_batch_effects` records from the fit it
            decomposed, and raises if neither is present. Supplying it explicitly
            is only needed for a model decomposed before that record existed, or
            to score against a different reference.
        reference_batches: batch labels for the raw-array form of ``reference``
            (ignored otherwise). Defaults to ``batch_0``, ``batch_1``, ...
        top_n_genes: if given, restrict each factor's correlation to that
            factor's own top-``N`` genes by loading. Default ``None`` uses every
            gene, which is the intended statistic; see above.
        eps: numerical-stability constant passed through to the log-ratios.
        log_base: log base for the log-ratios. Irrelevant to the result — a rank
            correlation is invariant to any monotone rescaling — and kept only
            so the underlying table matches what you would get from
            :func:`get_batch_specific_genes_from_gene_scale`.
        reference_mode: ``mean_other_batches`` (default) or ``global_mean``,
            passed straight through. With exactly two batches the two columns
            are mirror images either way, so the two scores of a factor are
            exact negatives of each other.

    Returns:
        A DataFrame indexed by ``model.factor_names[0]`` (one row per kept
        layer-0 factor), with one column per batch label holding the Spearman
        correlation of that factor's loadings with that batch's gene-scale
        log-ratio, plus

        * ``top_batch`` — the batch whose column is largest for that factor;
        * ``top_score`` — that correlation (negative only if the factor is
          anti-correlated with *every* batch's contrast);
        * ``abs_top_score`` — its magnitude, for ranking regardless of sign.

        ``attrs['n_genes_used']``, ``attrs['reference_mode']`` and
        ``attrs['top_n_genes']`` record the settings. Rows are in
        ``factor_names[0]`` order, not sorted by score.

    Raises:
        ValueError: the reference and the model disagree on the gene axis (both
            shapes are named in the message); the reference ``gene_scale`` has a
            single batch row, so no contrast exists; ``model.pmeans['L0W']`` is
            missing or its gene dimension does not match ``adata.n_vars``; or
            ``top_n_genes`` is not positive.
        TypeError: ``reference`` is not a model, DataFrame or array.

    Example:
        >>> decomposed = scdef.scDEF.load("ifn_batch_model")
        >>> ref = scdef.scDEF.load("ifn_model")          # fitted with batch_key='stim'
        >>> aff = scdef.tl.get_factor_batch_gene_scale_affinity(decomposed, ref)
        >>> aff.sort_values("abs_top_score", ascending=False).head()
        >>> # then LOOK at the genes before concluding anything:
        >>> decomposed.adata.uns["L0_signatures"][aff["abs_top_score"].idxmax()][:10]
    """
    if "L0W" not in model.pmeans:
        raise ValueError(
            "model.pmeans['L0W'] is missing; this needs a fitted (decomposed) " "model."
        )
    W_full = np.asarray(model.pmeans["L0W"], dtype=float)
    if W_full.ndim != 2:
        raise ValueError(f"pmeans['L0W'] must be 2d, got shape {W_full.shape}.")
    n_vars = int(model.adata.n_vars)
    if W_full.shape[1] != n_vars:
        raise ValueError(
            f"pmeans['L0W'] has shape {W_full.shape} but the model's adata has "
            f"{n_vars} genes; the second dimension must match adata.n_vars."
        )

    ratios = _resolve_reference_log_ratios(
        model,
        reference,
        reference_batches=reference_batches,
        eps=eps,
        log_base=log_base,
        reference_mode=reference_mode,
    )

    kept = np.asarray(model.factor_lists[0], dtype=int)
    if kept.size == 0:
        raise ValueError("The model has no kept layer-0 factors to score.")
    if np.any(kept < 0) or np.any(kept >= W_full.shape[0]):
        raise ValueError(
            f"model.factor_lists[0] indexes outside pmeans['L0W'] "
            f"(shape {W_full.shape}); the model's factor lists and posterior "
            "means are out of sync."
        )
    W = W_full[kept, :]
    names = list(model.factor_names[0])
    if len(names) != kept.size:
        names = [f"L0_{i}" for i in range(kept.size)]

    R = ratios.to_numpy(dtype=float)
    batch_cols = [str(c) for c in ratios.columns]

    if top_n_genes is None:
        scores = _spearman_matrix(W, R)
        n_genes_used = int(W.shape[1])
    else:
        n = int(top_n_genes)
        if n <= 0:
            raise ValueError(f"top_n_genes must be positive, got {top_n_genes!r}.")
        n = min(n, W.shape[1])
        scores = np.full((W.shape[0], R.shape[1]), np.nan, dtype=float)
        for i in range(W.shape[0]):
            idx = np.argsort(-W[i, :], kind="stable")[:n]
            scores[i, :] = _spearman_matrix(W[i : i + 1, idx], R[idx, :])[0]
        n_genes_used = n

    out = pd.DataFrame(scores, index=pd.Index(names, name="factor"), columns=batch_cols)
    with np.errstate(invalid="ignore"):
        all_nan = np.all(~np.isfinite(scores), axis=1)
        best = np.full(scores.shape[0], -1, dtype=int)
        best[~all_nan] = np.nanargmax(scores[~all_nan], axis=1)
    out[TOP_BATCH_COL] = [batch_cols[j] if j >= 0 else "" for j in best]
    out[TOP_SCORE_COL] = [
        scores[i, j] if j >= 0 else np.nan for i, j in enumerate(best)
    ]
    out[ABS_TOP_SCORE_COL] = np.abs(out[TOP_SCORE_COL].to_numpy(dtype=float))
    out.attrs["n_genes_used"] = n_genes_used
    out.attrs["reference_mode"] = str(reference_mode)
    out.attrs["top_n_genes"] = top_n_genes
    out.attrs["batch_columns"] = batch_cols
    return out
