"""Tests for get_factor_batch_gene_scale_affinity.

Nothing is fitted anywhere in this file: both models are constructed and then
have their posterior means hand-set, so the planted relationship between a
factor's gene loadings and the reference fit's per-batch ``gene_scale``
contrast is exactly known.
"""

import numpy as np
import pandas as pd
import pytest

anndata = pytest.importorskip("anndata")
scd = pytest.importorskip("scdef")


N_GENES = 200
N_CELLS = 60
LAYER_SIZES = [4, 2]

# The planted per-gene batch contrast, in log space. Batch "A" gets exp(+v),
# batch "B" gets exp(-v), so the log-ratio of A against the other batch is a
# strictly increasing function of v.
_RNG = np.random.default_rng(0)
CONTRAST = _RNG.normal(0.0, 1.0, size=N_GENES)


def _adata(n_cells=N_CELLS, n_genes=N_GENES, batches=None, gene_prefix="gene"):
    rng = np.random.default_rng(3)
    X = rng.poisson(1.0, size=(n_cells, n_genes)).astype(float)
    ad = anndata.AnnData(X)
    ad.obs_names = [f"cell{i}" for i in range(n_cells)]
    ad.var_names = [f"{gene_prefix}{i}" for i in range(n_genes)]
    if batches is not None:
        ad.obs["batch"] = np.asarray(batches, dtype=str)
    return ad


def _reference_model(gene_scale, n_genes=N_GENES, gene_prefix="gene"):
    """A reference-like scDEF carrying a hand-set per-batch ``gene_scale``."""
    batches = np.array(["A"] * (N_CELLS // 2) + ["B"] * (N_CELLS - N_CELLS // 2))
    ad = _adata(n_genes=n_genes, batches=batches, gene_prefix=gene_prefix)
    model = scd.scDEF(ad, layer_sizes=list(LAYER_SIZES), batch_key="batch", seed=0)
    model.pmeans["gene_scale"] = np.asarray(gene_scale, dtype=float)
    return model


def _decomposed_model(W0, n_genes=N_GENES, gene_prefix="gene"):
    """A decomposed-like scDEF with hand-set layer-0 gene loadings."""
    ad = _adata(n_genes=n_genes, gene_prefix=gene_prefix)
    model = scd.scDEF(ad, layer_sizes=list(LAYER_SIZES), seed=0)
    model.factor_lists = [np.arange(s, dtype=int) for s in LAYER_SIZES]
    model.set_factor_names()
    model.pmeans["L0W"] = np.asarray(W0, dtype=float)
    return model


def _planted_pair():
    """Reference with the planted contrast, decomposed model with 4 factors.

    * ``L0_0`` loads monotonically with the contrast   -> score near +1 for "A"
    * ``L0_1`` loads monotonically against it          -> score near +1 for "B"
    * ``L0_2`` is an unrelated random programme        -> score near 0
    * ``L0_3`` is flat                                 -> NaN (no rank variance)
    """
    base = np.full(N_GENES, 2.0)
    gene_scale = np.vstack([base * np.exp(CONTRAST), base * np.exp(-CONTRAST)])

    rng = np.random.default_rng(11)
    W0 = np.zeros((LAYER_SIZES[0], N_GENES))
    # A monotone (non-linear) transform of the contrast: Spearman is exactly 1,
    # Pearson would not be.
    W0[0] = np.exp(CONTRAST)
    W0[1] = np.exp(-CONTRAST)
    W0[2] = rng.gamma(1.0, 1.0, size=N_GENES)
    W0[3] = np.full(N_GENES, 0.5)
    return _reference_model(gene_scale), _decomposed_model(W0)


def test_planted_factor_scores_near_one_and_unrelated_near_zero():
    ref, dec = _planted_pair()
    aff = scd.tl.get_factor_batch_gene_scale_affinity(dec, ref)

    assert list(aff.index) == ["L0_0", "L0_1", "L0_2", "L0_3"]
    assert list(aff.columns[:2]) == ["A", "B"]
    assert aff.attrs["n_genes_used"] == N_GENES

    # Planted match: an exactly monotone relationship -> Spearman == 1.
    assert aff.loc["L0_0", "A"] == pytest.approx(1.0, abs=1e-12)
    assert aff.loc["L0_0", "B"] == pytest.approx(-1.0, abs=1e-12)
    assert aff.loc["L0_0", "top_batch"] == "A"
    assert aff.loc["L0_0", "top_score"] == pytest.approx(1.0, abs=1e-12)
    assert aff.loc["L0_0", "abs_top_score"] == pytest.approx(1.0, abs=1e-12)

    # Mirror-image factor: same magnitude, other batch.
    assert aff.loc["L0_1", "B"] == pytest.approx(1.0, abs=1e-12)
    assert aff.loc["L0_1", "top_batch"] == "B"

    # Unrelated programme: near zero.
    assert abs(aff.loc["L0_2", "A"]) < 0.15
    assert aff.loc["L0_2", "abs_top_score"] < 0.15

    # The planted factor must outrank the unrelated one by a wide margin.
    assert aff.loc["L0_0", "abs_top_score"] > 5 * aff.loc["L0_2", "abs_top_score"]

    # Flat loadings have no ranks to correlate: NaN, not an error, and no
    # spurious top_batch.
    assert np.isnan(aff.loc["L0_3", "A"])
    assert np.isnan(aff.loc["L0_3", "top_score"])
    assert aff.loc["L0_3", "top_batch"] == ""


def test_only_kept_factors_are_scored_and_named_contiguously():
    ref, dec = _planted_pair()
    # Drop the mirror factor: L0W stays full width, factor_lists shrinks.
    dec.factor_lists = [np.array([0, 2, 3]), np.arange(LAYER_SIZES[1])]
    dec.set_factor_names()
    aff = scd.tl.get_factor_batch_gene_scale_affinity(dec, ref)
    assert list(aff.index) == ["L0_0", "L0_1", "L0_2"]
    # Contiguous "L0_1" is original factor 2, the unrelated programme.
    assert aff.loc["L0_0", "A"] == pytest.approx(1.0, abs=1e-12)
    assert abs(aff.loc["L0_1", "A"]) < 0.15


def test_dataframe_and_array_reference_match_the_model_reference():
    ref, dec = _planted_pair()
    from_model = scd.tl.get_factor_batch_gene_scale_affinity(dec, ref)

    ratios = scd.tl.get_batch_specific_genes_from_gene_scale(ref)
    from_frame = scd.tl.get_factor_batch_gene_scale_affinity(dec, ratios)

    from_array = scd.tl.get_factor_batch_gene_scale_affinity(
        dec, np.asarray(ref.pmeans["gene_scale"]), reference_batches=["A", "B"]
    )

    for other in (from_frame, from_array):
        pd.testing.assert_frame_equal(from_model, other, check_like=False)


def test_array_reference_without_labels_gets_positional_names():
    ref, dec = _planted_pair()
    aff = scd.tl.get_factor_batch_gene_scale_affinity(
        dec, np.asarray(ref.pmeans["gene_scale"])
    )
    assert list(aff.columns[:2]) == ["batch_0", "batch_1"]


def test_top_n_genes_is_opt_in_and_restricts_the_gene_set():
    ref, dec = _planted_pair()
    aff = scd.tl.get_factor_batch_gene_scale_affinity(dec, ref, top_n_genes=20)
    assert aff.attrs["top_n_genes"] == 20
    assert aff.attrs["n_genes_used"] == 20
    # Restricted to its own top genes the planted factor is still monotone.
    assert aff.loc["L0_0", "A"] == pytest.approx(1.0, abs=1e-12)

    with pytest.raises(ValueError, match="top_n_genes must be positive"):
        scd.tl.get_factor_batch_gene_scale_affinity(dec, ref, top_n_genes=0)


def test_gene_axis_length_mismatch_raises_naming_both_shapes():
    ref, dec = _planted_pair()
    short_ref = _reference_model(
        np.vstack(
            [
                np.exp(CONTRAST[: N_GENES - 10]),
                np.exp(-CONTRAST[: N_GENES - 10]),
            ]
        ),
        n_genes=N_GENES - 10,
    )
    with pytest.raises(ValueError, match="not on the same gene axis") as excinfo:
        scd.tl.get_factor_batch_gene_scale_affinity(dec, short_ref)
    message = str(excinfo.value)
    assert str(N_GENES - 10) in message and str(N_GENES) in message


def test_gene_axis_order_mismatch_raises_even_at_equal_length():
    ref, dec = _planted_pair()
    renamed = _reference_model(
        np.asarray(ref.pmeans["gene_scale"]), gene_prefix="other_gene"
    )
    with pytest.raises(ValueError, match="not on the same gene axis"):
        scd.tl.get_factor_batch_gene_scale_affinity(dec, renamed)


def test_dataframe_reference_on_wrong_gene_axis_raises():
    ref, dec = _planted_pair()
    ratios = scd.tl.get_batch_specific_genes_from_gene_scale(ref)
    shuffled = ratios.iloc[::-1]
    with pytest.raises(ValueError, match="not on the model's gene axis"):
        scd.tl.get_factor_batch_gene_scale_affinity(dec, shuffled)


def test_single_row_reference_gene_scale_raises():
    ref, dec = _planted_pair()
    one_row = _reference_model(np.exp(CONTRAST).reshape(1, -1))
    with pytest.raises(ValueError, match="single batch row"):
        scd.tl.get_factor_batch_gene_scale_affinity(dec, one_row)

    # Same for the raw-array and DataFrame forms.
    with pytest.raises(ValueError, match="single batch row"):
        scd.tl.get_factor_batch_gene_scale_affinity(
            dec, np.exp(CONTRAST).reshape(1, -1)
        )
    ratios = scd.tl.get_batch_specific_genes_from_gene_scale(ref)
    with pytest.raises(ValueError, match="at least two batches"):
        scd.tl.get_factor_batch_gene_scale_affinity(dec, ratios[["A"]])


def test_decomposed_model_passed_as_its_own_reference_raises():
    """The classic mistake: the decomposed fit has one shared gene_scale row."""
    ref, dec = _planted_pair()
    dec.pmeans["gene_scale"] = np.ones((1, N_GENES))
    with pytest.raises(ValueError, match="single batch row"):
        scd.tl.get_factor_batch_gene_scale_affinity(dec, dec)


def test_missing_reference_raises_with_actionable_message():
    ref, dec = _planted_pair()
    with pytest.raises(ValueError, match="No reference gene_scale information"):
        scd.tl.get_factor_batch_gene_scale_affinity(dec)


def test_reference_read_from_uns_when_present():
    ref, dec = _planted_pair()
    expected = scd.tl.get_factor_batch_gene_scale_affinity(dec, ref)
    dec.adata.uns["reference_gene_scale"] = np.asarray(ref.pmeans["gene_scale"])
    dec.adata.uns["reference_gene_scale_batches"] = ["A", "B"]
    pd.testing.assert_frame_equal(
        expected, scd.tl.get_factor_batch_gene_scale_affinity(dec)
    )


def test_bad_reference_type_raises_typeerror():
    ref, dec = _planted_pair()
    with pytest.raises(TypeError, match="must be a fitted scDEF model"):
        scd.tl.get_factor_batch_gene_scale_affinity(dec, "the reference")


def test_helper_columns_come_after_the_batch_block():
    """The column contract the factor_diagnostics wiring reads."""
    ref, dec = _planted_pair()
    aff = scd.tl.get_factor_batch_gene_scale_affinity(dec, ref)
    assert list(aff.columns) == ["A", "B", "top_batch", "top_score", "abs_top_score"]
    assert aff.attrs["batch_columns"] == ["A", "B"]
    # top_score is always one of the per-batch entries of its row.
    for name, row in aff.iterrows():
        if row["top_batch"]:
            assert row["top_score"] == row[row["top_batch"]]
            assert row["abs_top_score"] == abs(row["top_score"])
