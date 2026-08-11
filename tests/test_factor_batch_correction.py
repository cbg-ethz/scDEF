"""Batch-technical flagging, the score/label correction, and the batch report.

Models are built by setting the posterior means directly rather than fitting, so
which factors are siblings, which cells score on what, and which parents are kept
are all exact and the assertions are about behaviour rather than about a fit.
"""

import numpy as np
import pytest

scanpy = pytest.importorskip("scanpy")
scd = pytest.importorskip("scdef")


def _make_model(layer_sizes=(4, 2), n_cells=40, n_batches=2, batch_key=None):
    adata = scanpy.datasets.pbmc3k()[:n_cells].copy()
    adata.X = adata.X.toarray()
    adata.obs["batch"] = [f"b{i % n_batches + 1}" for i in range(adata.n_obs)]
    model = scd.scDEF(adata, layer_sizes=list(layer_sizes), batch_key=batch_key, seed=0)
    model.init_var_params(init_budgets=True, init_alpha=True, nmf_init=False)
    model.set_posterior_means()
    model.factor_lists = [np.arange(s, dtype=int) for s in layer_sizes]
    model.set_factor_names()
    model.annotate_adata()
    return model


def _corrected(model, key="X_L0_batch_corrected"):
    """The correction's outputs, read back off the model (it returns None)."""
    return (
        np.asarray(model.adata.obsm[key], dtype=float),
        list(model.adata.uns[f"{key}_factors"]),
    )


def _set_parents(model, parent_of_l0, layer=1):
    """Make ``L{layer}W`` put each L0 factor under the given parent row."""
    key = f"{model.layer_names[layer]}W"
    w = np.full_like(np.asarray(model.pmeans[key], dtype=float), 0.01)
    for l0, parent in enumerate(parent_of_l0):
        w[parent, l0] = 1.0
    model.pmeans[key] = w


def _set_scores(model, columns):
    """Set ``L0z`` columns (one array per kept L0 factor) and re-annotate."""
    z = np.asarray(model.pmeans["L0z"], dtype=float).copy()
    for slot, values in enumerate(columns):
        z[:, model.factor_lists[0][slot]] = values
    model.pmeans["L0z"] = z
    model.annotate_adata()


# --------------------------------------------------------------------------
# flagging and the getter
# --------------------------------------------------------------------------


def test_get_batch_technical_factors_roundtrip():
    model = _make_model()
    assert scd.tl.get_batch_technical_factors(model) == []

    flagged = [model.factor_names[0][0], model.factor_names[0][2]]
    scd.tl.set_batch_technical_factors(model, flagged)
    assert sorted(scd.tl.get_batch_technical_factors(model)) == sorted(flagged)


def test_batch_technical_and_technical_flags_are_independent():
    model = _make_model()
    scd.tl.factor_diagnostics(model)
    bt = [model.factor_names[0][0]]
    tech = model.factor_names[0][1]

    scd.tl.set_batch_technical_factors(model, bt)
    model.adata.uns["factor_obs"].loc[tech, "technical"] = True

    assert scd.tl.get_batch_technical_factors(model) == bt
    assert tech in scd.tl.get_technical_factors(model)
    assert tech not in scd.tl.get_batch_technical_factors(model)


def test_set_batch_technical_factors_takes_no_top_layer():
    """The roll-up target belongs to decompose_batch_effects, not to flagging."""
    model = _make_model()
    with pytest.raises(TypeError):
        scd.tl.set_batch_technical_factors(
            model, [model.factor_names[0][0]], top_layer=1
        )
    scd.tl.set_batch_technical_factors(model, [model.factor_names[0][0]])
    assert "batch_technical_top_layer" not in model.adata.uns


def test_set_batch_technical_factors_rejects_unknown_names():
    model = _make_model()
    with pytest.raises(ValueError):
        scd.tl.set_batch_technical_factors(model, ["not_a_factor"])


# --------------------------------------------------------------------------
# factor_batch_correction: score space
# --------------------------------------------------------------------------


def test_nothing_flagged_returns_scores_unchanged_and_writes_no_labels():
    model = _make_model()
    scd.tl.factor_batch_correction(model)
    matrix, labels = _corrected(model)

    np.testing.assert_allclose(matrix, model.adata.obsm["X_L0"])
    assert labels == [str(n) for n in model.factor_names[0]]
    assert model.adata.uns["X_L0_batch_corrected_dropped"] == []
    assert "L0_batch_corrected" not in model.adata.obs.columns
    assert "batch_corrected" not in model.adata.obs.columns


def test_flagged_siblings_merge_by_sum():
    model = _make_model(layer_sizes=(4, 2))
    _set_parents(model, [0, 0, 1, 1])
    n = model.adata.n_obs
    a, b = np.linspace(1, 2, n), np.linspace(3, 4, n)
    _set_scores(model, [a, b, np.ones(n), np.ones(n)])

    names = [str(x) for x in model.factor_names[0]]
    scd.tl.set_batch_technical_factors(model, names[:2])
    scd.tl.factor_batch_correction(model)
    matrix, labels = _corrected(model)

    merged = f"{names[0]}+{names[1]}"
    assert merged in labels
    assert matrix.shape[1] == model.adata.obsm["X_L0"].shape[1] - 1
    np.testing.assert_allclose(matrix[:, labels.index(merged)], a + b)
    assert model.adata.uns["X_L0_batch_corrected_dropped"] == []
    assert (
        model.adata.uns["X_L0_batch_corrected_members"][labels.index(merged)]
        == names[:2]
    )


def test_flagged_siblings_merge_by_max():
    model = _make_model(layer_sizes=(4, 2))
    _set_parents(model, [0, 0, 1, 1])
    n = model.adata.n_obs
    a = np.linspace(1, 5, n)
    b = a[::-1].copy()
    _set_scores(model, [a, b, np.ones(n), np.ones(n)])

    names = [str(x) for x in model.factor_names[0]]
    scd.tl.set_batch_technical_factors(model, names[:2])
    scd.tl.factor_batch_correction(model, reduce="max")
    matrix, labels = _corrected(model)
    np.testing.assert_allclose(
        matrix[:, labels.index(f"{names[0]}+{names[1]}")], np.maximum(a, b)
    )


def test_lone_flagged_factor_is_dropped():
    """No opposite-batch half to merge with, so the column leaves entirely."""
    model = _make_model(layer_sizes=(4, 2))
    _set_parents(model, [0, 1, 1, 1])
    names = [str(x) for x in model.factor_names[0]]

    scd.tl.set_batch_technical_factors(model, [names[0]])
    scd.tl.factor_batch_correction(model)
    matrix, labels = _corrected(model)

    assert names[0] not in labels
    assert matrix.shape[1] == model.adata.obsm["X_L0"].shape[1] - 1
    assert model.adata.uns["X_L0_batch_corrected_dropped"] == [names[0]]
    assert model.adata.uns["X_L0_batch_corrected_batch_technical"] == [names[0]]


def test_unflagged_factors_are_untouched():
    model = _make_model(layer_sizes=(4, 2))
    _set_parents(model, [0, 0, 1, 1])
    names = [str(x) for x in model.factor_names[0]]
    x0 = model.adata.obsm["X_L0"].copy()

    scd.tl.set_batch_technical_factors(model, names[:2])
    scd.tl.factor_batch_correction(model)
    matrix, labels = _corrected(model)
    for keep in names[2:]:
        np.testing.assert_allclose(
            matrix[:, labels.index(keep)], x0[:, names.index(keep)]
        )


def test_all_factors_flagged_and_dropped_raises():
    model = _make_model(layer_sizes=(2, 2))
    _set_parents(model, [0, 1])  # each alone under its own parent
    names = [str(x) for x in model.factor_names[0]]
    scd.tl.set_batch_technical_factors(model, names)
    with pytest.raises(ValueError, match="no columns"):
        scd.tl.factor_batch_correction(model)


def test_reduce_must_be_sum_or_max():
    model = _make_model()
    with pytest.raises(ValueError):
        scd.tl.factor_batch_correction(model, reduce="mean")


# --------------------------------------------------------------------------
# factor_batch_correction: labels
# --------------------------------------------------------------------------


def test_labels_merge_pairs_and_roll_singletons_up_to_the_parent():
    model = _make_model(layer_sizes=(4, 2))
    # L0_0 and L0_1 are siblings under parent 0; L0_2 is alone under parent 1.
    _set_parents(model, [0, 0, 1, 1])
    n = model.adata.n_obs
    # Give each factor a block of cells so every label is exercised.
    cols = [np.zeros(n) for _ in range(4)]
    for slot in range(4):
        cols[slot][slot::4] = 10.0
    _set_scores(model, cols)

    names = [str(x) for x in model.factor_names[0]]
    parents = [str(x) for x in model.factor_names[1]]
    scd.tl.set_batch_technical_factors(model, [names[0], names[1], names[2]])
    scd.tl.factor_batch_correction(model)

    base = model.adata.obs["L0"].astype(str)
    sib = model.adata.obs["L0_batch_corrected"].astype(str)
    par = model.adata.obs["batch_corrected"].astype(str)

    # The pair keeps a merged sibling label, and points at its parent.
    pair_cells = base.isin(names[:2])
    assert set(sib[pair_cells]) == {f"{names[0]}+{names[1]}"}
    assert set(par[pair_cells]) == {parents[0]}

    # The lone factor rolls up to its parent in BOTH columns.
    lone = base == names[2]
    assert set(sib[lone]) == {parents[1]}
    assert set(par[lone]) == {parents[1]}

    # Untouched factors keep their own label in both.
    other = base == names[3]
    assert set(sib[other]) == {names[3]}
    assert set(par[other]) == {names[3]}


# --------------------------------------------------------------------------
# parent resolution
# --------------------------------------------------------------------------


def test_filtered_out_parent_cannot_win_the_argmax():
    """Regression: grouping must use kept parents only.

    An earlier version took ``argmax`` over every row of ``L1W`` including
    filtered-out factors, so two factors that share a *kept* parent could be
    assigned different ones and fail to merge.
    """
    model = _make_model(layer_sizes=(4, 3))
    model.factor_lists[1] = np.array([0, 2], dtype=int)  # L1 row 1 is filtered out
    model.set_factor_names()

    w = np.full_like(np.asarray(model.pmeans["L1W"], dtype=float), 0.01)
    # Both L0_0 and L0_1 have their largest *kept* weight on row 0, but L0_0's
    # overall largest is on the filtered row 1.
    w[1, 0], w[0, 0], w[2, 0] = 10.0, 5.0, 0.1
    w[0, 1], w[2, 1] = 5.0, 0.1
    w[2, 2] = w[2, 3] = 5.0
    model.pmeans["L1W"] = w

    names = [str(x) for x in model.factor_names[0]]
    scd.tl.set_batch_technical_factors(model, names[:2])
    scd.tl.factor_batch_correction(model)
    _, labels = _corrected(model)

    # They share the kept parent, so they merge instead of being dropped as two
    # unrelated singletons.
    assert f"{names[0]}+{names[1]}" in labels
    assert model.adata.uns["X_L0_batch_corrected_dropped"] == []


def test_top_layer_changes_the_grouping():
    model = _make_model(layer_sizes=(4, 2, 2))
    # Different L1 parents...
    _set_parents(model, [0, 1, 0, 1], layer=1)
    # ...but the same L2 parent.
    w2 = np.full_like(np.asarray(model.pmeans["L2W"], dtype=float), 0.01)
    w2[0, 0] = w2[0, 1] = 1.0
    model.pmeans["L2W"] = w2

    names = [str(x) for x in model.factor_names[0]]
    scd.tl.set_batch_technical_factors(model, names[:2])

    scd.tl.factor_batch_correction(model, top_layer=1)
    _, labels1 = _corrected(model)
    dropped1 = list(model.adata.uns["X_L0_batch_corrected_dropped"])

    scd.tl.factor_batch_correction(model, top_layer=2)
    _, labels2 = _corrected(model)
    dropped2 = list(model.adata.uns["X_L0_batch_corrected_dropped"])

    assert sorted(dropped1) == sorted(names[:2])  # two unrelated singletons
    assert dropped2 == []  # siblings at L2
    assert f"{names[0]}+{names[1]}" in labels2
    assert f"{names[0]}+{names[1]}" not in labels1


def test_top_layer_out_of_range_raises():
    model = _make_model(layer_sizes=(4, 2))
    scd.tl.set_batch_technical_factors(model, [str(model.factor_names[0][0])])
    with pytest.raises(ValueError):
        scd.tl.factor_batch_correction(model, top_layer=5)


# --------------------------------------------------------------------------
# batch_structure_report
# --------------------------------------------------------------------------


def test_report_runs_on_a_model_never_fitted_with_a_batch_key():
    """A plain fit leaves batch structure in the factors; the report reads it."""
    model = _make_model(layer_sizes=(4, 2))
    assert model.batch_key is None

    rep = scd.tl.batch_structure_report(model, batch_key="batch")
    assert len(rep) == len(model.factor_names[0])
    for col in (
        "n_cells",
        "dom_batch",
        "frac_dom_batch",
        "eff_parents",
        "parent",
        "shape",
    ):
        assert col in rep.columns
    assert set(rep["shape"]) <= {
        "branch_split",
        "branch_skewed",
        "overlaid",
        "balanced",
    }
    # No reference profile available, so the gene-side columns are absent, not NaN.
    assert not [c for c in rep.columns if c.startswith("gene_scale_affinity")]
    assert "branch_summary" in rep.attrs


def test_report_gene_side_columns_appear_with_a_reference_profile():
    model = _make_model(layer_sizes=(4, 2))
    rng = np.random.default_rng(0)
    model.adata.uns["reference_gene_scale"] = rng.gamma(
        2.0, 1.0, (2, model.adata.n_vars)
    )
    model.adata.uns["reference_gene_scale_batches"] = ["b1", "b2"]

    rep = scd.tl.batch_structure_report(model, batch_key="batch")
    for col in (
        "gene_scale_affinity_b1",
        "gene_scale_affinity_b2",
        "gene_scale_affinity_max",
        "gene_scale_affinity_batch",
    ):
        assert col in rep.columns

    # Two batches give mirror-image contrasts, so max is the elementwise max.
    b1 = rep["gene_scale_affinity_b1"].to_numpy(dtype=float)
    b2 = rep["gene_scale_affinity_b2"].to_numpy(dtype=float)
    np.testing.assert_allclose(b1, -b2, atol=1e-8)
    np.testing.assert_allclose(
        rep["gene_scale_affinity_max"].to_numpy(dtype=float), np.maximum(b1, b2)
    )
    assert rep.attrs["gene_scale_affinity"].shape[0] == len(rep)


def test_report_raises_on_an_explicit_bad_reference():
    """Silently dropping the columns is only acceptable when none was asked for."""
    model = _make_model(layer_sizes=(4, 2))
    with pytest.raises((TypeError, ValueError)):
        scd.tl.batch_structure_report(model, batch_key="batch", reference="nonsense")


def _batch_metrics_populated(model):
    fo = model.adata.uns["factor_obs"]
    return "batch_purity" in fo.columns and bool(fo["batch_purity"].notna().any())


def test_diagnostics_default_to_the_models_own_batch_key():
    """A batch-aware model should not have to be told its own batch key twice."""
    with_key = _make_model(batch_key="batch")
    scd.tl.factor_diagnostics(with_key)
    assert _batch_metrics_populated(with_key)

    without_key = _make_model(batch_key=None)
    scd.tl.factor_diagnostics(without_key)
    assert not _batch_metrics_populated(without_key)

    # An explicit key still works on a model that carries none.
    scd.tl.factor_diagnostics(without_key, batch_key="batch", recompute=True)
    assert _batch_metrics_populated(without_key)


def test_diagnostics_skip_a_model_key_with_one_observed_value():
    """A degenerate key must yield no metrics, not an error."""
    model = _make_model(batch_key="batch")
    model.adata.obs["batch"] = "only_one"
    scd.tl.factor_diagnostics(model, recompute=True)
    assert not _batch_metrics_populated(model)


def test_filter_passes_the_models_batch_key_through():
    model = _make_model(batch_key="batch")
    scd.tl.filter(model)
    assert _batch_metrics_populated(model)


def test_report_defaults_to_the_models_own_batch_key():
    with_key = _make_model(batch_key="batch")
    from_default = scd.tl.batch_structure_report(with_key)
    explicit = scd.tl.batch_structure_report(with_key, batch_key="batch")
    assert list(from_default.index) == list(explicit.index)
    assert from_default["dom_batch"].tolist() == explicit["dom_batch"].tolist()


def test_report_without_any_batch_key_raises():
    model = _make_model(batch_key=None)
    with pytest.raises(ValueError, match="needs a batch_key"):
        scd.tl.batch_structure_report(model)


def test_report_requires_two_observed_batches():
    model = _make_model(layer_sizes=(4, 2))
    model.adata.obs["batch"] = "only_one"
    with pytest.raises(ValueError):
        scd.tl.batch_structure_report(model, batch_key="batch")

    with pytest.raises(KeyError):
        scd.tl.batch_structure_report(model, batch_key="absent_column")
