import numpy as np
import pandas as pd
import pytest

scanpy = pytest.importorskip("scanpy")
scd = pytest.importorskip("scdef")


def _make_model_with_pmeans(layer_sizes):
    adata = scanpy.datasets.pbmc3k()[:30].copy()
    adata.X = adata.X.toarray()
    model = scd.scDEF(adata, layer_sizes=layer_sizes, seed=0)
    model.init_var_params(init_budgets=True, init_alpha=True, nmf_init=False)
    model.set_posterior_means()
    model.factor_lists = [np.arange(s, dtype=int) for s in layer_sizes]
    return model


def test_should_collapse_adjacent_thresholds():
    model = _make_model_with_pmeans([12, 10, 4, 1])
    assert model._should_collapse_adjacent_layer_sizes([12, 10, 4, 1], 0, 0.8)
    assert not model._should_collapse_adjacent_layer_sizes([12, 8, 4, 1], 0, 0.8)
    assert model._should_collapse_adjacent_layer_sizes([12, 40, 35, 8, 1], 1, 0.8)
    assert not model._should_collapse_adjacent_layer_sizes([12, 40, 30, 8, 1], 1, 0.8)
    assert not model._should_collapse_adjacent_layer_sizes([13, 4, 3, 1], 1, 0.8)
    assert model._should_collapse_redundant_l1(0.8, 10)


def test_collapse_only_wide_upper_pairs():
    model = _make_model_with_pmeans([13, 11, 4, 3, 1])
    sizes, old_keep, n_dropped = model._collapse_redundant_adjacent_layer_sizes(
        [13, 11, 4, 3, 1], 0.8
    )
    assert n_dropped == 1
    assert sizes == [13, 4, 3, 1]
    assert old_keep == [0, 2, 3, 4]


def test_collapse_multiple_wide_adjacent_layers():
    model = _make_model_with_pmeans([12, 10, 9, 8, 1])
    sizes, old_keep, n_dropped = model._collapse_redundant_adjacent_layer_sizes(
        [12, 10, 9, 8, 1], 0.8
    )
    assert n_dropped == 2
    assert sizes == [12, 9, 1]
    assert old_keep == [0, 2, 4]


def test_build_collapsed_refit_init_shapes():
    layer_sizes = [5, 4, 3, 1]
    model = _make_model_with_pmeans(layer_sizes)
    init_z, init_w, init_brd, init_ard = model._build_collapsed_refit_init([0, 2, 3])

    assert len(init_z) == 3
    assert len(init_w) == 3
    assert init_z[0].shape == (model.n_cells, 5)
    assert init_w[0].shape == (5, model.n_genes)
    assert init_z[1].shape == (model.n_cells, 3)
    assert init_w[1].shape == (3, 5)
    assert init_w[2].shape == (1, 3)
    assert init_brd.shape in {(5,), (5, 1)}
    assert init_ard.shape in {(5,), (5, 1)}


def test_sanitize_layer_sizes_dedup_tracks_old_keep():
    model = _make_model_with_pmeans([5, 4, 3, 1])
    sizes, old_keep = model._sanitize_layer_sizes([17, 12, 6, 3, 3, 1])
    assert sizes == [17, 12, 6, 3, 1]
    assert old_keep == [0, 1, 2, 3, 5]


def test_build_collapsed_refit_init_dedup_promotes_root():
    layer_sizes = [5, 4, 3, 3, 2, 1]
    model = _make_model_with_pmeans(layer_sizes)
    fl = [np.arange(s, dtype=int) for s in layer_sizes]
    names = model.layer_names
    old_keep = [0, 1, 2, 4, 5]
    init_z, init_w, _, _ = model._build_collapsed_refit_init(
        old_keep, factor_lists=fl, layer_names=names
    )
    expected_top_w = model._compose_w_to_parent_layer(names, fl, 5, 4)
    expected_parent_w = model._compose_w_to_parent_layer(names, fl, 4, 2)
    assert len(init_z) == len(old_keep)
    assert init_z[-1].shape == (model.n_cells, 1)
    np.testing.assert_allclose(init_w[-1], expected_top_w, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(init_w[-2], expected_parent_w, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        init_z[-1],
        np.asarray(model.pmeans["L5z"], dtype=np.float32)[:, fl[5]],
        rtol=1e-5,
        atol=1e-5,
    )


def test_refit_dedup_uses_hierarchy_changed_init(monkeypatch):
    layer_sizes = [5, 4, 3, 3, 1, 1]
    model = _make_model_with_pmeans(layer_sizes)
    model._has_fit = True
    init_kwargs = {}

    def capture_init(**kwargs):
        init_kwargs.update(kwargs)

    monkeypatch.setattr(model, "init_var_params", capture_init)
    monkeypatch.setattr(model, "_learn", lambda **kwargs: None)
    monkeypatch.setattr(model, "_invalidate_cached_diagnostics", lambda: None)
    monkeypatch.setattr(model, "update_model_priors", lambda **kwargs: None)
    monkeypatch.setattr(model, "clear_runtime_cache", lambda **kwargs: None)

    model.fit(n_epoch=1, collapse_l1_fraction=None)

    assert model.layer_sizes == [5, 4, 3, 1]
    init_z = init_kwargs["init_z"]
    init_w = init_kwargs["init_w"]
    assert isinstance(init_z, list)
    assert len(init_z) == 4
    assert len(init_w) == 4
    assert init_z[-1].shape == (model.n_cells, 1)
    assert init_w[-1].shape == (1, 3)


def test_build_collapsed_refit_init_composes_skipped_layer():
    layer_sizes = [5, 4, 3, 1]
    model = _make_model_with_pmeans(layer_sizes)
    fl = [np.arange(s, dtype=int) for s in layer_sizes]
    names = model.layer_names
    expected = (
        np.asarray(model.pmeans["L2W"], dtype=np.float32)[np.ix_(fl[2], fl[1])]
        @ np.asarray(model.pmeans["L1W"], dtype=np.float32)[np.ix_(fl[1], fl[0])]
    )
    composed = model._compose_w_to_parent_layer(names, fl, 2, 0)
    np.testing.assert_allclose(composed, expected, rtol=1e-5, atol=1e-5)


def test_refit_drops_redundant_layers_from_layer_sizes(monkeypatch):
    layer_sizes = [12, 10, 4, 1]
    model = _make_model_with_pmeans(layer_sizes)
    model._has_fit = True
    updated_sizes = []

    def capture_update(self, **kwargs):
        updated_sizes.append(list(kwargs.get("layer_sizes", self.layer_sizes)))

    monkeypatch.setattr(scd.scDEF, "update_model_size", capture_update)
    monkeypatch.setattr(model, "_learn", lambda **kwargs: None)
    monkeypatch.setattr(model, "init_var_params", lambda **kwargs: None)
    monkeypatch.setattr(model, "_invalidate_cached_diagnostics", lambda: None)
    monkeypatch.setattr(model, "update_model_priors", lambda **kwargs: None)
    monkeypatch.setattr(model, "clear_runtime_cache", lambda **kwargs: None)

    model.fit(n_epoch=1, collapse_l1_fraction=0.8)

    assert updated_sizes[-1] == [12, 4, 1]


def test_refit_preserves_existing_alpha_after_resizing(monkeypatch):
    model = _make_model_with_pmeans([12, 10, 4, 1])
    model._has_fit = True
    model.set_alpha_from_cov = True
    model.alpha = 123.0

    monkeypatch.setattr(model, "_learn", lambda **kwargs: None)
    monkeypatch.setattr(model, "init_var_params", lambda **kwargs: None)
    monkeypatch.setattr(model, "_invalidate_cached_diagnostics", lambda: None)
    monkeypatch.setattr(model, "clear_runtime_cache", lambda **kwargs: None)

    model.fit(n_epoch=1, collapse_l1_fraction=0.8)

    assert model.layer_sizes == [12, 4, 1]
    assert model.alpha == 123.0


def test_refit_without_hierarchy_change_warm_starts_root(monkeypatch):
    layer_sizes = [5, 3, 1]
    model = _make_model_with_pmeans(layer_sizes)
    model._has_fit = True
    init_kwargs = {}

    def capture_init(**kwargs):
        init_kwargs.update(kwargs)

    monkeypatch.setattr(model, "init_var_params", capture_init)
    monkeypatch.setattr(model, "_learn", lambda **kwargs: None)
    monkeypatch.setattr(model, "_invalidate_cached_diagnostics", lambda: None)
    monkeypatch.setattr(model, "update_model_priors", lambda **kwargs: None)
    monkeypatch.setattr(model, "clear_runtime_cache", lambda **kwargs: None)

    model.fit(n_epoch=1, collapse_l1_fraction=None)

    init_z = init_kwargs["init_z"]
    init_w = init_kwargs["init_w"]
    assert len(init_z) == len(layer_sizes)
    assert len(init_w) == len(layer_sizes)
    np.testing.assert_allclose(
        init_z[-1],
        np.asarray(model.pmeans["L2z"], dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        init_w[-1],
        np.asarray(model.pmeans["L2W"], dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )


def test_find_sensible_top_layer_stops_before_ambiguous_merge():
    model = _make_model_with_pmeans([4, 3, 2, 1])
    model.adata.uns["factor_obs"] = pd.DataFrame(
        [
            {
                "child_layer": "L0",
                "parent_layer": "L1",
                "best_parent": "L1_0",
                "best_parent_prob": 0.98,
                "top_gap": 0.95,
                "n_eff_parents": 1.05,
                "K_parents": 3,
            },
            {
                "child_layer": "L0",
                "parent_layer": "L1",
                "best_parent": "L1_0",
                "best_parent_prob": 0.98,
                "top_gap": 0.95,
                "n_eff_parents": 1.05,
                "K_parents": 3,
            },
            {
                "child_layer": "L0",
                "parent_layer": "L1",
                "best_parent": "L1_1",
                "best_parent_prob": 0.98,
                "top_gap": 0.95,
                "n_eff_parents": 1.05,
                "K_parents": 3,
            },
            {
                "child_layer": "L0",
                "parent_layer": "L1",
                "best_parent": "L1_2",
                "best_parent_prob": 0.98,
                "top_gap": 0.95,
                "n_eff_parents": 1.05,
                "K_parents": 3,
            },
            {
                "child_layer": "L1",
                "parent_layer": "L2",
                "best_parent": "L2_0",
                "best_parent_prob": 0.5,
                "top_gap": 0.0,
                "n_eff_parents": 2.0,
                "K_parents": 2,
            },
            {
                "child_layer": "L1",
                "parent_layer": "L2",
                "best_parent": "L2_0",
                "best_parent_prob": 0.5,
                "top_gap": 0.0,
                "n_eff_parents": 2.0,
                "K_parents": 2,
            },
            {
                "child_layer": "L1",
                "parent_layer": "L2",
                "best_parent": "L2_0",
                "best_parent_prob": 0.5,
                "top_gap": 0.0,
                "n_eff_parents": 2.0,
                "K_parents": 2,
            },
        ],
        index=model.factor_names[0] + model.factor_names[1],
    )

    res = scd.tl.find_sensible_top_layer(
        model,
        n_eff_parents_max=1.5,
        min_clear_fraction=1.0,
    )

    assert res["recommended_layer_idx"] == 1
    assert res["recommended_layer"] == "L1"
    assert res["recommended_factors"] == model.factor_names[1]
    transitions = res["transition_diagnostics"]
    assert transitions.loc[0, "transition_ok"]
    assert not transitions.loc[1, "transition_ok"]
    assert model.adata.uns["sensible_top_layer"]["recommended_layer"] == "L1"


def test_find_sensible_top_layer_ignores_technical_factors():
    model = _make_model_with_pmeans([3, 2, 1])
    model.adata.uns["factor_obs"] = pd.DataFrame(
        [
            {
                "child_layer": "L0",
                "parent_layer": "L1",
                "best_parent": "L1_0",
                "best_parent_prob": 0.98,
                "top_gap": 0.95,
                "n_eff_parents": 1.05,
                "K_parents": 2,
                "technical": False,
            },
            {
                "child_layer": "L0",
                "parent_layer": "L1",
                "best_parent": "L1_1",
                "best_parent_prob": 0.98,
                "top_gap": 0.95,
                "n_eff_parents": 1.05,
                "K_parents": 2,
                "technical": False,
            },
            {
                "child_layer": "L0",
                "parent_layer": "L1",
                "best_parent": "L1_1",
                "best_parent_prob": 0.40,
                "top_gap": 0.0,
                "n_eff_parents": 2.0,
                "K_parents": 2,
                "technical": True,
            },
        ],
        index=model.factor_names[0],
    )

    res = scd.tl.find_sensible_top_layer(
        model,
        n_eff_parents_max=1.5,
        min_clear_fraction=1.0,
    )

    assert res["recommended_layer"] == "L1"
    transitions = res["transition_diagnostics"]
    assert transitions.loc[0, "n_parents"] == 2
    assert transitions.loc[0, "transition_ok"]


def test_refit_top_layer_truncates_to_sensible_top(monkeypatch):
    layer_sizes = [5, 4, 3, 2, 1]
    model = _make_model_with_pmeans(layer_sizes)
    model._has_fit = True
    fl = [np.arange(s, dtype=int) for s in layer_sizes]
    names = list(model.layer_names)
    expected_root_w = model._compose_w_to_parent_layer(names, fl, 4, 2)
    init_kwargs = {}

    def capture_init(**kwargs):
        init_kwargs.update(kwargs)

    monkeypatch.setattr(model, "init_var_params", capture_init)
    monkeypatch.setattr(model, "_learn", lambda **kwargs: None)
    monkeypatch.setattr(model, "_invalidate_cached_diagnostics", lambda: None)
    monkeypatch.setattr(model, "clear_runtime_cache", lambda **kwargs: None)

    model.fit(n_epoch=1, collapse_l1_fraction=None, refit_top_layer=2)

    assert model.layer_sizes == [5, 4, 3, 1]
    assert len(init_kwargs["init_w"]) == 4
    np.testing.assert_allclose(
        init_kwargs["init_w"][-1], expected_root_w, rtol=1e-5, atol=1e-5
    )


def test_find_sensible_top_factors_keeps_ambiguous_l0_frontier():
    model = _make_model_with_pmeans([4, 3, 2, 1])
    model.pmeans["L1W"] = np.array(
        [
            [0.45, 0.45, 0.01, 0.01],
            [0.01, 0.95, 0.01, 0.01],
            [0.01, 0.01, 0.01, 0.95],
        ],
        dtype=np.float32,
    )
    model.pmeans["L2W"] = np.array(
        [[0.45, 0.45, 0.01], [0.01, 0.01, 0.95]],
        dtype=np.float32,
    )
    model.adata.uns["factor_obs"] = pd.DataFrame(
        [
            {
                "child_layer": "L0",
                "parent_layer": "L1",
                "best_parent": "L1_0",
                "best_parent_prob": 0.98,
                "n_eff_parents": 1.05,
            },
            {
                "child_layer": "L0",
                "parent_layer": "L1",
                "best_parent": "L1_0",
                "best_parent_prob": 0.98,
                "n_eff_parents": 1.05,
            },
            {
                "child_layer": "L0",
                "parent_layer": "L1",
                "best_parent": "L1_1",
                "best_parent_prob": 0.98,
                "n_eff_parents": 1.05,
            },
            {
                "child_layer": "L0",
                "parent_layer": "L1",
                "best_parent": "L1_2",
                "best_parent_prob": 0.40,
                "n_eff_parents": 2.7,
            },
            {
                "child_layer": "L1",
                "parent_layer": "L2",
                "best_parent": "L2_0",
                "best_parent_prob": 0.95,
                "n_eff_parents": 1.1,
            },
            {
                "child_layer": "L1",
                "parent_layer": "L2",
                "best_parent": "L2_1",
                "best_parent_prob": 0.95,
                "n_eff_parents": 1.1,
            },
            {
                "child_layer": "L1",
                "parent_layer": "L2",
                "best_parent": "L2_1",
                "best_parent_prob": 0.95,
                "n_eff_parents": 1.1,
            },
        ],
        index=model.factor_names[0] + model.factor_names[1],
    )

    res = scd.tl.find_sensible_top_factors(model, n_eff_parents_max=1.5)

    assert res == ["L0_3"]
    factor_obs = model.adata.uns["factor_obs"]
    assert "is_sensible_top_factor" in factor_obs.columns
    assert "top_local_gate_pass" in factor_obs.columns
    assert "top_parent_weighted_child_n_eff" in factor_obs.columns
    assert "top_parent_n_clear_children" in factor_obs.columns
    assert "top_parent_weighted_gate_pass" in factor_obs.columns
    assert "top_parent_count_gate_pass" in factor_obs.columns
    assert "top_parent_gate_pass" in factor_obs.columns
    assert bool(factor_obs.loc["L0_0", "top_local_gate_pass"])
    assert np.isfinite(float(factor_obs.loc["L0_0", "top_parent_weighted_child_n_eff"]))
    assert np.isfinite(float(factor_obs.loc["L0_0", "top_parent_n_clear_children"]))
    assert not bool(factor_obs.loc["L0_3", "top_local_gate_pass"])


def test_find_sensible_top_factors_blocks_ambiguous_parent_children():
    model = _make_model_with_pmeans([2, 2, 2, 1])
    model.pmeans["L2W"] = np.array(
        [[0.5, 0.5], [0.01, 0.99]],
        dtype=np.float32,
    )
    model.adata.uns["factor_obs"] = pd.DataFrame(
        [
            {
                "child_layer": "L0",
                "parent_layer": "L1",
                "best_parent": "L1_0",
                "best_parent_prob": 0.98,
                "n_eff_parents": 1.05,
            },
            {
                "child_layer": "L0",
                "parent_layer": "L1",
                "best_parent": "L1_1",
                "best_parent_prob": 0.98,
                "n_eff_parents": 1.05,
            },
            {
                "child_layer": "L1",
                "parent_layer": "L2",
                "best_parent": "L2_0",
                "best_parent_prob": 0.95,
                "n_eff_parents": 1.1,
            },
            {
                "child_layer": "L1",
                "parent_layer": "L2",
                "best_parent": "L2_0",
                "best_parent_prob": 0.55,
                "n_eff_parents": 2.0,
            },
        ],
        index=model.factor_names[0] + model.factor_names[1],
    )

    res = scd.tl.find_sensible_top_factors(model, n_eff_parents_max=1.5)

    assert res == ["L1_0", "L1_1"]
    factor_obs = model.adata.uns["factor_obs"]
    assert bool(factor_obs.loc["L1_0", "is_sensible_top_factor"])


def test_find_sensible_top_factors_ignores_technical_starts():
    model = _make_model_with_pmeans([3, 3, 1])
    model.pmeans["L1W"] = np.array(
        [
            [0.99, 0.01, 0.01],
            [0.01, 0.99, 0.01],
            [0.01, 0.01, 0.99],
        ],
        dtype=np.float32,
    )
    model.adata.uns["factor_obs"] = pd.DataFrame(
        [
            {
                "child_layer": "L0",
                "parent_layer": "L1",
                "best_parent": "L1_0",
                "best_parent_prob": 0.98,
                "n_eff_parents": 1.05,
                "technical": False,
            },
            {
                "child_layer": "L0",
                "parent_layer": "L1",
                "best_parent": "L1_1",
                "best_parent_prob": 0.40,
                "n_eff_parents": 2.5,
                "technical": False,
            },
            {
                "child_layer": "L0",
                "parent_layer": "L1",
                "best_parent": "L1_2",
                "best_parent_prob": 0.99,
                "n_eff_parents": 1.01,
                "technical": True,
            },
        ],
        index=model.factor_names[0],
    )

    res = scd.tl.find_sensible_top_factors(model, n_eff_parents_max=1.5)

    assert res == ["L0_1"]
    assert "L1_2" not in res


def test_find_sensible_top_factors_count_escape_hatch_unblocks_parent_with_clear_siblings():
    """Mirrors the pbmc3k picture: a multi-parent ambiguous child (L1_8 analog)
    sits with high W weight under both L2_4 and L2_0. The parent gate's
    weighted-average score is dragged above threshold for both L2 parents, so
    the old behavior would block every L1 child from ascending. With the
    count escape hatch the L2_0-analog (which has 2 clear best-parent
    children of its own) is accepted while the L2_4-analog (only 1 clear
    best-parent child) stays blocked."""
    model = _make_model_with_pmeans([4, 4, 2, 1])
    model.pmeans["L1W"] = np.eye(4, dtype=np.float32)
    model.pmeans["L2W"] = np.array(
        [
            [0.3, 0.6, 0.05, 0.05],
            [0.05, 0.6, 0.175, 0.175],
        ],
        dtype=np.float32,
    )
    model.adata.uns["factor_obs"] = pd.DataFrame(
        [
            {
                "child_layer": "L0",
                "parent_layer": "L1",
                "best_parent": "L1_0",
                "best_parent_prob": 0.98,
                "n_eff_parents": 1.05,
            },
            {
                "child_layer": "L0",
                "parent_layer": "L1",
                "best_parent": "L1_1",
                "best_parent_prob": 0.98,
                "n_eff_parents": 1.05,
            },
            {
                "child_layer": "L0",
                "parent_layer": "L1",
                "best_parent": "L1_2",
                "best_parent_prob": 0.98,
                "n_eff_parents": 1.05,
            },
            {
                "child_layer": "L0",
                "parent_layer": "L1",
                "best_parent": "L1_3",
                "best_parent_prob": 0.98,
                "n_eff_parents": 1.05,
            },
            {
                "child_layer": "L1",
                "parent_layer": "L2",
                "best_parent": "L2_0",
                "best_parent_prob": 0.95,
                "n_eff_parents": 1.05,
            },
            {
                "child_layer": "L1",
                "parent_layer": "L2",
                "best_parent": "L2_1",
                "best_parent_prob": 0.45,
                "n_eff_parents": 2.5,
            },
            {
                "child_layer": "L1",
                "parent_layer": "L2",
                "best_parent": "L2_1",
                "best_parent_prob": 0.95,
                "n_eff_parents": 1.05,
            },
            {
                "child_layer": "L1",
                "parent_layer": "L2",
                "best_parent": "L2_1",
                "best_parent_prob": 0.95,
                "n_eff_parents": 1.05,
            },
        ],
        index=model.factor_names[0] + model.factor_names[1],
    )

    res = scd.tl.find_sensible_top_factors(model, n_eff_parents_max=1.5)

    assert res == ["L1_0", "L1_1"]
    factor_obs = model.adata.uns["factor_obs"]

    res_strict = scd.tl.find_sensible_top_factors(
        model, n_eff_parents_max=1.5, min_clear_children=99
    )
    assert res_strict == ["L1_0", "L1_1", "L1_2", "L1_3"]


def test_find_sensible_top_factors_reclassifies_with_different_params():
    model = _make_model_with_pmeans([4, 4, 2, 1])
    model.pmeans["L1W"] = np.eye(4, dtype=np.float32)
    model.pmeans["L2W"] = np.array(
        [
            [0.3, 0.6, 0.05, 0.05],
            [0.05, 0.6, 0.175, 0.175],
        ],
        dtype=np.float32,
    )
    model.adata.uns["factor_obs"] = pd.DataFrame(
        [
            {
                "child_layer": "L0",
                "parent_layer": "L1",
                "best_parent": "L1_0",
                "best_parent_prob": 0.98,
                "n_eff_parents": 1.05,
            },
            {
                "child_layer": "L0",
                "parent_layer": "L1",
                "best_parent": "L1_1",
                "best_parent_prob": 0.98,
                "n_eff_parents": 1.05,
            },
            {
                "child_layer": "L0",
                "parent_layer": "L1",
                "best_parent": "L1_2",
                "best_parent_prob": 0.98,
                "n_eff_parents": 1.05,
            },
            {
                "child_layer": "L0",
                "parent_layer": "L1",
                "best_parent": "L1_3",
                "best_parent_prob": 0.98,
                "n_eff_parents": 1.05,
            },
            {
                "child_layer": "L1",
                "parent_layer": "L2",
                "best_parent": "L2_0",
                "best_parent_prob": 0.95,
                "n_eff_parents": 1.05,
            },
            {
                "child_layer": "L1",
                "parent_layer": "L2",
                "best_parent": "L2_1",
                "best_parent_prob": 0.45,
                "n_eff_parents": 2.5,
            },
            {
                "child_layer": "L1",
                "parent_layer": "L2",
                "best_parent": "L2_1",
                "best_parent_prob": 0.95,
                "n_eff_parents": 1.05,
            },
            {
                "child_layer": "L1",
                "parent_layer": "L2",
                "best_parent": "L2_1",
                "best_parent_prob": 0.95,
                "n_eff_parents": 1.05,
            },
        ],
        index=model.factor_names[0] + model.factor_names[1],
    )

    relaxed = scd.tl.find_sensible_top_factors(model, n_eff_parents_max=1.5)
    assert relaxed == ["L1_0", "L1_1"]

    strict = scd.tl.find_sensible_top_factors(
        model, n_eff_parents_max=1.5, min_clear_children=99
    )
    assert strict == ["L1_0", "L1_1", "L1_2", "L1_3"]


def test_find_sensible_top_layer_reports_clear_children_counts():
    """The L1->L2 transition exposes per-parent ``n_clear_children`` in
    ``parent_diagnostics`` and the count gate flips ``clear_parent`` for
    parents that have enough clear best-parent children even when the
    weighted average is above threshold."""
    model = _make_model_with_pmeans([4, 4, 2, 1])
    model.pmeans["L1W"] = np.eye(4, dtype=np.float32)
    model.pmeans["L2W"] = np.array(
        [
            [0.3, 0.6, 0.05, 0.05],
            [0.05, 0.6, 0.175, 0.175],
        ],
        dtype=np.float32,
    )
    model.adata.uns["factor_obs"] = pd.DataFrame(
        [
            {
                "child_layer": "L0",
                "parent_layer": "L1",
                "best_parent": "L1_0",
                "best_parent_prob": 0.98,
                "n_eff_parents": 1.05,
            },
            {
                "child_layer": "L0",
                "parent_layer": "L1",
                "best_parent": "L1_1",
                "best_parent_prob": 0.98,
                "n_eff_parents": 1.05,
            },
            {
                "child_layer": "L0",
                "parent_layer": "L1",
                "best_parent": "L1_2",
                "best_parent_prob": 0.98,
                "n_eff_parents": 1.05,
            },
            {
                "child_layer": "L0",
                "parent_layer": "L1",
                "best_parent": "L1_3",
                "best_parent_prob": 0.98,
                "n_eff_parents": 1.05,
            },
            {
                "child_layer": "L1",
                "parent_layer": "L2",
                "best_parent": "L2_0",
                "best_parent_prob": 0.95,
                "n_eff_parents": 1.05,
            },
            {
                "child_layer": "L1",
                "parent_layer": "L2",
                "best_parent": "L2_1",
                "best_parent_prob": 0.45,
                "n_eff_parents": 2.5,
            },
            {
                "child_layer": "L1",
                "parent_layer": "L2",
                "best_parent": "L2_1",
                "best_parent_prob": 0.95,
                "n_eff_parents": 1.05,
            },
            {
                "child_layer": "L1",
                "parent_layer": "L2",
                "best_parent": "L2_1",
                "best_parent_prob": 0.95,
                "n_eff_parents": 1.05,
            },
        ],
        index=model.factor_names[0] + model.factor_names[1],
    )

    res = scd.tl.find_sensible_top_layer(
        model,
        n_eff_parents_max=1.5,
        min_clear_fraction=0.5,
    )

    parent_diag = res["parent_diagnostics"]
    l1_to_l2 = parent_diag[parent_diag["child_layer"] == "L1"].set_index(
        "parent_factor"
    )
    assert l1_to_l2.loc["L2_0", "n_clear_children"] == 1
    assert l1_to_l2.loc["L2_1", "n_clear_children"] == 2
    assert not bool(l1_to_l2.loc["L2_0", "clear_parent"])
    assert bool(l1_to_l2.loc["L2_1", "clear_parent"])

    transitions = res["transition_diagnostics"]
    l1_row = transitions[transitions["child_layer"] == "L1"].iloc[0]
    assert l1_row["max_n_clear_children"] == 2
    assert l1_row["clear_fraction"] == 0.5
    assert bool(l1_row["transition_ok"])


def test_refit_top_factors_rebuilds_geometric_frontier(monkeypatch):
    model = _make_model_with_pmeans([5, 4, 3, 1])
    model._has_fit = True
    init_kwargs = {}

    def capture_init(**kwargs):
        init_kwargs.update(kwargs)

    monkeypatch.setattr(model, "init_var_params", capture_init)
    monkeypatch.setattr(model, "_learn", lambda **kwargs: None)
    monkeypatch.setattr(model, "_invalidate_cached_diagnostics", lambda: None)
    monkeypatch.setattr(model, "clear_runtime_cache", lambda **kwargs: None)

    model.fit(
        n_epoch=1,
        collapse_l1_fraction=None,
        refit_top_factors=["L2_0", "L0_4"],
    )

    assert model.layer_sizes[0] == 5
    assert model.layer_sizes[-2:] == [2, 1]
    assert len(init_kwargs["init_w"]) == len(model.layer_sizes)
    assert len(init_kwargs["init_z"]) == len(model.layer_sizes)
    assert init_kwargs["init_w"][-2].shape == (2, model.layer_sizes[-3])
    assert init_kwargs["init_z"][-2].shape == (model.n_cells, 2)


def test_refit_top_factors_preserves_current_depth_not_schedule(monkeypatch):
    model = _make_model_with_pmeans([5, 4, 3, 1])
    model._has_fit = True
    # Simulate a stale/expanded schedule. Refit should still preserve current
    # non-root depth (3 layers) plus root.
    model.n_layers_schedule = 6
    init_kwargs = {}

    def capture_init(**kwargs):
        init_kwargs.update(kwargs)

    monkeypatch.setattr(model, "init_var_params", capture_init)
    monkeypatch.setattr(model, "_learn", lambda **kwargs: None)
    monkeypatch.setattr(model, "_invalidate_cached_diagnostics", lambda: None)
    monkeypatch.setattr(model, "clear_runtime_cache", lambda **kwargs: None)

    model.fit(
        n_epoch=1,
        collapse_l1_fraction=None,
        refit_top_factors=["L2_0", "L0_4"],
    )

    assert model.layer_sizes == [5, 3, 2, 1]
    assert len(model.layer_sizes) == 4
    assert len(init_kwargs["init_w"]) == 4
    assert init_kwargs["init_w"][-2].shape == (2, 3)


def test_from_reference_initializes_new_model_hierarchy(monkeypatch):
    ref = _make_model_with_pmeans([5, 3, 1])
    adata = ref.adata.copy()
    adata.obs["modality"] = ["scrna"] * (adata.n_obs // 2) + ["spatial"] * (
        adata.n_obs - adata.n_obs // 2
    )

    model = scd.scDEF.from_reference(
        reference_model=ref,
        adata=adata,
        batch_key="modality",
        reference_obs="scrna",
        query_obs="spatial",
    )
    init_kwargs = {}

    def capture_init(**kwargs):
        init_kwargs.update(kwargs)

    monkeypatch.setattr(model, "init_var_params", capture_init)
    monkeypatch.setattr(model, "_learn", lambda **kwargs: None)
    monkeypatch.setattr(model, "_invalidate_cached_diagnostics", lambda: None)
    monkeypatch.setattr(model, "clear_runtime_cache", lambda **kwargs: None)

    model.fit(n_epoch=1, collapse_l1_fraction=None)

    assert model.layer_sizes == ref.layer_sizes
    assert model.n_batches == 2
    assert init_kwargs["init_budgets"] is True
    assert len(init_kwargs["init_w"]) == ref.n_layers
    np.testing.assert_allclose(
        init_kwargs["init_w"][0],
        np.asarray(ref.pmeans["L0W"], dtype=np.float32)[ref.factor_lists[0]],
        rtol=1e-5,
        atol=1e-5,
    )
    assert init_kwargs["init_z"][0].shape == (adata.n_obs, ref.layer_sizes[0])


def test_from_hierarchy_initializes_explicit_w_matrices(monkeypatch):
    ref = _make_model_with_pmeans([5, 3, 1])
    w_matrices = [
        np.asarray(ref.pmeans["L0W"], dtype=np.float32),
        np.asarray(ref.pmeans["L1W"], dtype=np.float32),
        np.asarray(ref.pmeans["L2W"], dtype=np.float32),
    ]
    model = scd.scDEF.from_hierarchy(ref.adata, w_matrices)
    init_kwargs = {}

    def capture_init(**kwargs):
        init_kwargs.update(kwargs)

    monkeypatch.setattr(model, "init_var_params", capture_init)
    monkeypatch.setattr(model, "_learn", lambda **kwargs: None)
    monkeypatch.setattr(model, "_invalidate_cached_diagnostics", lambda: None)
    monkeypatch.setattr(model, "clear_runtime_cache", lambda **kwargs: None)

    model.fit(n_epoch=1, collapse_l1_fraction=None)

    assert model.layer_sizes == [5, 3, 1]
    assert init_kwargs["init_budgets"] is True
    assert len(init_kwargs["init_w"]) == 3
    np.testing.assert_allclose(init_kwargs["init_w"][1], w_matrices[1])


def test_from_hierarchy_accepts_reference_model(monkeypatch):
    ref = _make_model_with_pmeans([5, 3, 1])
    ref.factor_lists = [
        np.array([0, 2, 4], dtype=int),
        np.array([1, 2], dtype=int),
        np.array([0], dtype=int),
    ]
    ref.set_factor_names()

    model = scd.scDEF.from_hierarchy(ref.adata, ref)
    init_kwargs = {}

    def capture_init(**kwargs):
        init_kwargs.update(kwargs)

    monkeypatch.setattr(model, "init_var_params", capture_init)
    monkeypatch.setattr(model, "_learn", lambda **kwargs: None)
    monkeypatch.setattr(model, "_invalidate_cached_diagnostics", lambda: None)
    monkeypatch.setattr(model, "clear_runtime_cache", lambda **kwargs: None)

    model.fit(n_epoch=1, collapse_l1_fraction=None)

    assert model.layer_sizes == [3, 2, 1]
    np.testing.assert_allclose(
        init_kwargs["init_w"][0],
        np.asarray(ref.pmeans["L0W"], dtype=np.float32)[ref.factor_lists[0]],
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        init_kwargs["init_w"][1],
        np.asarray(ref.pmeans["L1W"], dtype=np.float32)[
            np.ix_(ref.factor_lists[1], ref.factor_lists[0])
        ],
        rtol=1e-5,
        atol=1e-5,
    )


def test_shell_graph_omits_width_one_root():
    model = _make_model_with_pmeans([5, 3, 1])
    for layer_idx, layer_name in enumerate(model.layer_names):
        model.adata.obs[layer_name] = model.factor_names[layer_idx][0]

    graph = scd.pl.make_graph(
        model,
        show_signatures=False,
        shell=True,
        top_genes=[0, 0],
    )
    graph_body = "\n".join(graph.body)

    assert model.factor_names[-1][0] not in graph_body
    assert model.factor_names[-2][0] in graph_body


@pytest.mark.parametrize(
    "learn_budgets_on_refit,expected_stop",
    [(False, 1.0), (True, 0.0)],
)
def test_refit_budget_learning_option(
    monkeypatch, learn_budgets_on_refit, expected_stop
):
    model = _make_model_with_pmeans([5, 3, 1])
    model._has_fit = True
    learn_kwargs = {}

    def capture_learn(**kwargs):
        learn_kwargs.update(kwargs)

    monkeypatch.setattr(model, "_learn", capture_learn)
    monkeypatch.setattr(model, "update_model_size", lambda **kwargs: None)
    monkeypatch.setattr(model, "update_model_priors", lambda **kwargs: None)
    monkeypatch.setattr(model, "_invalidate_cached_diagnostics", lambda: None)
    monkeypatch.setattr(model, "clear_runtime_cache", lambda **kwargs: None)

    orig_init = model.init_var_params

    def wrapped_init(**kwargs):
        assert kwargs["init_budgets"] is False
        orig_init(**kwargs)

    monkeypatch.setattr(model, "init_var_params", wrapped_init)

    model.fit(
        n_epoch=1,
        collapse_l1_fraction=None,
        learn_budgets_on_refit=learn_budgets_on_refit,
    )

    assert float(learn_kwargs["stop_cell_budgets"]) == expected_stop
    assert float(learn_kwargs["stop_gene_budgets"]) == expected_stop


def test_first_fit_does_not_collapse_layer_sizes(monkeypatch):
    layer_sizes = [12, 10, 4, 1]
    model = _make_model_with_pmeans(layer_sizes)
    model._has_fit = False
    updated_sizes = []

    orig_update = scd.scDEF.update_model_size

    def wrapped_update(self, **kwargs):
        if kwargs.get("layer_sizes") is not None:
            updated_sizes.append(list(kwargs["layer_sizes"]))
        return orig_update(self, **kwargs)

    monkeypatch.setattr(scd.scDEF, "update_model_size", wrapped_update)
    monkeypatch.setattr(model, "_learn", lambda **kwargs: None)
    monkeypatch.setattr(model, "init_var_params", lambda **kwargs: None)
    monkeypatch.setattr(model, "_invalidate_cached_diagnostics", lambda: None)
    monkeypatch.setattr(model, "clear_runtime_cache", lambda **kwargs: None)

    model.fit(n_epoch=1, collapse_l1_fraction=0.8)

    assert updated_sizes == []
