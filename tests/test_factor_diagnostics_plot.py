import numpy as np
import pandas as pd
import pytest

scanpy = pytest.importorskip("scanpy")
scd = pytest.importorskip("scdef")


def _make_model(layer_sizes=(3, 2)):
    adata = scanpy.datasets.pbmc3k()[:40].copy()
    adata.X = adata.X.toarray()
    adata.obs["patient_id"] = np.random.choice(["P1", "P2", "P3"], size=adata.n_obs)
    model = scd.scDEF(adata, layer_sizes=list(layer_sizes), seed=0)
    model.init_var_params(init_budgets=True, init_alpha=True, nmf_init=False)
    model.set_posterior_means()
    model.factor_lists = [np.arange(s, dtype=int) for s in layer_sizes]
    model.set_factor_names()
    model.annotate_adata()
    return model


def test_factor_diagnostics_n_cells():
    model = _make_model()
    scd.tl.factor_diagnostics(model)
    fo = model.adata.uns["factor_obs"]
    assert "n_cells" in fo.columns
    assert fo["n_cells"].sum() == model.n_cells * (model.n_layers - 1)
    assert np.all(fo["n_cells"].to_numpy(dtype=int) >= 0)
    assert "confident_signatures" in model.adata.uns


def test_factor_obs_n_cells_matches_graph_assignments():
    from scdef.tools.factor import count_hard_assigned_cells

    model = _make_model()
    scd.tl.factor_diagnostics(model)
    fo = model.adata.uns["factor_obs"]

    for layer_idx in range(model.n_layers - 1):
        layer_name = model.layer_names[layer_idx]
        for name_idx, slot in enumerate(model.factor_lists[layer_idx]):
            factor_name = model.factor_names[layer_idx][name_idx]
            obs_count = int(
                np.sum(model.adata.obs[layer_name].astype(str) == str(factor_name))
            )
            if factor_name in fo.index:
                assert int(fo.at[factor_name, "n_cells"]) == obs_count
            assert obs_count == count_hard_assigned_cells(model, layer_idx, int(slot))

    g = scd.pl.make_graph(
        model,
        assignments=True,
        n_cells_label=True,
        show_signatures=False,
        show=False,
    )
    assert g is not None
    for row in fo.itertuples():
        if int(getattr(row, "n_cells", 0)) <= 0:
            continue
        assert f"({int(row.n_cells)} cells)" in g.source


def test_make_graph_bottom_layer():
    model = _make_model()
    scd.tl.factor_diagnostics(model)

    g_all = scd.pl.make_graph(model, show_signatures=False, show=False)
    for name in model.factor_names[0]:
        assert name in g_all.source

    g_upper = scd.pl.make_graph(
        model, bottom_layer=1, show_signatures=False, show=False
    )
    for name in model.factor_names[0]:
        assert name not in g_upper.source
    for name in model.factor_names[1]:
        assert name in g_upper.source


def test_factor_diagnostics_default_uses_ard_size_not_color():
    model = _make_model()
    scd.tl.factor_diagnostics(model)
    ax = scd.pl.factor_diagnostics(model, show=False)
    assert ax.get_xlabel() == "BRD"
    assert ax.figure.axes == [ax]


def test_factor_diagnostics_batch_color_with_ard_size():
    model = _make_model()
    scd.tl.factor_diagnostics(model, batch_key="patient_id")
    # batch_panel=False keeps the single-panel layout this test is about; with
    # batch diagnostics present the default now adds the per-batch panel.
    ax = scd.pl.factor_diagnostics(
        model, batch_purity_max=0.5, batch_panel=False, show=False
    )
    assert len(ax.figure.axes) == 2
    assert ax.get_xlabel() == "BRD"


def test_factor_diagnostics_batch_panel_auto():
    """With batch diagnostics and default axes, two panels are drawn."""
    import numpy as np

    model = _make_model()
    scd.tl.factor_diagnostics(model, batch_key="patient_id")
    axes = scd.pl.factor_diagnostics(model, show=False)
    assert isinstance(axes, np.ndarray) and len(axes) == 2
    assert axes[0].get_xlabel() == "BRD"
    assert axes[1].get_xlabel() == "Dominant batch fraction"
    # Explicit axes opt out of the second panel.
    single = scd.pl.factor_diagnostics(model, x="n_cells", show=False)
    assert not isinstance(single, np.ndarray)


def test_factor_diagnostics_batch_panel_absent_without_batch_key():
    """No batch diagnostics -> single panel, unchanged behaviour."""
    import numpy as np

    model = _make_model()
    scd.tl.factor_diagnostics(model)
    ax = scd.pl.factor_diagnostics(model, show=False)
    assert not isinstance(ax, np.ndarray)
    assert ax.get_xlabel() == "BRD"


def test_factor_diagnostics_signature_confidence_color():
    model = _make_model()
    scd.tl.factor_diagnostics(model)
    ax = scd.pl.factor_diagnostics(
        model,
        color="signature_confidence",
        show=False,
    )
    assert len(ax.figure.axes) == 2
    assert ax.get_xlabel() == "BRD"


def test_get_effective_factors_brd_exceptional():
    model = _make_model(layer_sizes=(4, 2))
    l0 = model.layer_names[0]
    names = list(model.factor_names[0])
    model.adata.uns["factor_obs"] = pd.DataFrame(
        {
            "child_layer": [l0] * 4,
            "BRD": [2.0, 5.0, 0.5, 1.5],
            "ARD": [1.0, 1.0, 1.0, 1.0],
            "avg_n_eff_parents": [2.0, 2.0, 1.0, 1.0],
            "clarity_score_01": [0.1, 0.1, 0.9, 0.9],
            "original_factor_idx": [0, 1, 2, 3],
        },
        index=names,
    )
    model.pmeans["L0z"] = np.ones((model.n_cells, 4), dtype=float)
    keep = model.get_effective_factors(
        brd_min=1.0,
        ard_min=0.0,
        min_cells=0,
        n_eff_parents_max=1.5,
        brd_exceptional=4.0,
    )
    assert 0 not in keep  # high neff, BRD below exceptional
    assert 1 in keep  # high neff but BRD >= brd_exceptional
    assert 2 not in keep  # low BRD
    assert 3 in keep  # passes BRD and neff


def test_get_effective_factors_brd_exceptional_disabled_by_default():
    model = _make_model(layer_sizes=(2, 2))
    l0 = model.layer_names[0]
    names = list(model.factor_names[0])
    model.adata.uns["factor_obs"] = pd.DataFrame(
        {
            "child_layer": [l0] * 2,
            "BRD": [5.0, 1.5],
            "ARD": [1.0, 1.0],
            "avg_n_eff_parents": [2.0, 1.0],
            "clarity_score_01": [0.1, 0.9],
            "original_factor_idx": [0, 1],
        },
        index=names,
    )
    model.pmeans["L0z"] = np.ones((model.n_cells, 2), dtype=float)
    keep = model.get_effective_factors(
        brd_min=1.0,
        ard_min=0.0,
        min_cells=0,
        n_eff_parents_max=1.5,
    )
    assert 0 not in keep
    assert 1 in keep


def test_filter_factors_keep_forces_l0_factor():
    model = _make_model(layer_sizes=(4, 2))
    l0 = model.layer_names[0]
    names = list(model.factor_names[0])
    model.adata.uns["factor_obs"] = pd.DataFrame(
        {
            "child_layer": [l0] * 4,
            "BRD": [0.2, 5.0, 1.5, 1.5],
            "ARD": [1.0, 1.0, 1.0, 1.0],
            "avg_n_eff_parents": [1.0, 2.0, 1.0, 1.0],
            "clarity_score_01": [0.9, 0.1, 0.9, 0.9],
            "original_factor_idx": [0, 1, 2, 3],
        },
        index=names,
    )
    model.pmeans["L0z"] = np.ones((model.n_cells, 4), dtype=float)
    model.filter_factors(
        brd_min=1.0,
        ard_min=0.0,
        min_cells_lower=0,
        annotate=False,
        keep=[names[0]],
    )
    assert 0 in model.factor_lists[0]
    assert 1 not in model.factor_lists[0]


def test_factor_diagnostics_plot_brd_exceptional_vline():
    model = _make_model(layer_sizes=(3, 2))
    l0 = model.layer_names[0]
    names = list(model.factor_names[0])
    model.adata.uns["factor_obs"] = pd.DataFrame(
        {
            "child_layer": [l0] * 3,
            "BRD": [1.5, 4.5, 0.5],
            "ARD": [1.0, 1.0, 1.0],
            "avg_n_eff_parents": [1.0, 2.0, 1.0],
            "K_parents": [3, 3, 3],
            "original_factor_idx": [0, 1, 2],
        },
        index=names,
    )
    ax = scd.pl.factor_diagnostics(
        model,
        brd_min=1.0,
        brd_exceptional=4.0,
        n_eff_parents_max=1.5,
        show=False,
    )
    vline_xs = sorted(
        line.get_xdata()[0]
        for line in ax.lines
        if line.get_xdata()[0] == line.get_xdata()[1]
    )
    assert vline_xs == [1.0, 4.0]
