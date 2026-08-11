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


def test_batch_purity_soft_computed():
    model = _make_model()
    scd.tl.factor_diagnostics(model, batch_key="patient_id")
    fo = model.adata.uns["factor_obs"]
    assert "batch_purity_soft" in fo.columns
    soft = fo["batch_purity_soft"].to_numpy(dtype=float)
    hard = fo["batch_purity"].to_numpy(dtype=float)
    finite_soft = soft[np.isfinite(soft)]
    finite_hard = hard[np.isfinite(hard)]
    assert finite_soft.size > 0
    assert finite_hard.size > 0
    assert np.all((finite_soft >= 0.0) & (finite_soft <= 1.0))
    assert np.all((finite_hard >= 0.0) & (finite_hard <= 1.0))


def test_batch_purity_soft_plot_filter():
    model = _make_model()
    scd.tl.factor_diagnostics(model, batch_key="patient_id")
    ax = scd.pl.factor_diagnostics(model, batch_purity_soft_max=1.0, show=False)
    assert ax is not None


def test_factor_diagnostics_batch_limits_include_zero_cell_factors():
    """Batch-colored plots draw zero-cell factors so autoscale includes them."""
    model = _make_model(layer_sizes=(3, 2))
    l0 = model.layer_names[0]
    names = list(model.factor_names[0])
    x_vals = np.array([5.0, 4.0, 0.01], dtype=float)
    y_vals = np.array([1.0, 1.1, 8.0], dtype=float)
    model.adata.uns["factor_obs"] = pd.DataFrame(
        {
            "child_layer": [l0, l0, l0],
            "BRD": x_vals,
            "ARD": [1.0, 0.5, 0.0],
            "n_eff_parents": y_vals,
            "avg_n_eff_parents": y_vals,
            "K_parents": [3, 3, 3],
            "batch_purity": [0.05, 0.05, np.nan],
            "original_factor_idx": [0, 1, 2],
        },
        index=names,
    )
    ax = scd.pl.factor_diagnostics(
        model,
        batch_purity_max=0.1,
        annotate_factors=True,
        show=False,
    )
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    assert xlim[0] <= x_vals[2] <= xlim[1]
    assert ylim[0] <= y_vals[2] <= ylim[1]
    assert names[2] in {t.get_text() for t in ax.texts}


def test_set_technical_factors_batch_purity_soft_max():
    model = _make_model(layer_sizes=(4, 2))
    scd.tl.factor_diagnostics(model, batch_key="patient_id")
    scd.tl.set_technical_factors(
        model,
        brd_min=0.0,
        ard_min=0.0,
        min_cells_lower=0,
        batch_purity_soft_max=1.0,
    )
    n_loose = int(model.adata.uns["factor_obs"]["technical"].sum())
    model.adata.uns["factor_obs"]["technical"] = False
    scd.tl.set_technical_factors(
        model,
        brd_min=0.0,
        ard_min=0.0,
        min_cells_lower=0,
        batch_purity_soft_max=0.0,
    )
    n_strict = int(model.adata.uns["factor_obs"]["technical"].sum())
    assert n_strict >= n_loose


def test_get_effective_factors_batch_purity_soft_max():
    model = _make_model(layer_sizes=(4, 2))
    scd.tl.factor_diagnostics(model, batch_key="patient_id")
    keep_loose = model.get_effective_factors(
        brd_min=0.0,
        ard_min=0.0,
        min_cells=0,
        batch_purity_soft_max=1.0,
    )
    keep_strict = model.get_effective_factors(
        brd_min=0.0,
        ard_min=0.0,
        min_cells=0,
        batch_purity_soft_max=0.0,
    )
    assert len(keep_strict) <= len(keep_loose)

    fo = model.adata.uns["factor_obs"]
    l0 = fo[fo["child_layer"] == model.layer_names[0]]
    idx = l0["original_factor_idx"].to_numpy(dtype=int)
    soft = l0["batch_purity_soft"].to_numpy(dtype=float)
    for k in keep_strict:
        row = np.where(idx == k)[0]
        if row.size:
            assert soft[row[0]] <= 0.0
