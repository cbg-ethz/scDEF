import numpy as np
import pytest

scanpy = pytest.importorskip("scanpy")
scd = pytest.importorskip("scdef")


def _make_model(layer_sizes=(3, 2)):
    adata = scanpy.datasets.pbmc3k()[:30].copy()
    adata.X = adata.X.toarray()
    model = scd.scDEF(adata, layer_sizes=list(layer_sizes), seed=0)
    model.init_var_params(init_budgets=True, init_alpha=True, nmf_init=False)
    model.set_posterior_means()
    model.factor_lists = [np.arange(s, dtype=int) for s in layer_sizes]
    model.set_factor_names()
    scd.tl.factor_diagnostics(model)
    if "technical" not in model.adata.uns["factor_obs"].columns:
        model.adata.uns["factor_obs"]["technical"] = False
    return model


def test_annotate_assignments_include_technical_factors():
    model = _make_model(layer_sizes=(2, 1))
    tech_name = model.factor_names[0][0]
    model.adata.uns["factor_obs"].loc[tech_name, "technical"] = True

    z = np.asarray(model.pmeans["L0z"], dtype=float).copy()
    z[:, model.factor_lists[0][0]] = 10.0
    z[:, model.factor_lists[0][1]] = 0.0
    model.pmeans["L0z"] = z

    model.annotate_adata()
    assert (model.adata.obs["L0"] == tech_name).all()


def test_normalize_cellscores_includes_all_factors():
    model = _make_model(layer_sizes=(2, 1))
    tech_name = model.factor_names[0][0]
    model.adata.uns["factor_obs"].loc[tech_name, "technical"] = True

    z = np.asarray(model.pmeans["L0z"], dtype=float).copy()
    z[:, model.factor_lists[0][0]] = 3.0
    z[:, model.factor_lists[0][1]] = 1.0
    model.pmeans["L0z"] = z
    model.annotate_adata()

    probs = model.adata.obsm["X_L0_probs"]
    assert np.all(probs[:, 0] > 0)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)


def test_drop_technical_filters_factor_lists_and_reannotates():
    model = _make_model(layer_sizes=(3, 2))
    tech_name = model.factor_names[0][0]
    model.adata.uns["factor_obs"].loc[tech_name, "technical"] = True

    scd.tl.drop_technical(model)
    assert len(model.factor_lists[0]) == 2
    assert 0 not in model.factor_lists[0]
    assert not model.adata.uns["factor_obs"]["technical"].any()
    assert "X_L0" in model.adata.obsm
    assert model.adata.obsm["X_L0"].shape[1] == 2


def test_drop_technical_raises_when_layer_would_be_empty():
    model = _make_model(layer_sizes=(1, 1))
    scd.tl.set_technical_factors(model, factors=[model.factor_names[0][0]])
    with pytest.raises(ValueError, match="no factors left"):
        scd.tl.drop_technical(model)
