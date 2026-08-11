import numpy as np
import pytest

scanpy = pytest.importorskip("scanpy")
scd = pytest.importorskip("scdef")


def _make_model(layer_sizes=(3, 2, 1)):
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


def test_get_layer_term_means_excludes_technical_l0():
    model = _make_model()
    tech_name = model.factor_names[0][0]
    model.adata.uns["factor_obs"].loc[tech_name, "technical"] = True

    full = scd.tl.factor._get_layer_term_means(model, layer_idx=1, drop_factors=[])
    filt = scd.tl.factor._get_layer_term_means(model, layer_idx=1, drop_factors=None)

    w0 = np.asarray(model.pmeans["L0W"])[model.factor_lists[0]]
    chain = np.asarray(model.pmeans["L1W"])[model.factor_lists[1]][
        :, model.factor_lists[0]
    ]
    expected = chain[:, 1:].dot(w0[1:])
    np.testing.assert_allclose(filt, expected, rtol=1e-5, atol=1e-5)
    assert not np.allclose(full, filt)


def test_get_rankings_upper_layer_uses_non_technical_l0():
    model = _make_model()
    tech_name = model.factor_names[0][0]
    model.adata.uns["factor_obs"].loc[tech_name, "technical"] = True

    rankings, _ = model.get_rankings(
        layer_idx=1, top_genes=10, genes=True, return_scores=True, sorted_scores=False
    )
    w0 = np.asarray(model.pmeans["L0W"])[model.factor_lists[0]]
    chain = np.asarray(model.pmeans["L1W"])[model.factor_lists[1]][
        :, model.factor_lists[0]
    ]
    expected_top_genes = np.array(model.adata.var_names)[
        chain[0, 1:].dot(w0[1:]).argsort()[::-1][:10]
    ].tolist()
    assert rankings[0] == expected_top_genes
