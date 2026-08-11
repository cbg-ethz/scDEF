from unittest.mock import patch

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
    model.annotate_adata()
    rng = np.random.default_rng(0)
    for idx, name in enumerate(model.layer_names):
        model.adata.obsm[f"X_{name}"] = np.abs(
            rng.standard_normal((adata.n_obs, len(model.factor_lists[idx])))
        )
    return model


def test_tl_umap_computes_layers_descending():
    model = _make_model(layer_sizes=(3, 2, 1))
    seen = []

    def _neighbors(adata, **kwargs):
        rep = kwargs.get("use_rep", "")
        for idx, name in enumerate(model.layer_names):
            if rep == f"X_{name}":
                seen.append(idx)
                break

    def _umap(adata):
        adata.obsm["X_umap"] = np.zeros((adata.n_obs, 2))

    with patch("scanpy.pp.neighbors", side_effect=_neighbors), patch(
        "scanpy.tl.umap", side_effect=_umap
    ):
        scd.tl.umap(model)

    assert seen == [1, 0]


def test_tl_umap_restores_original_x_umap():
    model = _make_model(layer_sizes=(3, 2))
    n = model.adata.n_obs
    original = np.column_stack([np.arange(n, dtype=float), np.arange(n) + 0.5])
    model.adata.obsm["X_umap"] = original.copy()

    def _neighbors(adata, **kwargs):
        pass

    def _umap(adata):
        adata.obsm["X_umap"] = np.full((adata.n_obs, 2), 99.0)

    with patch("scanpy.pp.neighbors", side_effect=_neighbors), patch(
        "scanpy.tl.umap", side_effect=_umap
    ):
        scd.tl.umap(model, layers=[1, 0])

    np.testing.assert_allclose(model.adata.obsm["X_umap"], original)
    assert "X_umap_L0" in model.adata.obsm
    assert "X_umap_L1" in model.adata.obsm


def test_tl_umap_restores_neighbors_and_obsp():
    import scipy.sparse as sp

    model = _make_model(layer_sizes=(3, 2))
    n = model.adata.n_obs
    model.adata.uns["neighbors"] = {
        "params": {"use_rep": "X_pca", "n_neighbors": 5},
    }
    model.adata.obsp["connectivities"] = sp.eye(n, format="csr")
    model.adata.obsp["distances"] = sp.csr_matrix(np.zeros((n, n)))

    def _neighbors(adata, **kwargs):
        adata.uns["neighbors"] = {
            "params": {"use_rep": kwargs.get("use_rep"), "n_neighbors": 99},
        }
        adata.obsp["connectivities"] = sp.csr_matrix(np.full((n, n), 7.0))

    def _umap(adata):
        adata.obsm["X_umap"] = np.zeros((adata.n_obs, 2))
        adata.uns["umap"] = {"params": {"a": 0.0}}

    with patch("scanpy.pp.neighbors", side_effect=_neighbors), patch(
        "scanpy.tl.umap", side_effect=_umap
    ):
        scd.tl.umap(model, layers=[0, 1])

    assert model.adata.uns["neighbors"]["params"]["use_rep"] == "X_pca"
    assert model.adata.uns["neighbors"]["params"]["n_neighbors"] == 5
    assert "umap" not in model.adata.uns
    np.testing.assert_array_equal(
        model.adata.obsp["connectivities"].toarray(), sp.eye(n).toarray()
    )


def test_tl_umap_removes_x_umap_when_absent_initially():
    model = _make_model(layer_sizes=(3, 2))
    assert "X_umap" not in model.adata.obsm

    def _neighbors(adata, **kwargs):
        pass

    def _umap(adata):
        adata.obsm["X_umap"] = np.zeros((adata.n_obs, 2))

    with patch("scanpy.pp.neighbors", side_effect=_neighbors), patch(
        "scanpy.tl.umap", side_effect=_umap
    ):
        scd.tl.umap(model, layers=[1, 0])

    assert "X_umap" not in model.adata.obsm
    assert "neighbors" not in model.adata.uns
    assert "connectivities" not in model.adata.obsp
    assert "X_umap_L0" in model.adata.obsm


def test_pl_umap_restores_original_x_umap_after_plot():
    import matplotlib.pyplot as plt

    model = _make_model(layer_sizes=(3, 2))
    n = model.adata.n_obs
    l0_name = model.layer_names[0]
    l1_name = model.layer_names[1]
    original = np.column_stack([np.zeros(n), np.ones(n)])
    model.adata.obsm["X_umap"] = original.copy()
    model.adata.obsm[f"X_umap_{l0_name}"] = np.column_stack(
        [np.full(n, 2.0), np.zeros(n)]
    )
    model.adata.obsm[f"X_umap_{l1_name}"] = np.column_stack(
        [np.full(n, 9.0), np.full(n, 8.0)]
    )
    model.adata.obs["_test"] = "x"

    seen_umap = []

    def _fake_umap(adata, **kwargs):
        seen_umap.append(np.asarray(adata.obsm["X_umap"], dtype=float).copy())
        return kwargs.get("ax")

    with patch("scanpy.pl.umap", side_effect=_fake_umap):
        scd.pl.umap(model, color=["_test"], layers=[1, 0], show=False)
    plt.close("all")

    np.testing.assert_allclose(seen_umap[0], model.adata.obsm[f"X_umap_{l1_name}"])
    np.testing.assert_allclose(seen_umap[1], model.adata.obsm[f"X_umap_{l0_name}"])
    np.testing.assert_allclose(model.adata.obsm["X_umap"], original)
