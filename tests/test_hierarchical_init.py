import numpy as np
import pytest

scanpy = pytest.importorskip("scanpy")
sc = scanpy
scd = pytest.importorskip("scdef")


def _make_model(layer_sizes=(4, 2, 1), seed=0):
    adata = scanpy.datasets.pbmc3k()[:60].copy()
    adata.X = adata.X.toarray()
    sc.pp.pca(adata, n_comps=20)
    model = scd.scDEF(adata, layer_sizes=list(layer_sizes), seed=seed)
    return model


def test_get_hierarchical_init_returns_z_and_w():
    model = _make_model(layer_sizes=(4, 2, 1))
    init_z, init_w = model.get_hierarchical_init()
    assert len(init_z) == model.n_layers
    assert len(init_w) == model.n_layers
    assert init_z[0].shape == (model.n_cells, 4)
    assert init_w[0] is None
    assert init_w[1].shape == (2, 4)
    assert init_w[2].shape == (1, 2)


def test_hierarchical_init_w_upper_layers_are_containment():
    model = _make_model(layer_sizes=(4, 2, 1))
    init_z, init_w = model.get_hierarchical_init(z_on=1.0, z_off=0.1)
    assert init_w[0] is None
    for layer_idx in range(1, model.n_layers):
        w = init_w[layer_idx]
        active = w >= 0.5
        assert np.all(active.sum(axis=0) <= 1)
        assert np.all((w[active] - 1.0) < 1e-6)
        assert np.all((w[~active] - 0.1) < 1e-6)
