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


def _spy_fit(model, **kwargs):
    """Fit, recording whether the hierarchical initialization was actually used."""
    called = {"hierarchical": False}
    original = model.get_hierarchical_init

    def spy(*args, **kw):
        called["hierarchical"] = True
        return original(*args, **kw)

    model.get_hierarchical_init = spy
    model.fit(n_epoch=2, n_rounds=1, **kwargs)
    return called["hierarchical"]


def test_hierarchical_init_is_the_default_when_pca_is_present():
    """`fit()` should reach for the better initialization without being asked."""
    model = _make_model()
    assert "X_pca" in model.adata.obsm
    assert _spy_fit(model) is True


def test_fit_falls_back_when_no_pca_is_available():
    """A model with no PCA must still fit.

    The default is `None` rather than `True` precisely so this call works: a hard
    `True` would raise KeyError out of `get_hierarchical_init` and break the
    simplest possible usage, `scd.scDEF(adata).fit()`.
    """
    adata = scanpy.datasets.pbmc3k()[:60].copy()
    adata.X = adata.X.toarray()
    assert "X_pca" not in adata.obsm
    model = scd.scDEF(adata, layer_sizes=[4, 2, 1], seed=0)
    assert _spy_fit(model) is False


def test_nmf_init_still_works_under_the_new_default():
    """`fit(nmf_init=True)` must not trip the mutual-exclusion check."""
    model = _make_model()
    assert _spy_fit(model, nmf_init=True) is False


def test_hierarchical_init_can_be_switched_off():
    model = _make_model()
    assert _spy_fit(model, hierarchical_init=False) is False


def test_explicit_true_still_raises_without_pca():
    """Asking for it by name must fail loudly rather than fall back silently."""
    adata = scanpy.datasets.pbmc3k()[:60].copy()
    adata.X = adata.X.toarray()
    model = scd.scDEF(adata, layer_sizes=[4, 2, 1], seed=0)
    with pytest.raises(KeyError, match="X_pca"):
        model.fit(n_epoch=2, n_rounds=1, hierarchical_init=True)


def test_explicit_both_still_raises():
    model = _make_model()
    with pytest.raises(ValueError, match="only one of"):
        model.fit(n_epoch=2, n_rounds=1, nmf_init=True, hierarchical_init=True)
