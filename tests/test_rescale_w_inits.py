"""Tests for refit W warm-start rescaling by layer size."""
import logging

import numpy as np
import pytest

from scdef.models._scdef import scDEF


def test_rescale_w_inits_scales_by_old_over_new():
    model = object.__new__(scDEF)
    model.layer_names = ["L0", "L1"]
    model.logger = logging.getLogger("test_rescale_w_inits")

    init_w = [
        np.full((2, 3), 2.0, dtype=np.float32),
        np.full((1, 2), 4.0, dtype=np.float32),
    ]
    out = scDEF._rescale_w_inits_for_layer_sizes(
        model, old_layer_sizes=[10, 5], new_layer_sizes=[5, 5], init_w=init_w
    )
    np.testing.assert_allclose(out[0], 4.0, rtol=1e-6)
    np.testing.assert_allclose(out[1], 4.0, rtol=1e-6)


def test_refit_old_layer_sizes_for_w_uses_old_keep():
    model = object.__new__(scDEF)
    init_w = [np.zeros((3, 2), dtype=np.float32), np.zeros((2, 3), dtype=np.float32)]
    old_sizes = model._refit_old_layer_sizes_for_w(
        [12, 10, 4, 1],
        [8, 4, 1],
        init_w,
        old_keep=[0, 2],
        n_original=4,
    )
    assert old_sizes == [12, 4]


def test_refit_scales_w_after_filter(monkeypatch):
    scanpy = pytest.importorskip("scanpy")
    scd = pytest.importorskip("scdef")

    adata = scanpy.datasets.pbmc3k()[:30].copy()
    adata.X = adata.X.toarray()
    model = scd.scDEF(adata, layer_sizes=[8, 4, 1], seed=0)
    model.init_var_params(init_budgets=True, init_alpha=True, nmf_init=False)
    model.set_posterior_means()
    n_genes = model.pmeans["L0W"].shape[1]
    model.pmeans["L0W"] = np.ones((8, n_genes), dtype=np.float32) * 2.0
    model.pmeans["L1W"] = np.ones((4, 8), dtype=np.float32) * 3.0
    model.factor_lists = [
        np.array([0, 1, 2, 3, 4], dtype=int),
        np.array([0, 1, 2], dtype=int),
        np.array([0], dtype=int),
    ]
    model._has_fit = True

    init_kwargs = {}

    def capture_init(**kwargs):
        init_kwargs.update(kwargs)

    monkeypatch.setattr(model, "init_var_params", capture_init)
    monkeypatch.setattr(model, "_learn", lambda **kwargs: None)
    monkeypatch.setattr(model, "_invalidate_cached_diagnostics", lambda: None)
    monkeypatch.setattr(model, "clear_runtime_cache", lambda **kwargs: None)

    model.fit(n_epoch=1, collapse_l1_fraction=None)

    assert init_kwargs["init_w"][0].shape == (5, n_genes)
    np.testing.assert_allclose(
        init_kwargs["init_w"][0],
        2.0 * (8.0 / 5.0),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        init_kwargs["init_w"][1],
        3.0 * (4.0 / 3.0),
        rtol=1e-5,
    )
