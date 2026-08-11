"""Tests for refit BRD/ARD warm-start rescaling."""
import numpy as np

from scdef.models._scdef import scDEF


def test_rescale_relevance_preserves_ratios_and_geometric_mean():
    model = object.__new__(scDEF)
    values = np.array([1.3, 134.0, 34.0, 3.0], dtype=np.float64)
    out = scDEF._rescale_relevance_init(model, values, target_mean=1.0, max_ratio=None)

    geo = np.exp(np.mean(np.log(out)))
    assert np.isclose(geo, 1.0, rtol=1e-5)
    ratios_in = values / values[0]
    ratios_out = out / out[0]
    np.testing.assert_allclose(ratios_out, ratios_in, rtol=1e-5)


def test_rescale_relevance_caps_spread():
    model = object.__new__(scDEF)
    values = np.array([1.0, 100.0], dtype=np.float64)
    out = scDEF._rescale_relevance_init(model, values, target_mean=1.0, max_ratio=10.0)
    assert np.isclose(np.exp(np.mean(np.log(out))), 1.0, rtol=1e-5)
    assert out.max() / out.min() <= 10.0 + 1e-6
