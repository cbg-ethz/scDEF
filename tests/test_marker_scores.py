"""Marker scores (cNMF gene spectra scores) from usages: equal to an explicit OLS on the standardised counts per
million, for dense and sparse counts, and the top-scoring genes of each factor are the genes up-regulated in the
cells that use it."""
import numpy as np
import scipy.sparse as sp

from scdef.tools.factor import marker_scores_from_usages


def _toy(seed=0, n_per=120, n_genes=60, k=3, n_up=5):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(k), n_per)
    rate = np.full((k * n_per, n_genes), 2.0)
    for g in range(k):
        rate[groups == g, g * n_up : (g + 1) * n_up] *= 6.0
    counts = rng.poisson(rate * rng.uniform(0.5, 1.5, size=(k * n_per, 1)))
    usages = np.full((k * n_per, k), 0.05)
    usages[np.arange(k * n_per), groups] = 0.9
    usages += rng.uniform(0, 0.05, size=usages.shape)
    usages /= usages.sum(1, keepdims=True)
    return counts, usages, groups, n_up


def _explicit(counts, usages):
    c = counts.astype(float)
    cpm = c / c.sum(1, keepdims=True) * 1e6
    z = (cpm - cpm.mean(0)) / cpm.std(0, ddof=1)
    beta, *_ = np.linalg.lstsq(usages, z, rcond=None)
    return beta


def test_matches_explicit_ols_dense_and_sparse():
    counts, usages, _, _ = _toy()
    ref = _explicit(counts, usages)
    np.testing.assert_allclose(
        marker_scores_from_usages(counts, usages), ref, rtol=1e-6, atol=1e-8
    )
    np.testing.assert_allclose(
        marker_scores_from_usages(sp.csr_matrix(counts), usages),
        ref,
        rtol=1e-6,
        atol=1e-8,
    )


def test_top_genes_are_the_up_regulated_genes():
    counts, usages, _, n_up = _toy()
    scores = marker_scores_from_usages(counts, usages)
    for k in range(usages.shape[1]):
        top = set(np.argsort(-scores[k])[:n_up])
        assert top == set(range(k * n_up, (k + 1) * n_up))
