"""Budget prior vs init behavior for scDEF with and without batch_key."""

import numpy as np
import pytest
from anndata import AnnData

import scdef as scd


def _toy_adata(n_cells=40, n_genes=20, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.poisson(lam=5.0, size=(n_cells, n_genes)).astype(float)
    batch = np.array(["A"] * (n_cells // 2) + ["B"] * (n_cells - n_cells // 2))
    return AnnData(X, obs={"batch": batch})


def _gene_scale_init_mean(model):
    gp = np.array(model.global_params[0])
    return np.exp(gp[0])


@pytest.mark.parametrize("batch_key", [None, "batch"])
def test_gene_prior_is_mean_over_var(batch_key):
    adata = _toy_adata()
    model = scd.scDEF(adata, batch_key=batch_key, layer_sizes=[4, 2], seed=0)
    gene_size = np.sum(adata.X, axis=0)
    expected_scalar = float(np.mean(gene_size)) / float(np.var(gene_size))

    ratio = np.asarray(model.gene_ratio)
    if batch_key is None:
        np.testing.assert_allclose(float(ratio), expected_scalar, rtol=1e-5)
    else:
        for i, label in enumerate(model.batches):
            cells = np.where(adata.obs["batch"].values == label)[0]
            batch_gene_size = np.sum(adata.X[cells], axis=0)
            expected = float(np.mean(batch_gene_size)) / float(np.var(batch_gene_size))
            np.testing.assert_allclose(ratio[i, 0], expected, rtol=1e-5)


@pytest.mark.parametrize("batch_key", [None, "batch"])
def test_gene_init_uses_expression_anchor_not_prior(batch_key):
    adata = _toy_adata()
    model = scd.scDEF(adata, batch_key=batch_key, layer_sizes=[4, 2], seed=0)
    model.init_var_params(init_budgets=True, init_alpha=False)

    gene_init = _gene_scale_init_mean(model)
    expected = 1.0 / np.asarray(model.gene_ratio_init)
    np.testing.assert_allclose(gene_init, expected, rtol=0.1)

    prior_mean = 1.0 / np.asarray(model.gene_ratio)
    if prior_mean.ndim == 0:
        prior_mean = np.full_like(gene_init, float(prior_mean))
    # Init should generally differ from the flat prior mean across genes.
    assert not np.allclose(gene_init, prior_mean, rtol=0.05)
