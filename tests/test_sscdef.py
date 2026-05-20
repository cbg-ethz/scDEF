import numpy as np
import pytest

scanpy = pytest.importorskip("scanpy")
scd = pytest.importorskip("scdef")


def test_sscdef_supervised_top_z_and_make_graph():
    adata = scanpy.datasets.pbmc3k()[:80].copy()
    adata.X = adata.X.toarray()
    scanpy.pp.filter_genes(adata, min_cells=3)
    adata = adata[:, :120]
    np.random.seed(0)
    types = np.random.choice(["A", "B"], size=adata.n_obs)
    adata.obs["cell_type"] = types

    model = scd.sscDEF(
        adata,
        top_key="cell_type",
        n_layers=2,
        n_factors=4,
        seed=1,
    )
    assert model.n_layers == 2
    assert model.layer_sizes == [4, 2]
    assert model.layer_names[-1] == "cell_type"
    assert len(model.w_priors) == 2
    assert model.w_priors[1][0].shape == (2, 4)

    top = model._supervised_top_z
    assert top.shape == (adata.n_obs, 2)
    for i in range(adata.n_obs):
        assert top[i].sum() == pytest.approx(1.0 + model._Z_OFF_TYPE, rel=1e-5)
        assert top[i].max() == pytest.approx(model._Z_ON)

    model.fit(n_epoch=2, n_rounds=1)
    model.annotate_adata()

    z_top = np.array(model.pmeans[f"{model.layer_names[-1]}z"])
    np.testing.assert_allclose(z_top, top, rtol=1e-5, atol=1e-5)

    start, end = model._layer_column_range(model.supervised_top_layer_idx)
    z_shapes, z_rates = model.local_params[1]
    z_rates_np = np.array(z_rates)[:, start:end]
    z_mean = np.exp(np.array(z_shapes)[:, start:end] + 0.5 * np.exp(z_rates_np) ** 2)
    np.testing.assert_allclose(z_mean, top, rtol=1e-3, atol=1e-3)

    scd.tl.set_confident_signatures(model, mc_samples=10)
    g = scd.pl.make_graph(model, show_signatures=True, top_genes=3)
    assert g is not None


def test_sscdef_layer_sizes_match_scdef_geometric_without_root():
    import anndata as ad

    n_factors, n_layers, top_k = 20, 3, 5
    expected = scd.sscDEF._geometric_layer_sizes_no_root(n_factors, top_k, n_layers)
    adata = ad.AnnData(np.ones((10, 5), dtype=np.float32))
    scdef_model = scd.scDEF(
        adata, n_factors=n_factors, top_factors=top_k, n_layers=n_layers, seed=0
    )
    # scDEF appends a width-1 root when top_factors > 1
    assert scdef_model.layer_sizes == expected + [1]
    assert expected[-1] == top_k
