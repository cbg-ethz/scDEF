import numpy as np
import pytest

scanpy = pytest.importorskip("scanpy")
scd = pytest.importorskip("scdef")


def test_build_z_init_l0_other_mass_none_uses_k0_fraction():
    markers = {"A": ["CD3D"], "B": ["MS4A1"]}
    adata = scanpy.datasets.pbmc3k()[:50].copy()
    adata.X = adata.X.toarray()
    scanpy.pp.filter_genes(adata, min_cells=3)
    adata = adata[:, :150]

    model = scd.iscDEF(
        adata,
        markers_dict=markers,
        markers_layer=0,
        add_other=1,
        seed=1,
    )
    k0 = model.layer_sizes[0]
    init_l0 = model._build_z_init_l0_from_marker_scores(
        other_mass=None, other_mode="uniform"
    )
    assert k0 == 3
    np.testing.assert_allclose(init_l0[:, 2], 1.0, rtol=1e-4)
    np.testing.assert_allclose(init_l0.sum(axis=1), k0, rtol=0.05)


def test_build_z_init_l0_markers_layer_zero_with_other():
    markers = {
        "A": ["CD3D", "CD3E"],
        "B": ["MS4A1"],
    }
    adata = scanpy.datasets.pbmc3k()[:100].copy()
    adata.X = adata.X.toarray()
    scanpy.pp.filter_genes(adata, min_cells=3)
    adata = adata[:, :200]

    model = scd.iscDEF(
        adata,
        markers_dict=markers,
        markers_layer=0,
        add_other=2,
        seed=1,
    )
    init_l0 = model._build_z_init_l0_from_marker_scores(
        temperature=1.0,
        other_mass=0.1,
    )

    assert init_l0.shape == (adata.n_obs, model.layer_sizes[0])
    assert model.layer_sizes[0] == 4
    row_sums = init_l0.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.full(adata.n_obs, 4.0), rtol=0.05)
    assert np.all(init_l0[:, 2] == init_l0[:, 3])
    assert np.all(init_l0[:, 2] > 0)


def test_build_z_init_all_layers_markers_layer_positive():
    adata = scanpy.datasets.pbmc3k()[:40].copy()
    adata.X = adata.X.toarray()
    scanpy.pp.filter_genes(adata, min_cells=3)
    adata = adata[:, :130]
    g = list(adata.var_names[:8])
    markers = {"A": g[:4], "B": g[4:8]}
    model = scd.iscDEF(
        adata,
        markers_dict=markers,
        markers_layer=2,
        add_other=1,
        decay_factor=2,
        seed=1,
    )
    assert model.n_layers == 4
    assert model.layer_sizes == [12, 6, 3, 1]
    inits = model._build_z_init_all_hierarchical_layers_from_marker_scores()
    assert len(inits) == model.n_layers
    assert inits[-1] is None
    for ell in range(3):
        assert inits[ell].shape == (adata.n_obs, model.layer_sizes[ell])
        np.testing.assert_allclose(
            inits[ell].sum(axis=1),
            np.full(adata.n_obs, float(model.layer_sizes[ell])),
            rtol=1e-4,
        )


def test_iscdef_markers_layer_positive_allows_multiple_add_other():
    adata = scanpy.datasets.pbmc3k()[:40].copy()
    adata.X = adata.X.toarray()
    scanpy.pp.filter_genes(adata, min_cells=3)
    adata = adata[:, :130]
    g = list(adata.var_names[:8])
    markers = {"A": g[:4], "B": g[4:8]}
    model = scd.iscDEF(
        adata,
        markers_dict=markers,
        markers_layer=2,
        add_other=2,
        decay_factor=2,
        seed=1,
    )
    assert model.n_markers == 4
    assert model.layer_sizes == [16, 8, 4, 1]
    init_l0 = model._build_z_init_l0_from_marker_scores()
    assert init_l0.shape == (adata.n_obs, 16)
    other_cols = init_l0[:, 8:16]
    assert np.all(other_cols > 0)
    assert np.allclose(other_cols[:, :4], other_cols[:, 4:5], rtol=1e-5)


def test_other_z_init_inverse_union_varies_by_cell():
    adata = scanpy.datasets.pbmc3k()[:80].copy()
    adata.X = adata.X.toarray()
    scanpy.pp.filter_genes(adata, min_cells=3)
    adata = adata[:, :200]
    g = list(adata.var_names[:8])
    markers = {"A": g[:4], "B": g[4:8]}

    model = scd.iscDEF(
        adata,
        markers_dict=markers,
        markers_layer=0,
        add_other=1,
        seed=1,
    )
    init_inv = model._build_z_init_l0_from_marker_scores(other_mode="inverse_union")
    init_uni = model._build_z_init_l0_from_marker_scores(other_mode="uniform")

    other_col = model.layer_sizes[0] - 1
    assert np.std(init_inv[:, other_col]) > 1e-4
    np.testing.assert_allclose(
        init_uni[:, other_col], np.full(adata.n_obs, init_uni[0, other_col]), rtol=1e-5
    )
    np.testing.assert_allclose(init_inv.sum(axis=1), model.layer_sizes[0], rtol=1e-4)

    union_scores = model._compute_union_marker_scores()
    affinity = model._other_affinity_from_union_scores(union_scores, temperature=1.0)
    corr = np.corrcoef(init_inv[:, other_col], affinity)[0, 1]
    assert corr > 0.5


def test_fit_uses_score_init_not_nmf(caplog):
    adata = scanpy.datasets.pbmc3k()[:60].copy()
    adata.X = adata.X.toarray()
    scanpy.pp.filter_genes(adata, min_cells=3)
    adata = adata[:, :120]
    genes = list(adata.var_names[:4])
    markers = {"A": genes[:2], "B": genes[2:4]}

    model = scd.iscDEF(adata, markers_dict=markers, markers_layer=0, seed=1)
    with caplog.at_level("WARNING"):
        model.fit(
            n_epoch=1,
            n_rounds=1,
            z_init_from_score_genes=True,
            nmf_init=True,
        )
    assert any("does not use nmf_init" in r.message for r in caplog.records)
    layer_name = model.layer_names[0]
    z_mean = np.array(model.pmeans[f"{layer_name}z"])
    assert z_mean.shape == (adata.n_obs, model.layer_sizes[0])
    assert np.all(np.isfinite(z_mean))
