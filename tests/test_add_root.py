import numpy as np
import pytest

scanpy = pytest.importorskip("scanpy")
scd = pytest.importorskip("scdef")


def _small_adata():
    adata = scanpy.datasets.pbmc3k()[:40].copy()
    adata.X = adata.X.toarray()
    scanpy.pp.filter_genes(adata, min_cells=3)
    return adata[:, :130]


def test_iscdef_markers_layer_positive_defaults_add_root():
    adata = _small_adata()
    g = list(adata.var_names[:8])
    markers = {"A": g[:4], "B": g[4:8]}
    model = scd.iscDEF(
        adata,
        markers_dict=markers,
        markers_layer=2,
        decay_factor=2,
        seed=1,
    )
    assert model.add_root is True
    assert model.layer_sizes[-1] == 1


def test_iscdef_add_root_appends_layer():
    adata = _small_adata()
    g = list(adata.var_names[:8])
    markers = {"A": g[:4], "B": g[4:8]}
    model = scd.iscDEF(
        adata,
        markers_dict=markers,
        markers_layer=2,
        add_other=1,
        decay_factor=2,
        add_root=True,
        seed=1,
    )
    assert model.add_root is True
    assert model.n_layers == 4
    assert model.layer_sizes == [12, 6, 3, 1]
    assert model.layer_names[-1] == "root"
    assert model.factor_names[model.markers_layer] == ["A", "B", "other0"]
    assert model.factor_names[-1] == ["root_0"]


def test_iscdef_add_root_two_phase_fit():
    adata = _small_adata()
    g = list(adata.var_names[:8])
    markers = {"A": g[:4], "B": g[4:8]}
    model = scd.iscDEF(
        adata,
        markers_dict=markers,
        markers_layer=2,
        decay_factor=2,
        add_root=True,
        seed=1,
    )
    model.fit(n_epoch=2, n_rounds=1)
    assert model.root_epochs == 10
    assert len(model.elbos) == 2


def test_iscdef_markers_layer_zero_disables_add_root_by_default():
    adata = _small_adata()
    genes = list(adata.var_names[:4])
    markers = {"A": genes[:2], "B": genes[2:4]}
    model = scd.iscDEF(adata, markers_dict=markers, markers_layer=0, seed=1)
    assert model.add_root is False


def test_iscdef_add_root_rejected_for_markers_layer_zero():
    adata = _small_adata()
    genes = list(adata.var_names[:4])
    markers = {"A": genes[:2], "B": genes[2:4]}
    with pytest.raises(ValueError, match="add_root=True requires markers_layer > 0"):
        scd.iscDEF(
            adata,
            markers_dict=markers,
            markers_layer=0,
            add_root=True,
            seed=1,
        )


def test_sscdef_add_root_layer_sizes_and_two_phase_fit():
    adata = _small_adata()
    np.random.seed(0)
    adata.obs["cell_type"] = np.random.choice(["A", "B"], size=adata.n_obs)
    model = scd.sscDEF(
        adata,
        top_key="cell_type",
        n_layers=2,
        n_factors=4,
        add_root=True,
        seed=1,
    )
    assert model.add_root is True
    assert model.n_layers == 3
    assert model.layer_sizes == [4, 2, 1]
    assert model.supervised_top_layer_idx == 1
    assert model.layer_names == ["L0", "cell_type", "root"]

    model.fit(n_epoch=2, n_rounds=1)
    assert model.root_epochs == 10
    assert len(model.elbos) == 2

    top = model._supervised_top_z
    z_top = np.array(
        model.pmeans[f"{model.layer_names[model.supervised_top_layer_idx]}z"]
    )
    np.testing.assert_allclose(z_top, top, rtol=1e-5, atol=1e-5)
