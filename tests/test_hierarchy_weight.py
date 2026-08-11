import numpy as np
import pytest

scanpy = pytest.importorskip("scanpy")
scd = pytest.importorskip("scdef")


def test_scdef_hierarchy_weight_scales_cov_alpha():
    adata = scanpy.datasets.pbmc3k()[:60].copy()
    adata.X = adata.X.toarray()

    base = scd.scDEF(
        adata,
        layer_sizes=[8, 1],
        set_alpha_from_cov=True,
        hierarchy_weight=1.0,
        seed=1,
    )
    scaled = scd.scDEF(
        adata,
        layer_sizes=[8, 1],
        set_alpha_from_cov=True,
        hierarchy_weight=0.25,
        seed=1,
    )
    median_lib = float(np.median(base.batch_lib_sizes))
    assert base.alpha == pytest.approx(median_lib / 8.0)
    assert scaled.alpha == pytest.approx(0.25 * median_lib / 8.0)


def test_iscdef_default_hierarchy_weight():
    adata = scanpy.datasets.pbmc3k()[:40].copy()
    adata.X = adata.X.toarray()
    genes = list(adata.var_names[:4])
    markers = {"A": genes[:2], "B": genes[2:4]}

    model = scd.iscDEF(
        adata,
        markers_dict=markers,
        markers_layer=0,
        set_alpha_from_cov=True,
        seed=1,
    )
    assert model.hierarchy_weight == 0.25
    median_lib = float(np.median(model.batch_lib_sizes))
    assert model.alpha == pytest.approx(0.25 * median_lib / model.layer_sizes[0])
