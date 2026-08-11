import numpy as np
import pytest

scanpy = pytest.importorskip("scanpy")
scd = pytest.importorskip("scdef")


@pytest.fixture(scope="module")
def pbmc_adata():
    adata = scanpy.datasets.pbmc3k()[:80].copy()
    adata.X = adata.X.toarray()
    rng = np.random.default_rng(0)
    adata.obs["patient_id"] = rng.choice(["P1", "P2", "P3"], size=adata.n_obs)
    return adata


@pytest.fixture(scope="module")
def batch_corrected_model(pbmc_adata):
    """Stage 1: fit with batch_key to get a batch-corrected hierarchy."""
    model = scd.scDEF(pbmc_adata, batch_key="patient_id", layer_sizes=[6, 3], seed=0)
    model.fit(n_epoch=5)
    return model


def test_decompose_returns_fitted_model(batch_corrected_model):
    model = scd.scDEF.decompose_batch_effects(
        batch_corrected_model,
        n_epoch=5,
    )
    assert hasattr(model, "elbos") and len(model.elbos) > 0
    assert model._has_fit
    # The gene side of batch_key is always discarded: gene_scale is a single shared
    # row. The cell side (per-batch cell_scale prior) is kept by default.
    assert model.n_gene_scale_batches == 1
    assert np.asarray(model.pmeans["gene_scale"]).shape[0] == 1
    assert model.batch_key == batch_corrected_model.batch_key
    assert model.batch_gene_scale is False
    assert model.n_batches == batch_corrected_model.n_batches


def test_decompose_batch_cell_scale_false_drops_batch_key(batch_corrected_model):
    """batch_cell_scale=False reproduces the historical batch_key=None construction."""
    model = scd.scDEF.decompose_batch_effects(
        batch_corrected_model,
        batch_cell_scale=False,
        n_epoch=5,
    )
    assert model._has_fit
    assert model.n_batches == 1
    assert model.batch_key is None
    assert model.n_gene_scale_batches == 1
    assert np.unique(np.asarray(model.batch_lib_ratio)).size == 1


def test_decompose_keeps_per_batch_cell_prior(batch_corrected_model):
    """With the default, cell_scale shrinks toward a per-batch Gamma prior."""
    model = scd.scDEF.decompose_batch_effects(
        batch_corrected_model,
        n_epoch=5,
    )
    lib_ratio = np.asarray(model.batch_lib_ratio).reshape(-1)
    assert np.unique(lib_ratio).size == model.n_batches > 1


def test_decompose_upper_z_frozen(batch_corrected_model):
    """L1+ z should remain fixed (structural constraint from Stage 1)."""
    ref_factor_lists = [
        np.asarray(f, dtype=int) for f in batch_corrected_model.factor_lists
    ]

    model = scd.scDEF.decompose_batch_effects(
        batch_corrected_model,
        n_epoch=5,
    )

    for layer_idx in range(1, batch_corrected_model.n_layers):
        name = batch_corrected_model.layer_names[layer_idx]
        keep = ref_factor_lists[layer_idx]
        expected_z = np.asarray(
            batch_corrected_model.pmeans[f"{name}z"], dtype=np.float32
        )[:, keep]

        new_name = model.layer_names[layer_idx]
        actual_z = np.asarray(model.pmeans[f"{new_name}z"])

        np.testing.assert_allclose(
            actual_z,
            expected_z,
            rtol=1e-4,
            err_msg=f"z at layer {layer_idx} ({new_name}) should be frozen",
        )


def test_decompose_l1_w_can_change(batch_corrected_model):
    """L1 W should be re-learned (allowed to diverge from reference)."""
    ref_factor_lists = [
        np.asarray(f, dtype=int) for f in batch_corrected_model.factor_lists
    ]
    ref_name = batch_corrected_model.layer_names[1]
    keep = ref_factor_lists[1]
    parent_keep = ref_factor_lists[0]
    ref_w1 = np.asarray(batch_corrected_model.pmeans[f"{ref_name}W"], dtype=np.float32)[
        np.ix_(keep, parent_keep)
    ]

    model = scd.scDEF.decompose_batch_effects(
        batch_corrected_model,
        n_epoch=5,
    )

    new_w1 = np.asarray(model.pmeans[f"{model.layer_names[1]}W"])
    assert not np.allclose(
        new_w1, ref_w1, rtol=1e-2
    ), "L1 W should diverge from reference after re-learning"


def test_decompose_l0_w_differs_from_reference(batch_corrected_model):
    ref_w0 = np.asarray(
        batch_corrected_model.pmeans[f"{batch_corrected_model.layer_names[0]}W"],
        dtype=np.float32,
    )[batch_corrected_model.factor_lists[0]]

    model = scd.scDEF.decompose_batch_effects(
        batch_corrected_model,
        n_epoch=5,
    )

    new_w0 = np.asarray(model.pmeans[f"{model.layer_names[0]}W"])
    assert not np.allclose(
        new_w0, ref_w0, rtol=1e-2
    ), "L0 W should differ from reference after re-learning"


def test_decompose_preserves_layer_structure(batch_corrected_model):
    model = scd.scDEF.decompose_batch_effects(
        batch_corrected_model,
        n_epoch=5,
    )
    assert model.n_layers == batch_corrected_model.n_layers
    ref_sizes = [len(f) for f in batch_corrected_model.factor_lists]
    new_sizes = list(model.layer_sizes)
    assert new_sizes == ref_sizes


def test_decompose_requires_enough_layers():
    adata = scanpy.datasets.pbmc3k()[:40].copy()
    adata.X = adata.X.toarray()
    model = scd.scDEF(adata, layer_sizes=[4], seed=0)
    model.fit(n_epoch=3)
    with pytest.raises(ValueError, match="at least 2 layers"):
        scd.scDEF.decompose_batch_effects(model, n_epoch=3)


def test_decompose_with_nmf_init(batch_corrected_model):
    model = scd.scDEF.decompose_batch_effects(
        batch_corrected_model,
        n_epoch=5,
        nmf_init=True,
    )
    assert hasattr(model, "elbos") and len(model.elbos) > 0
    assert model._has_fit


def test_decompose_warm_starts_gene_scale_from_reference(batch_corrected_model):
    ref_gs = np.asarray(batch_corrected_model.pmeans["gene_scale"], dtype=np.float32)
    if ref_gs.ndim == 1:
        ref_gs = ref_gs[None, :]
    expected = np.exp(np.mean(np.log(np.clip(ref_gs, 1e-6, None)), axis=0))

    model = scd.scDEF.decompose_batch_effects(
        batch_corrected_model,
        n_epoch=1,
    )
    new_gs = np.asarray(model.pmeans["gene_scale"], dtype=np.float32).reshape(-1)
    corr = np.corrcoef(np.log1p(new_gs), np.log1p(expected))[0, 1]
    assert corr > 0.9


def test_decompose_resets_brd_ard(batch_corrected_model):
    ref_brd = np.asarray(batch_corrected_model.pmeans["brd"], dtype=np.float32)[
        batch_corrected_model.factor_lists[0]
    ]
    ref_ard = np.asarray(
        batch_corrected_model.pmeans["factor_means"], dtype=np.float32
    )[batch_corrected_model.factor_lists[0]]

    model = scd.scDEF.decompose_batch_effects(
        batch_corrected_model,
        n_epoch=5,
    )
    new_brd = np.asarray(model.pmeans["brd"])
    new_ard = np.asarray(model.pmeans["factor_means"])
    assert not np.allclose(new_brd, ref_brd, rtol=1e-2)
    assert not np.allclose(new_ard, ref_ard, rtol=1e-2)


@pytest.fixture(scope="module")
def three_layer_model(pbmc_adata):
    """3-layer model with batch correction for top_layer tests."""
    model = scd.scDEF(pbmc_adata, batch_key="patient_id", layer_sizes=[8, 4, 2], seed=0)
    model.fit(n_epoch=5)
    return model


def test_decompose_top_layer_2(three_layer_model):
    """top_layer=2: L0 and L1 fully re-learned, L2 z frozen, L2 W re-learned."""
    model = scd.scDEF.decompose_batch_effects(
        three_layer_model,
        top_layer=2,
        n_epoch=5,
    )
    assert model._has_fit
    assert model.n_layers == three_layer_model.n_layers

    # L2 z should be frozen
    ref_factor_lists = [
        np.asarray(f, dtype=int) for f in three_layer_model.factor_lists
    ]
    keep = ref_factor_lists[2]
    expected_z = np.asarray(
        three_layer_model.pmeans[f"{three_layer_model.layer_names[2]}z"],
        dtype=np.float32,
    )[:, keep]
    actual_z = np.asarray(model.pmeans[f"{model.layer_names[2]}z"])
    np.testing.assert_allclose(
        actual_z,
        expected_z,
        rtol=1e-4,
        err_msg="L2 z should be frozen with top_layer=2",
    )

    # L1 z should NOT be frozen (it's below top_layer)
    keep_l1 = ref_factor_lists[1]
    ref_z1 = np.asarray(
        three_layer_model.pmeans[f"{three_layer_model.layer_names[1]}z"],
        dtype=np.float32,
    )[:, keep_l1]
    actual_z1 = np.asarray(model.pmeans[f"{model.layer_names[1]}z"])
    assert not np.allclose(
        actual_z1, ref_z1, rtol=1e-2
    ), "L1 z should be re-learned when top_layer=2"
