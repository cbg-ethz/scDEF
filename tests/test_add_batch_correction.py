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
def reference_model(pbmc_adata):
    model = scd.scDEF(pbmc_adata, layer_sizes=[4, 2], seed=0)
    model.fit(n_epoch=3)
    return model


def test_add_batch_correction_returns_fitted_model(reference_model, pbmc_adata):
    model = scd.scDEF.add_batch_correction(
        reference_model,
        batch_key="patient_id",
        n_epoch=3,
    )
    assert model.n_batches == 3
    assert model.batch_key == "patient_id"
    assert hasattr(model, "elbos") and len(model.elbos) > 0


def test_freeze_w_preserves_w(reference_model, pbmc_adata):
    ref_w = {}
    for idx, name in enumerate(reference_model.layer_names):
        ref_w[name] = np.array(reference_model.pmeans[f"{name}W"], copy=True)

    model = scd.scDEF.add_batch_correction(
        reference_model,
        batch_key="patient_id",
        freeze_w=True,
        n_epoch=3,
    )

    for idx, name in enumerate(model.layer_names):
        kept = np.asarray(reference_model.factor_lists[idx], dtype=int)
        if idx == 0:
            expected = ref_w[name][kept]
        else:
            parent_kept = np.asarray(reference_model.factor_lists[idx - 1], dtype=int)
            expected = ref_w[name][np.ix_(kept, parent_kept)]
        np.testing.assert_allclose(
            model.pmeans[f"{name}W"],
            expected,
            rtol=1e-4,
            err_msg=f"W for {name} drifted despite freeze_w=True",
        )


def test_freeze_w_false_allows_w_to_change(reference_model, pbmc_adata):
    ref_w = {}
    for name in reference_model.layer_names:
        ref_w[name] = np.array(reference_model.pmeans[f"{name}W"], copy=True)

    model = scd.scDEF.add_batch_correction(
        reference_model,
        batch_key="patient_id",
        freeze_w=False,
        n_epoch=5,
    )

    any_changed = False
    for idx, name in enumerate(model.layer_names):
        kept = np.asarray(reference_model.factor_lists[idx], dtype=int)
        if idx == 0:
            expected = ref_w[name][kept]
        else:
            parent_kept = np.asarray(reference_model.factor_lists[idx - 1], dtype=int)
            expected = ref_w[name][np.ix_(kept, parent_kept)]
        if not np.allclose(model.pmeans[f"{name}W"], expected, rtol=1e-4):
            any_changed = True
    assert any_changed, "W should drift when freeze_w=False"


def test_add_batch_correction_clears_stale_factor_obs_full(reference_model):
    scd.tl.factor_diagnostics(reference_model)
    assert "factor_obs_full" in reference_model.adata.uns
    assert "factor_obs" in reference_model.adata.uns

    model = scd.scDEF.add_batch_correction(
        reference_model,
        batch_key="patient_id",
        n_epoch=3,
    )
    assert "factor_obs_full" not in model.adata.uns
    assert "factor_obs" not in model.adata.uns


def test_add_batch_correction_custom_adata(reference_model, pbmc_adata):
    subset = pbmc_adata[:60].copy()
    subset.obs["patient_id"] = np.random.default_rng(1).choice(
        ["P1", "P2"], size=subset.n_obs
    )
    model = scd.scDEF.add_batch_correction(
        reference_model,
        batch_key="patient_id",
        adata=subset,
        n_epoch=3,
    )
    assert model.n_cells == subset.n_obs
    assert model.n_batches == 2
