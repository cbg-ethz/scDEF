"""The fused training step is cached on the instance; the key must be exact.

`jit` re-traces when argument shapes change, but is blind to values captured in a
closure. The fused step closes over both learning rates (baked into the optax
transforms), `freeze_w` (which decides a branch at trace time) and the W slot
range (via n_layers). Reusing a cached step after any of those changed would keep
training with the previous setting while looking perfectly healthy, so each one
is part of the key and is checked here.
"""

import pytest

scanpy = pytest.importorskip("scanpy")
scd = pytest.importorskip("scdef")


@pytest.fixture(scope="module")
def model():
    adata = scanpy.datasets.pbmc3k()[:40].copy()
    adata.X = adata.X.toarray()
    scanpy.pp.filter_genes(adata, min_cells=3)
    adata = adata[:, :80].copy()
    return scd.scDEF(adata, layer_sizes=[4, 2, 1], seed=0)


def _steps(model, **kw):
    args = dict(num_samples=3, lr=1e-1, local_lr=1e-2, freeze_w=False)
    args.update(kw)
    return model._get_or_build_learn_step_fns(**args)


def test_same_config_hits_the_cache(model):
    a = _steps(model)
    b = _steps(model)
    assert a[2] is b[2] and a[3] is b[3], "identical config should reuse the steps"
    assert a[0] is b[0] and a[1] is b[1], "and the optimizers they close over"


@pytest.mark.parametrize(
    "changed",
    [
        {"lr": 5e-2},
        {"local_lr": 5e-3},
        {"freeze_w": True},
        {"num_samples": 4},
    ],
)
def test_config_change_rebuilds(model, changed):
    """A stale step would silently train with the previous setting."""
    baseline = _steps(model)
    other = _steps(model, **changed)
    assert (
        other[2] is not baseline[2] or other[3] is not baseline[3]
    ), f"changing {changed} must not reuse the cached step"


def test_optimizers_match_the_cached_steps(model):
    """The caller inits optimizer state from the returned transforms.

    They have to be the same objects the steps closed over, or the state layout
    could disagree with what the compiled step expects.
    """
    local_opt, global_opt, _, _ = _steps(model)
    assert local_opt.init(model.local_params) is not None
    assert global_opt.init(model.global_params) is not None
    again = _steps(model)
    assert again[0] is local_opt and again[1] is global_opt


def test_resetting_the_gradient_cache_rebuilds_the_steps(model):
    """The fused steps close over the gradient callables.

    `clear_runtime_cache` and `sscDEF.fit` reset `_learn_jit_cache` so the
    gradients are re-traced against the current model. A step cached before the
    reset would keep calling the old gradients, making that reset a no-op.
    """
    before = _steps(model)
    model._learn_jit_cache = None
    after = _steps(model)
    assert after[2] is not before[2] and after[3] is not before[3]
    model.clear_runtime_cache()
    assert model._learn_step_cache is None


def test_save_and_load_after_fit(model, tmp_path):
    """The step cache holds optax closures, which cannot be pickled.

    `fit` ends by clearing the runtime caches; fill the step cache again so the
    save sees the worst case.
    """
    model.fit(n_epoch=2)
    assert model._learn_step_cache is None, "fit should clear the step cache"
    _steps(model)
    assert model._learn_step_cache is not None
    model.save(tmp_path / "m", overwrite=True)
    loaded = scd.scDEF.load(tmp_path / "m", adata=model.adata)
    assert loaded._learn_step_cache is None and loaded._learn_jit_cache is None
    assert [list(f) for f in loaded.factor_lists] == [
        list(f) for f in model.factor_lists
    ]
