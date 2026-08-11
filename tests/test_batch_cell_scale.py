"""Decoupling the cell side of ``batch_key`` from the gene side.

``batch_key`` used to control two independent things at once:

* the **cell side** -- ``batch_indices_onehot``, ``batch_lib_sizes`` and the per-batch
  ``batch_lib_ratio`` that ``cell_scale`` shrinks toward. Gene-independent, so it can
  only carry sequencing depth.
* the **gene side** -- a ``gene_scale`` row per batch via per-batch ``gene_ratio`` /
  ``gene_ratio_init``. Gene-specific, so it can absorb gene programmes.

``batch_gene_scale=False`` keeps the cell side and drops the gene side. These tests do
no fitting: they assert on what ``load_adata`` / ``init_var_params`` set, and evaluate
the ELBO exactly once per configuration.
"""

import inspect

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
ad = pytest.importorskip("anndata")
pd = pytest.importorskip("pandas")
scd = pytest.importorskip("scdef")

from scdef.models.extend import (  # noqa: E402
    _resolve_decompose_batch_kwargs,
    decompose_batch_effects,
)

N_PER_BATCH = 30
N_GENES = 40
DEPTH_MULTIPLIER = 3.0


@pytest.fixture(scope="module")
def batched_adata():
    """Two batches that differ ~3x in library size."""
    rng = np.random.default_rng(0)
    n = 2 * N_PER_BATCH
    X = rng.poisson(1.0, size=(n, N_GENES)).astype(float)
    X[N_PER_BATCH:] *= DEPTH_MULTIPLIER
    obs = pd.DataFrame(
        {"b": ["A"] * N_PER_BATCH + ["B"] * N_PER_BATCH},
        index=[f"c{i}" for i in range(n)],
    )
    var = pd.DataFrame(index=[f"g{i}" for i in range(N_GENES)])
    return ad.AnnData(X, obs=obs, var=var)


def _make(adata, **kwargs):
    return scd.scDEF(
        adata.copy(), layer_sizes=[4, 2], seed=0, logginglevel=50, **kwargs
    )


@pytest.fixture(scope="module")
def models(batched_adata):
    return {
        "none": _make(batched_adata),
        "per_batch": _make(batched_adata, batch_key="b"),
        "shared": _make(batched_adata, batch_key="b", batch_gene_scale=False),
    }


# --------------------------------------------------------------------------------
# Cell side is kept when the gene side is switched off
# --------------------------------------------------------------------------------


def test_shared_gene_scale_keeps_per_batch_cell_prior(models):
    m = models["shared"]
    assert m.batch_key == "b"
    assert m.n_batches == 2
    assert m.batch_gene_scale is False

    lib_ratio = np.asarray(m.batch_lib_ratio).reshape(-1)
    a = np.unique(lib_ratio[:N_PER_BATCH])
    b = np.unique(lib_ratio[N_PER_BATCH:])
    assert (
        a.size == 1 and b.size == 1
    ), "batch_lib_ratio must be constant within a batch"
    assert not np.isclose(
        a[0], b[0]
    ), "batches differ in depth, so their lib ratios must differ"

    # The library sizes themselves reflect the depth gap.
    lib = np.asarray(m.batch_lib_sizes)
    ratio = float(np.median(lib[N_PER_BATCH:]) / np.median(lib[:N_PER_BATCH]))
    assert ratio == pytest.approx(DEPTH_MULTIPLIER, rel=0.2)


def test_shared_gene_scale_still_builds_per_batch_onehot(models):
    m = models["shared"]
    onehot = np.asarray(m.batch_indices_onehot)
    assert onehot.shape == (2 * N_PER_BATCH, 2)
    assert np.array_equal(onehot.sum(axis=1), np.ones(2 * N_PER_BATCH))
    assert onehot[:N_PER_BATCH, 0].all()
    assert onehot[N_PER_BATCH:, 1].all()


def test_no_batch_key_has_a_single_pooled_cell_prior(models):
    lib_ratio = np.asarray(models["none"].batch_lib_ratio).reshape(-1)
    assert np.unique(lib_ratio).size == 1


# --------------------------------------------------------------------------------
# Gene side: single shared row, bit-for-bit identical to the no-batch_key path
# --------------------------------------------------------------------------------


def test_shared_gene_scale_shapes_are_single_row(models):
    m = models["shared"]
    assert m.n_gene_scale_batches == 1
    assert np.asarray(m.gene_ratio).shape == ()
    assert np.asarray(m.gene_ratio_init).shape == (1, N_GENES)
    assert np.asarray(m.global_params[0]).shape == (2, 1, N_GENES)
    assert np.asarray(m.pmeans["gene_scale"]).shape == (1, N_GENES)
    assert np.asarray(m.pvars["gene_scale"]).shape == (1, N_GENES)

    gene_onehot = np.asarray(m._gene_scale_onehot())
    assert gene_onehot.shape == (2 * N_PER_BATCH, 1)
    assert np.array_equal(gene_onehot, np.ones_like(gene_onehot))


def test_shared_gene_side_matches_no_batch_key_bit_for_bit(models):
    ref, shared = models["none"], models["shared"]
    assert np.array_equal(np.asarray(ref.gene_ratio), np.asarray(shared.gene_ratio))
    assert np.array_equal(
        np.asarray(ref.gene_ratio_init), np.asarray(shared.gene_ratio_init)
    )
    assert np.array_equal(
        np.asarray(ref.global_params[0]), np.asarray(shared.global_params[0])
    )
    assert np.array_equal(
        np.asarray(ref.pmeans["gene_scale"]), np.asarray(shared.pmeans["gene_scale"])
    )


def test_per_batch_gene_scale_is_unchanged(models):
    m = models["per_batch"]
    assert m.batch_gene_scale is True
    assert m.n_gene_scale_batches == 2
    assert np.asarray(m.gene_ratio).shape == (2, N_GENES)
    assert np.asarray(m.gene_ratio_init).shape == (2, N_GENES)
    assert np.asarray(m.global_params[0]).shape == (2, 2, N_GENES)
    assert np.asarray(m.pmeans["gene_scale"]).shape == (2, N_GENES)
    # The per-batch gene side selects each cell's own row.
    assert np.array_equal(
        np.asarray(m._gene_scale_onehot()), np.asarray(m.batch_indices_onehot)
    )
    # Per-batch gene_ratio rows are constant per batch but differ between batches
    # (pre-existing behaviour: the prior rate carries no per-gene structure).
    gr = np.asarray(m.gene_ratio)
    assert np.unique(gr[0]).size == 1 and np.unique(gr[1]).size == 1
    assert not np.isclose(gr[0, 0], gr[1, 0])


def test_per_batch_cell_prior_identical_across_gene_side_settings(models):
    """Switching the gene side off must not perturb the cell side."""
    assert np.array_equal(
        np.asarray(models["per_batch"].batch_lib_ratio),
        np.asarray(models["shared"].batch_lib_ratio),
    )
    assert np.array_equal(
        np.asarray(models["per_batch"].batch_indices_onehot),
        np.asarray(models["shared"].batch_indices_onehot),
    )


def test_gene_side_default_is_on(batched_adata):
    sig = inspect.signature(scd.scDEF.__init__)
    assert sig.parameters["batch_gene_scale"].default is True


# --------------------------------------------------------------------------------
# Parameter layout self-consistency
# --------------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["none", "per_batch", "shared"])
def test_global_params_layout_self_consistent(models, key):
    m = models[key]
    # Layout: [gene_scale, BRD, W_0 ... W_{n-1}, ARD/wm, alpha]
    assert len(m.global_params) == 2 + m.n_layers + 2

    gene_scale_params = np.asarray(m.global_params[0])
    assert gene_scale_params.ndim == 3 and gene_scale_params.shape[0] == 2
    n_rows, n_genes = gene_scale_params.shape[1], gene_scale_params.shape[2]
    assert n_rows == m.n_gene_scale_batches
    assert n_genes == m.n_genes

    # The onehot used in the likelihood must have exactly one column per row of
    # gene_scale, otherwise `onehot.dot(gene_scale)` is ill-formed.
    assert np.asarray(m._gene_scale_onehot()).shape == (m.n_cells, n_rows)
    assert np.asarray(m.pmeans["gene_scale"]).shape == (n_rows, n_genes)

    # Cell side is always one row per cell regardless of the gene side.
    assert np.asarray(m.local_params[0]).shape == (2, m.n_cells, 1)
    assert np.asarray(m.batch_lib_ratio).shape[0] == m.n_cells


# --------------------------------------------------------------------------------
# One forward ELBO evaluation per configuration (no optimisation)
# --------------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["none", "per_batch", "shared"])
def test_single_elbo_evaluation_is_finite(models, key):
    m = models[key]
    value = m.batch_elbo(
        jax.random.PRNGKey(0),
        jnp.array(m.X),
        jnp.arange(m.n_cells),
        m.local_params,
        m.global_params,
        1,  # num_samples
        1.0,  # annealing_parameter
        jnp.zeros(m.n_layers),  # stop_gradients
        False,  # stop_cell_budgets
        False,  # stop_gene_budgets
        float(m.alpha),
    )
    value = float(value)
    assert np.ndim(value) == 0
    assert np.isfinite(value), f"ELBO for {key} is not finite: {value}"


def test_annotate_writes_one_gene_scale_column_when_shared(models):
    shared = models["shared"]
    shared.annotate_adata()
    assert "gene_scale" in shared.adata.var.columns
    assert "gene_scale_1" not in shared.adata.var.columns

    per_batch = models["per_batch"]
    per_batch.annotate_adata()
    assert "gene_scale_0" in per_batch.adata.var.columns
    assert "gene_scale_1" in per_batch.adata.var.columns


# --------------------------------------------------------------------------------
# decompose_batch_effects keyword resolution (no decomposition is run)
# --------------------------------------------------------------------------------


class _StubReference:
    """Minimal stand-in exposing only what the resolver reads."""

    def __init__(self, batch_key):
        self.batch_key = batch_key
        self.logger = None


def test_decompose_batch_cell_scale_default_is_on():
    sig = inspect.signature(decompose_batch_effects)
    assert sig.parameters["batch_cell_scale"].default is True


def test_resolver_keeps_reference_batch_key(batched_adata):
    out = _resolve_decompose_batch_kwargs(
        _StubReference("b"), batched_adata, batch_cell_scale=True
    )
    assert out == {"batch_key": "b", "batch_gene_scale": False}


def test_resolver_reads_batch_key_off_a_real_model(models, batched_adata):
    out = _resolve_decompose_batch_kwargs(
        models["per_batch"], batched_adata, batch_cell_scale=True
    )
    assert out == {"batch_key": "b", "batch_gene_scale": False}


def test_resolver_off_reproduces_historical_construction(batched_adata):
    out = _resolve_decompose_batch_kwargs(
        _StubReference("b"), batched_adata, batch_cell_scale=False
    )
    assert out == {"batch_key": None}


def test_resolver_degrades_when_reference_has_no_batch_key(batched_adata):
    out = _resolve_decompose_batch_kwargs(
        _StubReference(None), batched_adata, batch_cell_scale=True
    )
    assert out == {"batch_key": None}


def test_resolver_degrades_when_key_missing_from_target(batched_adata):
    out = _resolve_decompose_batch_kwargs(
        _StubReference("not_a_column"), batched_adata, batch_cell_scale=True
    )
    assert out == {"batch_key": None}


def test_resolver_degrades_when_fewer_than_two_batches(batched_adata):
    single = batched_adata[: N_PER_BATCH // 2].copy()
    assert single.obs["b"].nunique() == 1
    out = _resolve_decompose_batch_kwargs(
        _StubReference("b"), single, batch_cell_scale=True
    )
    assert out == {"batch_key": None}


def test_resolver_kwargs_build_the_intended_model(batched_adata):
    """The resolved kwargs, applied to the constructor, give cell side on / gene side off."""
    kwargs = _resolve_decompose_batch_kwargs(
        _StubReference("b"), batched_adata, batch_cell_scale=True
    )
    model = _make(batched_adata, **kwargs)
    assert model.n_batches == 2
    assert model.n_gene_scale_batches == 1
    assert np.unique(np.asarray(model.batch_lib_ratio)).size == 2
    assert np.asarray(model.pmeans["gene_scale"]).shape == (1, N_GENES)


def test_resolver_logs_when_degrading(batched_adata):
    messages = []

    class _Logger:
        def info(self, msg):
            messages.append(msg)

    _resolve_decompose_batch_kwargs(
        _StubReference(None), batched_adata, batch_cell_scale=True, logger=_Logger()
    )
    assert messages and "without a batch key" in messages[0]


# --------------------------------------------------------------------------------
# Backwards compatibility with models pickled before these attributes existed
# --------------------------------------------------------------------------------


def test_legacy_model_without_new_attributes_still_works(models):
    m = models["per_batch"]
    saved_flag = m.batch_gene_scale
    saved_onehot = m.gene_batch_onehot
    try:
        del m.batch_gene_scale
        del m.gene_batch_onehot
        assert m.n_gene_scale_batches == m.n_batches
        assert np.array_equal(
            np.asarray(m._gene_scale_onehot()), np.asarray(m.batch_indices_onehot)
        )
    finally:
        m.batch_gene_scale = saved_flag
        m.gene_batch_onehot = saved_onehot
