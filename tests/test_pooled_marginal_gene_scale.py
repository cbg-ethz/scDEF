"""Unit tests for the pooled-marginal ``gene_scale`` warm start.

No fitting happens here: both the reference and the target model are stand-ins with
hand-set attributes, which is all
:func:`scdef.models.extend._resolve_decompose_gene_scale` reads.
"""

import numpy as np
import pytest

extend = pytest.importorskip("scdef.models.extend")


class _FakeLogger:
    def __init__(self):
        self.messages = []

    def info(self, msg):
        self.messages.append(str(msg))


class _FakeAdata:
    def __init__(self, n_vars):
        self.n_vars = int(n_vars)


class _FakeReference:
    """Only ``pmeans['gene_scale']`` is read by the resolution path."""

    def __init__(self, gene_scale):
        self.pmeans = {"gene_scale": np.asarray(gene_scale, dtype=np.float32)}


class _FakeTarget:
    """Stand-in for the freshly constructed decomposed model.

    ``X`` and ``batch_lib_sizes`` are exactly what ``scDEF.load_adata`` would set,
    and ``batch_lib_sizes`` is what ``init_var_params(init_budgets=True)`` reads.
    """

    def __init__(self, X, n_gene_scale_batches=1):
        self.X = np.asarray(X, dtype=np.float64)
        self.batch_lib_sizes = self.X.sum(axis=1)
        self.n_gene_scale_batches = int(n_gene_scale_batches)
        self.adata = _FakeAdata(self.X.shape[1])
        self.logger = _FakeLogger()


def _expected_u(target, z0, w0):
    """``U_g`` recomputed independently of the implementation."""
    lib = np.asarray(target.batch_lib_sizes, dtype=np.float64)
    cell_scale = np.clip(lib / np.mean(lib), 1e-3, 1e2)
    z0 = np.clip(np.asarray(z0, dtype=np.float64), 1e-3, 1e6)
    w0 = np.clip(np.asarray(w0, dtype=np.float64), 1e-3, 1e8)
    recon = (z0 @ w0) * cell_scale[:, None]
    return recon.sum(axis=0)


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    n_cells, n_factors, n_genes = 40, 3, 12
    z0 = rng.gamma(2.0, 0.5, size=(n_cells, n_factors)).astype(np.float32)
    w0 = rng.gamma(2.0, 0.5, size=(n_factors, n_genes)).astype(np.float32)
    # Gene abundances spanning several orders of magnitude, so a level error in the
    # warm start would be visible.
    X = rng.poisson((z0 @ w0) * np.geomspace(1.0, 500.0, n_genes)[None, :]).astype(
        np.float64
    )
    ref_gs = np.stack(
        [
            rng.gamma(2.0, 5.0, size=n_genes).astype(np.float32),
            rng.gamma(2.0, 50.0, size=n_genes).astype(np.float32),
        ]
    )
    return {
        "init_z": [z0],
        "init_w": [w0],
        "target": _FakeTarget(X),
        "reference": _FakeReference(ref_gs),
    }


def test_reproduces_pooled_counts_exactly(toy):
    """s_g * U_g must equal the observed pooled counts, gene by gene."""
    arr = extend._resolve_decompose_gene_scale(
        toy["reference"],
        toy["target"],
        toy["init_z"],
        toy["init_w"],
        "reference",
        nmf_init=False,
    )
    assert arr.shape == (1, toy["target"].X.shape[1])

    profile = np.asarray(arr[0], dtype=np.float64)
    u = _expected_u(toy["target"], toy["init_z"][0], toy["init_w"][0])
    reconstructed = profile * u
    observed = toy["target"].X.sum(axis=0)

    np.testing.assert_allclose(reconstructed, observed, rtol=1e-5)
    # And in the units the offline bake-off used: zero dex on every gene.
    dex = np.abs(np.log10(reconstructed / observed))
    assert dex.max() < 1e-5


def test_differs_from_geometric_mean(toy):
    """The pooled MLE is not the geometric-mean profile it replaces."""
    pooled = extend._resolve_decompose_gene_scale(
        toy["reference"],
        toy["target"],
        toy["init_z"],
        toy["init_w"],
        "reference",
        nmf_init=False,
    )
    geometric = extend._resolve_init_gene_scale_array(
        toy["reference"], "reference", n_batches=1, n_genes=toy["target"].adata.n_vars
    )
    assert not np.allclose(pooled, geometric, rtol=1e-2)


def test_nmf_init_falls_back_to_geometric_mean(toy):
    """Under nmf_init, init_w is ignored, so the pooled MLE must not be used."""
    arr = extend._resolve_decompose_gene_scale(
        toy["reference"],
        toy["target"],
        toy["init_z"],
        toy["init_w"],
        "reference",
        nmf_init=True,
    )
    geometric = extend._resolve_init_gene_scale_array(
        toy["reference"], "reference", n_batches=1, n_genes=toy["target"].adata.n_vars
    )
    np.testing.assert_allclose(arr, geometric, rtol=0, atol=0)
    assert any("nmf_init=True" in m for m in toy["target"].logger.messages)


def test_zero_u_uses_fallback_and_logs(toy):
    """A zero U_g must yield the fallback value, not a division blow-up.

    Note the clips make this unreachable through ordinary inputs -- z is floored at
    1e-3 and W at 1e-3, so every gene has *some* support -- which is why the guard is
    exercised here with a degenerate factor axis.
    """
    n_genes = toy["target"].X.shape[1]
    target = _FakeTarget(toy["target"].X)
    n_cells = target.X.shape[0]
    fallback = np.full(n_genes, 7.0)

    profile = extend._pooled_marginal_gene_scale(
        target,
        [np.zeros((n_cells, 0), dtype=np.float32)],
        [np.zeros((0, n_genes), dtype=np.float32)],
        fallback_profile=fallback,
        logger=target.logger,
    )

    np.testing.assert_allclose(profile, fallback, rtol=1e-6)
    assert np.isfinite(profile).all()
    assert any("no factor support" in m for m in target.logger.messages)
    assert any(f"{n_genes} of {n_genes}" in m for m in target.logger.messages)


def test_non_finite_u_uses_fallback(toy):
    """Non-finite U_g (degenerate libraries) must not leak nan into the warm start."""
    n_genes = toy["target"].X.shape[1]
    target = _FakeTarget(toy["target"].X)
    # All-zero libraries make lib / mean(lib) nan, so U is nan for every gene.
    target.batch_lib_sizes = np.zeros(target.X.shape[0])
    fallback = np.full(n_genes, 7.0)

    profile = extend._pooled_marginal_gene_scale(
        target,
        toy["init_z"],
        toy["init_w"],
        fallback_profile=fallback,
        logger=target.logger,
    )
    np.testing.assert_allclose(profile, fallback, rtol=1e-6)
    assert np.isfinite(profile).all()


def test_one_row_reference_is_still_level_corrected(toy):
    """A one-row reference used to be copied verbatim; it must now be re-levelled."""
    n_genes = toy["target"].X.shape[1]
    one_row = np.full((1, n_genes), 3.0, dtype=np.float32)
    reference = _FakeReference(one_row)

    arr = extend._resolve_decompose_gene_scale(
        reference,
        toy["target"],
        toy["init_z"],
        toy["init_w"],
        "reference",
        nmf_init=False,
    )
    profile = np.asarray(arr[0], dtype=np.float64)

    # Not the straight copy the geometric branch would have produced ...
    assert not np.allclose(profile, one_row[0], rtol=1e-2)
    # ... and still exact on the pooled marginal.
    u = _expected_u(toy["target"], toy["init_z"][0], toy["init_w"][0])
    np.testing.assert_allclose(profile * u, toy["target"].X.sum(axis=0), rtol=1e-5)


def test_ignores_reference_batch_rows(toy):
    """The profile must not depend on the reference's per-batch gene_scale rows."""
    base = extend._resolve_decompose_gene_scale(
        toy["reference"],
        toy["target"],
        toy["init_z"],
        toy["init_w"],
        "reference",
        nmf_init=False,
    )
    swapped_rows = np.asarray(toy["reference"].pmeans["gene_scale"])[::-1] * 1000.0
    other = extend._resolve_decompose_gene_scale(
        _FakeReference(swapped_rows),
        toy["target"],
        toy["init_z"],
        toy["init_w"],
        "reference",
        nmf_init=False,
    )
    np.testing.assert_allclose(base, other, rtol=0, atol=0)


def test_upper_bound_is_1e7():
    """Abundant genes must not be truncated at the historical 1e6 bound."""
    n_genes = 4
    # Tiny factor support against large counts -> pooled MLE above 1e6.
    z0 = np.full((5, 1), 1.0, dtype=np.float32)
    w0 = np.full((1, n_genes), 1e-3, dtype=np.float32)
    X = np.full((5, n_genes), 2000.0)
    target = _FakeTarget(X)
    profile = extend._pooled_marginal_gene_scale(
        target, [z0], [w0], fallback_profile=np.ones(n_genes), logger=target.logger
    )
    assert profile.max() > 1e6
    assert profile.max() <= extend._GENE_SCALE_MAX

    # And the geometric branch carries the same bound.
    huge = extend._resolve_init_gene_scale_array(
        _FakeReference(np.full((2, n_genes), 6e6, dtype=np.float32)),
        "reference",
        n_batches=1,
        n_genes=n_genes,
    )
    assert huge.max() > 1e6


def test_prior_and_explicit_array_paths(toy):
    assert (
        extend._resolve_decompose_gene_scale(
            toy["reference"],
            toy["target"],
            toy["init_z"],
            toy["init_w"],
            "prior",
            nmf_init=False,
        )
        is None
    )

    n_genes = toy["target"].adata.n_vars
    explicit = np.linspace(1.0, 2.0, n_genes).astype(np.float32)
    arr = extend._resolve_decompose_gene_scale(
        toy["reference"],
        toy["target"],
        toy["init_z"],
        toy["init_w"],
        explicit,
        nmf_init=False,
    )
    np.testing.assert_allclose(arr, explicit[None, :], rtol=1e-6)

    with pytest.raises(ValueError, match="must be 'reference', 'prior', or an array"):
        extend._resolve_decompose_gene_scale(
            toy["reference"],
            toy["target"],
            toy["init_z"],
            toy["init_w"],
            "nonsense",
            nmf_init=False,
        )
