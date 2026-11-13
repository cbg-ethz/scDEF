"""Unannotated cells must not crash the plotting and scoring paths.

Users routinely have a few cells with no annotation, which arrives as a NaN in the
``obs`` column. Before these guards, a NaN reached ``markers[obs]`` (KeyError), was
matched to a factor by ``assign_obs_to_factors`` and then concatenated into a
graph label (TypeError), or produced non-finite input to ``pdist`` (ValueError).
"""

import numpy as np
import pytest

scanpy = pytest.importorskip("scanpy")
scd = pytest.importorskip("scdef")

from scdef.utils import data_utils, factor_utils  # noqa: E402
from scdef.plotting.graph import _get_base_label  # noqa: E402


@pytest.fixture(scope="module")
def model_with_nan_obs():
    """A tiny fitted model whose annotation column has unannotated cells."""
    adata = scanpy.datasets.pbmc3k()[:80].copy()
    adata.X = adata.X.toarray()
    scanpy.pp.filter_genes(adata, min_cells=3)
    adata = adata[:, :120].copy()

    rng = np.random.default_rng(0)
    labels = rng.choice(["A", "B"], size=adata.n_obs).astype(object)
    labels[:12] = np.nan  # unannotated cells
    adata.obs["celltype"] = labels

    model = scd.scDEF(adata, n_layers=2, n_factors=4, seed=1)
    model.fit(n_epoch=5, n_rounds=1)
    # The documented invariant: factor_diagnostics refreshes the diagnostics AND
    # the confident signatures, so nothing else needs set_confident_signatures.
    scd.tl.factor_diagnostics(model)
    return model


def test_assign_obs_to_factors_skips_nan(model_with_nan_obs):
    """A NaN annotation must not become a factor's label."""
    # Returns two flat dicts: obs value -> factor, and factor -> obs value.
    assignments, matches = factor_utils.assign_obs_to_factors(
        model_with_nan_obs, ["celltype"]
    )
    assert set(assignments).issubset({"A", "B"})
    for mapping in (assignments, matches):
        for key, value in mapping.items():
            assert not (isinstance(key, float) and np.isnan(key))
            assert not (isinstance(value, float) and np.isnan(value))


def test_make_graph_with_nan_derived_annotations(model_with_nan_obs):
    """The full path the tutorials use: assign annotations, then draw the graph."""
    _, matches = factor_utils.assign_obs_to_factors(model_with_nan_obs, ["celltype"])
    graph = scd.pl.make_graph(model_with_nan_obs, factor_annotations=matches)
    assert graph is not None


def test_get_base_label_coerces_non_string_annotation():
    """A non-string annotation must survive the `label += ...` callers.

    Checked with ``n_cells_label=False``: the True branch formats the label into an
    f-string, which stringifies a NaN on its own and so would pass either way.
    """
    label = _get_base_label("L0_0", {"L0_0": float("nan")}, False, 5)
    assert isinstance(label, str)
    label += "<br/>signature"  # the concatenation that raised TypeError on a float
    assert label.endswith("signature")


def test_get_signature_scores_tolerates_missing_marker_lists(model_with_nan_obs):
    """An obs value absent from `markers` leaves its row at zero, and does not raise."""
    var_names = list(model_with_nan_obs.adata.var_names)
    markers = {"A": var_names[:5]}  # deliberately no entry for "B" or for NaN
    obs_vals = ["A", "B", np.nan]
    mats = data_utils.get_signature_scores(
        model_with_nan_obs, "celltype", obs_vals, markers, top_genes=5
    )
    assert len(mats) == model_with_nan_obs.n_layers
    for mat in mats:
        assert mat.shape[0] == len(obs_vals)
        assert np.all(np.isfinite(mat))
        assert np.all(mat[1] == 0)  # "B" has no marker list
        assert np.all(mat[2] == 0)  # NaN has no marker list


def test_obs_scores_runs_with_unannotated_cells(model_with_nan_obs):
    """`prepare_obs_factor_scores` clusters without tripping on non-finite input."""
    obs_mats, obs_clusters, obs_vals = data_utils.prepare_obs_factor_scores(
        model_with_nan_obs,
        "celltype",
        data_utils.get_assignment_fracs,
    )
    assert "celltype" in obs_clusters
    assert len(obs_clusters["celltype"]) == len(obs_vals["celltype"])
    for mat in obs_mats["celltype"]:
        assert np.all(np.isfinite(mat))


def test_prepare_obs_factor_scores_single_value_skips_ward(model_with_nan_obs):
    """Ward needs two rows; one annotation value must keep its natural order."""
    model = model_with_nan_obs
    model.adata.obs["only_one"] = "solo"
    _, obs_clusters, obs_vals = data_utils.prepare_obs_factor_scores(
        model, "only_one", data_utils.get_assignment_fracs
    )
    assert list(obs_clusters["only_one"]) == list(range(len(obs_vals["only_one"])))


def test_set_technical_factors_keeps_the_signature_cache(model_with_nan_obs):
    """Flagging factors technical must not force a signature rebuild.

    The cache is per factor and computed without reference to the flags, so
    flipping one cannot invalidate it. `get_biological_signature` applies the flags
    at read time instead. Before this, `set_technical_factors` popped the cache and
    `make_graph(show_signatures=True)` raised until it was rebuilt by hand.
    """
    model = model_with_nan_obs
    scd.tl.factor_diagnostics(model)
    assert "confident_signatures" in model.adata.uns

    flagged = [model.factor_names[0][0]]
    scd.tl.set_technical_factors(model, flagged)

    assert "confident_signatures" in model.adata.uns, "cache was dropped"
    assert flagged[0] in scd.tl.get_technical_factors(model)
    # The invariant this buys: no separate set_confident_signatures needed.
    assert scd.pl.make_graph(model, show_signatures=True) is not None


def test_drop_technical_does_clear_the_signature_cache(model_with_nan_obs):
    """The counterpart: dropping factors reassigns `factor_lists`, so it must clear.

    Upper-layer signatures are drawn through the hierarchy, which reads
    `factor_lists`, so they go stale when the kept set changes. This is why the
    cache-keeping change above applies to `set_technical_factors` only.
    """
    model = model_with_nan_obs
    scd.tl.factor_diagnostics(model)
    scd.tl.set_technical_factors(model, [model.factor_names[0][0]])
    assert "confident_signatures" in model.adata.uns

    scd.tl.drop_technical(model)
    assert "confident_signatures" not in model.adata.uns
