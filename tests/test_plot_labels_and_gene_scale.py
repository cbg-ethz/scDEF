"""Two plotting changes made for the 0.8.0 tutorials (2026-11-13).

- ``make_graph(gene_score=..., gene_score_scale="layer")`` scales a gene across the factors of each
  layer, so the factors that use it most are brightest. The default, ``"factor"``, keeps the score
  relative to the top gene of each factor, as in Figure 3D of the paper.
- ``obs_cell_factor_heatmap`` names each group beside its block of the group track.
- ``obs_scores`` (through ``layers_obs``) titles its panels with the layer names when the titles
  "Layer i" overlap, and plots a subset of layers given by ``layers``.
"""
import types

import matplotlib.pyplot as plt
import numpy as np
import pytest
from anndata import AnnData

from scdef.plotting.factors import layers_obs, obs_cell_factor_heatmap
from scdef.plotting.graph import _get_gene_scores

from test_obs_cell_factor_heatmap import _make_plot_model


class _GeneModel:
    """Two layers; gene g1 is a small share of every factor but used almost only by f1."""

    def __init__(self):
        self.adata = AnnData(np.zeros((2, 3), dtype=np.float32))
        self.adata.var_names = ["g0", "g1", "g2"]
        self.factor_names = [["f0", "f1"], ["u0"]]
        self._scores = {
            "f0": np.array([10.0, 0.01, 1.0]),
            "f1": np.array([50.0, 2.0, 0.0]),
            "u0": np.array([60.0, 2.01, 1.0]),
        }

    def get_signatures_dict(
        self, scores=False, layer_normalize=False, drop_factors=None
    ):
        out = {}
        for n, v in self._scores.items():
            v = v.astype(float)
            if layer_normalize:
                v = (v - v.min()) / (v.max() - v.min())
            out[n] = v
        return {n: [] for n in out}, out


def test_factor_scale_is_relative_to_the_top_gene_of_each_factor():
    s = _get_gene_scores(_GeneModel(), "g1", None)
    assert s["f1"] == pytest.approx(2.0 / 50.0)  # min-max over the genes of the factor
    assert s["f0"] == pytest.approx(0.0)  # g1 is the smallest gene of f0


def test_layer_scale_is_relative_to_the_gene_across_the_layer():
    s = _get_gene_scores(_GeneModel(), "g1", None, scale="layer")
    assert s["f1"] == pytest.approx(1.0)
    assert s["f0"] == pytest.approx(0.01 / 2.0)
    assert s["u0"] == pytest.approx(1.0)  # the only factor of its layer


def test_unknown_scale_is_refused():
    with pytest.raises(ValueError):
        _get_gene_scores(_GeneModel(), "g1", None, scale="gene")


def test_heatmap_names_each_group():
    fig = obs_cell_factor_heatmap(
        _make_plot_model(),
        subset_obs_key="patient",
        subset_obs="P1",
        group_obs_key="treatment",
        show=False,
    )
    texts = {t.get_text() for ax in fig.axes for t in ax.texts}
    assert {"A", "B"} <= texts


def _layers_model(sizes):
    model = types.SimpleNamespace()
    model.n_layers = len(sizes)
    model.layer_names = [f"L{i}" for i in range(len(sizes))]
    model.factor_lists = [np.arange(n) for n in sizes]
    model.factor_names = [[f"{i}_{k}" for k in range(n)] for i, n in enumerate(sizes)]
    model.get_layer_factor_orders = lambda: [np.arange(n) for n in sizes]
    obs_mats = {
        "type": {
            i: np.random.default_rng(i).random((2, n)) for i, n in enumerate(sizes)
        }
    }
    return model, obs_mats, {"type": np.arange(2)}, {"type": ["a", "b"]}


def _titles(**kwargs):
    model, *mats = _layers_model(kwargs.pop("sizes"))
    layers_obs(model, "type", *mats, show=False, **kwargs)
    titles = [ax.get_title() for ax in plt.gcf().axes if ax.get_title()]
    plt.close("all")
    return titles


def test_layer_titles_kept_when_they_fit():
    assert _titles(sizes=[4, 4, 4], figsize=(10, 4)) == [
        "Layer 0",
        "Layer 1",
        "Layer 2",
    ]


def test_layer_titles_shortened_when_they_overlap():
    assert _titles(sizes=[40, 2, 2], figsize=(4, 2)) == ["L0", "L1", "L2"]


def test_layers_subset_is_plotted_by_position():
    # the columns were indexed by layer number, so a subset not starting at layer 0 raised IndexError
    assert _titles(sizes=[6, 4, 3], layers=[1, 2], figsize=(10, 4)) == [
        "Layer 1",
        "Layer 2",
    ]
