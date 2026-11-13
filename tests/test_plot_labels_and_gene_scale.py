"""Two plotting changes made for the 0.8.0 tutorials (2026-11-13).

- ``make_graph(gene_score=..., gene_score_scale="layer")`` scales a gene across the factors of each
  layer, so the factors that use it most are brightest. The default, ``"factor"``, keeps the score
  relative to the top gene of each factor, as in Figure 3D of the paper.
- ``obs_cell_factor_heatmap`` names each group beside its block of the group track.
"""
import numpy as np
import pytest
from anndata import AnnData

from scdef.plotting.factors import obs_cell_factor_heatmap
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

    def get_signatures_dict(self, scores=False, layer_normalize=False, drop_factors=None):
        out = {}
        for n, v in self._scores.items():
            v = v.astype(float)
            if layer_normalize:
                v = (v - v.min()) / (v.max() - v.min())
            out[n] = v
        return {n: [] for n in out}, out


def test_factor_scale_is_relative_to_the_top_gene_of_each_factor():
    s = _get_gene_scores(_GeneModel(), "g1", None)
    assert s["f1"] == pytest.approx(2.0 / 50.0)   # min-max over the genes of the factor
    assert s["f0"] == pytest.approx(0.0)          # g1 is the smallest gene of f0


def test_layer_scale_is_relative_to_the_gene_across_the_layer():
    s = _get_gene_scores(_GeneModel(), "g1", None, scale="layer")
    assert s["f1"] == pytest.approx(1.0)
    assert s["f0"] == pytest.approx(0.01 / 2.0)
    assert s["u0"] == pytest.approx(1.0)          # the only factor of its layer


def test_unknown_scale_is_refused():
    with pytest.raises(ValueError):
        _get_gene_scores(_GeneModel(), "g1", None, scale="gene")


def test_heatmap_names_each_group():
    fig = obs_cell_factor_heatmap(_make_plot_model(), subset_obs_key="patient", subset_obs="P1",
                                  group_obs_key="treatment", show=False)
    texts = {t.get_text() for ax in fig.axes for t in ax.texts}
    assert {"A", "B"} <= texts
