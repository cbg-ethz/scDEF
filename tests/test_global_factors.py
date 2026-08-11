import numpy as np
import pandas as pd
import pytest

from anndata import AnnData

from scdef.tools.factor import (
    set_global_factors,
    get_global_signature,
    __build_consensus_signature,
)
from scdef.tools.hierarchy import (
    make_biological_hierarchy,
    make_global_hierarchy,
    make_hierarchies,
)


def _make_model_with_factor_obs(factor_obs: pd.DataFrame):
    class _Model:
        pass

    model = _Model()
    model.n_layers = 3
    model.layer_names = ["L0", "L1", "top"]
    model.factor_names = [
        ["L0_a", "L0_global"],
        ["L1_a"],
        ["top_A"],
    ]
    model.factor_lists = [
        np.array([0, 1], dtype=int),
        np.array([0], dtype=int),
        np.array([0], dtype=int),
    ]
    model.n_genes = 5
    model.adata = AnnData(np.zeros((3, model.n_genes), dtype=np.float32))
    model.adata.var_names = [f"g{i}" for i in range(model.n_genes)]
    model.adata.uns["factor_obs"] = factor_obs
    model.pmeans = {}
    model._fit_revision = 1

    def get_rankings(layer_idx=0, genes=True, return_scores=False):
        rankings = [["g0", "g1"], ["g2", "g3", "g4"]]
        scores = [np.array([1.0, 0.5]), np.array([0.8, 0.6, 0.2])]
        if return_scores:
            return rankings, scores
        return rankings

    def get_relevances_dict():
        return {"L0_a": 0.3, "L0_global": 0.7}

    model.get_rankings = get_rankings
    model.get_relevances_dict = get_relevances_dict
    return model


def _factor_obs_df():
    rows = [
        dict(
            child_layer="L0",
            original_factor_idx=0,
            parent_layer="L1",
            best_parent="L1_a",
            best_parent_prob=0.9,
            n_eff_parents=1.1,
            clarity_score_01=0.8,
            K_parents=1,
            avg_n_eff_parents=1.2,
            technical=False,
            **{"global": False},
        ),
        dict(
            child_layer="L0",
            original_factor_idx=1,
            parent_layer="L1",
            best_parent="L1_a",
            best_parent_prob=0.85,
            n_eff_parents=1.2,
            clarity_score_01=0.4,
            K_parents=1,
            avg_n_eff_parents=2.0,
            technical=False,
            **{"global": False},
        ),
        dict(
            child_layer="L1",
            original_factor_idx=0,
            parent_layer="top",
            best_parent="top_A",
            best_parent_prob=0.95,
            n_eff_parents=1.05,
            clarity_score_01=0.85,
            K_parents=1,
            avg_n_eff_parents=np.nan,
            technical=False,
            **{"global": False},
        ),
    ]
    return pd.DataFrame(rows, index=["L0_a", "L0_global", "L1_a"])


def test_set_global_factors_marks_rows():
    model = _make_model_with_factor_obs(_factor_obs_df())
    set_global_factors(model, factors=["L0_global"])
    assert model.adata.uns["factor_obs"].loc["L0_global", "global"]
    assert not model.adata.uns["factor_obs"].loc["L0_a", "global"]


def test_make_global_hierarchy_star():
    model = _make_model_with_factor_obs(_factor_obs_df())
    set_global_factors(model, factors=["L0_global"])
    h = make_global_hierarchy(model)
    assert h == {"global_top": ["L0_global"]}
    assert model.adata.uns["global_hierarchy"] == h


def test_make_biological_hierarchy_drops_global_and_technical(monkeypatch):
    model = _make_model_with_factor_obs(_factor_obs_df())
    model.adata.uns["factor_obs"].loc["L0_a", "technical"] = True
    set_global_factors(model, factors=["L0_global"])

    captured = {}

    def fake_get_hierarchy(m, simplified=True, drop_factors=None):
        captured["drop_factors"] = list(drop_factors or [])
        return {}

    monkeypatch.setattr(
        "scdef.tools.hierarchy.get_hierarchy",
        fake_get_hierarchy,
    )
    make_biological_hierarchy(model)
    assert set(captured["drop_factors"]) == {"L0_a", "L0_global"}


def test_get_global_signature_consensus():
    model = _make_model_with_factor_obs(_factor_obs_df())
    set_global_factors(model, factors=["L0_global"])
    make_global_hierarchy(model)
    genes, scores = get_global_signature(model, top_genes=3, return_scores=True)
    assert len(genes) == 3
    assert genes[0] == "g2"
    assert len(scores) == 3


def test_make_hierarchies_includes_global(monkeypatch):
    model = _make_model_with_factor_obs(_factor_obs_df())
    set_global_factors(model, factors=["L0_global"])
    monkeypatch.setattr(
        "scdef.tools.hierarchy.make_biological_hierarchy",
        lambda m: m.adata.uns.__setitem__("biological_hierarchy", {}),
    )
    monkeypatch.setattr(
        "scdef.tools.hierarchy.make_technical_hierarchy",
        lambda m: m.adata.uns.__setitem__("technical_hierarchy", {}),
    )
    make_hierarchies(model)
    assert "global_hierarchy" in model.adata.uns


def test_set_global_factors_does_not_call_annotate():
    model = _make_model_with_factor_obs(_factor_obs_df())

    def annotate():
        raise AssertionError("annotate_adata should not run for set_global_factors")

    model.annotate_adata = annotate
    set_global_factors(model, factors=["L0_global"])
