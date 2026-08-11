import numpy as np
import pandas as pd
import pytest

from anndata import AnnData

from scdef.tools.lineage import get_lineage_factors, get_global_factors


def _make_model_with_factor_obs(factor_obs: pd.DataFrame):
    """Minimal model shell for lineage tool tests."""

    class _Model:
        pass

    model = _Model()
    model.n_layers = 3
    model.layer_names = ["L0", "L1", "top"]
    model.factor_names = [
        ["L0_a", "L0_b", "L0_other", "L0_global"],
        ["L1_a", "L1_b", "L1_ambig"],
        ["top_A", "top_B"],
    ]
    model.factor_lists = [
        np.array([0, 1, 2, 3], dtype=int),
        np.array([0, 1, 2], dtype=int),
        np.array([0, 1], dtype=int),
    ]
    model.adata = AnnData(np.zeros((5, 10), dtype=np.float32))
    model.adata.uns["factor_obs"] = factor_obs
    return model


def _build_factor_obs() -> pd.DataFrame:
    rows = [
        # L0 factors
        dict(
            child_layer="L0",
            original_factor_idx=0,
            parent_layer="L1",
            best_parent="L1_a",
            best_parent_prob=0.9,
            n_eff_parents=1.2,
            clarity_score_01=0.8,
            K_parents=3,
            avg_n_eff_parents=1.25,
            technical=False,
        ),
        dict(
            child_layer="L0",
            original_factor_idx=1,
            parent_layer="L1",
            best_parent="L1_a",
            best_parent_prob=0.85,
            n_eff_parents=1.3,
            clarity_score_01=0.75,
            K_parents=3,
            avg_n_eff_parents=1.3,
            technical=False,
        ),
        dict(
            child_layer="L0",
            original_factor_idx=2,
            parent_layer="L1",
            best_parent="L1_b",
            best_parent_prob=0.88,
            n_eff_parents=1.1,
            clarity_score_01=0.82,
            K_parents=3,
            avg_n_eff_parents=1.15,
            technical=False,
        ),
        dict(
            child_layer="L0",
            original_factor_idx=3,
            parent_layer="L1",
            best_parent="L1_ambig",
            best_parent_prob=0.9,
            n_eff_parents=1.2,
            clarity_score_01=0.4,
            K_parents=3,
            avg_n_eff_parents=2.2,
            technical=False,
        ),
        # L1 factors
        dict(
            child_layer="L1",
            original_factor_idx=0,
            parent_layer="top",
            best_parent="top_A",
            best_parent_prob=0.95,
            n_eff_parents=1.1,
            clarity_score_01=0.85,
            K_parents=2,
            avg_n_eff_parents=np.nan,
            technical=False,
        ),
        dict(
            child_layer="L1",
            original_factor_idx=1,
            parent_layer="top",
            best_parent="top_B",
            best_parent_prob=0.92,
            n_eff_parents=1.15,
            clarity_score_01=0.8,
            K_parents=2,
            avg_n_eff_parents=np.nan,
            technical=False,
        ),
        dict(
            child_layer="L1",
            original_factor_idx=2,
            parent_layer="top",
            best_parent="top_A",
            best_parent_prob=0.55,
            n_eff_parents=2.5,
            clarity_score_01=0.3,
            K_parents=2,
            avg_n_eff_parents=np.nan,
            technical=False,
        ),
    ]
    index = [
        "L0_a",
        "L0_b",
        "L0_other",
        "L0_global",
        "L1_a",
        "L1_b",
        "L1_ambig",
    ]
    return pd.DataFrame(rows, index=index)


def test_get_lineage_factors_returns_lineage_specific_l0():
    model = _make_model_with_factor_obs(_build_factor_obs())
    lineage = get_lineage_factors(model, "top_A", layer_idx=0)
    assert set(lineage) == {"L0_a", "L0_b"}
    assert "L0_global" not in lineage
    assert "L0_other" not in lineage


def test_get_lineage_factors_rejects_ambiguous_intermediate_parent():
    model = _make_model_with_factor_obs(_build_factor_obs())
    # L0_other -> L1_b -> top_B is fine; L0 with path through L1_ambig fails at L1 step
    obs = model.adata.uns["factor_obs"]
    obs.loc["L0_other", "best_parent"] = "L1_ambig"
    lineage = get_lineage_factors(model, "top_A", layer_idx=0, prob_min=0.5)
    assert "L0_other" not in lineage


def test_get_global_factors_uses_avg_n_eff_parents_on_l0():
    model = _make_model_with_factor_obs(_build_factor_obs())
    global_factors = get_global_factors(model, layer_idx=0, n_eff_parents_min=1.5)
    assert global_factors == ["L0_global"]


def test_get_global_factors_l1_uses_local_n_eff_parents():
    model = _make_model_with_factor_obs(_build_factor_obs())
    global_l1 = get_global_factors(model, layer_idx=1, n_eff_parents_min=2.0)
    assert global_l1 == ["L1_ambig"]


def test_get_lineage_factors_direct_children_of_l1():
    model = _make_model_with_factor_obs(_build_factor_obs())
    lineage = get_lineage_factors(model, "L1_a", layer_idx=0)
    assert set(lineage) == {"L0_a", "L0_b"}


def test_get_lineage_factors_l1_from_top_b():
    model = _make_model_with_factor_obs(_build_factor_obs())
    lineage = get_lineage_factors(model, "top_B", layer_idx=1)
    assert lineage == ["L1_b"]


def test_get_lineage_factors_layer_idx_not_below_ancestor_raises():
    model = _make_model_with_factor_obs(_build_factor_obs())
    with pytest.raises(ValueError, match="layer_idx .* must be below"):
        get_lineage_factors(model, "L1_a", layer_idx=1)


def test_get_lineage_factors_unknown_top_raises():
    model = _make_model_with_factor_obs(_build_factor_obs())
    with pytest.raises(ValueError, match="Unknown factor label"):
        get_lineage_factors(model, "not_a_factor", layer_idx=0)


def test_missing_factor_obs_raises():
    model = _make_model_with_factor_obs(_build_factor_obs())
    del model.adata.uns["factor_obs"]
    with pytest.raises(KeyError, match="factor_obs"):
        get_global_factors(model, layer_idx=0)
