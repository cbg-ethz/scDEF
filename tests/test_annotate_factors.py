import numpy as np
import pandas as pd
import pytest

from anndata import AnnData

from scdef.tools.factor import annotate_factors, get_factor_annotations


def _make_model_with_factor_obs():
    class _Model:
        pass

    model = _Model()
    model.n_layers = 1
    model.layer_names = ["L0"]
    model.factor_names = [["L0_0", "L0_1", "L0_2"]]
    model.factor_lists = [np.array([0, 1, 2], dtype=int)]
    model.layer_sizes = [3]
    model.adata = AnnData(np.zeros((5, 3), dtype=np.float32))
    model.adata.uns["factor_obs"] = pd.DataFrame(
        {
            "child_layer": ["L0", "L0", "L0"],
            "original_factor_idx": [0, 1, 2],
        },
        index=["L0_0", "L0_1", "L0_2"],
    )
    return model


def test_annotate_factors_sets_column():
    model = _make_model_with_factor_obs()
    annotate_factors(model, {"L0_1": "stem-like", "L0_2": "cycling"})
    assert "annotation" in model.adata.uns["factor_obs"].columns
    assert model.adata.uns["factor_obs"].loc["L0_1", "annotation"] == "stem-like"
    assert model.adata.uns["factor_obs"].loc["L0_2", "annotation"] == "cycling"
    assert pd.isna(model.adata.uns["factor_obs"].loc["L0_0", "annotation"])


def test_get_factor_annotations_lookup():
    model = _make_model_with_factor_obs()
    annotate_factors(model, {"L0_1": "stem-like"})
    out = get_factor_annotations(model, ["L0_0", "L0_1", "L0_2"])
    assert out == [None, "stem-like", None]


def test_annotate_factors_unknown_raises():
    model = _make_model_with_factor_obs()
    with pytest.raises(ValueError, match="Unknown factor"):
        annotate_factors(model, {"missing": "x"})
