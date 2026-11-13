"""``scDEF.learn`` runs the workflow of the paper (fit, filter, fit, filter) with the paper's filter arguments, passes
the model's batch key to both filters, lets the caller replace filter arguments, and returns the model;
``final_loss`` is the mean loss over the last iterations of the latest fit."""
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import scdef as scd
import scdef.tools.factor as tf


def _model(batch_key=None):
    rng = np.random.default_rng(0)
    counts = rng.poisson(2.0, size=(60, 30)).astype(float)
    obs = pd.DataFrame(
        {"batch": np.repeat(["a", "b"], 30)}, index=[f"c{i}" for i in range(60)]
    )
    adata = AnnData(counts, obs=obs)
    adata.layers["counts"] = counts.copy()
    return scd.scDEF(
        adata, counts_layer="counts", batch_key=batch_key, n_factors=6, n_layers=3
    )


def _record(monkeypatch, model):
    calls = []
    monkeypatch.setattr(model, "fit", lambda **kw: calls.append(("fit", kw)))
    monkeypatch.setattr(
        tf,
        "filter_factors",
        lambda m, batch_key=None, **kw: calls.append(("filter", batch_key, kw)),
    )
    return calls


@pytest.mark.parametrize("batch_key", [None, "batch"])
def test_learn_runs_the_paper_steps(monkeypatch, batch_key):
    model = _model(batch_key)
    calls = _record(monkeypatch, model)
    assert model.learn() is model
    assert [c[0] for c in calls] == ["fit", "filter", "fit", "filter"]
    assert calls[0][1] == {"hierarchical_init": True}
    # the refit warm-starts; None resolves to the same initial concentration without the warning
    assert calls[2][1] == {"hierarchical_init": None}
    assert calls[1][1] == batch_key and calls[3][1] == batch_key
    assert calls[1][2] == dict(
        brd_min=1.0, ard_min=0.0, n_eff_parents_max=10.1, filter_up=True
    )
    assert calls[3][2] == dict(
        brd_min=1.0,
        ard_min=0.0,
        n_eff_parents_max=1.5,
        filter_up=True,
        min_cells_upper=1,
        brd_exceptional=4,
    )


def test_learn_overrides(monkeypatch):
    model = _model()
    calls = _record(monkeypatch, model)
    model.learn(second_filter=dict(n_eff_parents_max=2.0), n_epoch=10)
    assert calls[0][1] == {"n_epoch": 10, "hierarchical_init": True}
    assert (
        calls[3][2]["n_eff_parents_max"] == 2.0 and calls[3][2]["brd_exceptional"] == 4
    )
    assert calls[1][2]["n_eff_parents_max"] == 10.1


def test_final_loss():
    model = _model()
    with pytest.raises(ValueError):
        model.final_loss()
    model.elbos = [np.arange(10.0), np.arange(200.0)]
    assert model.final_loss() == pytest.approx(np.arange(100.0, 200.0).mean())
    assert model.final_loss(n_iter=10) == pytest.approx(np.arange(190.0, 200.0).mean())


def test_refit_warns_only_when_hierarchical_init_is_requested(caplog):
    import logging

    model = _model()
    model.adata.obsm["X_pca"] = np.random.default_rng(1).normal(
        size=(model.adata.n_obs, 5)
    )
    model.fit(n_epoch=2, hierarchical_init=True)
    with caplog.at_level(logging.WARNING, logger=model.logger.name):
        model.fit(n_epoch=2)
    assert "ignored on refit" not in caplog.text
    with caplog.at_level(logging.WARNING, logger=model.logger.name):
        model.fit(n_epoch=2, hierarchical_init=True)
    assert "ignored on refit" in caplog.text
