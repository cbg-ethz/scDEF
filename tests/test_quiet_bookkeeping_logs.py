"""The messages that only report which fields of ``adata`` were written, that a cache was cleared or
recomputed, or that the graph was rebuilt are DEBUG, not INFO, so that a fit and filter print a few
lines at the default level (0.8.0, 2026-11-13). The convergence messages stay at INFO."""
import logging

import numpy as np
import pandas as pd
from anndata import AnnData

import scdef as scd

BOOKKEEPING = (
    "Updated adata",
    "Updated scDEF graph",
    "cleared name-keyed caches",
    "cached diagnostics",
    "W warm start by",
)


def test_fit_and_filter_log_no_bookkeeping_at_info(caplog):
    rng = np.random.default_rng(0)
    counts = rng.poisson(2.0, size=(60, 30)).astype(float)
    adata = AnnData(counts, obs=pd.DataFrame(index=[f"c{i}" for i in range(60)]))
    adata.layers["counts"] = counts.copy()
    model = scd.scDEF(adata, counts_layer="counts", n_factors=6, n_layers=3)
    with caplog.at_level(logging.DEBUG, logger=model.logger.name):
        model.fit(n_epoch=2)
        scd.tl.filter_factors(model, brd_min=0.0, ard_min=0.0)
        model.fit(n_epoch=2)
    info = [r.getMessage() for r in caplog.records if r.levelno >= logging.INFO]
    debug = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
    assert not [m for m in info if any(b in m for b in BOOKKEEPING)]
    assert any("Updated adata" in m for m in debug)
