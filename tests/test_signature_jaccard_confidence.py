import numpy as np
import pytest

from scdef.tools.factor import (
    _compute_signature_jaccard_confidences,
    _jaccard_gene_lists,
    _weighted_signature_jaccard,
    get_stored_confident_signatures,
)

scanpy = pytest.importorskip("scanpy")
scd = pytest.importorskip("scdef")


def test_jaccard_gene_lists():
    assert _jaccard_gene_lists(["a", "b"], ["a", "b"]) == 1.0
    assert _jaccard_gene_lists(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3)
    assert _jaccard_gene_lists([], []) == 1.0


def test_weighted_signature_jaccard_favors_top_genes():
    assert _weighted_signature_jaccard(["a", "b"], [9.0, 1.0], ["a", "b"]) == 1.0
    assert _weighted_signature_jaccard(["a", "b"], [9.0, 1.0], ["a"]) == pytest.approx(
        0.9
    )
    assert _weighted_signature_jaccard(["a", "b"], [9.0, 1.0], ["b"]) == pytest.approx(
        0.1
    )
    assert _weighted_signature_jaccard(
        ["a", "b"], [9.0, 1.0], ["a", "c"]
    ) == pytest.approx(0.9 / 1.5)


def test_set_confident_signatures_stores_signature_jaccard():
    adata = scanpy.datasets.pbmc3k()[:40].copy()
    adata.X = adata.X.toarray()
    model = scd.scDEF(adata, layer_sizes=[3, 2], seed=0)
    model.init_var_params(init_budgets=True, init_alpha=True, nmf_init=False)
    model.set_posterior_means()
    model.factor_lists = [np.arange(3, dtype=int), np.arange(2, dtype=int)]
    model.set_factor_names()

    scd.tl.set_confident_signatures(model, mc_samples=20, random_seed=1)
    cache = model.adata.uns["confident_signatures"]
    layer0 = cache["by_layer"]["0"]
    assert "signature_confidences" in layer0
    for factor_name in model.factor_names[0]:
        assert factor_name in layer0["signature_confidences"]
        val = layer0["signature_confidences"][factor_name]
        if len(layer0["signatures"][factor_name]) > 0:
            assert 0.0 <= val <= 1.0
        else:
            assert np.isnan(val)

    sigs, sig_conf = get_stored_confident_signatures(
        model, layer_idx=0, return_signature_confidences=True
    )
    assert sigs
    assert sig_conf == layer0["signature_confidences"]


def test_signature_jaccard_perfect_when_draws_match_reference():
    adata = scanpy.datasets.pbmc3k()[:20].copy()
    adata.X = adata.X.toarray()
    model = scd.scDEF(adata, layer_sizes=[2], seed=0)
    model.init_var_params(init_budgets=True, init_alpha=True, nmf_init=False)
    model.set_posterior_means()
    model.factor_lists = [np.arange(2, dtype=int)]
    model.set_factor_names()

    ref = ["gene_a", "gene_b"]
    calls = {"n": 0}

    def fake_sample(rng, factor_idx, layer_idx, top_genes=10, return_scores=False):
        calls["n"] += 1
        return list(ref)

    model.get_signature_sample = fake_sample
    out = _compute_signature_jaccard_confidences(
        model,
        layer_idx=0,
        signatures={model.factor_names[0][0]: ref, model.factor_names[0][1]: []},
        combined_scores={model.factor_names[0][0]: [1.0, 1.0]},
        mc_samples=5,
        random_seed=0,
    )
    assert out[model.factor_names[0][0]] == pytest.approx(1.0)
    assert np.isnan(out[model.factor_names[0][1]])
    assert calls["n"] == 5
