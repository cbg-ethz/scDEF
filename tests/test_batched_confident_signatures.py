import time

import numpy as np
import pytest

from scdef.tools.factor import (
    _collect_hierarchy_mc_scores,
    _confidence_mean_score,
    _confident_signatures_from_mc_scores,
    _get_layer_term_means,
    _hierarchy_gene_scores_draw,
    _signature_jaccard_from_mc_scores,
    get_confident_signatures,
)

scanpy = pytest.importorskip("scanpy")
scd = pytest.importorskip("scdef")


def _make_model(n_cells=60, layer_sizes=(4, 3)):
    adata = scanpy.datasets.pbmc3k()[:n_cells].copy()
    adata.X = adata.X.toarray()
    model = scd.scDEF(adata, layer_sizes=list(layer_sizes), seed=0)
    model.init_var_params(init_budgets=True, init_alpha=True, nmf_init=False)
    model.set_posterior_means()
    model.factor_lists = [
        np.arange(layer_sizes[0], dtype=int),
        np.arange(layer_sizes[1], dtype=int),
    ]
    model.set_factor_names()
    return model


def test_hierarchy_gene_scores_draw_matches_get_signature_sample():
    from jax import random

    model = _make_model()
    base_rng = random.PRNGKey(7)
    n_genes = len(model.adata.var_names)
    for layer_idx in range(1, model.n_layers):
        for factor_idx in range(len(model.factor_names[layer_idx])):
            key = random.fold_in(base_rng, layer_idx * 100 + factor_idx)
            scores, _ = _hierarchy_gene_scores_draw(model, key, max_layer_idx=layer_idx)
            key = random.fold_in(base_rng, layer_idx * 100 + factor_idx)
            _, sampled = model.get_signature_sample(
                key,
                factor_idx=factor_idx,
                layer_idx=layer_idx,
                top_genes=n_genes,
                return_scores=True,
            )
            np.testing.assert_allclose(
                scores[layer_idx][factor_idx],
                sampled,
                rtol=1e-6,
                atol=1e-6,
            )


def test_confident_signatures_from_mc_matches_factorwise_reference():
    model = _make_model()
    mc_samples = 30
    random_seed = 3
    layer_idx = 1
    mc_by_layer = _collect_hierarchy_mc_scores(
        model, mc_samples=mc_samples, random_seed=random_seed, max_layer_idx=layer_idx
    )
    mc_scores = mc_by_layer[layer_idx]

    batched_sigs, batched_confs = _confident_signatures_from_mc_scores(
        model,
        layer_idx=layer_idx,
        mc_scores=mc_scores,
        confidence_threshold=0.9,
        tau_quantile=0.99,
        min_effect=None,
        max_genes=None,
        return_confidences=True,
    )

    term_names = np.asarray(model.adata.var_names)
    term_means = _get_layer_term_means(model, layer_idx)
    for factor_idx, factor_name in enumerate(model.factor_names[layer_idx]):
        mu = term_means[factor_idx]
        tau = float(np.quantile(mu, 0.99))
        sample_arr = mc_scores[:, factor_idx, :]
        confidences = np.mean(sample_arr > tau, axis=0)
        keep_idx = np.where(confidences >= 0.9)[0]
        ref_genes = term_names[keep_idx].tolist()
        assert batched_sigs[factor_name] == ref_genes
        np.testing.assert_allclose(
            batched_confs[factor_name],
            confidences[keep_idx],
            rtol=1e-12,
            atol=1e-12,
        )


def test_get_confident_signatures_upper_layer_uses_batched_mc():
    model = _make_model()
    sigs_a, confs_a = get_confident_signatures(
        model,
        layer_idx=1,
        mc_samples=25,
        random_seed=11,
        return_confidences=True,
    )
    mc_by_layer = _collect_hierarchy_mc_scores(
        model, mc_samples=25, random_seed=11, max_layer_idx=1
    )
    sigs_b, confs_b = _confident_signatures_from_mc_scores(
        model,
        layer_idx=1,
        mc_scores=mc_by_layer[1],
        confidence_threshold=0.9,
        tau_quantile=0.99,
        min_effect=None,
        max_genes=None,
        return_confidences=True,
    )
    assert sigs_a == sigs_b
    for factor_name in model.factor_names[1]:
        np.testing.assert_allclose(
            confs_a[factor_name], confs_b[factor_name], rtol=1e-12, atol=1e-12
        )


def test_signature_jaccard_reuses_same_mc_draws():
    model = _make_model()
    mc_by_layer = _collect_hierarchy_mc_scores(
        model, mc_samples=20, random_seed=5, max_layer_idx=1
    )
    mc_scores = mc_by_layer[1]
    sigs, confs = _confident_signatures_from_mc_scores(
        model,
        layer_idx=1,
        mc_scores=mc_scores,
        confidence_threshold=0.9,
        tau_quantile=0.99,
        min_effect=None,
        max_genes=None,
        return_confidences=True,
    )
    term_means = _get_layer_term_means(model, 1)
    combined = {}
    for factor_idx, factor_name in enumerate(model.factor_names[1]):
        genes = sigs[factor_name]
        if len(genes) == 0:
            combined[factor_name] = []
            continue
        gene_to_idx = {g: i for i, g in enumerate(model.adata.var_names)}
        gene_idx = [gene_to_idx[g] for g in genes]
        mean_arr = term_means[factor_idx, gene_idx]
        combined[factor_name] = _confidence_mean_score(
            confs[factor_name], mean_arr
        ).tolist()

    jaccard = _signature_jaccard_from_mc_scores(
        model,
        layer_idx=1,
        signatures=sigs,
        combined_scores=combined,
        mc_scores=mc_scores,
    )
    for factor_name, val in jaccard.items():
        if len(sigs[factor_name]) == 0:
            assert np.isnan(val)
        else:
            assert 0.0 <= val <= 1.0


def test_set_confident_signatures_end_to_end():
    model = _make_model()
    scd.tl.set_confident_signatures(model, mc_samples=20, random_seed=2)
    cache = model.adata.uns["confident_signatures"]
    assert "1" in cache["by_layer"]
    layer1 = cache["by_layer"]["1"]
    assert "signature_confidences" in layer1
    for factor_name in model.factor_names[1]:
        assert factor_name in layer1["signatures"]
        assert factor_name in layer1["signature_confidences"]


def test_batched_upper_layer_mc_is_faster_than_factorwise_sampling():
    from jax import random

    model = _make_model(n_cells=120, layer_sizes=(6, 4))
    mc_samples = 40
    layer_idx = 1
    n_genes = model.adata.var_names.size

    t0 = time.perf_counter()
    base_rng = random.PRNGKey(0)
    for factor_idx in range(len(model.factor_names[layer_idx])):
        for s_idx in range(mc_samples):
            rng = random.fold_in(base_rng, factor_idx * mc_samples + s_idx)
            model.get_signature_sample(
                rng,
                factor_idx=factor_idx,
                layer_idx=layer_idx,
                top_genes=n_genes,
                return_scores=True,
            )
        for s_idx in range(mc_samples):
            rng = random.fold_in(
                base_rng, layer_idx * 1_000_003 + factor_idx * mc_samples + s_idx
            )
            model.get_signature_sample(
                rng,
                factor_idx=factor_idx,
                layer_idx=layer_idx,
                top_genes=10,
            )
    legacy_elapsed = time.perf_counter() - t0

    t0 = time.perf_counter()
    mc_by_layer = _collect_hierarchy_mc_scores(
        model, mc_samples=mc_samples, random_seed=0, max_layer_idx=layer_idx
    )
    sigs, confs = _confident_signatures_from_mc_scores(
        model,
        layer_idx=layer_idx,
        mc_scores=mc_by_layer[layer_idx],
        confidence_threshold=0.9,
        tau_quantile=0.99,
        min_effect=None,
        max_genes=None,
        return_confidences=True,
    )
    term_means = _get_layer_term_means(model, layer_idx)
    combined = {}
    for factor_idx, factor_name in enumerate(model.factor_names[layer_idx]):
        genes = sigs[factor_name]
        if len(genes) == 0:
            combined[factor_name] = []
            continue
        gene_to_idx = {g: i for i, g in enumerate(model.adata.var_names)}
        gene_idx = [gene_to_idx[g] for g in genes]
        combined[factor_name] = _confidence_mean_score(
            confs[factor_name], term_means[factor_idx, gene_idx]
        ).tolist()
    _signature_jaccard_from_mc_scores(
        model,
        layer_idx=layer_idx,
        signatures=sigs,
        combined_scores=combined,
        mc_scores=mc_by_layer[layer_idx],
    )
    batched_elapsed = time.perf_counter() - t0

    assert batched_elapsed < legacy_elapsed * 0.5, (
        f"expected batched MC to be much faster "
        f"(batched={batched_elapsed:.3f}s, legacy={legacy_elapsed:.3f}s)"
    )
