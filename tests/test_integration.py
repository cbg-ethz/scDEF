import scdef as scd

import scanpy as sc
import numpy as np
import pandas as pd
import scdef.plotting.graph as graph_plot
from pathlib import Path
import pytest
import os
import matplotlib.pyplot as plt
from unittest.mock import patch


def test_scdef():
    # Ground truth
    true_hierarchy = {
        "T": ["CD8 T", "Memory CD4 T", "Naive CD4 T"],
        "Mono": ["FCGR3A+ Mono", "CD14+ Mono", "DC"],
        "Platelet": [],
        "B": [],
        "CD8 T": [],
        "Memory CD4 T": [],
        "Naive CD4 T": [],
        "NK": [],
        "FCGR3A+ Mono": [],
        "CD14+ Mono": [],
        "DC": [],
    }

    markers = {
        "Naive CD4 T": ["IL7R"],
        "Memory CD4 T": ["IL7R"],
        "CD14+ Mono": ["CD14", "LYZ"],
        "B": ["MS4A1"],
        "CD8 T": ["CD8A", "CD2"],
        "NK": ["GNLY", "NKG7"],
        "FCGR3A+ Mono": ["FCGR3A", "MS4A7"],
        "DC": ["FCER1A", "CST3"],
        "Platelet": ["PPBP"],
    }

    # Download data
    adata = sc.datasets.pbmc3k()

    # Add random annotations
    n_cells = adata.shape[0]
    ctypes = np.random.choice(list(markers.keys()), size=n_cells)
    annotations = pd.DataFrame(index=adata.obs.index)
    annotations["ctypes"] = ctypes
    batches = np.random.choice(["A", "B", "C"], size=n_cells)
    annotations["batches"] = batches

    map_coarse = {}
    for c in annotations["ctypes"].astype("category").cat.categories:
        if c.endswith(" T"):
            map_coarse[c] = "T"
        elif c.endswith("Mono") or c == "DC":
            map_coarse[c] = "Mono"
        else:
            map_coarse[c] = c

    adata.obs["celltypes"] = annotations["ctypes"]
    adata.obs["batches"] = annotations["batches"]

    adata.obs["celltypes_coarse"] = (
        adata.obs["celltypes"].map(map_coarse).astype("category")
    )

    # Filter data
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata = adata[np.random.randint(adata.shape[0], size=200)]
    adata.var["mt"] = adata.var_names.str.startswith(
        "MT-"
    )  # annotate the group of mitochondrial genes as 'mt'
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )
    adata = adata[adata.obs.n_genes_by_counts < 2500, :]
    adata = adata[adata.obs.pct_counts_mt < 5, :]
    adata.raw = adata
    raw_adata = adata.raw
    raw_adata = raw_adata.to_adata()
    raw_adata.X = raw_adata.X.toarray()
    adata.layers["counts"] = adata.X.toarray()  # Keep the counts
    adata.X = adata.X.toarray()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=300
    )
    raw_adata = raw_adata[:, adata.var.highly_variable]
    adata = adata[:, adata.var.highly_variable]

    model = scd.scDEF(
        raw_adata,
        layer_sizes=[10, 5, 2],
        seed=1,
        batch_key="batches",
    )
    assert hasattr(model, "adata")

    model.fit(n_epoch=10)

    model.filter_factors(brd_min=1.0, min_cells_lower=0)  # make sure we keep factors

    model.logger.info(model.factor_lists)

    model.fit(n_epoch=10)

    lr_batch = scd.tl.get_batch_specific_genes_from_gene_scale(model)
    assert lr_batch.shape == (model.adata.n_vars, model.n_batches)
    assert list(lr_batch.columns) == [str(b) for b in model.batches]
    assert np.all(np.isfinite(lr_batch.to_numpy(dtype=float)))
    lr_global = scd.tl.get_batch_specific_genes_from_gene_scale(
        model, reference="global_mean"
    )
    assert lr_global.shape == lr_batch.shape

    expected_fit_passes = 1 + int(getattr(model, "root_epochs", 0) > 0)
    assert len(model.elbos) == expected_fit_passes
    assert "L0" in model.adata.obs.columns
    assert "L1" in model.adata.obs.columns
    assert "L2" in model.adata.obs.columns

    scd.pl.qc(model, show=False)

    scd.pl.multilevel_paga(
        model, figsize=(16, 4), reuse_pos=True, frameon=False, show=False
    )

    scd.tl.set_confident_signatures(model)
    scd.pl.signatures_scores(model, "celltypes", markers, top_genes=10, show=False)
    # Confidence-based signatures are used (and capped by top_genes) in tools/plots.
    top_k = 5
    confident_all = {}
    for layer_idx in range(model.n_layers):
        layer_sigs = scd.tl.get_stored_confident_signatures(
            model, layer_idx=layer_idx, max_genes=top_k
        )
        confident_all.update(layer_sigs)
        for factor_name in model.factor_names[layer_idx]:
            assert len(layer_sigs.get(factor_name, [])) <= top_k
    cached_sigs = scd.tl.set_factor_signatures(model, top_genes=top_k)
    for factor_name, sig in confident_all.items():
        assert cached_sigs[factor_name] == sig
    ranked_terms, ranked_scores = model.get_rankings(
        layer_idx=0, top_genes=top_k, return_scores=True
    )
    expected_terms, expected_scores = scd.tl.get_stored_confident_signatures(
        model,
        layer_idx=0,
        max_genes=top_k,
        return_combined_scores=True,
    )
    for factor_idx, factor_name in enumerate(model.factor_names[0]):
        assert ranked_terms[factor_idx] == expected_terms[factor_name]
        assert np.allclose(ranked_scores[factor_idx], expected_scores[factor_name])

    for mode in ["f1", "fracs", "weights", "prob"]:
        scd.pl.obs_scores(
            model,
            ["celltypes", "celltypes_coarse"],
            mode=mode,
            hierarchy=true_hierarchy,
            show=False,
        )
    assert "obs_scores" in model.adata.uns
    assert "fracs" in model.adata.uns["obs_scores"]
    assert "prob" in model.adata.uns["obs_scores"]
    ranked = scd.tl.get_obs_score_rankings(
        model,
        layer=0,
        obs_key="celltypes",
        obs_values=["B", "NK"],
        score_model="fracs",
    )
    assert isinstance(ranked, pd.DataFrame)
    assert len(ranked) == 2 * len(model.factor_names[0])
    assert set(ranked["obs_value"].unique()) == {"B", "NK"}
    assert "score" in ranked.columns
    for obs_value in ["B", "NK"]:
        sub = ranked[ranked["obs_value"] == obs_value]
        assert sub["score"].is_monotonic_decreasing
    specific = scd.tl.get_obs_value_specific_factors(
        model,
        layer=0,
        obs_key="celltypes",
        obs_values=["B", "NK"],
        score_model="fracs",
        min_specificity=0.0,
    )
    assert set(specific.keys()) == {"B", "NK"}
    specific_df = scd.tl.get_obs_value_specific_factors(
        model,
        layer=0,
        obs_key="celltypes",
        obs_values=["B", "NK"],
        score_model="fracs",
        min_specificity=0.0,
        return_scores=True,
    )
    assert isinstance(specific_df, pd.DataFrame)
    assert "specificity" in specific_df.columns
    entropy_cols = scd.tl.set_cell_entropies(model)
    assert len(entropy_cols) == model.n_layers
    for idx in range(model.n_layers):
        col = f"{model.layer_names[idx]}_entropy"
        assert col in model.adata.obs.columns
        eff_col = f"{model.layer_names[idx]}_effective_n_factors"
        assert eff_col in model.adata.obs.columns
    jsd_df = scd.tl.compute_within_group_pairwise_dissimilarity(
        model,
        layer=0,
        obs_key="celltypes",
        metric="jsd",
    )
    assert isinstance(jsd_df, pd.DataFrame)
    assert "mean_distance" in jsd_df.columns
    assert "within_group_pairwise_dissimilarity" in model.adata.uns
    scd.pl.within_group_pairwise_dissimilarity(
        model,
        layer=0,
        obs_key="celltypes",
        metric="jsd",
        kind="box",
        show=False,
    )

    scd.tl.factor_diagnostics(model, recompute=True)
    scd.tl.set_technical_factors(model, factors=[model.factor_names[0][0]])
    scd.tl.make_hierarchies(model)
    with patch(
        "scdef.plotting.graph._get_confident_signature_rankings_layer",
        wraps=graph_plot._get_confident_signature_rankings_layer,
    ) as mocked_conf_sig:
        scd.pl.biological_hierarchy(model, show_label=False)
        assert mocked_conf_sig.called
    scd.pl.technical_hierarchy(model, show_label=False)

    simplified = scd.tl.get_hierarchy(model, simplified=True)
    g = scd.pl.make_graph(model, hierarchy=simplified, show_confidences=True)

    assignments, matches = scd.utils.factor_utils.assign_obs_to_factors(
        model,
        ["celltypes", "celltypes_coarse"],
        factor_names=scd.utils.hierarchy_utils.get_nodes_from_hierarchy(simplified),
    )
    g = scd.pl.make_graph(model, hierarchy=simplified, factor_annotations=matches)

    roots = [
        k for k, v in simplified.items() if isinstance(v, (list, tuple)) and len(v) > 0
    ]
    if len(roots) > 0:
        r0 = roots[0]
        c0 = simplified[r0][0]
        g_path = scd.pl.make_graph(
            model,
            hierarchy=simplified,
            path=[r0, c0],
            path_color="blue",
            show_label=False,
        )
        assert g_path is not None
        g_path_gene = scd.pl.make_graph(
            model,
            hierarchy=simplified,
            path={"nodes": [r0, c0]},
            gene_score=str(model.adata.var_names[0]),
            color_edges=True,
            path_color="#ff00aa",
            show_label=False,
        )
        assert g_path_gene is not None

    if len(simplified.keys()) > 0:
        k = list(simplified.keys())[0]
        g = scd.pl.make_graph(
            model, hierarchy=simplified, top_factor=k, factor_annotations=matches
        )

    signatures, scores = model.get_signatures_dict(scores=True, sorted_scores=False)
    sizes = model.get_sizes_dict()

    scd.tl.umap(model)
    scd.pl.umap(
        model,
        color=["celltypes", "celltypes_coarse"],
        fontsize=16,
        legend_fontsize=14,
        show=False,
    )

    scd.pl.factors_bars(model, ["celltypes", "celltypes_coarse"], show=False)


def test_iscdef_refit_after_filtering():
    markers = {
        "Naive CD4 T": ["IL7R"],
        "Memory CD4 T": ["IL7R"],
        "CD14+ Mono": ["CD14", "LYZ"],
        "B": ["MS4A1"],
        "CD8 T": ["CD8A", "CD2"],
        "NK": ["GNLY", "NKG7"],
        "FCGR3A+ Mono": ["FCGR3A", "MS4A7"],
        "DC": ["FCER1A", "CST3"],
        "Platelet": ["PPBP"],
    }

    adata = sc.datasets.pbmc3k()
    np.random.seed(7)
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata = adata[np.random.randint(adata.shape[0], size=200)]
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )
    adata = adata[adata.obs.n_genes_by_counts < 2500, :]
    adata = adata[adata.obs.pct_counts_mt < 5, :]
    adata.raw = adata
    raw_adata = adata.raw.to_adata()
    raw_adata.X = raw_adata.X.toarray()
    adata.X = adata.X.toarray()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=300
    )
    raw_adata = raw_adata[:, adata.var.highly_variable]

    model = scd.iscDEF(
        raw_adata,
        markers_dict=markers,
        markers_layer=3,
        add_other=1,
        seed=1,
    )

    model.fit(n_epoch=3, n_rounds=1)
    model.filter_factors(brd_min=1.0, min_cells_lower=0)
    upper_assignments_before = {}
    for layer_idx in range(1, model.n_layers):
        layer_name = model.layer_names[layer_idx]
        upper_assignments_before[layer_name] = np.argmax(
            model.adata.obsm[f"X_{layer_name}"], axis=1
        )
    model.fit(n_epoch=3, n_rounds=1)

    assert len(model.factor_lists) == model.n_layers
    assert all(len(factors) > 0 for factors in model.factor_lists)
    for layer_name, assignments_before in upper_assignments_before.items():
        assignments_after = np.argmax(model.adata.obsm[f"X_{layer_name}"], axis=1)
        agreement = np.mean(assignments_before == assignments_after)
        assert agreement >= 0.7, (
            f"Upper-layer assignments for {layer_name} changed too much "
            f"after refit (agreement={agreement:.3f})."
        )


def test_scdef_entropy_annealing_updates():
    adata = sc.datasets.pbmc3k()
    np.random.seed(31)
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata = adata[np.random.randint(adata.shape[0], size=80)]
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )
    adata = adata[adata.obs.n_genes_by_counts < 2500, :]
    adata = adata[adata.obs.pct_counts_mt < 5, :]
    adata.raw = adata
    raw_adata = adata.raw.to_adata()
    raw_adata.X = raw_adata.X.toarray()
    adata.X = adata.X.toarray()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=150
    )
    raw_adata = raw_adata[:, adata.var.highly_variable]

    model = scd.scDEF(
        raw_adata,
        layer_sizes=[6, 3, 1],
        seed=1,
    )

    model.fit(
        n_epoch=20,
        root_epochs=0,
        annealing=1.0,
        entropy_anneal=True,
        entropy_window=5,
        entropy_check_every=1,
        # Force frequent increases (relative_change is always < huge threshold),
        # so this test is deterministic and does not depend on loss dynamics.
        entropy_rel_change_low=1e9,
        entropy_rel_change_high=1e10,
        entropy_increase_factor=1.2,
        entropy_decrease_factor=0.9,
        entropy_max_annealing=3.0,
        entropy_min_annealing=1.0,
        entropy_optimizer_reset_threshold=0.05,
    )

    trace = np.asarray(model.adata.uns["entropy_annealing_trace"])
    trace_epochs = np.asarray(model.adata.uns["entropy_annealing_trace_epochs"])
    assert trace.shape[0] == trace_epochs.shape[0]
    assert trace.shape[0] == len(model.elbos[0])
    assert np.all(trace >= 1.0)
    assert np.all(trace <= 3.0 + 1e-9)
    # Should have increased at least once from the initial value.
    assert np.max(trace) > 1.0

    # QC plot should include available annealing traces without error.
    fig = scd.pl.qc(model, show=False)
    assert fig is not None
    plt.close(fig)


def test_scdef_assign_confident():
    adata = sc.datasets.pbmc3k()
    np.random.seed(17)
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata = adata[np.random.randint(adata.shape[0], size=120)]
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )
    adata = adata[adata.obs.n_genes_by_counts < 2500, :]
    adata = adata[adata.obs.pct_counts_mt < 5, :]
    adata.raw = adata
    raw_adata = adata.raw.to_adata()
    raw_adata.X = raw_adata.X.toarray()
    adata.X = adata.X.toarray()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=200
    )
    raw_adata = raw_adata[:, adata.var.highly_variable]

    # Top layer has K=1 so it is a true single-factor "root" that can
    # act as the stem-cell fallback for cells with no confident
    # multi-factor assignment at lower layers.
    model = scd.scDEF(
        raw_adata,
        layer_sizes=[8, 4, 1],
        seed=1,
    )
    model.fit(n_epoch=3)

    n_cells = model.adata.n_obs
    n_layers = model.n_layers
    top_layer_name = model.layer_names[-1]

    # Default call uses the gap-based (winner minus runner-up) metric,
    # which is invariant to layer size K_k:
    #   per-sample gap^(s) = ẑ^(s)_{f*} - max_{g != f*} ẑ^(s)_g
    #   effect_size   = E_s[gap^(s)]
    #   posterior_sd  = SD_s[gap^(s)]
    #   confidence    = quantile_{1 - credible_level}({gap^(s)})
    # and picks the FINEST multi-factor layer whose confidence clears tau.
    tau = 0.3
    credible_level = 0.9
    scd.tl.assign_confident(
        model, n_samples=200, tau=tau, credible_level=credible_level
    )

    effect_mat = np.asarray(model.adata.obsm["confident_effect_size"])
    sd_mat = np.asarray(model.adata.obsm["confident_posterior_sd"])
    conf_mat = np.asarray(model.adata.obsm["confident_confidence"])
    winner_mass_mat = np.asarray(model.adata.obsm["confident_winner_mass"])
    winner_prob_mat = np.asarray(model.adata.obsm["confident_winner_probability"])
    ent_conf_mat = np.asarray(model.adata.obsm["confident_entropy_confidence"])
    argmax_mat = np.asarray(model.adata.obsm["confident_argmax_factor"])
    for mat in (
        effect_mat,
        sd_mat,
        conf_mat,
        winner_mass_mat,
        winner_prob_mat,
        ent_conf_mat,
        argmax_mat,
    ):
        assert mat.shape == (n_cells, n_layers)
    # gap can go slightly negative in cells where posterior identity is
    # not stable, but it's bounded in [-1, 1]; confidence is clipped to
    # [0, 1].
    assert np.all((effect_mat >= -1.0) & (effect_mat <= 1.0))
    assert np.all((sd_mat >= 0.0) & (sd_mat <= 1.0))
    assert np.all((conf_mat >= 0.0) & (conf_mat <= 1.0))
    assert np.all((winner_mass_mat >= 0.0) & (winner_mass_mat <= 1.0))
    assert np.all((winner_prob_mat >= 0.0) & (winner_prob_mat <= 1.0))
    assert np.all((ent_conf_mat >= 0.0) & (ent_conf_mat <= 1.0))

    # Single-factor layers have ẑ ≡ 1 and no runner-up: gap is defined
    # as 1 so confidence = 1 (stem-cell fallback can always win).
    for layer_idx in range(n_layers):
        if len(model.factor_lists[layer_idx]) == 1:
            np.testing.assert_allclose(effect_mat[:, layer_idx], 1.0)
            np.testing.assert_allclose(sd_mat[:, layer_idx], 0.0, atol=1e-6)
            np.testing.assert_allclose(conf_mat[:, layer_idx], 1.0)
            np.testing.assert_allclose(winner_mass_mat[:, layer_idx], 1.0)
            np.testing.assert_allclose(winner_prob_mat[:, layer_idx], 1.0)
            np.testing.assert_allclose(ent_conf_mat[:, layer_idx], 1.0)
        if len(model.factor_lists[layer_idx]) > 0:
            assert np.all(argmax_mat[:, layer_idx] >= 0)
            assert np.all(argmax_mat[:, layer_idx] < len(model.factor_lists[layer_idx]))

    # For multi-factor layers, winner_mass is in [1/K_k, 1).
    for layer_idx in range(n_layers):
        K_k = len(model.factor_lists[layer_idx])
        if K_k >= 2:
            assert np.all(winner_mass_mat[:, layer_idx] >= 1.0 / K_k - 1e-9)
            assert np.all(winner_mass_mat[:, layer_idx] < 1.0)

    # Per-layer obs columns exist.
    for layer_idx in range(n_layers):
        layer_name = model.layer_names[layer_idx]
        assert f"confident_confidence_{layer_name}" in model.adata.obs.columns
        assert f"confident_argmax_{layer_name}" in model.adata.obs.columns

    # Best-layer summaries.
    for col in [
        "confident_best_layer",
        "confident_best_layer_idx",
        "confident_best_factor_idx",
        "confident_factor",
        "confident_best_effect_size",
        "confident_best_posterior_sd",
        "confident_best_confidence",
        "confident_depth_score",
    ]:
        assert col in model.adata.obs.columns

    best_layer = model.adata.obs["confident_best_layer"].astype(str).to_numpy()
    best_conf = model.adata.obs["confident_best_confidence"].to_numpy()
    best_effect = model.adata.obs["confident_best_effect_size"].to_numpy()
    best_sd = model.adata.obs["confident_best_posterior_sd"].to_numpy()
    best_label = model.adata.obs["confident_factor"].astype(str).to_numpy()

    # Every cell has SOME assignment (root is a universal fallback).
    assert not np.any(best_layer == "")
    assert not np.any(np.isnan(best_conf))
    assert not np.any(np.isnan(best_effect))
    assert not np.any(np.isnan(best_sd))

    # Finest-that-clears invariant: for each cell, the chosen layer is
    # either (a) the finest multi-factor layer with conf >= tau, or
    # (b) the top single-factor fallback layer.
    layer_name_to_idx = {name: i for i, name in enumerate(model.layer_names)}
    best_layer_idx_obs = np.asarray(
        model.adata.obs["confident_best_layer_idx"], dtype=int
    )
    np.testing.assert_array_equal(
        best_layer_idx_obs,
        np.array([layer_name_to_idx[str(x)] for x in best_layer], dtype=int),
    )
    depth_obs = np.asarray(model.adata.obs["confident_depth_score"], dtype=float)
    assert np.all((depth_obs >= 0.0) & (depth_obs <= 1.0))
    assert not np.any(np.isnan(depth_obs))
    for cell_idx in range(n_cells):
        lname = best_layer[cell_idx]
        chosen_idx = layer_name_to_idx[lname]
        np.testing.assert_allclose(
            depth_obs[cell_idx],
            chosen_idx / float(n_layers - 1),
            rtol=0.0,
            atol=0.0,
        )
        K_chosen = len(model.factor_lists[chosen_idx])
        if K_chosen >= 2:
            assert best_conf[cell_idx] >= tau
            for finer_idx in range(chosen_idx):
                if len(model.factor_lists[finer_idx]) >= 2:
                    assert conf_mat[cell_idx, finer_idx] < tau
        else:
            assert lname == top_layer_name

    # best_label should be a valid factor name in the chosen layer.
    for cell_idx in range(n_cells):
        lname = best_layer[cell_idx]
        layer_idx = model.layer_names.index(lname)
        assert best_label[cell_idx] in model.factor_names[layer_idx]

    # uns metadata is populated.
    meta = model.adata.uns["confident"]
    assert meta["tau"] == tau
    assert meta["n_samples"] == 200
    assert meta["metric"] == "empirical_lower_quantile_winner_runner_up_gap"
    assert meta["selection_rule"] == "finest_layer_clearing_tau"
    np.testing.assert_allclose(meta["credible_level"], credible_level)
    np.testing.assert_allclose(meta["quantile_level"], 1.0 - credible_level)
    assert list(meta["layer_names"]) == list(model.layer_names)
    assert "depth_score" in meta

    g_conf = scd.pl.make_graph(
        model,
        confident_assignments=True,
        show_signatures=False,
        show_label=False,
    )
    assert g_conf is not None
    with pytest.raises(KeyError, match="missing_prefix_factor"):
        scd.pl.make_graph(
            model,
            confident_assignments=True,
            confident_key="missing_prefix",
            show_signatures=False,
        )

    # Fallback semantics: with tau = 1.0, NO multi-factor layer can
    # clear the bar (the gap is strictly < 1 whenever K_k >= 2, since
    # the runner-up has some positive mass), so every cell must fall
    # back to the single-factor root.
    scd.tl.assign_confident(model, n_samples=100, tau=1.0, key_added="cf_hi")
    assert "cf_hi_best_layer" in model.adata.obs.columns
    assert "cf_hi_best_layer_idx" in model.adata.obs.columns
    assert "cf_hi_depth_score" in model.adata.obs.columns
    best_layer_hi = model.adata.obs["cf_hi_best_layer"].astype(str).to_numpy()
    best_conf_hi = model.adata.obs["cf_hi_best_confidence"].to_numpy()
    depth_hi = np.asarray(model.adata.obs["cf_hi_depth_score"], dtype=float)
    idx_hi = np.asarray(model.adata.obs["cf_hi_best_layer_idx"], dtype=int)
    assert np.all(best_layer_hi == top_layer_name)
    assert np.all(idx_hi == n_layers - 1)
    np.testing.assert_allclose(best_conf_hi, 1.0)
    np.testing.assert_allclose(depth_hi, 1.0)

    g_cf_hi_graph = scd.pl.make_graph(
        model,
        confident_assignments=True,
        confident_key="cf_hi",
        show_signatures=False,
        show_label=False,
    )
    assert g_cf_hi_graph is not None

    with pytest.raises(ValueError, match="at least two batch rows"):
        scd.tl.get_batch_specific_genes_from_gene_scale(model)


def test_scdef_path_pipeline_and_plotting():
    adata = sc.datasets.pbmc3k()
    np.random.seed(23)
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata = adata[np.random.randint(adata.shape[0], size=100)]
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )
    adata = adata[adata.obs.n_genes_by_counts < 2500, :]
    adata = adata[adata.obs.pct_counts_mt < 5, :]
    adata.raw = adata
    raw_adata = adata.raw.to_adata()
    raw_adata.X = raw_adata.X.toarray()
    adata.X = adata.X.toarray()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=200
    )
    raw_adata = raw_adata[:, adata.var.highly_variable]

    model = scd.scDEF(
        raw_adata,
        layer_sizes=[8, 4, 1],
        seed=1,
    )
    model.fit(n_epoch=3)

    # Add a simple condition label to exercise faceting in path_embedding.
    rng = np.random.default_rng(23)
    model.adata.obs["condition"] = pd.Categorical(
        rng.choice(["untreated", "treated", "pseudo_escape"], size=model.adata.n_obs)
    )

    # Build a multilayer embedding for path visualization.
    scd.tl.multilayer_umap(model, key_added="multilayer", normalize_per_layer=True)

    # Transition paths (de novo) + path scores.
    tpaths = scd.tl.build_transition_paths(
        model,
        rel_parent_weight=0.2,
        max_path_len=5,
        max_paths_per_pair=3,
    )
    assert isinstance(tpaths, list)
    assert "transition_paths" in model.adata.uns
    scd.tl.score_paths(
        model,
        paths_key="transition_paths",
        key_added="transition_paths",
        min_affinity=0.05,
    )

    tpos = np.asarray(model.adata.obsm["transition_paths_positions"])
    taff = np.asarray(model.adata.obsm["transition_paths_affinities"])
    assert tpos.shape[0] == model.adata.n_obs
    assert taff.shape == tpos.shape
    assert np.all((taff >= 0.0) & (taff <= 1.0))
    if tpos.shape[1] > 0:
        assert np.all(np.isnan(tpos) | ((tpos >= 0.0) & (tpos <= 1.0)))

    # Differentiation paths + path scores.
    dpaths = scd.tl.build_differentiation_paths(model, rel_parent_weight=0.25)
    assert isinstance(dpaths, list)
    assert "differentiation_paths" in model.adata.uns
    scd.tl.score_paths(
        model,
        paths_key="differentiation_paths",
        key_added="differentiation_paths",
        min_affinity=0.05,
    )
    dpos = np.asarray(model.adata.obsm["differentiation_paths_positions"])
    daff = np.asarray(model.adata.obsm["differentiation_paths_affinities"])
    assert dpos.shape[0] == model.adata.n_obs
    assert daff.shape == dpos.shape
    assert np.all((daff >= 0.0) & (daff <= 1.0))
    if dpos.shape[1] > 0:
        assert np.all(np.isnan(dpos) | ((dpos >= 0.0) & (dpos <= 1.0)))

    # Plotting helper: auto path-id selection and faceting by condition.
    if tpos.shape[1] > 0:
        fig = scd.pl.path_embedding(
            model,
            path_id="auto",
            paths_key="transition_paths",
            score_key="transition_paths",
            basis="umap_multilayer",
            obs_key="condition",
            obs_order=["untreated", "treated", "pseudo_escape"],
            ncols=2,
            show=False,
        )
        assert fig is not None
        plt.close(fig)

    if dpos.shape[1] > 0:
        fig = scd.pl.path_embedding(
            model,
            path_id="auto",
            paths_key="differentiation_paths",
            score_key="differentiation_paths",
            basis="umap_multilayer",
            show=False,
        )
        assert fig is not None
        plt.close(fig)

        scd.tl.set_confident_signatures(model)
        fig_ph = scd.pl.plot_path_trajectory_heatmap(
            model,
            path_id=0,
            paths_key="differentiation_paths",
            score_key="differentiation_paths",
            genes_per_factor=2,
            show=False,
        )
        assert fig_ph is not None
        plt.close(fig_ph)

        g1, g2 = list(model.adata.var_names[:2])
        fig_pg = scd.pl.plot_path_trajectory_heatmap(
            model,
            path_id=0,
            paths_key="differentiation_paths",
            score_key="differentiation_paths",
            genes=[g1, g2],
            normalize=False,
            show=False,
        )
        assert fig_pg is not None
        plt.close(fig_pg)


def test_scdef_load_and_plotting_pipeline():
    model_dir = Path(__file__).resolve().parent / "pretrained_scdef_model"
    state_file = model_dir / "model_state.pkl"
    if not state_file.exists():
        pytest.skip(
            "Pretrained model artifact not found. "
            "Place it under tests/pretrained_scdef_model/ (with model_state.pkl)."
        )

    try:
        loaded = scd.scDEF.load(model_dir)
    except ValueError as e:
        pytest.skip(
            f"Pretrained artifact found but missing AnnData for loading: {e}. "
            "Include adata.h5ad in the artifact or adapt test to pass adata explicitly."
        )

    loaded.set_posterior_variances()

    # Cached PAGA compute + plotting from loaded model.
    layers = [i for i in range(loaded.n_layers - 1) if len(loaded.factor_lists[i]) > 1]
    if len(layers) > 0:
        scd.tl.multilevel_paga(
            loaded,
            neighbors_rep=f"X_{loaded.layer_names[0]}",
            layers=layers,
            reuse_pos=True,
        )
        assert "multilevel_paga" in loaded.adata.uns
        scd.pl.multilevel_paga(loaded, layers=layers, show=False)

    # Trajectory heatmap from loaded model along an L0 path.
    if len(loaded.factor_names[0]) >= 2:
        scd.tl.set_confident_signatures(loaded)
        factor_path = loaded.factor_names[0][: min(3, len(loaded.factor_names[0]))]
        scd.pl.plot_trajectory_heatmap(
            loaded,
            factor_path=factor_path,
            layer_idx=0,
            genes_per_factor=2,
            annotation_obs_key=None,
            show=False,
        )

    # Factor-gene uncertainty boxplots (L0 and upper layer when available).
    if len(loaded.factor_names[0]) > 0:
        scd.pl.factor_gene_uncertainty_boxplot(
            loaded,
            factor=loaded.factor_names[0][0],
            layer_idx=0,
            max_genes=20,
            sort_by="confidence",
            color_by_confidence=True,
            show_confidence_cutoff_line=True,
            confidence_include_threshold=0.9,
            show_tau_quantile_line=True,
            show=False,
        )
    upper_layers = [
        i for i in range(1, loaded.n_layers) if len(loaded.factor_names[i]) > 0
    ]
    if len(upper_layers) > 0:
        upper_idx = upper_layers[0]
        scd.pl.factor_gene_uncertainty_boxplot(
            loaded,
            factor=loaded.factor_names[upper_idx][0],
            layer_idx=upper_idx,
            max_genes=20,
            sort_by="confidence",
            color_by_confidence=True,
            show_confidence_cutoff_line=True,
            confidence_include_threshold=0.9,
            show_tau_quantile_line=True,
            mc_samples_upper=20,
            random_seed=0,
            show=False,
        )


def test_scdef_gsea_and_graph_enrichments():
    pytest.importorskip("gseapy")
    adata = sc.datasets.pbmc3k()
    np.random.seed(21)
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata = adata[np.random.randint(adata.shape[0], size=120)]
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )
    adata = adata[adata.obs.n_genes_by_counts < 2500, :]
    adata = adata[adata.obs.pct_counts_mt < 5, :]
    adata.raw = adata
    raw_adata = adata.raw.to_adata()
    raw_adata.X = raw_adata.X.toarray()
    adata.X = adata.X.toarray()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=200
    )
    raw_adata = raw_adata[:, adata.var.highly_variable]

    model = scd.scDEF(
        raw_adata,
        layer_sizes=[8, 4, 2],
        seed=1,
    )
    model.fit(n_epoch=3)
    scd.tl.set_confident_signatures(model)
    # Keep this test offline and deterministic (no Enrichr network dependency).
    custom_gene_sets = {
        "Mock_Pathway_A": list(model.adata.var_names[:80]),
        "Mock_Pathway_B": list(model.adata.var_names[40:120]),
    }
    enr = scd.tl.gsea(
        model,
        libs=[],
        custom_gene_sets=custom_gene_sets,
        layers=[0],
        top_genes=50,
        cutoff=0.05,
        outdir=None,
    )
    assert isinstance(enr, pd.DataFrame)
    assert "factor_enrichments" in model.adata.uns

    g = scd.pl.make_graph(
        model,
        show_enrichments=True,
        show_signatures=False,
    )
    assert g is not None
