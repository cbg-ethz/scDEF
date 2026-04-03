import scdef as scd

import scanpy as sc
import numpy as np
import pandas as pd
import scdef.plotting.graph as graph_plot
from pathlib import Path
import pytest
from unittest.mock import patch
import os


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
        batch_key="Experiment",
    )
    assert hasattr(model, "adata")

    model.fit(n_epoch=10)

    model.filter_factors(brd_min=1.0, min_cells_lower=0)  # make sure we keep factors

    model.logger.info(model.factor_lists)

    model.fit(n_epoch=10)

    assert len(model.elbos) == 1
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

    for mode in ["f1", "fracs", "weights"]:
        scd.pl.obs_scores(
            model,
            ["celltypes", "celltypes_coarse"],
            mode=mode,
            hierarchy=true_hierarchy,
            show=False,
        )
    assert "obs_scores" in model.adata.uns
    assert "fracs" in model.adata.uns["obs_scores"]
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


def test_scdef_alpha_annealing_fit():
    adata = sc.datasets.pbmc3k()
    np.random.seed(13)
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
    alpha_before = float(model.alpha)
    with patch.object(
        model, "_compute_median_parents", return_value=(3.0, 3)
    ) as mocked:
        model.fit(
            n_epoch=3,
            anneal_alpha=True,
            alpha_burn_in=2,
            check_every=1,
            target_parents=1.5,
            max_elbo_drop=1.0,
            damping=0.5,
        )

    assert model.alpha > alpha_before
    assert mocked.call_count == 1
    assert len(model.elbos) == 1
    assert "L0" in model.adata.obs.columns


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


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true",
    reason="Network-dependent enrichment test is skipped on GitHub Actions.",
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
    enr = scd.tl.gsea(
        model,
        libs=["KEGG_2019_Human"],
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
