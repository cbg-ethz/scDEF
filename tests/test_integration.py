import scdef as scd

import scanpy as sc
import numpy as np
import pandas as pd


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

    model.fit(n_epoch=3)

    model.filter_factors(thres=0.0, min_cells_lower=0)  # make sure we keep factors

    model.logger.info(model.factor_lists)

    assert len(model.elbos) == 2
    assert "L0" in model.adata.obs.columns
    assert "L1" in model.adata.obs.columns
    assert "L2" in model.adata.obs.columns

    scd.pl.plot_qc(model, show=False)

    scd.pl.plot_multilevel_paga(
        model, figsize=(16, 4), reuse_pos=True, frameon=False, show=False
    )

    scd.pl.plot_signatures_scores(model, "celltypes", markers, top_genes=10, show=False)

    for mode in ["f1", "fracs", "weights"]:
        scd.pl.plot_obs_scores(
            model,
            ["celltypes", "celltypes_coarse"],
            mode=mode,
            hierarchy=true_hierarchy,
            show=False,
        )

    simplified = scd.tl.get_hierarchy(model, simplified=True)
    g = scd.pl.make_graph(model, hierarchy=simplified)

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

    scd.pl.plot_umaps(
        model,
        color=["celltypes", "celltypes_coarse"],
        fontsize=16,
        legend_fontsize=14,
        show=False,
    )

    scd.pl.plot_factors_bars(model, ["celltypes", "celltypes_coarse"], show=False)
