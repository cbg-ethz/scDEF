import numpy as np
import pytest

scanpy = pytest.importorskip("scanpy")
sc = scanpy
scd = pytest.importorskip("scdef")
from scdef.tools.factor import (  # noqa: E402
    _kept_layer_w,
    hard_assignment_name_indices,
    tree_assignment_name_indices,
)


@pytest.fixture(scope="module")
def model():
    adata = scanpy.datasets.pbmc3k()[:150].copy()
    adata.X = adata.X.toarray()
    sc.pp.pca(adata, n_comps=20)
    m = scd.scDEF(adata, layer_sizes=[8, 4, 2, 1], seed=0)
    m.fit(n_epoch=5, hierarchical_init=True)
    return m


def test_tree_rule_is_the_default(model):
    assert model.upper_assignment == "tree"


def test_upper_clusters_follow_the_tree(model):
    names = model.factor_names
    for layer_idx in range(1, model.n_layers):
        parent = np.argmax(_kept_layer_w(model, layer_idx), axis=0)
        below = np.array([names[layer_idx - 1].index(f)
                          for f in model.adata.obs[model.layer_names[layer_idx - 1]].astype(str)])
        here = np.array([names[layer_idx].index(f)
                         for f in model.adata.obs[model.layer_names[layer_idx]].astype(str)])
        np.testing.assert_array_equal(here, parent[below])


def test_layer_zero_is_unchanged(model):
    np.testing.assert_array_equal(tree_assignment_name_indices(model, 0),
                                  hard_assignment_name_indices(model, 0))


def test_argmax_rule_restores_the_old_behaviour(model):
    model.upper_assignment = "argmax"
    try:
        model.annotate_adata()
        for layer_idx in range(1, model.n_layers):
            got = model.adata.obs[model.layer_names[layer_idx]].astype(str).to_numpy()
            want = np.array(model.factor_names[layer_idx])[hard_assignment_name_indices(model, layer_idx)]
            np.testing.assert_array_equal(got, want)
    finally:
        model.upper_assignment = "tree"
        model.annotate_adata()


def test_graph_counts_the_cells_it_wedges(model):
    # The diagnostics partition upper layers by each layer's argmax; the graph must
    # label each node with the cells in adata.obs, which its wedges are drawn from.
    scd.tl.factor_diagnostics(model)
    g = scd.pl.make_graph(model, n_cells_label=True, show_signatures=False, show=False)
    for layer_idx in range(model.n_layers - 1):
        counts = model.adata.obs[model.layer_names[layer_idx]].astype(str).value_counts()
        for name, n in counts.items():
            assert f"({int(n)} cells)" in g.source, (name, n)


def test_unknown_rule_raises(model):
    model.upper_assignment = "nearest"
    try:
        with pytest.raises(ValueError, match="upper_assignment"):
            model.annotate_adata()
    finally:
        model.upper_assignment = "tree"
        model.annotate_adata()


@pytest.fixture(scope="module")
def decomposed():
    adata = scanpy.datasets.pbmc3k()[:150].copy()
    adata.X = adata.X.toarray()
    sc.pp.pca(adata, n_comps=20)
    rng = np.random.default_rng(0)
    adata.obs["batch"] = rng.choice(["A", "B"], size=adata.n_obs)
    ref = scd.scDEF(adata, batch_key="batch", layer_sizes=[8, 4, 2, 1], seed=0)
    ref.fit(n_epoch=5, hierarchical_init=True)
    return scd.scDEF.decompose_batch_effects(ref, top_layer=2, n_epoch=5)


def test_decomposition_records_its_anchor(decomposed):
    assert decomposed.decomposition_top_layer == 2


def test_anchor_layer_uses_the_frozen_z(decomposed):
    got = decomposed.adata.obs["L2"].astype(str).to_numpy()
    want = np.array(decomposed.factor_names[2])[hard_assignment_name_indices(decomposed, 2)]
    np.testing.assert_array_equal(got, want)


def test_layers_below_the_anchor_follow_the_tree(decomposed):
    parent = np.argmax(_kept_layer_w(decomposed, 1), axis=0)
    l0 = np.array([decomposed.factor_names[0].index(f) for f in decomposed.adata.obs["L0"].astype(str)])
    l1 = np.array([decomposed.factor_names[1].index(f) for f in decomposed.adata.obs["L1"].astype(str)])
    np.testing.assert_array_equal(l1, parent[l0])
