"""Transition-path scoring follows the drawn tree.

A transition path rises from one layer-0 factor to a shared ancestor and descends to
another. Its affinity is the share of a cell's layer-0 weight below that ancestor
(strongest-parent chains, as ``make_graph`` draws them), and its position uses only the
two layer-0 endpoints. Before 2026-11-12 the affinity divided by every layer's mass, so a
3-node path could not exceed 1/3, and the ancestor pulled every position to the middle.

Hand-built hierarchy: L0_0, L0_1 and L0_2 under L1_0; L0_3 under L1_1; both under L2_0.
"""
from types import SimpleNamespace

import numpy as np
import pytest
from anndata import AnnData

import scdef as scd


def _model():
    l0 = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],  # source
            [0.0, 0.0, 1.0, 0.0],  # target
            [0.5, 0.0, 0.5, 0.0],  # halfway
            [0.1, 0.8, 0.1, 0.0],  # intermediate state below the same ancestor
            [0.0, 0.0, 0.0, 1.0],  # other branch
        ]
    )
    l1 = np.array([[0.9, 0.1], [0.9, 0.1], [0.9, 0.1], [0.9, 0.1], [0.1, 0.9]])
    l2 = np.ones((5, 1))
    adata = AnnData(np.zeros((5, 3)))
    adata.obsm["X_L0"], adata.obsm["X_L1"], adata.obsm["X_L2"] = l0, l1, l2
    # (L1, L0). L0_2 also links strongly to L1_1 (2.9, below its 3.0 to L1_0), so the route
    # L0_0-L1_0-L2_0-L1_1-L0_2 has a weight product of 78 against 9 for the direct one.
    w1 = np.array([[3.0, 3.0, 3.0, 0.1], [0.1, 0.1, 2.9, 3.0]])
    w2 = np.array([[3.0, 3.0]])
    return SimpleNamespace(
        n_layers=3,
        layer_names=["L0", "L1", "L2"],
        factor_names=[["L0_0", "L0_1", "L0_2", "L0_3"], ["L1_0", "L1_1"], ["L2_0"]],
        factor_lists=[np.arange(4), np.arange(2), np.arange(1)],
        pmeans={"L1W": w1, "L2W": w2},
        adata=adata,
    )


def test_shortest_path_comes_first():
    m = _model()
    paths = scd.tl.build_transition_paths(
        m, sources=["L0_0"], targets=["L0_2"], max_path_len=7, rel_parent_weight=0.0
    )
    assert [
        "L0_0",
        "L1_0",
        "L2_0",
        "L1_1",
        "L0_2",
    ] in paths  # the higher-product route exists
    assert paths[0] == ["L0_0", "L1_0", "L0_2"]  # but the direct one comes first


def test_affinity_and_position_follow_the_tree():
    m = _model()
    scd.tl.build_transition_paths(m, sources=["L0_0"], targets=["L0_2"], max_path_len=3)
    scd.tl.score_paths(m, paths_key="transition_paths")
    aff = m.adata.obsm["transition_paths_affinities"][:, 0]
    pos = m.adata.obsm["transition_paths_positions"][:, 0]
    # every cell below L1_0 belongs to the path, including the intermediate state
    np.testing.assert_allclose(aff[:4], 1.0)
    assert aff[4] == pytest.approx(0.0)
    np.testing.assert_allclose(pos[:3], [0.0, 1.0, 0.5])
    assert pos[3] == pytest.approx(0.5)  # placed by its remaining source/target weight
    assert np.all((aff >= 0) & (aff <= 1))


def test_differentiation_scoring_is_unchanged():
    m = _model()
    nodes = ["L2_0", "L1_0", "L0_0"]
    m.adata.uns["dp"] = {
        "paths": [{"path_id": 0, "nodes": nodes, "type": "differentiation"}]
    }
    scd.tl.score_paths(m, paths_key="dp")
    mats = [
        m.adata.obsm[k] / m.adata.obsm[k].sum(1, keepdims=True)
        for k in ("X_L0", "X_L1", "X_L2")
    ]
    s = np.vstack([mats[2][:, 0], mats[1][:, 0], mats[0][:, 0]]).T
    expected_aff = s.sum(1) / sum(x.sum(1) for x in mats)
    expected_pos = (s / s.sum(1, keepdims=True)) @ np.linspace(0, 1, 3)
    np.testing.assert_allclose(m.adata.obsm["dp_affinities"][:, 0], expected_aff)
    np.testing.assert_allclose(m.adata.obsm["dp_positions"][:, 0], expected_pos)
