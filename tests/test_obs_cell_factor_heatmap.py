import numpy as np
import pandas as pd
import pytest

from anndata import AnnData

from scdef.plotting.factors import (
    _hierarchical_row_order,
    obs_cell_factor_heatmap,
)


def _make_plot_model():
    class _Model:
        pass

    n_cells = 12
    model = _Model()
    model.n_layers = 2
    model.layer_names = ["L0", "L1"]
    model.factor_names = [["f0", "f1", "f2"], ["u0"]]
    model.factor_lists = [np.array([0, 1, 2], dtype=int), np.array([0], dtype=int)]
    model.adata = AnnData(np.zeros((n_cells, 3), dtype=np.float32))
    model.adata.obs_names = [f"c{i}" for i in range(n_cells)]
    model.adata.obs["patient"] = ["P1"] * 6 + ["P2"] * 6
    model.adata.obs["treatment"] = ["A", "A", "A", "B", "B", "B"] * 2
    scores = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.8, 0.2, 0.0],
            [0.0, 1.0, 0.0],
            [0.1, 0.9, 0.0],
            [0.2, 0.8, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.1, 0.9],
            [0.1, 0.0, 0.9],
            [0.5, 0.5, 0.0],
            [0.4, 0.6, 0.0],
            [0.6, 0.4, 0.0],
        ],
        dtype=float,
    )
    model.adata.obsm["X_L0"] = scores
    den = np.clip(scores.sum(axis=1, keepdims=True), 1e-12, None)
    model.adata.obsm["X_L0_probs"] = scores / den

    def get_layer_factor_orders():
        return [np.array([0, 1, 2]), np.array([0])]

    model.get_layer_factor_orders = get_layer_factor_orders
    return model


def test_hierarchical_row_order_single_row():
    data = np.array([[1.0, 2.0, 3.0]])
    assert np.array_equal(_hierarchical_row_order(data), np.array([0]))


def _heatmap_array_from_fig(fig):
    for ax in fig.axes:
        if not ax.images:
            continue
        arr = ax.images[0].get_array()
        if getattr(arr, "ndim", 0) == 2 and arr.shape[1] > 1:
            return arr
    raise AssertionError("No heatmap axis found in figure.")


def test_obs_cell_factor_heatmap_builds_matrix():
    model = _make_plot_model()
    fig = obs_cell_factor_heatmap(
        model,
        subset_obs_key="patient",
        subset_obs="P1",
        group_obs_key="treatment",
        layer_idx=0,
        values="score",
        cluster_cells=True,
        show=False,
    )
    assert fig is not None
    assert _heatmap_array_from_fig(fig).shape == (6, 3)


def test_obs_cell_factor_heatmap_multiple_subsets():
    model = _make_plot_model()
    fig = obs_cell_factor_heatmap(
        model,
        subset_obs_key="patient",
        subset_obs=["P1", "P2"],
        group_obs_key="treatment",
        figsize=(8, 6),
        show=False,
    )
    heatmaps = []
    for ax in fig.axes:
        if ax.images and ax.images[0].get_array().ndim == 2:
            arr = ax.images[0].get_array()
            if arr.shape[1] > 1:
                heatmaps.append(arr)
    assert len(heatmaps) == 2
    assert heatmaps[0].shape == (6, 3)
    assert heatmaps[1].shape == (6, 3)


def test_obs_cell_factor_heatmap_group_track_sharey_with_heatmap():
    model = _make_plot_model()
    fig = obs_cell_factor_heatmap(
        model,
        subset_obs_key="patient",
        subset_obs="P1",
        group_obs_key="treatment",
        show_group_track=True,
        show=False,
    )
    image_axes = [ax for ax in fig.axes if ax.images]
    assert len(image_axes) == 2
    assert image_axes[0].get_ylim() == image_axes[1].get_ylim()


def test_obs_cell_factor_heatmap_group_track_left_of_heatmap():
    model = _make_plot_model()
    fig = obs_cell_factor_heatmap(
        model,
        subset_obs_key="patient",
        subset_obs="P1",
        group_obs_key="treatment",
        show_group_track=True,
        show=False,
    )
    image_axes = [ax for ax in fig.axes if ax.images]
    assert len(image_axes) >= 2
    track_ax = min(image_axes, key=lambda ax: ax.get_position().x0)
    hm_ax = max(image_axes, key=lambda ax: ax.get_position().width)
    assert track_ax.get_position().x0 < hm_ax.get_position().x0


def test_obs_cell_factor_heatmap_show_false_returns_figure():
    import matplotlib.pyplot as plt

    model = _make_plot_model()
    fig = obs_cell_factor_heatmap(
        model,
        subset_obs_key="patient",
        subset_obs="P1",
        group_obs_key="treatment",
        show=False,
    )
    assert fig is not None
    assert isinstance(fig, plt.Figure)


def test_obs_cell_factor_heatmap_show_true_returns_none():
    model = _make_plot_model()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.close("all")
    result = obs_cell_factor_heatmap(
        model,
        subset_obs_key="patient",
        subset_obs="P1",
        group_obs_key="treatment",
        show=True,
    )
    assert result is None


def test_obs_cell_factor_heatmap_show_annotations():
    model = _make_plot_model()
    model.adata.uns["factor_obs"] = pd.DataFrame(
        index=["f0", "f1", "f2"],
        data={"annotation": ["ann0", "ann1", "ann2"]},
    )
    fig = obs_cell_factor_heatmap(
        model,
        subset_obs_key="patient",
        subset_obs="P1",
        group_obs_key="treatment",
        factors=["f0", "f1"],
        show_annotations=True,
        show=False,
    )
    axes = [ax for ax in fig.axes if hasattr(ax, "get_xlabel")]
    assert any(len(ax.xaxis.get_major_ticks()) > 0 for ax in axes)


def test_obs_cell_factor_heatmap_subset_factors():
    model = _make_plot_model()
    fig = obs_cell_factor_heatmap(
        model,
        subset_obs_key="patient",
        subset_obs="P1",
        group_obs_key="treatment",
        factors=["f2", "f0"],
        sort_layer_factors=True,
        show=False,
    )
    arr = _heatmap_array_from_fig(fig)
    assert arr.shape == (6, 2)


def test_obs_cell_factor_heatmap_unknown_factor():
    model = _make_plot_model()
    with pytest.raises(ValueError, match="not found"):
        obs_cell_factor_heatmap(
            model,
            subset_obs_key="patient",
            subset_obs="P1",
            group_obs_key="treatment",
            factors=["missing"],
            show=False,
        )


def test_obs_cell_factor_heatmap_prob_mode():
    model = _make_plot_model()
    fig = obs_cell_factor_heatmap(
        model,
        subset_obs_key="patient",
        subset_obs="P1",
        group_obs_key="treatment",
        values="prob",
        cluster_cells=False,
        show=False,
    )
    assert fig is not None


def test_obs_cell_factor_heatmap_missing_subset():
    model = _make_plot_model()
    with pytest.raises(ValueError, match="No cells found"):
        obs_cell_factor_heatmap(
            model,
            subset_obs_key="patient",
            subset_obs="missing",
            group_obs_key="treatment",
            show=False,
        )


def test_obs_cell_factor_heatmap_missing_obsm():
    model = _make_plot_model()
    del model.adata.obsm["X_L0"]
    with pytest.raises(KeyError, match="X_L0"):
        obs_cell_factor_heatmap(
            model,
            subset_obs_key="patient",
            subset_obs="P1",
            group_obs_key="treatment",
            show=False,
        )
