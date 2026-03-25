"""Trajectory and PAGA tooling utilities for scDEF."""

from typing import Optional, List, TYPE_CHECKING
import copy
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc

if TYPE_CHECKING:
    from scdef.models._scdef import scDEF


def multilevel_paga(
    model: "scDEF",
    neighbors_rep: str = "X_L0",
    layers: Optional[List[int]] = None,
    reuse_pos: bool = True,
    layout: str = "fa",
    random_seed: int = 0,
    **paga_kwargs,
) -> None:
    """Compute and cache multilevel PAGA results for plotting."""
    if layers is None:
        layers = [
            i
            for i in range(model.n_layers - 1, -1, -1)
            if len(model.factor_lists[i]) > 1
        ]
    if len(layers) == 0:
        model.adata.uns["multilevel_paga"] = {
            "neighbors_rep": neighbors_rep,
            "layers": [],
            "reuse_pos": reuse_pos,
            "layout": layout,
            "results": {},
        }
        return

    sc.pp.neighbors(model.adata, use_rep=neighbors_rep)
    results = {}
    pos = None
    old_layer_name = None
    old_paga = copy.deepcopy(model.adata.uns.get("paga", None))

    for layer_idx in layers:
        layer_name = model.layer_names[layer_idx]

        if old_layer_name is not None and reuse_pos:
            matches = sc._utils.identify_groups(
                model.adata.obs[layer_name], model.adata.obs[old_layer_name]
            )
            pos = []
            np.random.seed(random_seed)
            prev_pos = model.adata.uns["paga"]["pos"]
            coarse_categories = model.adata.obs[old_layer_name].cat.categories
            for c in model.adata.obs[layer_name].cat.categories:
                idx = coarse_categories.get_loc(matches[c][0])
                pos_i = prev_pos[idx] + np.random.random(2)
                pos.append(pos_i)
            pos = np.array(pos)

        sc.tl.paga(model.adata, groups=layer_name)
        sc.pl.paga(
            model.adata,
            init_pos=pos,
            layout=layout,
            show=False,
            **paga_kwargs,
        )
        plt.close()
        results[layer_name] = {
            "paga": copy.deepcopy(model.adata.uns["paga"]),
            "pos": np.array(model.adata.uns["paga"]["pos"]),
            "layer_idx": int(layer_idx),
        }
        old_layer_name = layer_name

    if old_paga is None:
        model.adata.uns.pop("paga", None)
    else:
        model.adata.uns["paga"] = old_paga

    model.adata.uns["multilevel_paga"] = {
        "neighbors_rep": neighbors_rep,
        "layers": [int(i) for i in layers],
        "reuse_pos": bool(reuse_pos),
        "layout": layout,
        "results": results,
    }
