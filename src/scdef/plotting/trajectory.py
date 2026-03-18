"""Trajectory/PAGA plotting utilities for scDEF."""

from typing import Optional, List, Tuple, Any, TYPE_CHECKING
import copy
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc

from ..tools.trajectory import multilevel_paga as compute_multilevel_paga

if TYPE_CHECKING:
    from scdef.models._scdef import scDEF


def multilevel_paga(
    model: "scDEF",
    neighbors_rep: Optional[str] = "X_L0",
    layers: Optional[List[int]] = None,
    figsize: Optional[Tuple[float, float]] = (16, 4),
    reuse_pos: Optional[bool] = True,
    recompute: bool = False,
    fontsize: Optional[int] = 12,
    show: Optional[bool] = True,
    **paga_kwargs: Any,
) -> None:
    """Plot cached multilevel PAGA graphs across scDEF layers."""
    if layers is None:
        layers = [
            i
            for i in range(model.n_layers - 1, -1, -1)
            if len(model.factor_lists[i]) > 1
        ]

    if len(layers) == 0:
        print("Cannot run PAGA on 0 layers.")
        return

    cache = model.adata.uns.get("multilevel_paga", None)
    cache_layers = [] if cache is None else cache.get("layers", [])
    need_recompute = (
        recompute
        or cache is None
        or cache.get("neighbors_rep") != neighbors_rep
        or cache.get("reuse_pos") != bool(reuse_pos)
        or any(int(layer) not in cache_layers for layer in layers)
    )
    if need_recompute:
        compute_multilevel_paga(
            model,
            neighbors_rep=neighbors_rep,
            layers=layers,
            reuse_pos=reuse_pos,
            **paga_kwargs,
        )
        cache = model.adata.uns.get("multilevel_paga", None)

    if cache is None or "results" not in cache:
        raise RuntimeError(
            "Multilevel PAGA cache not found. Run scdef.tl.multilevel_paga(model) first."
        )

    n_layers = len(layers)
    fig, axes = plt.subplots(1, n_layers, figsize=figsize)
    if n_layers == 1:
        axes = [axes]
    old_paga = copy.deepcopy(model.adata.uns.get("paga", None))
    layout = cache.get("layout", "fa")
    for i, layer_idx in enumerate(layers):
        ax = axes[i]
        layer_name = model.layer_names[layer_idx]
        if layer_name not in cache["results"]:
            raise KeyError(
                f"Cached PAGA for layer '{layer_name}' not found. Recompute with scdef.tl.multilevel_paga."
            )
        entry = cache["results"][layer_name]
        model.adata.uns["paga"] = copy.deepcopy(entry["paga"])
        sc.pl.paga(
            model.adata,
            init_pos=np.array(entry["pos"]),
            layout=layout,
            ax=ax,
            show=False,
            **paga_kwargs,
        )
        ax.set_title(f"Layer {layer_idx} PAGA", fontsize=fontsize)
    if old_paga is None:
        model.adata.uns.pop("paga", None)
    else:
        model.adata.uns["paga"] = old_paga
    if show:
        plt.show()
