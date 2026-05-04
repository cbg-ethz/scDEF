"""Trajectory/PAGA plotting utilities for scDEF."""

from typing import Optional, List, Tuple, Any, TYPE_CHECKING, Sequence, Union, Dict
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import scanpy as sc
from scipy.ndimage import uniform_filter1d
from sklearn.preprocessing import minmax_scale

from ..tools.trajectory import multilevel_paga as compute_multilevel_paga


def _gene_row_values(expr_vals: np.ndarray, *, normalize: bool) -> np.ndarray:
    """Per-cell gene expression row for heatmap: optional min–max to ``[0, 1]``."""
    v = np.asarray(expr_vals, dtype=float).ravel()
    if normalize:
        return np.asarray(minmax_scale(v), dtype=float)
    return v
from ..tools.factor import get_stored_confident_signatures


def _factor_layer_and_slot(model: "scDEF", factor_name: str) -> Tuple[int, int]:
    """Return (layer_idx, slot_idx) for a filtered factor name."""
    for li in range(model.n_layers):
        names = list(model.factor_names[li])
        if factor_name in names:
            return int(li), int(names.index(factor_name))
    raise ValueError(
        f"Factor '{factor_name}' not found in any layer's factor_names. "
        "Paths must use current filtered factor names."
    )


def _ma_has_any_unmasked(a: np.ma.MaskedArray) -> bool:
    """True if at least one entry is not masked out."""
    m = a.mask
    if m is np.ma.nomask:
        return True
    return not bool(np.all(m))


def _path_index_in_uns_paths(paths: List[Dict[str, Any]], path_id: int) -> int:
    """Map user ``path_id`` to the column index in ``score_paths`` matrices."""
    pid = int(path_id)
    for i, p in enumerate(paths):
        if int(p.get("path_id", i)) == pid:
            return i
    if 0 <= pid < len(paths):
        return pid
    raise IndexError(
        f"path_id {path_id} not found among {len(paths)} paths "
        f"(checked path_id fields and list indices)."
    )


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


def plot_trajectory_heatmap(
    model: "scDEF",
    factor_path: Sequence[Union[int, str]],
    layer_idx: int = 0,
    l0_path: Optional[Sequence[Union[int, str]]] = None,
    genes_per_factor: Optional[int] = 3,
    smoothing: int = 50,
    figwidth: float = 8,
    gene_height: float = 0.28,
    block_spacing: int = 1,
    annotation_obs_key: Optional[Union[str, Sequence[str]]] = None,
    subset_obs_key: Optional[str] = None,
    subset_obs: Optional[Union[str, Sequence[str]]] = None,
    heatmap_cmap: str = "RdYlBu_r",
    factor_heatmap_cmap: str = "viridis",
    colorbar_gap: float = 0.16,
    xlabel: str = "Cells",
    normalize: bool = True,
    save: Optional[str] = None,
    show: bool = True,
):
    """Plot stacked trajectory heatmap from precomputed confident signatures.

    Requires ``scd.tl.set_confident_signatures(model)`` to be run beforehand.

    Cell sorting:
        Cells are restricted to those assigned to ``factor_path`` (and optional
        ``subset_obs`` filter), then ordered by a soft trajectory progress score.
        For each selected cell, the function takes its factor probabilities
        (from ``X_<layer>_probs``), keeps only the columns for the path factors,
        and computes a probability-weighted average of path positions
        ``0..len(path)-1``. This gives a continuous progress coordinate used to
        sort cells from early to late along the path.

        If ``layer_idx > 0`` and ``l0_path`` is provided, the sorting weights are
        computed on layer 0 probabilities instead. If that yields near-zero total
        path weight for all selected cells, sorting falls back to the
        ``layer_idx`` path probabilities.

    ``normalize`` (default True): min–max scale each **gene** row to ``[0, 1]``
    after smoothing; if False, use smoothed raw expression. Factor score rows
    are always min–max scaled.
    """
    layer_name = model.layer_names[layer_idx]
    kept_names = list(model.factor_names[layer_idx])

    def _resolve_path_names(
        path: Sequence[Union[int, str]], layer_names: List[str], layer_idx_resolve: int
    ) -> List[str]:
        resolved: List[str] = []
        for f in path:
            if isinstance(f, str):
                if f not in layer_names:
                    raise ValueError(
                        f"Factor '{f}' not found in layer {layer_idx_resolve}."
                    )
                resolved.append(f)
            else:
                f_idx = int(f)
                if f_idx < 0 or f_idx >= len(layer_names):
                    raise IndexError(
                        f"Factor index {f_idx} out of bounds for layer {layer_idx_resolve}."
                    )
                resolved.append(layer_names[f_idx])
        return resolved

    path_names = _resolve_path_names(factor_path, kept_names, layer_idx)
    if len(path_names) == 0:
        raise ValueError("factor_path must contain at least one factor.")
    if block_spacing < 0:
        raise ValueError("block_spacing must be >= 0.")
    if colorbar_gap < 0.0:
        raise ValueError("colorbar_gap must be >= 0.")

    subset_mask = np.ones(model.adata.n_obs, dtype=bool)
    if subset_obs is not None and subset_obs_key is None:
        raise ValueError("subset_obs_key must be provided when subset_obs is set.")
    if subset_obs_key is not None:
        if subset_obs_key not in model.adata.obs.columns:
            raise KeyError(f"{subset_obs_key} not found in adata.obs.")
        if subset_obs is None:
            raise ValueError("subset_obs must be provided when subset_obs_key is set.")
        if isinstance(subset_obs, str):
            subset_vals = [subset_obs]
        else:
            subset_vals = [str(v) for v in subset_obs]
        if len(subset_vals) == 0:
            raise ValueError("subset_obs must contain at least one value.")
        subset_mask = (
            model.adata.obs[subset_obs_key].astype(str).isin(subset_vals).values
        )

    path_mask = model.adata.obs[layer_name].isin(path_names).values & subset_mask
    if np.count_nonzero(path_mask) == 0:
        raise ValueError(
            "No cells found for the provided factor_path after applying subset filters."
        )

    sort_layer_name = layer_name
    sort_names = path_names
    sort_kept_names = kept_names
    use_l0_sort = layer_idx > 0 and l0_path is not None
    if use_l0_sort:
        l0_kept_names = list(model.factor_names[0])
        sort_layer_name = model.layer_names[0]
        sort_names = _resolve_path_names(l0_path, l0_kept_names, 0)
        if len(sort_names) == 0:
            raise ValueError("l0_path must contain at least one factor.")
        sort_kept_names = l0_kept_names

    X_probs = np.asarray(model.adata.obsm[f"X_{sort_layer_name}_probs"], dtype=float)
    factor_pos = {name: idx for idx, name in enumerate(sort_kept_names)}
    path_cols = np.array([factor_pos[name] for name in sort_names], dtype=int)
    path_weights = X_probs[path_mask][:, path_cols]
    if np.all(np.sum(path_weights, axis=1) <= 1e-12):
        if use_l0_sort and hasattr(model, "logger"):
            model.logger.warning(
                "l0_path provided but selected cells have near-zero weight on this "
                "path. Falling back to layer %s factor_path ordering.",
                layer_idx,
            )
        X_probs_fb = np.asarray(model.adata.obsm[f"X_{layer_name}_probs"], dtype=float)
        factor_pos_fb = {name: idx for idx, name in enumerate(kept_names)}
        path_cols_fb = np.array([factor_pos_fb[name] for name in path_names], dtype=int)
        path_weights = X_probs_fb[path_mask][:, path_cols_fb]
    denom = np.maximum(np.sum(path_weights, axis=1), 1e-12)
    ranks = np.arange(path_weights.shape[1], dtype=float)
    progress = np.sum(path_weights * ranks[None, :], axis=1) / denom
    selected_cells = np.where(path_mask)[0]
    sorted_cells = selected_cells[np.argsort(progress)]
    t_path = model.adata[sorted_cells].copy()

    confident_sigs = get_stored_confident_signatures(
        model, layer_idx=layer_idx, max_genes=genes_per_factor
    )

    score_cols = [f"{name}_score" for name in path_names]
    missing_scores = [col for col in score_cols if col not in t_path.obs.columns]
    if len(missing_scores) > 0:
        raise KeyError(f"Score columns not found in adata.obs: {missing_scores}")

    block_matrices: List[Tuple[np.ma.MaskedArray, np.ma.MaskedArray]] = []
    block_labels: List[List[str]] = []
    block_has_genes: List[bool] = []
    for factor_name in path_names:
        block_rows: List[np.ndarray] = []
        labels: List[str] = []
        kinds: List[str] = []
        score_col = f"{factor_name}_score"
        score_vals = uniform_filter1d(
            np.asarray(t_path.obs[score_col].values, dtype=float), size=smoothing
        )
        block_rows.append(minmax_scale(score_vals))
        labels.append(factor_name)
        kinds.append("factor")

        genes = [
            g for g in confident_sigs.get(factor_name, []) if g in t_path.var_names
        ]
        for gene in genes:
            expr = t_path[:, [gene]].X
            if hasattr(expr, "toarray"):
                expr = expr.toarray()
            expr_vals = uniform_filter1d(
                np.asarray(expr, dtype=float).ravel(), size=smoothing
            )
            block_rows.append(_gene_row_values(expr_vals, normalize=normalize))
            labels.append(f"  {gene}")
            kinds.append("gene")

        block_arr = np.vstack(block_rows)
        factor_rows = np.asarray([k == "factor" for k in kinds], dtype=bool)
        gene_rows = np.asarray([k == "gene" for k in kinds], dtype=bool)

        genes_masked = np.ma.array(
            block_arr, mask=np.broadcast_to(~gene_rows[:, None], block_arr.shape)
        )
        factors_masked = np.ma.array(
            block_arr, mask=np.broadcast_to(~factor_rows[:, None], block_arr.shape)
        )

        block_matrices.append((genes_masked, factors_masked))
        block_labels.append(labels)
        block_has_genes.append(bool(np.any(gene_rows)))

    if len(block_matrices) == 0:
        raise ValueError("No rows to plot. Check factor_path and confident signatures.")

    obs_keys: List[str] = []
    if annotation_obs_key is not None:
        if isinstance(annotation_obs_key, str):
            obs_keys = [annotation_obs_key]
        else:
            obs_keys = [str(k) for k in annotation_obs_key]
        if len(obs_keys) == 0:
            raise ValueError(
                "annotation_obs_key must be a non-empty string or sequence of strings."
            )

    obs_tracks: List[Dict[str, object]] = []
    for key in obs_keys:
        if key not in t_path.obs.columns:
            raise KeyError(f"{key} not found in adata.obs.")
        cat = pd.Categorical(t_path.obs[key])
        categories = cat.categories.tolist()
        uns_key = f"{key}_colors"
        if uns_key in t_path.uns:
            cat_to_color = dict(
                zip(
                    t_path.uns.get(f"{key}_categories", categories),
                    t_path.uns[uns_key],
                )
            )
        else:
            cmap_fb = plt.get_cmap("tab10", max(len(categories), 1))
            cat_to_color = {c: cmap_fb(i) for i, c in enumerate(categories)}
        rgb = np.array([mpl.colors.to_rgb(cat_to_color[c]) for c in t_path.obs[key]])
        obs_tracks.append(
            {
                "key": key,
                "categories": categories,
                "cat_to_color": cat_to_color,
                "rgb": rgb,
            }
        )

    nrows = len(obs_tracks) + len(block_matrices)
    height_ratios = [0.3] * len(obs_tracks)
    for labels in block_labels:
        height_ratios.append(len(labels) * gene_height)

    total_height = sum(height_ratios) + 1.5
    fig = plt.figure(figsize=(figwidth, total_height))
    gs = fig.add_gridspec(
        nrows=nrows,
        ncols=4,
        height_ratios=height_ratios,
        width_ratios=[figwidth - 1.6, 0.22, 0.24, 0.22],
        hspace=0.05 + 0.08 * block_spacing,
        wspace=colorbar_gap,
    )

    row = 0
    for obs in obs_tracks:
        ax_obs = fig.add_subplot(gs[row, 0])
        ax_obs_cb1 = fig.add_subplot(gs[row, 1])
        ax_obs_cb_gap = fig.add_subplot(gs[row, 2])
        ax_obs_cb2 = fig.add_subplot(gs[row, 3])
        ax_obs_cb1.axis("off")
        ax_obs_cb_gap.axis("off")
        ax_obs_cb2.axis("off")
        ax_obs.imshow(
            obs["rgb"][np.newaxis, :, :], aspect="auto", interpolation="nearest"
        )
        ax_obs.set_yticks([0])
        ax_obs.set_yticklabels([obs["key"]], fontsize=9)
        ax_obs.set_xticks([])
        legend_patches = [
            mpl.patches.Patch(color=obs["cat_to_color"][c], label=c)
            for c in obs["categories"]
        ]
        ax_obs.legend(
            handles=legend_patches,
            fontsize=8,
            frameon=False,
            loc="lower left",
            bbox_to_anchor=(0.0, 1.05),
            ncol=max(1, min(len(obs["categories"]), 8)),
            borderaxespad=0,
        )
        row += 1

    hm_row_start = row
    im_genes_ref = None
    im_factors_ref = None
    any_gene_rows = False
    for i, labels in enumerate(block_labels):
        ax_hm = fig.add_subplot(gs[hm_row_start + i, 0])
        genes_masked, factors_masked = block_matrices[i]

        im_genes = ax_hm.imshow(
            genes_masked, aspect="auto", cmap=heatmap_cmap, interpolation="nearest"
        )
        im_factors = ax_hm.imshow(
            factors_masked,
            aspect="auto",
            cmap=factor_heatmap_cmap,
            interpolation="nearest",
        )
        ax_hm.set_yticks(range(len(labels)))
        ax_hm.set_yticklabels(labels, fontsize=8)
        if i == len(block_labels) - 1:
            ax_hm.set_xlabel(xlabel, fontsize=10)
        ax_hm.set_xticks([])

        im_factors_ref = im_factors
        if block_has_genes[i]:
            im_genes_ref = im_genes
            any_gene_rows = True

        if i < len(block_labels) - 1:
            ax_hm.axhline(len(labels) - 0.5, color="white", linewidth=1.0, alpha=0.8)

    ax_hm_cb_gene = fig.add_subplot(gs[hm_row_start:, 1])
    ax_hm_cb_factor = fig.add_subplot(gs[hm_row_start:, 3])
    gene_cb_label = (
        "Smoothed gene expression in path (min–max)"
        if normalize
        else "Smoothed gene expression in path (raw)"
    )
    if any_gene_rows and im_genes_ref is not None:
        plt.colorbar(im_genes_ref, cax=ax_hm_cb_gene, label=gene_cb_label)
        ax_hm_cb_gene.yaxis.labelpad = 10
        ax_hm_cb_gene.yaxis.label.set_size(8)
        ax_hm_cb_gene.tick_params(labelsize=7)
    else:
        ax_hm_cb_gene.axis("off")
    if im_factors_ref is not None:
        plt.colorbar(
            im_factors_ref, cax=ax_hm_cb_factor, label="Smoothed factor score in path"
        )
        ax_hm_cb_factor.yaxis.labelpad = 10
        ax_hm_cb_factor.yaxis.label.set_size(8)
        ax_hm_cb_factor.tick_params(labelsize=7)
    else:
        ax_hm_cb_factor.axis("off")

    if save:
        plt.savefig(save, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
        return None
    return fig


def _sorted_cells_for_multilayer_path(
    model: "scDEF",
    nodes: List[str],
    pobj: Dict[str, Any],
    paths_key: str,
    p_list_idx: int,
    score_key: str,
    min_sort_affinity: float,
    subset_mask: np.ndarray,
    path_mass_eps: float,
) -> np.ndarray:
    """Order cells along ``nodes`` using score_paths output if available."""
    adata = model.adata
    pos_key = f"{score_key}_positions"
    aff_key = f"{score_key}_affinities"

    if (
        pos_key in adata.obsm
        and aff_key in adata.obsm
        and int(np.asarray(adata.obsm[pos_key]).shape[1]) > p_list_idx
    ):
        pos_col = np.asarray(adata.obsm[pos_key][:, p_list_idx], dtype=float)
        aff_col = np.asarray(adata.obsm[aff_key][:, p_list_idx], dtype=float)
        scored = (
            subset_mask & np.isfinite(pos_col) & (aff_col >= float(min_sort_affinity))
        )
        if np.count_nonzero(scored) > 0:
            sel = np.where(scored)[0]
            return sel[np.argsort(pos_col[sel])]

    l0_name = model.layer_names[0]
    obs_l0 = adata.obs[l0_name].astype(str).to_numpy()
    if str(pobj.get("type", "")) == "transition" and pobj.get("source") is not None:
        src = str(pobj["source"])
        tgt = str(pobj.get("target", ""))
        anchor = subset_mask & np.isin(obs_l0, [src, tgt])
    else:
        if len(nodes) == 0:
            raise ValueError("Path has no nodes.")
        anchor = subset_mask & (obs_l0 == str(nodes[-1]))

    prob_cols: List[np.ndarray] = []
    for name in nodes:
        li, si = _factor_layer_and_slot(model, name)
        lnm = model.layer_names[li]
        pk = f"X_{lnm}_probs"
        if pk not in adata.obsm:
            raise KeyError(
                f"Missing '{pk}' in adata.obsm. Run `model.annotate_adata()` first."
            )
        Xp = np.asarray(adata.obsm[pk], dtype=float)
        prob_cols.append(Xp[:, si])
    s_arr = np.stack(prob_cols, axis=1)
    mass = np.sum(s_arr, axis=1)
    ranks = np.arange(s_arr.shape[1], dtype=float)
    denom_mass = np.maximum(mass, float(path_mass_eps))
    w = s_arr / denom_mass[:, None]
    prog = np.sum(w * ranks[None, :], axis=1) / max(float(s_arr.shape[1] - 1), 1.0)
    valid = anchor & (mass > float(path_mass_eps))
    if np.count_nonzero(valid) == 0:
        raise ValueError(
            "No cells after subset/anchor filters. "
            "Try lowering min_sort_affinity, run score_paths with matching "
            "paths_key and key_added, or relax subset_obs filters."
        )
    sel = np.where(valid)[0]
    return sel[np.argsort(prog[sel])]


def plot_path_trajectory_heatmap(
    model: "scDEF",
    path_id: int,
    paths_key: str = "differentiation_paths",
    score_key: Optional[str] = None,
    min_sort_affinity: float = 0.05,
    path_mass_eps: float = 1e-12,
    genes_per_factor: Optional[int] = 3,
    smoothing: int = 50,
    figwidth: float = 8,
    gene_height: float = 0.28,
    block_spacing: int = 1,
    ytick_prefix_layer: bool = True,
    genes: Optional[Sequence[str]] = None,
    annotation_obs_key: Optional[Union[str, Sequence[str]]] = None,
    subset_obs_key: Optional[str] = None,
    subset_obs: Optional[Union[str, Sequence[str]]] = None,
    heatmap_cmap: str = "RdYlBu_r",
    factor_heatmap_cmap: str = "viridis",
    colorbar_gap: float = 0.16,
    xlabel: str = "Cells",
    normalize: bool = True,
    save: Optional[str] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """Trajectory heatmap along a stored multi-layer path (differentiation or transition).

    Uses ``adata.uns[paths_key]['paths'][path_id]`` factor ``nodes`` (root→leaf
    for differentiation paths) only for **cell ordering** along the path.

    Default mode: per-node factor scores from ``adata.obs['<factor>_score']``
    (``annotate_adata`` / fit) plus confident genes from
    ``scd.tl.set_confident_signatures(model)``.

    **Custom genes** (``genes=[...]``): one heatmap block, one row per gene; no
    factor score rows and no confident-signature cache required.

    Cell order:
        Prefer ``scd.tl.score_paths`` matrices ``{score_key}_positions`` and
        ``{score_key}_affinities`` (same column order as ``paths``). Cells must
        meet ``min_sort_affinity`` and have finite positions. If no such cells
        remain, falls back to the same logic as :func:`plot_trajectory_heatmap`:
        anchor cells on the L0 terminus (differentiation) or transition
        source/target on L0, then sort by probability-weighted index along
        ``nodes`` using ``X_<layer>_probs``.

    If ``genes`` is set to a non-empty list of gene names, skips per-factor blocks
    (no ``<factor>_score`` rows and no confident signatures). Renders a single
    heatmap block with one row per gene (same path-based cell order). ``genes_per_factor``
    is ignored in that mode.

    Args:
        model: fitted scDEF model.
        path_id: ``path_id`` field stored with the path, or a list index.
        paths_key: ``adata.uns`` key from ``build_differentiation_paths`` or
            ``build_transition_paths``.
        score_key: prefix for ``obsm`` position/affinity matrices from
            ``score_paths``; defaults to ``paths_key``.
        min_sort_affinity: when using ``score_paths`` output, minimum affinity
            along the path for a cell to be included and ordered.
        path_mass_eps: minimum summed path-node probability mass in fallback mode.
        ytick_prefix_layer: if True, factor row labels are ``'<layer> <name>'``.
        genes: optional list of ``adata.var_names`` entries to plot as rows only.
        normalize: min–max scale each gene expression row after smoothing (default
            True); factor score rows are always scaled. Same as
            :func:`plot_trajectory_heatmap`.
    """
    if paths_key not in model.adata.uns:
        raise KeyError(f"'{paths_key}' not found in model.adata.uns.")
    paths_raw = model.adata.uns[paths_key].get("paths", [])
    paths: List[Dict[str, Any]] = list(paths_raw)
    if len(paths) == 0:
        raise ValueError(f"No paths in adata.uns['{paths_key}']['paths'].")

    sk = paths_key if score_key is None else str(score_key)
    p_list_idx = _path_index_in_uns_paths(paths, int(path_id))
    pobj = paths[p_list_idx]
    nodes = list(pobj.get("nodes", []))
    if len(nodes) == 0:
        raise ValueError(f"Path {path_id} has empty 'nodes'.")

    if block_spacing < 0:
        raise ValueError("block_spacing must be >= 0.")
    if colorbar_gap < 0.0:
        raise ValueError("colorbar_gap must be >= 0.")

    subset_mask = np.ones(model.adata.n_obs, dtype=bool)
    if subset_obs is not None and subset_obs_key is None:
        raise ValueError("subset_obs_key must be provided when subset_obs is set.")
    if subset_obs_key is not None:
        if subset_obs_key not in model.adata.obs.columns:
            raise KeyError(f"{subset_obs_key} not found in model.adata.obs.")
        if subset_obs is None:
            raise ValueError("subset_obs must be provided when subset_obs_key is set.")
        if isinstance(subset_obs, str):
            subset_vals = [subset_obs]
        else:
            subset_vals = [str(v) for v in subset_obs]
        if len(subset_vals) == 0:
            raise ValueError("subset_obs must contain at least one value.")
        subset_mask = (
            model.adata.obs[subset_obs_key].astype(str).isin(subset_vals).values
        )

    sorted_cells = _sorted_cells_for_multilayer_path(
        model,
        nodes,
        pobj,
        paths_key,
        p_list_idx,
        sk,
        float(min_sort_affinity),
        subset_mask,
        float(path_mass_eps),
    )
    t_path = model.adata[sorted_cells].copy()

    block_matrices: List[Tuple[np.ma.MaskedArray, np.ma.MaskedArray]] = []
    block_labels: List[List[str]] = []
    block_has_genes: List[bool] = []

    if genes is not None:
        gene_list = [str(g) for g in genes]
        if len(gene_list) == 0:
            raise ValueError("genes must be a non-empty sequence when provided.")
        missing = [g for g in gene_list if g not in t_path.var_names]
        if len(missing) > 0:
            raise KeyError(
                f"Genes not in adata.var_names (subset view): {missing[:25]}"
                + ("..." if len(missing) > 25 else "")
            )
        block_rows: List[np.ndarray] = []
        labels: List[str] = []
        kinds: List[str] = []
        for gene in gene_list:
            expr = t_path[:, [gene]].X
            if hasattr(expr, "toarray"):
                expr = expr.toarray()
            expr_vals = uniform_filter1d(
                np.asarray(expr, dtype=float).ravel(), size=smoothing
            )
            block_rows.append(_gene_row_values(expr_vals, normalize=normalize))
            labels.append(gene)
            kinds.append("gene")
        block_arr = np.vstack(block_rows)
        factor_rows = np.asarray([k == "factor" for k in kinds], dtype=bool)
        gene_rows = np.asarray([k == "gene" for k in kinds], dtype=bool)
        genes_masked = np.ma.array(
            block_arr, mask=np.broadcast_to(~gene_rows[:, None], block_arr.shape)
        )
        factors_masked = np.ma.array(
            block_arr, mask=np.broadcast_to(~factor_rows[:, None], block_arr.shape)
        )
        block_matrices.append((genes_masked, factors_masked))
        block_labels.append(labels)
        block_has_genes.append(True)
    else:
        for factor_name in nodes:
            li, _ = _factor_layer_and_slot(model, factor_name)
            layer_nm = model.layer_names[li]
            display_name = (
                f"{layer_nm} {factor_name}" if ytick_prefix_layer else str(factor_name)
            )
            confident_sigs = get_stored_confident_signatures(
                model, layer_idx=li, max_genes=genes_per_factor
            )
            score_col = f"{factor_name}_score"
            if score_col not in t_path.obs.columns:
                raise KeyError(
                    f"Column '{score_col}' not found in adata.obs. "
                    "Run `model.annotate_adata()` (or fit with annotation) first."
                )

            block_rows_f: List[np.ndarray] = []
            labels_f: List[str] = []
            kinds_f: List[str] = []
            score_vals = uniform_filter1d(
                np.asarray(t_path.obs[score_col].values, dtype=float), size=smoothing
            )
            block_rows_f.append(minmax_scale(score_vals))
            labels_f.append(display_name)
            kinds_f.append("factor")

            sig_genes = [
                g for g in confident_sigs.get(factor_name, []) if g in t_path.var_names
            ]
            for gene in sig_genes:
                expr = t_path[:, [gene]].X
                if hasattr(expr, "toarray"):
                    expr = expr.toarray()
                expr_vals = uniform_filter1d(
                    np.asarray(expr, dtype=float).ravel(), size=smoothing
                )
                block_rows_f.append(_gene_row_values(expr_vals, normalize=normalize))
                labels_f.append(f"  {gene}")
                kinds_f.append("gene")

            block_arr_f = np.vstack(block_rows_f)
            factor_rows_f = np.asarray([k == "factor" for k in kinds_f], dtype=bool)
            gene_rows_f = np.asarray([k == "gene" for k in kinds_f], dtype=bool)

            genes_masked_f = np.ma.array(
                block_arr_f,
                mask=np.broadcast_to(~gene_rows_f[:, None], block_arr_f.shape),
            )
            factors_masked_f = np.ma.array(
                block_arr_f,
                mask=np.broadcast_to(~factor_rows_f[:, None], block_arr_f.shape),
            )

            block_matrices.append((genes_masked_f, factors_masked_f))
            block_labels.append(labels_f)
            block_has_genes.append(bool(np.any(gene_rows_f)))

    if len(block_matrices) == 0:
        raise ValueError("No heatmap blocks to plot.")

    obs_keys: List[str] = []
    if annotation_obs_key is not None:
        if isinstance(annotation_obs_key, str):
            obs_keys = [annotation_obs_key]
        else:
            obs_keys = [str(k) for k in annotation_obs_key]
        if len(obs_keys) == 0:
            raise ValueError(
                "annotation_obs_key must be a non-empty string or sequence of strings."
            )

    obs_tracks: List[Dict[str, object]] = []
    for key in obs_keys:
        if key not in t_path.obs.columns:
            raise KeyError(f"{key} not found in t_path.obs.")
        cat = pd.Categorical(t_path.obs[key])
        categories = cat.categories.tolist()
        uns_key = f"{key}_colors"
        if uns_key in t_path.uns:
            cat_to_color = dict(
                zip(
                    t_path.uns.get(f"{key}_categories", categories),
                    t_path.uns[uns_key],
                )
            )
        else:
            cmap_fb = plt.get_cmap("tab10", max(len(categories), 1))
            cat_to_color = {c: cmap_fb(i) for i, c in enumerate(categories)}
        rgb = np.array([mpl.colors.to_rgb(cat_to_color[c]) for c in t_path.obs[key]])
        obs_tracks.append(
            {
                "key": key,
                "categories": categories,
                "cat_to_color": cat_to_color,
                "rgb": rgb,
            }
        )

    nrows = len(obs_tracks) + len(block_matrices)
    height_ratios = [0.3] * len(obs_tracks)
    for labels in block_labels:
        height_ratios.append(len(labels) * gene_height)

    total_height = sum(height_ratios) + 1.5
    fig = plt.figure(figsize=(figwidth, total_height))
    gs = fig.add_gridspec(
        nrows=nrows,
        ncols=4,
        height_ratios=height_ratios,
        width_ratios=[figwidth - 1.6, 0.22, 0.24, 0.22],
        hspace=0.05 + 0.08 * block_spacing,
        wspace=colorbar_gap,
    )

    row = 0
    for obs in obs_tracks:
        ax_obs = fig.add_subplot(gs[row, 0])
        ax_obs_cb1 = fig.add_subplot(gs[row, 1])
        ax_obs_cb_gap = fig.add_subplot(gs[row, 2])
        ax_obs_cb2 = fig.add_subplot(gs[row, 3])
        ax_obs_cb1.axis("off")
        ax_obs_cb_gap.axis("off")
        ax_obs_cb2.axis("off")
        ax_obs.imshow(
            obs["rgb"][np.newaxis, :, :], aspect="auto", interpolation="nearest"
        )
        ax_obs.set_yticks([0])
        ax_obs.set_yticklabels([obs["key"]], fontsize=9)
        ax_obs.set_xticks([])
        legend_patches = [
            mpl.patches.Patch(color=obs["cat_to_color"][c], label=c)
            for c in obs["categories"]
        ]
        ax_obs.legend(
            handles=legend_patches,
            fontsize=8,
            frameon=False,
            loc="lower left",
            bbox_to_anchor=(0.0, 1.05),
            ncol=max(1, min(len(obs["categories"]), 8)),
            borderaxespad=0,
        )
        row += 1

    hm_row_start = row
    im_genes_ref = None
    im_factors_ref = None
    any_gene_rows = False
    path_title = " -> ".join(nodes)
    for i, labels in enumerate(block_labels):
        ax_hm = fig.add_subplot(gs[hm_row_start + i, 0])
        genes_masked, factors_masked = block_matrices[i]

        im_genes = ax_hm.imshow(
            genes_masked, aspect="auto", cmap=heatmap_cmap, interpolation="nearest"
        )
        im_factors = ax_hm.imshow(
            factors_masked,
            aspect="auto",
            cmap=factor_heatmap_cmap,
            interpolation="nearest",
        )
        ax_hm.set_yticks(range(len(labels)))
        ax_hm.set_yticklabels(labels, fontsize=8)
        if i == len(block_labels) - 1:
            ax_hm.set_xlabel(xlabel, fontsize=10)
        ax_hm.set_xticks([])

        if _ma_has_any_unmasked(factors_masked):
            im_factors_ref = im_factors
        if block_has_genes[i]:
            im_genes_ref = im_genes
            any_gene_rows = True

        if i < len(block_labels) - 1:
            ax_hm.axhline(len(labels) - 0.5, color="white", linewidth=1.0, alpha=0.8)

    title_extra = " (custom genes)" if genes is not None else ""
    fig.suptitle(
        f"{paths_key}[{pobj.get('path_id', p_list_idx)}]  {path_title}{title_extra}",
        fontsize=10,
        y=0.995,
    )

    ax_hm_cb_gene = fig.add_subplot(gs[hm_row_start:, 1])
    ax_hm_cb_factor = fig.add_subplot(gs[hm_row_start:, 3])
    gene_cb_label_ph = (
        "Smoothed gene expression in path (min–max)"
        if normalize
        else "Smoothed gene expression in path (raw)"
    )
    if any_gene_rows and im_genes_ref is not None:
        plt.colorbar(im_genes_ref, cax=ax_hm_cb_gene, label=gene_cb_label_ph)
        ax_hm_cb_gene.yaxis.labelpad = 10
        ax_hm_cb_gene.yaxis.label.set_size(8)
        ax_hm_cb_gene.tick_params(labelsize=7)
    else:
        ax_hm_cb_gene.axis("off")
    if im_factors_ref is not None:
        plt.colorbar(
            im_factors_ref, cax=ax_hm_cb_factor, label="Smoothed factor score in path"
        )
        ax_hm_cb_factor.yaxis.labelpad = 10
        ax_hm_cb_factor.yaxis.label.set_size(8)
        ax_hm_cb_factor.tick_params(labelsize=7)
    else:
        ax_hm_cb_factor.axis("off")

    if save:
        plt.savefig(save, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
        return None
    return fig


def path_embedding(
    model: "scDEF",
    path_id: Union[int, str] = "auto",
    paths_key: str = "transition_paths",
    score_key: Optional[str] = None,
    basis: str = "umap_multilayer",
    min_affinity: float = 0.0,
    affinity_alpha_range: Tuple[float, float] = (0.15, 1.0),
    cmap: str = "viridis",
    point_size: float = 10.0,
    show_background: bool = True,
    background_color: str = "lightgray",
    background_alpha: float = 0.2,
    obs_key: Optional[str] = None,
    obs_order: Optional[Sequence[str]] = None,
    ncols: int = 3,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """Plot cells on an embedding colored by position along one path.

    This visualization uses outputs from ``scd.tl.score_paths``:
      - ``adata.obsm[f"{score_key}_positions"]``
      - ``adata.obsm[f"{score_key}_affinities"]``.

    Color encodes path position (0->1), and point alpha scales with
    path affinity. If ``obs_key`` is provided, draws one facet per
    category value.
    """
    if score_key is None:
        score_key = paths_key
    pos_key = f"{score_key}_positions"
    aff_key = f"{score_key}_affinities"

    if paths_key not in model.adata.uns:
        raise KeyError(f"'{paths_key}' not found in adata.uns.")
    path_objs = list(model.adata.uns[paths_key].get("paths", []))
    if len(path_objs) == 0:
        raise ValueError(f"No paths found in adata.uns['{paths_key}']['paths'].")
    if pos_key not in model.adata.obsm or aff_key not in model.adata.obsm:
        raise KeyError(
            f"Missing '{pos_key}' or '{aff_key}' in adata.obsm. "
            f"Run `scd.tl.score_paths(model, paths_key='{paths_key}', key_added='{score_key}')` first."
        )

    emb_key = basis if basis.startswith("X_") else f"X_{basis}"
    if emb_key not in model.adata.obsm:
        raise KeyError(
            f"Embedding '{emb_key}' not found in adata.obsm. "
            "Compute the embedding first."
        )
    emb = np.asarray(model.adata.obsm[emb_key], dtype=float)
    if emb.ndim != 2 or emb.shape[1] < 2:
        raise ValueError(f"Embedding '{emb_key}' must be a 2D array with >=2 columns.")

    # Resolve path id (supports "auto")
    if isinstance(path_id, str):
        if path_id != "auto":
            raise ValueError("path_id must be an int or 'auto'.")
        stats = list(model.adata.uns[paths_key].get("path_stats", []))
        if len(stats) == len(path_objs):
            ranked = sorted(
                stats,
                key=lambda s: (
                    s.get("n_cells_mid_region", 0),
                    s.get("n_cells_with_position", 0),
                    s.get("mean_affinity", 0.0),
                ),
                reverse=True,
            )
            resolved_path_id = int(ranked[0].get("path_id", 0))
        else:
            aff_all = np.asarray(model.adata.obsm[aff_key], dtype=float)
            pos_all = np.asarray(model.adata.obsm[pos_key], dtype=float)
            support = []
            for p in range(aff_all.shape[1]):
                valid = np.isfinite(pos_all[:, p]) & (
                    aff_all[:, p] >= float(min_affinity)
                )
                support.append(int(np.sum(valid)))
            resolved_path_id = int(np.argmax(np.asarray(support)))
    else:
        resolved_path_id = int(path_id)

    if resolved_path_id < 0 or resolved_path_id >= len(path_objs):
        raise IndexError(
            f"path_id {path_id} resolved to {resolved_path_id}, out of bounds for "
            f"{len(path_objs)} paths in '{paths_key}'."
        )

    positions = np.asarray(
        model.adata.obsm[pos_key][:, int(resolved_path_id)], dtype=float
    )
    affinities = np.asarray(
        model.adata.obsm[aff_key][:, int(resolved_path_id)], dtype=float
    )
    affinities = np.clip(affinities, 0.0, 1.0)
    mask_main = np.isfinite(positions) & (affinities >= float(min_affinity))

    a0, a1 = float(affinity_alpha_range[0]), float(affinity_alpha_range[1])
    if a0 < 0 or a1 > 1 or a1 < a0:
        raise ValueError("affinity_alpha_range must satisfy 0 <= min <= max <= 1.")

    def _plot_one(_ax: plt.Axes, subset_mask: np.ndarray, title: str) -> None:
        if show_background:
            _ax.scatter(
                emb[:, 0],
                emb[:, 1],
                s=float(point_size),
                c=background_color,
                alpha=float(background_alpha),
                linewidths=0,
                rasterized=True,
            )
        mask = mask_main & subset_mask
        if np.any(mask):
            alpha = a0 + (a1 - a0) * affinities[mask]
            rgba = plt.get_cmap(cmap)(np.clip(positions[mask], 0.0, 1.0))
            rgba[:, 3] = alpha
            _ax.scatter(
                emb[mask, 0],
                emb[mask, 1],
                s=float(point_size),
                c=rgba,
                linewidths=0,
                rasterized=True,
            )
        _ax.set_title(title)
        _ax.set_xlabel(f"{basis}_1")
        _ax.set_ylabel(f"{basis}_2")
        _ax.set_xticks([])
        _ax.set_yticks([])
        for spine in _ax.spines.values():
            spine.set_visible(False)

    pobj = path_objs[int(resolved_path_id)]
    nodes = " -> ".join(pobj.get("nodes", []))

    if obs_key is None:
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        else:
            fig = None
        _plot_one(
            ax,
            np.ones(model.adata.n_obs, dtype=bool),
            f"{paths_key}[{resolved_path_id}]  {nodes}",
        )
        cbar = plt.colorbar(
            mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap=cmap),
            ax=ax,
            fraction=0.046,
            pad=0.04,
        )
        cbar.set_label("Path position")
    else:
        if obs_key not in model.adata.obs.columns:
            raise KeyError(f"{obs_key} not found in adata.obs.")
        if ax is not None:
            raise ValueError("ax cannot be provided when obs_key faceting is enabled.")
        obs_vals = model.adata.obs[obs_key].astype(str).to_numpy()
        if obs_order is None:
            values = list(pd.unique(obs_vals))
        else:
            values = [str(v) for v in obs_order]
            missing = [v for v in values if v not in set(obs_vals)]
            if len(missing) > 0:
                raise ValueError(
                    f"obs_order values not present in obs['{obs_key}']: {missing}"
                )
        if int(ncols) <= 0:
            raise ValueError("ncols must be > 0.")
        n_panels = len(values)
        ncols_i = min(int(ncols), max(1, n_panels))
        nrows_i = int(np.ceil(n_panels / ncols_i))
        fig, axes = plt.subplots(
            nrows_i,
            ncols_i,
            figsize=(5.5 * ncols_i, 4.8 * nrows_i),
            squeeze=False,
        )
        flat_axes = axes.ravel()
        for i, value in enumerate(values):
            m = obs_vals == value
            _plot_one(
                flat_axes[i],
                m,
                f"{value} (n={int(np.sum(m))})",
            )
        for j in range(n_panels, len(flat_axes)):
            flat_axes[j].axis("off")
        fig.suptitle(f"{paths_key}[{resolved_path_id}]  {nodes}", y=0.995)
        cbar = fig.colorbar(
            mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap=cmap),
            ax=[axes[r, c] for r in range(nrows_i) for c in range(ncols_i)],
            fraction=0.02,
            pad=0.01,
        )
        cbar.set_label("Path position")
        fig.tight_layout()

    if show:
        plt.show()
        return None
    return fig
