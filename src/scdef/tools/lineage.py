"""Lineage-specific vs global (shared) factor selection from hierarchy diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

from scdef.tools.factor import _resolve_factor_obs_names

if TYPE_CHECKING:
    from scdef.models._scdef import scDEF


def _require_factor_obs(model: "scDEF") -> pd.DataFrame:
    if "factor_obs" not in model.adata.uns:
        raise KeyError(
            "model.adata.uns['factor_obs'] is missing. "
            "Run scd.tl.factor_diagnostics(model) first."
        )
    factor_obs = model.adata.uns["factor_obs"]
    required = {
        "child_layer",
        "original_factor_idx",
        "best_parent",
        "best_parent_prob",
        "n_eff_parents",
        "parent_layer",
    }
    missing = required - set(factor_obs.columns)
    if missing:
        raise KeyError(
            "factor_obs is missing required columns: "
            + ", ".join(sorted(missing))
            + ". Recompute with scd.tl.factor_diagnostics(model)."
        )
    return factor_obs


def _validate_layer_idx(model: "scDEF", layer_idx: int) -> None:
    layer_idx = int(layer_idx)
    if layer_idx < 0 or layer_idx >= model.n_layers - 1:
        raise ValueError(
            f"layer_idx must be in [0, {model.n_layers - 2}] "
            f"(child layers with parents); got {layer_idx}."
        )


def _build_factor_obs_lookup(
    factor_obs: pd.DataFrame, layer_names: Sequence[str]
) -> Dict[Tuple[str, str], int]:
    lookup: Dict[Tuple[str, str], int] = {}
    for idx in range(len(factor_obs)):
        key = (str(factor_obs.iloc[idx]["child_layer"]), str(factor_obs.index[idx]))
        lookup[key] = idx
    return lookup


def _current_name_for_row(model: "scDEF", row: pd.Series) -> Optional[str]:
    layer_name = str(row["child_layer"])
    if layer_name not in model.layer_names:
        return None
    layer_idx = model.layer_names.index(layer_name)
    orig = int(row["original_factor_idx"])
    for slot, fidx in enumerate(model.factor_lists[layer_idx]):
        if int(fidx) == orig:
            return model.factor_names[layer_idx][slot]
    return None


def _lineage_path_to_top(
    factor_obs: pd.DataFrame,
    start_name: str,
    start_layer_idx: int,
    top_factor_obs_name: str,
    layer_names: Sequence[str],
) -> Optional[List[int]]:
    """Row indices from ``start_layer_idx`` upward; None if path does not reach top."""
    lookup = _build_factor_obs_lookup(factor_obs, layer_names)
    top_layer_name = layer_names[-1]
    penultimate_idx = len(layer_names) - 2
    path: List[int] = []
    cur_name = str(start_name)
    for layer_idx in range(start_layer_idx, penultimate_idx + 1):
        key = (layer_names[layer_idx], cur_name)
        if key not in lookup:
            return None
        ridx = lookup[key]
        path.append(ridx)
        row = factor_obs.iloc[ridx]
        if layer_idx == penultimate_idx:
            if (
                str(row["parent_layer"]) == top_layer_name
                and str(row["best_parent"]) == top_factor_obs_name
            ):
                return path
            return None
        cur_name = str(row["best_parent"])
    return None


def _path_passes_thresholds(
    factor_obs: pd.DataFrame,
    path: Sequence[int],
    n_eff_parents_max: float,
    prob_min: float,
    exclude_technical: bool,
) -> bool:
    for ridx in path:
        row = factor_obs.iloc[ridx]
        if exclude_technical and bool(row.get("technical", False)):
            return False
        if float(row["n_eff_parents"]) > float(n_eff_parents_max):
            return False
        if float(row["best_parent_prob"]) < float(prob_min):
            return False
    return True


def _resolve_top_factor_label(model: "scDEF", top_factor_label: str) -> str:
    """Map a user top-layer name to the ``best_parent`` label used in ``factor_obs``."""
    label = str(top_factor_label)
    factor_obs = model.adata.uns["factor_obs"]
    if label in factor_obs.index:
        return label
    resolved, _unknown = _resolve_factor_obs_names(model, [label])
    if resolved:
        return resolved[0]
    if label in model.factor_names[model.n_layers - 1]:
        return label
    raise ValueError(
        f"Unknown top_factor_label `{top_factor_label}`. "
        f"Expected a name in model.factor_names[{model.n_layers - 1}] "
        f"or factor_obs.index."
    )


def _hierarchy_ambiguity_score(row: pd.Series, layer_idx: int) -> float:
    if layer_idx == 0 and "avg_n_eff_parents" in row.index:
        val = row["avg_n_eff_parents"]
        if pd.notna(val):
            return float(val)
    return float(row["n_eff_parents"])


def get_lineage_factors(
    model: "scDEF",
    top_factor_label: str,
    layer_idx: int = 0,
    n_eff_parents_max: float = 1.5,
    prob_min: float = 0.5,
    exclude_technical: bool = True,
) -> List[str]:
    """Return L{layer_idx} factors in the lineage of a top-layer population.

    A factor is included if following ``best_parent`` at each child layer reaches
    ``top_factor_label`` at the top, and every step on that path has
    ``n_eff_parents <= n_eff_parents_max`` and ``best_parent_prob >= prob_min``.

    Args:
        model: fitted scDEF / iscDEF / sscDEF model with ``factor_obs`` stored.
        top_factor_label: name of a top-layer factor (current ``model.factor_names``
            or a ``factor_obs`` index label).
        layer_idx: child layer to query (default 0 = L0).
        n_eff_parents_max: maximum effective parents at each step on the path.
        prob_min: minimum ``best_parent_prob`` at each step on the path.
        exclude_technical: drop factors marked technical in ``factor_obs``.

    Returns:
        List of factor names in the current model view at ``layer_idx``.
    """
    _validate_layer_idx(model, layer_idx)
    factor_obs = _require_factor_obs(model)

    top_obs_name = _resolve_top_factor_label(model, top_factor_label)

    child_layer = model.layer_names[layer_idx]
    candidates = factor_obs.index[factor_obs["child_layer"] == child_layer]

    out: List[str] = []
    for name in candidates:
        path = _lineage_path_to_top(
            factor_obs,
            str(name),
            layer_idx,
            top_obs_name,
            model.layer_names,
        )
        if path is None:
            continue
        if not _path_passes_thresholds(
            factor_obs,
            path,
            n_eff_parents_max=n_eff_parents_max,
            prob_min=prob_min,
            exclude_technical=exclude_technical,
        ):
            continue
        current = _current_name_for_row(model, factor_obs.loc[name])
        if current is not None:
            out.append(current)
    return out


def get_global_factors(
    model: "scDEF",
    layer_idx: int = 0,
    n_eff_parents_min: float = 1.5,
    exclude_technical: bool = True,
) -> List[str]:
    """Return L{layer_idx} factors shared across lineages (high effective parents).

    Uses ``avg_n_eff_parents`` for layer 0 when available; otherwise local
    ``n_eff_parents``. Factors with score >= ``n_eff_parents_min`` are returned.

    Args:
        model: fitted scDEF model with ``factor_obs`` stored.
        layer_idx: child layer to query (default 0 = L0).
        n_eff_parents_min: minimum effective-parent score for a global factor.
        exclude_technical: drop factors marked technical in ``factor_obs``.

    Returns:
        List of factor names in the current model view at ``layer_idx``.
    """
    _validate_layer_idx(model, layer_idx)
    factor_obs = _require_factor_obs(model)
    layer_idx = int(layer_idx)

    child_layer = model.layer_names[layer_idx]
    mask = factor_obs["child_layer"] == child_layer
    out: List[str] = []
    for name in factor_obs.index[mask]:
        row = factor_obs.loc[name]
        if exclude_technical and bool(row.get("technical", False)):
            continue
        score = _hierarchy_ambiguity_score(row, layer_idx)
        if score < float(n_eff_parents_min):
            continue
        current = _current_name_for_row(model, row)
        if current is not None:
            out.append(current)
    return out
