import numpy as np
import pandas as pd
import scdef.utils.hierarchy_utils as hierarchy_utils
from typing import Optional, Sequence, Dict, Any, Union, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from scdef.models._scdef import scDEF


def add_l0_lineage_aggregate_scores(
    per_factor: pd.DataFrame,
    layer_names: Sequence[str],
) -> pd.DataFrame:
    """Add lineage-averaged clarity and effective parents for layer-0 factors only.

    For each L0 factor, follows ``best_parent`` upward through layers L1 … L{n-2},
    collecting ``clarity_score_01`` and ``n_eff_parents`` from each factor along the
    path (same definitions as :func:`compute_hierarchy_scores`). Stores the mean
    over those positions in ``avg_clarity`` and ``avg_n_eff_parents`` on the L0 row
    only; other rows get NaN.

    This is called automatically from :func:`compute_hierarchy_scores`; it remains
    public for advanced use on a pre-built ``per_factor`` frame.

    This captures cases where an L0 factor maps cleanly to one L1 parent (low local
    ``n_eff_parents``) while that parent is ambiguous relative to L2, by letting
    lineage averages reflect uncertainty higher in the hierarchy.

    Args:
        per_factor: per-factor scores from :func:`compute_hierarchy_scores`. Factor
            identity for lookups uses **row index labels** (not the ``child_factor``
            column when present).
        layer_names: ordered model layer names (``model.layer_names``). The walk
            length is ``len(layer_names) - 1`` (one score per child layer, L0 through
            L{n-2}); layer order is not inferred from strings in the frame so
            non-lexicographic names stay correct.

    Returns:
        Copy of ``per_factor`` with two additional float columns.
    """
    out = per_factor.copy()
    out["avg_clarity"] = np.nan
    out["avg_n_eff_parents"] = np.nan

    n_layers = len(layer_names)
    if n_layers < 2 or len(out) == 0:
        return out

    l0_name = layer_names[0]
    lookup: Dict[Tuple[str, str], int] = {}
    for idx in range(len(out)):
        key = (str(out.iloc[idx]["child_layer"]), str(out.index[idx]))
        lookup[key] = idx

    expected_steps = n_layers - 1
    cl_series = out["child_layer"]
    l0_mask = cl_series == l0_name
    l0_positions = np.flatnonzero(l0_mask.to_numpy())

    for pos in l0_positions:
        cur_name = str(out.index[pos])
        clarities: list[float] = []
        neffs: list[float] = []
        for step in range(expected_steps):
            cur_layer_idx = step
            key = (layer_names[cur_layer_idx], cur_name)
            if key not in lookup:
                clarities = []
                break
            ridx = lookup[key]
            clarities.append(float(out.iloc[ridx]["clarity_score_01"]))
            neffs.append(float(out.iloc[ridx]["n_eff_parents"]))
            cur_name = str(out.iloc[ridx]["best_parent"])
        if len(clarities) == expected_steps:
            out.iat[pos, out.columns.get_loc("avg_clarity")] = float(np.mean(clarities))
            out.iat[pos, out.columns.get_loc("avg_n_eff_parents")] = float(
                np.mean(neffs)
            )

    return out


def compute_hierarchy_scores(
    model: "scDEF",
    use_filtered: bool = False,  # use model.factor_lists / model.factor_names
    filter_upper_layers: bool = True,
    factor_weight: str = "uniform",  # {"usage","uniform"}
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """Compute per-factor and global hierarchy scores from learned W matrices.

    Args:
        model: scDEF model instance
        use_filtered: whether to use model.factor_lists / model.factor_names
        filter_upper_layers: when use_filtered is False, whether to still use
            filtered factors for layers > 0 (both as parents and children)
        factor_weight: weighting scheme for factors, either "uniform" or "usage"
        eps: small epsilon value for numerical stability

    Returns:
        dict containing ``per_factor`` (index = child factor name, ``child_factor``
        column retained; includes ``avg_clarity`` and ``avg_n_eff_parents`` for L0),
        ``per_transition``, ``global_score``, and ``global_ambiguity``.
    """

    def _col_normalize(W):
        return W / (W.sum(axis=0, keepdims=True) + eps)

    def _usage_weights(child_layer_idx, child_factor_indices):
        # usage weight: sum over cells of per-cell normalized z mass (restricted to kept factors)
        lname = model.layer_names[child_layer_idx]
        Z = np.asarray(model.pmeans[f"{lname}z"][:, child_factor_indices], dtype=float)
        Z = Z / (
            Z.sum(axis=1, keepdims=True) + eps
        )  # per cell sum=1 across kept factors
        w = Z.sum(axis=0)  # sums to n_cells
        w = w / (w.sum() + eps)  # normalize to sum=1 for that transition
        return w

    rows = []
    trans_rows = []

    all_scores = []
    all_weights = []

    # child layers: 0..n_layers-2 (top layer has no parent)
    for child_layer in range(0, model.n_layers - 1):
        parent_layer = child_layer + 1
        child_name = model.layer_names[child_layer]
        parent_name = model.layer_names[parent_layer]

        use_filtered_child = use_filtered or (filter_upper_layers and child_layer > 0)
        use_filtered_parent = use_filtered or (filter_upper_layers and parent_layer > 0)

        if use_filtered_child:
            child_idx = np.array(model.factor_lists[child_layer], dtype=int)
            child_factor_names = list(model.factor_names[child_layer])
        else:
            child_idx = np.arange(model.layer_sizes[child_layer], dtype=int)
            child_factor_names = [f"{child_name}_{i}" for i in child_idx]
        if use_filtered_parent:
            parent_idx = np.array(model.factor_lists[parent_layer], dtype=int)
            parent_factor_names = list(model.factor_names[parent_layer])
        else:
            parent_idx = np.arange(model.layer_sizes[parent_layer], dtype=int)
            parent_factor_names = [f"{parent_name}_{i}" for i in parent_idx]

        # W_parent has shape (#parents x #children)
        W = np.asarray(model.pmeans[f"{parent_name}W"], dtype=float)[
            np.ix_(parent_idx, child_idx)
        ]
        P = _col_normalize(W)
        K = P.shape[0]
        n_children = P.shape[1]

        # weights within this transition
        if factor_weight == "uniform":
            w = np.ones(n_children, dtype=float) / max(n_children, 1)
        elif factor_weight == "usage":
            w = _usage_weights(child_layer, child_idx)
        else:
            raise ValueError("factor_weight must be one of {'usage','uniform'}")

        # per-child entropy etc.
        if K <= 1:
            H = np.zeros(n_children, dtype=float)
            ambiguity_frac = np.zeros(n_children, dtype=float)
            clarity = np.ones(n_children, dtype=float)
            n_eff = np.ones(n_children, dtype=float)
            top_gap = np.ones(n_children, dtype=float)
        else:
            H = -np.sum(P * np.log(P + eps), axis=0)  # [n_children]
            ambiguity_frac = H / np.log(K)  # [0,1]
            clarity = 1.0 - ambiguity_frac  # [0,1]
            n_eff = np.exp(H)  # [1,K]

            # top-2 gap diagnostic
            part = np.partition(P, kth=K - 2, axis=0)  # safe
            p2 = part[-2, :]
            p1 = part[-1, :]
            top_gap = p1 - p2

        best_parent_pos = np.argmax(P, axis=0)
        best_parent_prob = P[best_parent_pos, np.arange(n_children)]

        # accumulate global weighted scores (transition weights sum to 1, so global becomes average across transitions)
        all_scores.append(clarity)
        all_weights.append(w)

        trans_rows.append(
            {
                "transition": f"{parent_name} -> {child_name}",
                "K_parents": int(K),
                "n_children": int(n_children),
                "transition_score": float(np.sum(w * clarity)),
                "transition_ambiguity": float(np.sum(w * ambiguity_frac)),
            }
        )

        for j in range(n_children):
            rows.append(
                {
                    "child_layer": child_name,
                    "child_factor": child_factor_names[j],
                    "original_factor_idx": int(child_idx[j]),
                    "parent_layer": parent_name,
                    "K_parents": int(K),
                    "best_parent": parent_factor_names[int(best_parent_pos[j])],
                    "best_parent_prob": float(best_parent_prob[j]),
                    "top_gap": float(top_gap[j]),
                    "clarity_score_01": float(clarity[j]),  # comparable across datasets
                    "ambiguity_frac_01": float(
                        ambiguity_frac[j]
                    ),  # comparable across datasets
                    "n_eff_parents": float(
                        n_eff[j]
                    ),  # interpretable (#effective parents)
                    "n_eff_parents_frac": float(
                        n_eff[j] / max(K, 1)
                    ),  # scale-free (approx share of parents used)
                    "weight": float(w[j]),
                }
            )

    per_factor = pd.DataFrame(rows)
    per_transition = pd.DataFrame(trans_rows)

    if len(all_scores) == 0:
        global_score = 1.0
    else:
        scores_concat = np.concatenate(all_scores)
        weights_concat = np.concatenate(all_weights)
        global_score = float(
            np.sum(scores_concat * weights_concat) / (np.sum(weights_concat) + eps)
        )

    if len(per_factor) > 0 and "child_factor" in per_factor.columns:
        per_factor = per_factor.set_index("child_factor", drop=True)
        per_factor.index.name = "child_factor"
    per_factor = add_l0_lineage_aggregate_scores(per_factor, model.layer_names)

    return {
        "per_factor": per_factor,
        "per_transition": per_transition,
        "global_score": global_score,  # comparable across datasets
        "global_ambiguity": 1.0 - global_score,  # also comparable
        "weighting": factor_weight,
    }


def effective_parents_from_clarity(
    clarity_score: Union[float, np.ndarray],
    n_parents: int,
    clip: bool = True,
) -> Union[float, np.ndarray]:
    """Compute effective number of parents from a clarity score.

    This uses the same definitions as ``compute_hierarchy_scores``:
    ``clarity = 1 - H/log(K)`` and ``n_eff = exp(H)``.
    Therefore, for ``K > 1``:
    ``n_eff = K ** (1 - clarity)``.

    Args:
        clarity_score: clarity score(s), typically in [0, 1]
        n_parents: number of candidate parents ``K``
        clip: if True, clip clarity values to [0, 1]. If False, values
            outside [0, 1] raise a ValueError.

    Returns:
        Effective number of parents, with the same shape as clarity_score.
    """
    if n_parents < 1:
        raise ValueError("n_parents must be >= 1")

    clarity = np.asarray(clarity_score, dtype=float)

    if clip:
        clarity = np.clip(clarity, 0.0, 1.0)
    elif np.any((clarity < 0.0) | (clarity > 1.0)):
        raise ValueError("clarity_score must be in [0, 1] when clip=False")

    # Degenerate case: only one possible parent.
    if n_parents == 1:
        n_eff = np.ones_like(clarity, dtype=float)
    else:
        n_eff = np.power(float(n_parents), 1.0 - clarity)

    if np.ndim(n_eff) == 0:
        return float(n_eff)
    return n_eff


def clarity_from_effective_parents(
    n_eff_parents: Union[float, np.ndarray],
    n_parents: int,
    clip: bool = True,
) -> Union[float, np.ndarray]:
    """Compute clarity score from effective number of parents.

    Inverse of ``effective_parents_from_clarity`` under the same definitions
    used in ``compute_hierarchy_scores``:
    ``clarity = 1 - H/log(K)`` and ``n_eff = exp(H)``.
    Therefore, for ``K > 1``:
    ``clarity = 1 - log(n_eff)/log(K)``.

    Args:
        n_eff_parents: effective parent count(s)
        n_parents: number of candidate parents ``K``
        clip: if True, clip n_eff to [1, n_parents]. If False, values
            outside this range raise a ValueError.

    Returns:
        Clarity score(s), with the same shape as n_eff_parents.
    """
    if n_parents < 1:
        raise ValueError("n_parents must be >= 1")

    n_eff = np.asarray(n_eff_parents, dtype=float)

    if clip:
        n_eff = np.clip(n_eff, 1.0, float(n_parents))
    elif np.any((n_eff < 1.0) | (n_eff > float(n_parents))):
        raise ValueError("n_eff_parents must be in [1, n_parents] when clip=False")

    # Degenerate case: only one possible parent.
    if n_parents == 1:
        clarity = np.ones_like(n_eff, dtype=float)
    else:
        clarity = 1.0 - (np.log(n_eff) / np.log(float(n_parents)))

    if np.ndim(clarity) == 0:
        return float(clarity)
    return clarity


def _technical_factor_names(factor_obs: pd.DataFrame) -> set[str]:
    """Factor names marked technical in ``factor_obs``."""
    if "technical" not in factor_obs.columns:
        return set()
    technical = factor_obs["technical"].fillna(False).astype(bool)
    return set(factor_obs.index[technical].astype(str))


def find_sensible_top_layer(
    model: "scDEF",
    n_eff_parents_max: float = 1.5,
    min_best_parent_prob: Optional[float] = None,
    min_clear_fraction: float = 0.8,
    ignore_root: bool = True,
    use_filtered: bool = True,
    store: bool = True,
) -> Dict[str, Any]:
    """Find the coarsest hierarchy layer supported by confident merges.

    Uses existing ``factor_obs`` diagnostics (``n_eff_parents``,
    ``best_parent_prob``, ``top_gap``, etc.), which are computed from the W
    matrices by :func:`factor_diagnostics`. Moving upward stops at the first
    transition whose child factors are too ambiguous to merge; the layer just
    below that transition is reported as the sensible top.
    """
    n_layers = int(model.n_layers)
    visible_n_layers = n_layers
    if (
        bool(ignore_root)
        and n_layers > 1
        and int(model.layer_sizes[-1]) == 1
        and len(model.factor_lists[-1]) == 1
    ):
        visible_n_layers -= 1
    if visible_n_layers < 1:
        raise ValueError("Model must contain at least one visible layer.")
    if "factor_obs" not in model.adata.uns:
        from scdef.tools.factor import factor_diagnostics

        factor_diagnostics(model)
    factor_obs = model.adata.uns["factor_obs"]
    required = {"child_layer", "n_eff_parents", "best_parent_prob", "parent_layer"}
    missing = required.difference(factor_obs.columns)
    if missing:
        raise KeyError(
            "factor_obs is missing required hierarchy diagnostics: "
            + ", ".join(sorted(missing))
            + ". Re-run scd.tl.factor_diagnostics(model)."
        )
    technical_factors = _technical_factor_names(factor_obs)

    factor_rows = []
    transition_rows = []
    recommended_layer_idx = 0

    for child_layer in range(visible_n_layers - 1):
        parent_layer = child_layer + 1
        child_name = model.layer_names[child_layer]
        parent_name = model.layer_names[parent_layer]
        if use_filtered:
            child_factor_names = list(model.factor_names[child_layer])
        else:
            child_factor_names = factor_obs.index[
                factor_obs["child_layer"] == child_name
            ].tolist()
        child_factor_names = [
            name for name in child_factor_names if str(name) not in technical_factors
        ]
        rows = factor_obs[
            (factor_obs["child_layer"] == child_name)
            & (factor_obs.index.isin(child_factor_names))
        ].copy()
        if len(rows) == 0:
            clear_fraction = 0.0
            transition_ok = False
            n_eff = np.array([], dtype=float)
            best_prob = np.array([], dtype=float)
        else:
            rows = rows.loc[child_factor_names]
            n_eff = rows["n_eff_parents"].to_numpy(dtype=float)
            best_prob = rows["best_parent_prob"].to_numpy(dtype=float)
            clear = np.isfinite(n_eff) & (n_eff <= float(n_eff_parents_max))
            if min_best_parent_prob is not None:
                clear = clear & (
                    np.isfinite(best_prob) & (best_prob >= float(min_best_parent_prob))
                )
            rows["clear_merge"] = clear
            clear_fraction = float(np.mean(clear))
            transition_ok = clear_fraction >= float(min_clear_fraction)

        transition_rows.append(
            {
                "child_layer_idx": int(child_layer),
                "child_layer": child_name,
                "parent_layer_idx": int(parent_layer),
                "parent_layer": parent_name,
                "n_children": int(len(rows)),
                "n_parents": int(rows["K_parents"].iloc[0])
                if len(rows) > 0 and "K_parents" in rows.columns
                else int(len(model.factor_lists[parent_layer])),
                "clear_fraction": clear_fraction,
                "median_n_eff_parents": float(np.median(n_eff))
                if len(n_eff) > 0
                else np.nan,
                "max_n_eff_parents": float(np.max(n_eff)) if len(n_eff) > 0 else np.nan,
                "min_best_parent_prob": float(np.min(best_prob))
                if len(best_prob) > 0
                else np.nan,
                "transition_ok": bool(transition_ok),
            }
        )

        if len(rows) > 0:
            keep_cols = [
                c
                for c in [
                    "parent_layer",
                    "best_parent",
                    "best_parent_prob",
                    "top_gap",
                    "n_eff_parents",
                    "K_parents",
                    "clear_merge",
                ]
                if c in rows.columns
            ]
            out_rows = rows[keep_cols].copy()
            out_rows.insert(0, "child_factor", out_rows.index)
            out_rows.insert(0, "child_layer", child_name)
            out_rows.insert(0, "child_layer_idx", int(child_layer))
            out_rows.insert(3, "parent_layer_idx", int(parent_layer))
            factor_rows.extend(out_rows.to_dict(orient="records"))

        if transition_ok:
            recommended_layer_idx = parent_layer
        else:
            break

    recommended_factors = [
        name
        for name in model.factor_names[recommended_layer_idx]
        if str(name) not in technical_factors
    ]
    transitions = pd.DataFrame(transition_rows)
    per_factor = pd.DataFrame(factor_rows)
    result = {
        "recommended_layer_idx": int(recommended_layer_idx),
        "recommended_layer": model.layer_names[recommended_layer_idx],
        "recommended_factors": recommended_factors,
        "n_eff_parents_max": float(n_eff_parents_max),
        "min_best_parent_prob": min_best_parent_prob,
        "min_clear_fraction": float(min_clear_fraction),
        "transition_diagnostics": transitions,
        "factor_diagnostics": per_factor,
    }
    if store:
        model.adata.uns["sensible_top_layer"] = {
            k: v
            for k, v in result.items()
            if k not in {"transition_diagnostics", "factor_diagnostics"}
        }
        model.adata.uns["sensible_top_transition_diagnostics"] = transitions
        model.adata.uns["sensible_top_factor_diagnostics"] = per_factor
    return result


def find_sensible_top_factors(
    model: "scDEF",
    n_eff_parents_max: float = 1.5,
    min_best_parent_prob: Optional[float] = None,
    ignore_root: bool = True,
    use_filtered: bool = True,
    store: bool = True,
) -> Dict[str, Any]:
    """Find a mixed-depth frontier of factors whose upward merge is ambiguous.

    Starting from each L0 factor, follows ``best_parent`` upward through
    ``factor_obs`` while the merge is confident. The first factor before an
    ambiguous merge becomes a top factor. This lets rare lineages stop at L0
    while common lineages continue to coarser factors.
    """
    n_layers = int(model.n_layers)
    visible_n_layers = n_layers
    if (
        bool(ignore_root)
        and n_layers > 1
        and int(model.layer_sizes[-1]) == 1
        and len(model.factor_lists[-1]) == 1
    ):
        visible_n_layers -= 1
    if visible_n_layers < 1:
        raise ValueError("Model must contain at least one visible layer.")
    if "factor_obs" not in model.adata.uns:
        from scdef.tools.factor import factor_diagnostics

        factor_diagnostics(model)
    factor_obs = model.adata.uns["factor_obs"]
    required = {"child_layer", "n_eff_parents", "best_parent_prob", "best_parent"}
    missing = required.difference(factor_obs.columns)
    if missing:
        raise KeyError(
            "factor_obs is missing required hierarchy diagnostics: "
            + ", ".join(sorted(missing))
            + ". Re-run scd.tl.factor_diagnostics(model)."
        )
    technical_factors = _technical_factor_names(factor_obs)

    name_to_layer = {}
    for layer_idx in range(visible_n_layers):
        names = (
            list(model.factor_names[layer_idx])
            if use_filtered
            else [
                f"{model.layer_names[layer_idx]}_{i}"
                for i in range(model.layer_sizes[layer_idx])
            ]
        )
        for name in names:
            if str(name) in technical_factors:
                continue
            name_to_layer[str(name)] = int(layer_idx)

    def _is_clear(row: pd.Series) -> bool:
        n_eff = float(row["n_eff_parents"])
        best_prob = float(row["best_parent_prob"])
        if not np.isfinite(n_eff) or n_eff > float(n_eff_parents_max):
            return False
        if min_best_parent_prob is not None and (
            (not np.isfinite(best_prob)) or best_prob < float(min_best_parent_prob)
        ):
            return False
        return True

    top_factors = []
    path_rows = []
    for start in [
        name
        for name in list(model.factor_names[0])
        if str(name) not in technical_factors
    ]:
        cur = str(start)
        path = [cur]
        stop_reason = "top_visible_layer"
        while True:
            layer_idx = name_to_layer.get(cur)
            if layer_idx is None:
                stop_reason = "unknown_factor"
                break
            if layer_idx >= visible_n_layers - 1:
                stop_reason = "top_visible_layer"
                break
            if cur not in factor_obs.index:
                stop_reason = "missing_factor_obs"
                break
            row = factor_obs.loc[cur]
            if not _is_clear(row):
                stop_reason = "ambiguous_parent"
                break
            parent = str(row["best_parent"])
            if parent not in name_to_layer or name_to_layer[parent] <= layer_idx:
                stop_reason = "invalid_parent"
                break
            cur = parent
            path.append(cur)
        if cur not in top_factors:
            top_factors.append(cur)
        path_rows.append(
            {
                "start_factor": str(start),
                "top_factor": cur,
                "top_layer_idx": int(name_to_layer.get(cur, -1)),
                "top_layer": model.layer_names[name_to_layer[cur]]
                if cur in name_to_layer
                else None,
                "stop_reason": stop_reason,
                "path": path,
            }
        )

    top_factor_layers = {
        factor: int(name_to_layer[factor])
        for factor in top_factors
        if factor in name_to_layer
    }
    paths = pd.DataFrame(path_rows)
    result = {
        "top_factors": top_factors,
        "top_factor_layers": top_factor_layers,
        "n_eff_parents_max": float(n_eff_parents_max),
        "min_best_parent_prob": min_best_parent_prob,
        "paths": paths,
    }
    if store:
        model.adata.uns["sensible_top_factors"] = {
            k: v for k, v in result.items() if k != "paths"
        }
        model.adata.uns["sensible_top_factor_paths"] = paths
    return result


def make_biological_hierarchy(model: "scDEF") -> Dict[str, Sequence[str]]:
    """Make the biological hierarchy of the model.

    Args:
        model: scDEF model instance

    Returns:
        biological_hierarchy: dictionary containing the biological hierarchy
    """
    factor_obs = model.adata.uns["factor_obs"]
    drop_mask = factor_obs["technical"]
    if "global" in factor_obs.columns:
        drop_mask = drop_mask | factor_obs["global"]
    drop_factors = factor_obs.index[drop_mask].tolist()
    biological_hierarchy = get_hierarchy(model, drop_factors=drop_factors)
    model.adata.uns["biological_hierarchy"] = biological_hierarchy
    return biological_hierarchy


def make_technical_hierarchy(model: "scDEF") -> Dict[str, Sequence[str]]:
    """Make the technical hierarchy of the model.

    Args:
        model: scDEF model instance

    Returns:
        technical_hierarchy: dictionary containing the technical hierarchy
    """
    factor_obs = model.adata.uns["factor_obs"]
    technical_mask = factor_obs["technical"] & (factor_obs["child_layer"] == "L0")
    technical_factors = factor_obs[
        technical_mask
    ].index.tolist()  # only layer 0 factors
    # technical hierarchy is a root with all technical factors as direct children.
    # connection weights are proportional to the usage of each factor
    technical_hierarchy = dict()
    technical_hierarchy["tech_top"] = technical_factors
    model.adata.uns["technical_hierarchy"] = technical_hierarchy
    return technical_hierarchy


def make_global_hierarchy(model: "scDEF") -> Dict[str, Sequence[str]]:
    """Make the global (shared-across-lineages) hierarchy of the model.

    Args:
        model: scDEF model instance

    Returns:
        global_hierarchy: dictionary with synthetic root ``global_top`` and
            global layer-0 factors as direct children.
    """
    factor_obs = model.adata.uns["factor_obs"]
    if "global" not in factor_obs.columns:
        factor_obs["global"] = False
    global_mask = factor_obs["global"] & (
        factor_obs["child_layer"] == model.layer_names[0]
    )
    global_factors = factor_obs[global_mask].index.tolist()
    global_hierarchy: Dict[str, Sequence[str]] = {"global_top": global_factors}
    model.adata.uns["global_hierarchy"] = global_hierarchy
    return global_hierarchy


def make_hierarchies(model: "scDEF") -> None:
    """Store the biological, technical, and global hierarchies of the model.

    Args:
        model: scDEF model instance
    """
    make_biological_hierarchy(model)
    make_technical_hierarchy(model)
    make_global_hierarchy(model)


def get_hierarchy(
    model: "scDEF",
    simplified: Optional[bool] = True,
    drop_factors: Optional[Sequence[str]] = None,
) -> Dict[str, Sequence[str]]:
    """Get a dictionary containing the polytree contained in the scDEF graph.

    Args:
        simplified: whether to collapse single-child nodes
        drop_factors: factors to drop from the hierarchy
    Returns:
        hierarchy: the dictionary containing the hierarchy
    """
    hierarchy = dict()
    for layer_idx in range(0, model.n_layers - 1):
        # factor_names always matches factor_lists in order
        factor_names = model.factor_names[layer_idx]
        factors = model.factor_lists[layer_idx]
        # drop_factors are names, not indices
        if drop_factors is not None:
            kept_idx = [
                i for i, name in enumerate(factor_names) if name not in drop_factors
            ]
            factors = np.array([factors[i] for i in kept_idx], dtype=int)
            factor_names = [factor_names[i] for i in kept_idx]
        else:
            factors = np.array(factors, dtype=int)
            factor_names = list(factor_names)
        n_factors = len(factors)

        upper_factor_names = model.factor_names[layer_idx + 1]
        upper_factors = model.factor_lists[layer_idx + 1]
        if drop_factors is not None:
            upper_kept_idx = [
                i
                for i, name in enumerate(upper_factor_names)
                if name not in drop_factors
            ]
            upper_factors = np.array(
                [upper_factors[i] for i in upper_kept_idx], dtype=int
            )
            upper_factor_names = [upper_factor_names[i] for i in upper_kept_idx]
        else:
            upper_factors = np.array(upper_factors, dtype=int)
            upper_factor_names = list(upper_factor_names)

        # Check: left index is upper factor, right is lower factor
        # model.pmeans["<upper_layer_name>W"] shape: [n_upper, n_lower]
        mat = model.pmeans[f"{model.layer_names[layer_idx+1]}W"][
            np.ix_(upper_factors, factors)
        ]
        # mat: (len(upper_factors), len(factors))
        # Each lower factor will be assigned to a single upper factor:
        # For each factor (column), find the row (upper factor) with the highest value
        normalized_factor_weights = mat / np.sum(mat, axis=0, keepdims=True)
        assignments = []
        for factor_idx in range(n_factors):
            # for each lower factor (column), get the upper factor (row) with the highest normalized weight
            assignments.append(np.argmax(normalized_factor_weights[:, factor_idx]))
        assignments = np.array(assignments)

        for upper_layer_factor_idx, upper_layer_factor_name in enumerate(
            upper_factor_names
        ):
            assigned_lower = [
                factor_names[j]
                for j in range(n_factors)
                if assignments[j] == upper_layer_factor_idx
            ]
            hierarchy[upper_layer_factor_name] = assigned_lower

    if simplified:
        layer_sizes = [len(model.factor_names[idx]) for idx in range(model.n_layers)]
        hierarchy = hierarchy_utils.simplify_hierarchy(
            hierarchy, model.layer_names, layer_sizes, factor_names=model.factor_names
        )

    return hierarchy
