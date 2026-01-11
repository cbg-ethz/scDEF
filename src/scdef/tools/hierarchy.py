import numpy as np
import pandas as pd
import scdef.utils.hierarchy_utils as hierarchy_utils
from typing import Optional, Sequence, Mapping


def compute_hierarchy_scores(
    model,
    use_filtered: bool = True,  # use model.factor_lists / model.factor_names
    factor_weight: str = "uniform",  # {"usage","uniform"}
    eps: float = 1e-12,
):
    """
    Computes per-factor and global hierarchy scores from learned W matrices.

    Definitions (for each child factor j at a transition with K possible parents):
      P(:,j) = normalized column of W  => p(parent | child=j)

      H_j = -sum_i P_ij log P_ij
      ambiguity_frac_j = H_j / log(K)              in [0,1]
      clarity_score_j  = 1 - ambiguity_frac_j      in [0,1]   (COMPARABLE across different K)

      n_eff_parents_j = exp(H_j)                   in [1, K]  (interpretable "effective #parents")

    Global score:
      weighted mean of clarity_score_j across all children across all transitions,
      weights chosen by factor_weight.

    Returns:
      dict with:
        - per_factor: DataFrame (one row per factor that has a parent; i.e. all non-top layers)
        - per_transition: DataFrame (one row per layer transition)
        - global_score: float in [0,1]
        - global_ambiguity: float in [0,1] (= 1 - global_score)
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

        if use_filtered:
            child_idx = np.array(model.factor_lists[child_layer], dtype=int)
            parent_idx = np.array(model.factor_lists[parent_layer], dtype=int)
            child_factor_names = list(model.factor_names[child_layer])
            parent_factor_names = list(model.factor_names[parent_layer])
        else:
            child_idx = np.arange(model.layer_sizes[child_layer], dtype=int)
            parent_idx = np.arange(model.layer_sizes[parent_layer], dtype=int)
            child_factor_names = [f"{child_name}_{i}" for i in child_idx]
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

    return {
        "per_factor": per_factor,
        "per_transition": per_transition,
        "global_score": global_score,  # comparable across datasets
        "global_ambiguity": 1.0 - global_score,  # also comparable
        "weighting": factor_weight,
    }


def make_biological_hierarchy(model):
    """Make the biological hierarchy of the model."""
    technical_factors = model.adata.uns["factor_obs"][
        model.adata.uns["factor_obs"]["technical"]
    ].index.tolist()  # layer 0
    biological_hierarchy = get_hierarchy(model, drop_factors=technical_factors)
    model.adata.uns["biological_hierarchy"] = biological_hierarchy
    return biological_hierarchy


def make_technical_hierarchy(model):
    """Make the technical hierarchy of the model."""
    technical_factors = model.adata.uns["factor_obs"][
        model.adata.uns["factor_obs"]["technical"]
    ][
        model.adata.uns["factor_obs"]["child_layer"] == "L0"
    ].index.tolist()  # only layer 0 factors
    # technical hierarchy is a root with all technical factors as direct children.
    # connection weights are proportional to the usage of each factor
    technical_hierarchy = dict()
    technical_hierarchy["tech_top"] = technical_factors
    model.adata.uns["technical_hierarchy"] = technical_hierarchy
    return technical_hierarchy


def make_hierarchies(model):
    """Store the biological and technical hierarchies of the model."""
    make_biological_hierarchy(model)
    make_technical_hierarchy(model)


def get_hierarchy(
    model,
    simplified: Optional[bool] = True,
    drop_factors: Optional[Sequence[str]] = None,
) -> Mapping[str, Sequence[str]]:
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
