"""Factor analysis utilities for scDEF.

This module contains functions for analyzing factor-observation associations
and other factor-related computations.
"""

import numpy as np
import scipy.stats
from .data_utils import get_weight_scores


def compute_factor_obs_association_score(
    model, layer_idx, factor_name, obs_key, obs_val
):
    """Compute association score between a factor and observation value."""
    layer_name = model.layer_names[layer_idx]

    # Cells attached to factor
    adata_cells_in_factor = model.adata[
        np.where(model.adata.obs[f"{layer_name}"] == factor_name)[0]
    ]

    # Cells from obs_val
    adata_cells_from_obs = model.adata[np.where(model.adata.obs[obs_key] == obs_val)[0]]

    cells_from_obs = float(adata_cells_from_obs.shape[0])

    # Number of cells from obs_val that are not in factor
    cells_not_in_factor_from_obs = float(
        np.count_nonzero(adata_cells_from_obs.obs[f"{layer_name}"] != factor_name)
    )

    # Number of cells in factor that are obs_val
    cells_in_factor_from_obs = float(
        np.count_nonzero(adata_cells_in_factor.obs[obs_key] == obs_val)
    )

    # Number of cells in factor that are not obs_val
    cells_in_factor_not_from_obs = float(
        np.count_nonzero(adata_cells_in_factor.obs[obs_key] != obs_val)
    )

    from .score_utils import compute_fscore

    return compute_fscore(
        cells_in_factor_from_obs,
        cells_in_factor_not_from_obs,
        cells_not_in_factor_from_obs,
    )


def get_factor_obs_association_scores(model, obs_key, obs_val):
    """Get association scores for all factors and a given observation value."""
    scores = []
    factors = []
    layers = []
    for layer_idx in range(model.n_layers):
        n_factors = len(model.factor_lists[layer_idx])
        for factor in range(n_factors):
            factor_name = model.factor_names[layer_idx][factor]
            score = compute_factor_obs_association_score(
                model, layer_idx, factor_name, obs_key, obs_val
            )
            scores.append(score)
            factors.append(factor_name)
            layers.append(layer_idx)
    return scores, factors, layers


def compute_factor_obs_assignment_fracs(
    model, layer_idx, factor_name, obs_key, obs_val, total=False
):
    """Compute assignment fraction for a factor and observation value."""
    layer_name = model.layer_names[layer_idx]

    # Cells attached to factor
    adata_cells_in_factor = model.adata[
        np.where(model.adata.obs[f"{layer_name}"] == factor_name)[0]
    ]

    # Cells in factor
    cells_in_factor = float(adata_cells_in_factor.shape[0])

    # Cells from factor in obs
    cells_in_factor_from_obs = float(
        np.count_nonzero(adata_cells_in_factor.obs[obs_key] == obs_val)
    )

    score = 0.0
    score = cells_in_factor_from_obs
    if cells_in_factor != 0 and not total:
        score = cells_in_factor_from_obs / cells_in_factor

    return score


def get_factor_obs_assignment_fracs(model, obs_key, obs_val, total=False):
    """Get assignment fractions for all factors and a given observation value."""
    scores = []
    factors = []
    layers = []
    for layer_idx in range(model.n_layers):
        n_factors = len(model.factor_lists[layer_idx])
        for factor in range(n_factors):
            factor_name = model.factor_names[layer_idx][factor]
            score = compute_factor_obs_assignment_fracs(
                model, layer_idx, factor_name, obs_key, obs_val, total=total
            )
            scores.append(score)
            factors.append(factor_name)
            layers.append(layer_idx)
    return scores, factors, layers


def _compute_factor_obs_weight_score(model, layer_idx, factor_name, obs_key, obs_val):
    """Compute weight score for a factor and observation value."""
    layer_name = model.layer_names[layer_idx]

    # Cells from obs_val
    adata_cells_from_obs = model.adata[np.where(model.adata.obs[obs_key] == obs_val)[0]]
    adata_cells_not_from_obs = model.adata[
        np.where(model.adata.obs[obs_key] != obs_val)[0]
    ]

    # Weight of cells from obs in factor
    avg_in = np.mean(adata_cells_from_obs.obs[f"{factor_name}_score"])

    # Weight of cells not from obs in factor
    avg_out = np.mean(adata_cells_not_from_obs.obs[f"{factor_name}_score"])

    score = avg_in / np.sum(avg_in + avg_out)

    return score

def compute_factor_obs_weight_score(model, layer_idx, factor_name, obs_key, obs_val, eps=1e-8):
    """
    Compute a soft F1-like association score between a factor and an obs category,
    using normalized Z scores in the given layer.

    - Precision: fraction of factor mass in this obs category.
    - Recall:    average membership of cells in this obs category for this factor.
    - Score:     soft F1 = 2 * prec * rec / (prec + rec).
    """
    layer_name = model.layer_names[layer_idx]

    # Get Z for this layer: shape (n_cells, n_factors_layer)
    Z = np.asarray(model.adata.obsm[f"X_{layer_name}"], dtype=float)

    # Normalize per cell to get soft memberships p_{nk}
    Z_norm = Z / (Z.sum(axis=1, keepdims=True) + eps)

    # Find index of the factor in this layer
    factor_names = model.factor_names[layer_idx]
    try:
        factor_idx = factor_names.index(factor_name)
    except ValueError:
        raise ValueError(f"Factor name {factor_name} not found in layer {layer_name}")

    # Memberships for this factor across cells
    p = Z_norm[:, factor_idx]

    # Boolean mask for the obs category
    obs_vals = model.adata.obs[obs_key].values
    mask = (obs_vals == obs_val)

    if mask.sum() == 0:
        # No cells of this category
        return np.nan

    # Precision: how much of the factor's total mass lies in this obs category
    prec = p[mask].sum() / (p.sum() + eps)

    # Recall: average membership of cells in this obs category
    rec = p[mask].mean()

    # Soft F1
    score = 2.0 * prec * rec / (prec + rec + eps)

    return float(score)



def get_factor_obs_weight_scores(model, obs_key, obs_val):
    """Get weight scores for all factors and a given observation value."""
    scores = []
    factors = []
    layers = []
    for layer_idx in range(model.n_layers):
        n_factors = len(model.factor_lists[layer_idx])
        for factor in range(n_factors):
            factor_name = model.factor_names[layer_idx][factor]
            score = compute_factor_obs_weight_score(
                model, layer_idx, factor_name, obs_key, obs_val
            )
            scores.append(score)
            factors.append(factor_name)
            layers.append(layer_idx)
    return scores, factors, layers


def compute_factor_obs_entropies(model, obs_key):
    """Compute entropies for factor-observation associations."""
    mats = get_weight_scores(model, obs_key, model.adata.obs[obs_key].unique())
    mat = np.concatenate(mats, axis=1)
    factors = [model.factor_names[idx] for idx in range(model.n_layers)]
    flat_list = [item for sublist in factors for item in sublist]
    entropies = scipy.stats.entropy(mat, axis=0)
    return dict(zip(flat_list, entropies))


def assign_obs_to_factors(model, obs_keys, factor_names=[]):
    """Assign observations to factors based on association scores."""
    if not isinstance(obs_keys, list):
        obs_keys = [obs_keys]

    # Sort obs_keys from broad to specific
    sizes = [len(model.adata.obs[obs_key].unique()) for obs_key in obs_keys]
    obs_keys = np.array(obs_keys)[np.argsort(sizes)].tolist()

    obs_to_factor_assignments = []
    obs_to_factor_matches = []
    for obs_key in obs_keys:
        obskey_to_factor_assignments = dict()
        obskey_to_factor_matches = dict()
        for obs in model.adata.obs[obs_key].unique():
            scores, factors, layers = get_factor_obs_association_scores(
                model, obs_key, obs
            )
            if len(factor_names) > 0:
                # Subset to factor_names
                idx = np.array(
                    [i for i, factor in enumerate(factors) if factor in factor_names]
                )
                scores = np.array(scores)[idx]
                factors = np.array(factors)[idx]
                layers = np.array(layers)[idx]
            obskey_to_factor_assignments[obs] = factors[np.argmax(scores)]
            obskey_to_factor_matches[factors[np.argmax(scores)]] = obs
        obs_to_factor_assignments.append(obskey_to_factor_assignments)
        obs_to_factor_matches.append(obskey_to_factor_matches)

    # Join them all up
    from collections import ChainMap

    factor_annotation_assignments = ChainMap(*obs_to_factor_assignments)
    factor_annotation_matches = ChainMap(*obs_to_factor_matches)

    return dict(factor_annotation_assignments), dict(factor_annotation_matches)

def compute_factor_obs_correlation(model, layer_idx, factor_name, obs_key):
    """Compute correlation between a factor and an observation value."""
    layer_name = model.layer_names[layer_idx]
    Z = np.asarray(model.adata.obsm[f"X_{layer_name}"], dtype=float)
    factor_idx = model.factor_names[layer_idx].index(factor_name)
    p = Z[:, factor_idx]
    obs_vals = model.adata.obs[obs_key].values
    corr = np.corrcoef(p, obs_vals)[0, 1]
    return corr

def get_factor_obs_correlations(model, obs_key):
    """Compute correlations between factors and observation values."""
    corrs = []
    factors = []
    layers = []
    for layer_idx in range(model.n_layers):
        n_factors = len(model.factor_lists[layer_idx])
        for factor in range(n_factors):
            factor_name = model.factor_names[layer_idx][factor]
            corr = compute_factor_obs_correlation(model, layer_idx, factor_name, obs_key)
            corrs.append(corr)
            factors.append(factor_name)
            layers.append(layer_idx)
    return corrs, factors, layers


def compute_factor_hierarchy_scores(model):
    """
    For each factor in each layer except the last, compute the normalized assignment entropy
    to factors in layer i+1 using model.pmeans[f'L{i+1}W']. 
    Returns a nested dictionary: {layer_idx: {factor_name: entropy, ...}, ...}
    Entropies are normalized by the maximum theoretical entropy for that layer 
    (log2 of number of factors in layer i+1).
    """
    results = {}
    n_layers = model.n_layers

    for i in range(n_layers - 1):
        n_factors_l = len(model.factor_names[i])
        n_factors_lp1 = len(model.factor_names[i + 1])
        # model.pmeans[f'L{i+1}W']: shape = (n_factors_lp1, n_factors_l)
        weight_matrix = model.pmeans[f'L{i+1}W']
        max_entropy = np.log2(n_factors_lp1) if n_factors_lp1 > 1 else 1.0  # avoid log2(1)=0, fallback to 1

        factor_scores = {}
        for f_idx in range(n_factors_l):
            factor_name = model.factor_names[i][f_idx]
            # Get outgoing weights from factor f_idx in layer i to all in layer i+1
            outgoing_weights = weight_matrix[:, f_idx]
            if np.sum(outgoing_weights) == 0:
                entropy = np.nan
            else:
                probs = outgoing_weights / np.sum(outgoing_weights)
                probs_nonzero = probs[probs > 0]
                entropy_raw = -np.sum(probs_nonzero * np.log2(probs_nonzero))
                if max_entropy > 0:
                    entropy = entropy_raw / max_entropy
                else:
                    entropy = np.nan
            factor_scores[factor_name] = entropy
        results[i] = factor_scores
    return results