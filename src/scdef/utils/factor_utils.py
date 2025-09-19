"""Factor analysis utilities for scDEF.

This module contains functions for analyzing factor-observation associations
and other factor-related computations.
"""

import numpy as np
import scipy.stats
from typing import Optional, Sequence, Mapping
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
    model, layer_idx, factor_name, obs_key, obs_val
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
    if cells_in_factor != 0:
        score = cells_in_factor_from_obs / cells_in_factor

    return score


def get_factor_obs_assignment_fracs(model, obs_key, obs_val):
    """Get assignment fractions for all factors and a given observation value."""
    scores = []
    factors = []
    layers = []
    for layer_idx in range(model.n_layers):
        n_factors = len(model.factor_lists[layer_idx])
        for factor in range(n_factors):
            factor_name = model.factor_names[layer_idx][factor]
            score = compute_factor_obs_assignment_fracs(
                model, layer_idx, factor_name, obs_key, obs_val
            )
            scores.append(score)
            factors.append(factor_name)
            layers.append(layer_idx)
    return scores, factors, layers


def compute_factor_obs_weight_score(model, layer_idx, factor_name, obs_key, obs_val):
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
