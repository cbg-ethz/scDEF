"""Data preparation utilities for scDEF.

This module contains helper functions for data preparation and processing
that are used by the scDEF model.
"""

import numpy as np
import pandas as pd
import scanpy as sc
import decoupler
from scipy.cluster.hierarchy import ward, leaves_list
from scipy.spatial.distance import pdist
from typing import Optional, Sequence, Mapping
from . import score_utils


def get_assignment_fracs(model, obs_key, obs_vals):
    """Get assignment fractions for observations and factors."""
    signatures_dict = model.get_signatures_dict()
    n_obs = len(obs_vals)
    mats = [
        np.zeros((n_obs, len(model.factor_names[idx]))) for idx in range(model.n_layers)
    ]
    for i, obs in enumerate(obs_vals):
        from .factor_utils import get_factor_obs_assignment_fracs

        scores, factors, layers = get_factor_obs_assignment_fracs(model, obs_key, obs)
        for j in range(model.n_layers):
            indices = np.where(np.array(layers) == j)[0]
            mats[j][i] = np.array(scores)[indices]
    return mats


def get_assignment_scores(model, obs_key, obs_vals):
    """Get assignment scores for observations and factors."""
    signatures_dict = model.get_signatures_dict()
    n_obs = len(obs_vals)
    mats = [
        np.zeros((n_obs, len(model.factor_names[idx]))) for idx in range(model.n_layers)
    ]
    for i, obs in enumerate(obs_vals):
        from .factor_utils import get_factor_obs_association_scores

        scores, factors, layers = get_factor_obs_association_scores(model, obs_key, obs)
        for j in range(model.n_layers):
            indices = np.where(np.array(layers) == j)[0]
            mats[j][i] = np.array(scores)[indices]
    return mats


def get_weight_scores(model, obs_key, obs_vals):
    """Get weight scores for observations and factors."""
    signatures_dict = model.get_signatures_dict()
    n_obs = len(obs_vals)
    mats = [
        np.zeros((n_obs, len(model.factor_names[idx]))) for idx in range(model.n_layers)
    ]
    for i, obs in enumerate(obs_vals):
        from .factor_utils import get_factor_obs_weight_scores

        scores, factors, layers = get_factor_obs_weight_scores(model, obs_key, obs)
        for j in range(model.n_layers):
            indices = np.where(np.array(layers) == j)[0]
            mats[j][i] = np.array(scores)[indices]
    return mats


def get_signature_scores(model, obs_key, obs_vals, markers, top_genes=10):
    """Get signature scores for observations and factors."""
    signatures_dict = model.get_signatures_dict()
    n_obs = len(obs_vals)
    mats = [
        np.zeros((n_obs, len(model.factor_names[idx]))) for idx in range(model.n_layers)
    ]
    for i, obs in enumerate(obs_vals):
        markers_type = markers[obs]
        nonmarkers_type = [m for m in markers if m not in markers_type]
        for layer_idx in range(model.n_layers):
            for j, factor_name in enumerate(model.factor_names[layer_idx]):
                signature = signatures_dict[factor_name][:top_genes]
                mats[layer_idx][i, j] = score_utils.score_signature(
                    signature, markers_type, nonmarkers_type
                )
    return mats


def prepare_obs_factor_scores(
    model, obs_keys, get_scores_func, hierarchy=None, **kwargs
):
    """Prepare observation-factor scores for plotting."""
    if not isinstance(obs_keys, list):
        obs_keys = [obs_keys]

    factors = [model.factor_names[idx] for idx in range(model.n_layers)]
    flat_list = [item for sublist in factors for item in sublist]
    n_factors = len(flat_list)

    obs_mats = dict()
    obs_joined_mats = dict()
    obs_clusters = dict()
    obs_vals_dict = dict()
    for idx, obs_key in enumerate(obs_keys):
        obs_vals = model.adata.obs[obs_key].unique().tolist()

        # Don't keep non-hierarchical levels
        if idx > 0 and hierarchy is not None:
            obs_vals = [val for val in obs_vals if len(hierarchy[val]) > 0]

        obs_vals_dict[obs_key] = obs_vals
        n_obs = len(obs_vals)

        mats = get_scores_func(model, obs_key, obs_vals, **kwargs)

        if np.max(mats[-1]) > 1.0:
            for i in range(len(mats)):
                mats[i] = mats[i] / np.max(mats[i])

        # Cluster rows across columns in all mats
        joined_mats = np.hstack(mats)
        Z = ward(pdist(joined_mats))
        hclust_index = leaves_list(Z)

        obs_mats[obs_key] = mats
        obs_joined_mats[obs_key] = joined_mats
        obs_clusters[obs_key] = hclust_index
    return obs_mats, obs_clusters, obs_vals_dict


def prepare_pathway_factor_scores(
    model,
    pathways,
    top_genes=20,
    source="source",
    target="target",
    score="Combined score",
    z_score=True,
):
    """Prepare pathway-factor scores for plotting."""
    factors = [model.factor_names[idx] for idx in range(model.n_layers)]
    flat_list = [item for sublist in factors for item in sublist]
    n_factors = len(flat_list)

    obs_mats = dict()
    obs_joined_mats = dict()
    obs_clusters = dict()
    obs_vals_dict = dict()
    obs_vals_dict["Pathway"] = pathways[source].unique().tolist()

    n_pathways = len(obs_vals_dict["Pathway"])

    mats = []
    for layer in range(len(model.factor_names)):
        _n_factors = len(model.factor_names[layer])
        factor_vals = np.zeros((n_pathways, _n_factors))
        for i, factor in enumerate(model.factor_names[layer]):
            df = sc.get.rank_genes_groups_df(
                model.adata,
                group=factor,
                key=f"{model.layer_names[layer]}_signatures",
            )
            df = df.set_index("names")
            df = df.iloc[:top_genes]
            res = decoupler.get_ora_df(
                df, net=pathways, source=source, target=target, verbose=False
            )
            for term in res["Term"]:
                term_idx = np.where(np.array(obs_vals_dict["Pathway"]) == term)[0]
                factor_vals[term_idx, i] = res.loc[res["Term"] == term][score].values[0]

            if z_score:
                # Compute z-scores
                den = np.std(factor_vals, axis=0)
                den[den == 0] = 1e6
                factor_vals = (factor_vals - np.mean(factor_vals, axis=0)) / den[
                    None, :
                ]
        mats.append(factor_vals)

    # Cluster rows across columns in all mats
    joined_mats = np.hstack(mats)
    Z = ward(pdist(joined_mats))
    hclust_index = leaves_list(Z)

    obs_mats["Pathway"] = mats
    obs_joined_mats["Pathway"] = joined_mats
    obs_clusters["Pathway"] = hclust_index
    return obs_mats, obs_clusters, obs_vals_dict, joined_mats
