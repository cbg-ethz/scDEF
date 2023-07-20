from collections import ChainMap
from .score_utils import *


def get_hierarchy_from_clusters(clusters_levels, level_prefix="h"):
    # clusters_levels is a list of cell to cluster assignments in decreasing resolution
    # we will assign each cluster from higher resolution a cluster with lower resolution
    # by checking the cell assignment overlaps
    hierarchy = dict()
    level_prefixes = [level_prefix * level for level in range(len(clusters_levels))]
    for level, clusters in enumerate(clusters_levels[:-1]):
        clusters_up = clusters_levels[level + 1]
        unique_clusters = np.unique(clusters)
        unique_clusters_up = np.unique(clusters_up)
        for i, cluster in enumerate(unique_clusters):
            name = f"{level_prefixes[level]}{i}"
            # Get all cells attached to this cluster
            cells_a = np.where(np.array(clusters) == cluster)[0]
            scores = []
            for j, cluster_up in enumerate(unique_clusters_up):
                cells_b = np.where(np.array(clusters_up) == cluster_up)[0]

                scores.append(jaccard_similarity(cells_a, cells_b))
            upper_level = f"{level_prefixes[level+1]}{np.argmax(scores)}"
            if upper_level in hierarchy:
                hierarchy[upper_level].append(name)
            else:
                hierarchy[upper_level] = []
                hierarchy[upper_level].append(name)

    return hierarchy


def flatten_hierarchy(hierarchy_dict):
    def descend(factor):
        node = [factor]
        if factor in hierarchy_dict:
            children = list(hierarchy_dict[factor])
            for child in children:
                nodes = descend(child)
                node.extend(nodes)
        return node

    flattened = {}
    for factor in hierarchy_dict:
        flattened[factor] = descend(factor)[1:]

    return flattened


def get_nodes_from_hierarchy(hierarchy):
    hierarchy_nodes = None
    hierarchy_nodes = list(hierarchy.keys())
    vals = [item for sublist in list(hierarchy.values()) for item in sublist]
    hierarchy_nodes.extend(vals)
    hierarchy_nodes = list(set(hierarchy_nodes))  # remove duplicates
    return hierarchy_nodes


# Annotate hierarchy
def annotate_hierarchy(hierarchy, factor_annotation_matches):
    annotated_hierarchy = {}
    for hrc in hierarchy:
        factors = hierarchy[hrc]
        factor_annots = []
        for factor in factors:
            if factor in factor_annotation_matches:
                factor_annots.append(factor_annotation_matches[factor])
            else:
                factor_annots.append(factor)
        if hrc in factor_annotation_matches:
            annotated_hierarchy[factor_annotation_matches[hrc]] = factor_annots
        else:
            annotated_hierarchy[hrc] = factor_annots
    return annotated_hierarchy


# Compare hierarchies
def compare_hierarchies(inf_complete, true_complete):
    # Need the hierarchies to have all types
    flattened_inferred = flatten_hierarchy(inf_complete)
    flattened_true = flatten_hierarchy(true_complete)
    levels = list(flattened_true.keys())

    scores = []
    for hrc in levels:
        true_below = flattened_true[hrc]
        true_not_below = []
        for h in levels:
            if h != hrc and h not in true_below:
                true_not_below.append(h)
                for down in flattened_true[h]:
                    true_not_below.append(down)
        true_not_below = list(set(true_not_below))

        # Get inferred set of types below hrc
        inf_below = flattened_inferred[hrc]

        # true positive: marker for this type is in signature
        # true negative: marker for another type is not in signature
        tp = sum([1 for grp in inf_below if grp in true_below])
        fp = sum([1 for grp in inf_below if grp in true_not_below])
        fn = sum([1 for grp in true_below if grp not in inf_below])
        if fp == 0 and len(true_below) == 0:
            # the inf hierarchy does not disagree with the true but it also doesn't have TP because the true hierarchy is already flat
            score = 1
        else:
            score = compute_fscore(tp, fp, fn)

        scores.append(score)

    return np.mean(scores)


def simplify_hierarchy(hierarchy, layer_names, layer_sizes):
    simplified = hierarchy.copy()
    n_layers = len(layer_names)
    for layer_idx in range(1, n_layers):
        layer_name = layer_names[layer_idx]
        for factor_idx in range(layer_sizes[layer_idx]):
            factor_name = f"{layer_name}{factor_idx}"
            if factor_name in simplified:
                if len(hierarchy[factor_name]) == 1:
                    if hierarchy[factor_name][0] in hierarchy:
                        down_hrc = hierarchy[hierarchy[factor_name][0]]
                        if len(down_hrc) > 1:
                            simplified[factor_name] = down_hrc
                            del simplified[hierarchy[factor_name][0]]
                        else:
                            if layer_idx < n_layers - 1:
                                del simplified[factor_name]
                            else:
                                simplified[factor_name] = []
                    else:
                        del simplified[factor_name]

    return simplified


def get_cluster_obs_association_scores(adata, clusters, layer_names, obs_key, obs_val):
    scores = []
    factors = []
    layers = []
    n_layers = len(clusters)
    for layer_idx in range(n_layers):
        layer_name = layer_names[layer_idx]
        n_clusters = len(np.unique(clusters[layer_idx]))
        cluster_assignments = clusters[layer_idx]
        for factor_idx in range(n_clusters):
            cluster_name = f"{layer_name}{factor_idx}"
            score = compute_cluster_obs_association_score(
                adata, cluster_assignments, cluster_name, obs_key, obs_val
            )
            scores.append(score)
            factors.append(cluster_name)
            layers.append(layer_idx)
    return scores, factors, layers


def assign_obs_to_clusters(
    adata, clusters_levels, layer_names, obs_keys, cluster_names=None
):
    if not isinstance(obs_keys, list):
        obs_keys = [obs_keys]

    obs_to_factor_assignments = []
    obs_to_factor_matches = []
    for obs_key in obs_keys:
        obskey_to_factor_assignments = dict()
        obskey_to_factor_matches = dict()
        for obs in adata.obs[obs_key].unique():
            scores, factors, layers = get_cluster_obs_association_scores(
                adata, clusters_levels, layer_names, obs_key, obs
            )
            if cluster_names is not None:
                # Subset to factor_names
                idx = np.array(
                    [i for i, factor in enumerate(factors) if factor in cluster_names]
                )
                scores = np.array(scores)[idx]
                factors = np.array(factors)[idx]
                layers = np.array(layers)[idx]
            obskey_to_factor_assignments[obs] = factors[np.argmax(scores)]
            obskey_to_factor_matches[factors[np.argmax(scores)]] = obs
        obs_to_factor_assignments.append(obskey_to_factor_assignments)
        obs_to_factor_matches.append(obskey_to_factor_matches)

    # Join them all up
    factor_annotation_assignments = ChainMap(*obs_to_factor_assignments)
    factor_annotation_matches = ChainMap(*obs_to_factor_matches)

    return dict(factor_annotation_assignments), dict(factor_annotation_matches)


def complete_hierarchy(hierarchy, obs_vals):
    new_hierarchy = hierarchy.copy()
    for obs_val in obs_vals:
        if obs_val not in hierarchy:
            new_hierarchy[obs_val] = []
    return new_hierarchy
