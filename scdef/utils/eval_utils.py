import numpy as np
from ..utils import score_utils, hierarchy_utils


def evaluate_scdef_hierarchy(scd, obs_keys, true_hierarchy):
    hierarchy = scd.get_hierarchy()
    layer_sizes = [len(scd.factor_lists[i]) for i in range(scd.n_layers)]
    simplified = scd.simplify_hierarchy(hierarchy)
    assignments, matches = scd.assign_obs_to_factors(
        obs_keys, hierarchy_utils.get_nodes_from_hierarchy(simplified)
    )
    annotated = hierarchy_utils.annotate_hierarchy(simplified, matches)

    obs_vals = [
        scd.adata.obs[obs_key].astype("category").cat.categories for obs_key in obs_keys
    ]
    obs_vals = list(set([item for sublist in obs_vals for item in sublist]))

    completed_annotated = hierarchy_utils.complete_hierarchy(annotated, obs_vals)
    completed_true_hierarchy = hierarchy_utils.complete_hierarchy(
        true_hierarchy, obs_vals
    )
    return hierarchy_utils.compare_hierarchies(
        completed_annotated, completed_true_hierarchy
    )


def evaluate_hierarchy_from_cluster_levels(
    adata, obs_keys, clusters_levels, true_hierarchy
):
    hierarchy = hierarchy_utils.get_hierarchy_from_clusters(clusters_levels)
    layer_names = ["h" * level for level in range(len(clusters_levels))]
    layer_sizes = [len(np.unique(cluster)) for cluster in clusters_levels]
    simplified = hierarchy_utils.simplify_hierarchy(hierarchy, layer_names, layer_sizes)
    assignments, matches = hierarchy_utils.assign_obs_to_clusters(
        adata,
        clusters_levels,
        layer_names,
        obs_keys,
        hierarchy_utils.get_nodes_from_hierarchy(simplified),
    )
    annotated = hierarchy_utils.annotate_hierarchy(simplified, matches)

    obs_vals = [
        adata.obs[obs_key].astype("category").cat.categories for obs_key in obs_keys
    ]
    obs_vals = list(set([item for sublist in obs_vals for item in sublist]))

    completed_annotated = hierarchy_utils.complete_hierarchy(annotated, obs_vals)
    completed_true_hierarchy = hierarchy_utils.complete_hierarchy(
        true_hierarchy, obs_vals
    )
    return hierarchy_utils.compare_hierarchies(
        completed_annotated, completed_true_hierarchy
    )


def evaluate_cluster_signatures(
    adata, clusters_levels, signatures, obs_keys, markers, cluster_names=None
):
    layer_names = ["h" * level for level in range(len(clusters_levels))]

    # Assign clusters to obs
    assignments, matches = hierarchy_utils.assign_obs_to_clusters(
        adata, clusters_levels, layer_names, obs_keys, cluster_names=cluster_names
    )

    # Compare each type's inferred signature with the true signature
    signature_scores = []
    for celltype in markers:
        cluster_name = assignments[celltype]
        signature = signatures[cluster_name]
        markers_type = markers[celltype]
        nonmarkers_type = [m for m in markers if m not in markers_type]
        signature_scores.append(
            score_utils.core_signature(signature, markers_type, nonmarkers_type)
        )
    return signature_scores


def evaluate_scdef_signatures(scd, obs_keys, markers):
    # Assign factors to obs
    assignments, matches = scd.assign_obs_to_factors(obs_keys)
    signatures_dict = scd.get_signatures_dict()

    signature_scores = []
    for celltype in markers:
        factor_name = assignments[celltype]
        signature = signatures_dict[factor_name]
        markers_type = markers[celltype]
        nonmarkers_type = [m for m in markers if m not in markers_type]
        signature_scores.append(
            score_utils.score_signature(signature, markers_type, nonmarkers_type)
        )
    return signature_scores
